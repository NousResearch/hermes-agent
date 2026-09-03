#!/usr/bin/env python3
"""
MCP OAuth 2.1 Client Support

Implements the browser-based OAuth 2.1 authorization code flow with PKCE
for MCP servers that require OAuth authentication instead of static bearer
tokens.

Uses the MCP Python SDK's ``OAuthClientProvider`` (an ``httpx.Auth`` subclass)
which handles discovery, client identification, PKCE, token exchange,
refresh, and step-up authorization automatically.

Client identification follows the MCP 2026-07-28 spec: when the authorization
server advertises ``client_id_metadata_document_supported``, the SDK uses the
URL of Hermes' published Client ID Metadata Document (CIMD) as the
``client_id``; otherwise it falls back to RFC 7591 dynamic client registration,
which that spec revision deprecated.

This module provides the glue:
    - ``HermesTokenStorage``: persists tokens/client-info to disk so they
      survive across process restarts.
    - Callback server: ephemeral localhost HTTP server to capture the OAuth
      redirect with the authorization code.
    - ``build_oauth_auth()``: entry point called by ``mcp_tool.py`` that wires
      everything together and returns the ``httpx.Auth`` object.

Configuration in config.yaml::

    mcp_servers:
      my_server:
        url: "https://mcp.example.com/mcp"
        auth: oauth
        oauth:                                  # all fields optional
          client_id: "pre-registered-id"        # skip dynamic registration
          client_secret: "secret"               # confidential clients only
          scope: "read write"                   # default: server-provided
          redirect_port: 0                      # 0 = auto-pick free port
          redirect_uri: "https://proxy/callback"  # default: loopback callback
          redirect_host: "localhost"            # loopback hostname (WAF-safe)
          client_name: "My Custom Client"       # default: "Hermes Agent"
          client_metadata_url: "https://me/cimd.json"  # self-hosted CIMD
          cimd: false                           # force DCR for this server
"""

import asyncio
import contextvars
import json
import logging
import os
import re
import secrets
import socket
import stat
import sys
import threading
import time
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from hermes_constants import secure_parent_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports -- MCP SDK with OAuth support is optional
# ---------------------------------------------------------------------------

# Availability is detected WITHOUT importing the mcp SDK (which costs
# ~170 ms at module load). The actual classes are imported lazily on first
# use via _ensure_sdk_loaded(); the module-level names below are kept as
# placeholders so tests can patch them (patch.object requires the attribute
# to exist on the module).
import importlib.util as _importlib_util

_OAUTH_AVAILABLE = _importlib_util.find_spec("mcp") is not None
if not _OAUTH_AVAILABLE:
    logger.debug("MCP OAuth types not available -- OAuth MCP auth disabled")

# Lazily-bound SDK names (rebound by _ensure_sdk_loaded on first use).
# Annotated ``Any`` so quoted type annotations elsewhere in the file remain
# valid for static checkers while the runtime value starts as None.
OAuthClientProvider: Any = None
OAuthClientInformationFull: Any = None
OAuthClientMetadata: Any = None
OAuthMetadata: Any = None
OAuthToken: Any = None

# Cache of the real SDK classes so a test that temporarily patches one of the
# module-level names (and restores it to None afterwards) doesn't strand the
# module in a broken state.
_SDK_CLASSES: dict[str, Any] = {}
_SDK_LOAD_FAILED = False


def _ensure_sdk_loaded() -> bool:
    """Import the MCP SDK OAuth classes on first use and bind module globals.

    Returns True when the SDK classes are available. Module-level names that
    have been replaced (e.g. patched by tests) are left untouched; only names
    that are currently ``None`` are (re)bound to the real SDK classes.
    """
    global _SDK_LOAD_FAILED, _OAUTH_AVAILABLE
    if _SDK_LOAD_FAILED:
        return False
    if not _SDK_CLASSES:
        try:
            from mcp.client.auth import OAuthClientProvider as _Provider
            from mcp.shared.auth import (
                OAuthClientInformationFull as _InfoFull,
                OAuthClientMetadata as _ClientMeta,
                OAuthMetadata as _Meta,
                OAuthToken as _Token,
            )
        except ImportError:
            _SDK_LOAD_FAILED = True
            _OAUTH_AVAILABLE = False
            logger.debug("MCP OAuth types not available -- OAuth MCP auth disabled")
            return False
        _SDK_CLASSES.update(
            OAuthClientProvider=_Provider,
            OAuthClientInformationFull=_InfoFull,
            OAuthClientMetadata=_ClientMeta,
            OAuthMetadata=_Meta,
            OAuthToken=_Token,
        )
    g = globals()
    for _name, _cls in _SDK_CLASSES.items():
        if g.get(_name) is None:
            g[_name] = _cls
    return True

try:
    from pydantic import AnyUrl
except ImportError:
    AnyUrl = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OAuthNonInteractiveError(RuntimeError):
    """Raised when OAuth requires browser interaction in a non-interactive env."""


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Port used by the most recent build_oauth_auth() call.  Exposed so that
# tests can verify the callback server and the redirect_uri share a port.
_oauth_port: int | None = None
# Interactivity gate for OAuth stdin prompts. A ContextVar (NOT threading.local)
# is required: background MCP discovery sets this on the discovery thread, but
# the actual connect+OAuth runs on the dedicated `mcp-event-loop` thread via
# run_coroutine_threadsafe. asyncio copies the *calling context* into the
# scheduled coroutine, so a ContextVar propagates across that boundary while a
# threading.local would not — see #35927. Default True (interactive allowed).
_oauth_interactive_enabled: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "_oauth_interactive_enabled", default=True
)

# Forces _is_interactive() past the stdin-TTY check for flows driven from a
# GUI (dashboard/desktop REST): the browser + localhost callback server do all
# the work there, and the stdin paste fallback degrades harmlessly (EOF is
# swallowed by _paste_callback_reader). Suppression still wins — background
# discovery must never start a browser flow.
_oauth_interactive_forced: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "_oauth_interactive_forced", default=False
)


# Skip tokens accepted at the paste prompt — exit OAuth without auth.
_SKIP_TOKENS = frozenset({"skip", "cancel", "s", "n", "no", "q", "quit"})

# Sentinel value written to result["error"] when the user skipped via stdin.
# _wait_for_callback maps this to OAuthNonInteractiveError ("user_skipped")
# so the MCP setup path treats it as a non-fatal "continue without this
# server" rather than a hard failure.
_USER_SKIPPED_SENTINEL = "__hermes_user_skipped__"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_token_dir(hermes_home: str | Path | None = None) -> Path:
    """Return the directory for MCP OAuth token files.

    Uses HERMES_HOME so each profile gets its own OAuth tokens.
    Layout: ``HERMES_HOME/mcp-tokens/``
    """
    from hermes_constants import get_hermes_home

    base = Path(hermes_home) if hermes_home is not None else Path(get_hermes_home())
    return base / "mcp-tokens"


def _safe_filename(name: str) -> str:
    """Sanitize a server name for use as a filename (no path separators)."""
    return re.sub(r"[^\w\-]", "_", name).strip("_")[:128] or "default"


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Bound-but-not-listening sockets reserved for pending OAuth callback flows,
# keyed by port. Holding the socket from port-selection time until
# _wait_for_callback adopts it closes the TOCTOU window where another process
# could grab the port between _find_free_port() closing its probe socket and
# HTTPServer binding minutes later (#22161). Bounded FIFO so repeated
# build_oauth_auth calls (reconnect loops) cannot leak fds.
_reserved_sockets: "dict[int, socket.socket]" = {}
_MAX_RESERVED_SOCKETS = 8


def _park_reserved_socket(port: int, sock: socket.socket) -> None:
    """Hold *sock* bound to *port* until ``_wait_for_callback`` adopts it.

    Pinned CIMD sockets are never evicted: the published metadata document
    only declares the pinned ports, so losing one mid-flow silently converts
    a pinned reservation back into a stealable window — the exact race the
    parking exists to prevent (#22161). The FIFO cap applies to ephemeral
    reservations only; the pinned range is already bounded by ``_CIMD_PORTS``.
    """
    # Evict oldest ephemeral reservations past the cap (dict preserves
    # insertion order).
    while len(_reserved_sockets) >= _MAX_RESERVED_SOCKETS:
        stale_port = next(
            (p for p in _reserved_sockets if p not in _CIMD_PORTS), None
        )
        if stale_port is None:
            break  # only pinned sockets remain — never evict those
        stale = _reserved_sockets.pop(stale_port, None)
        if stale is None:
            continue
        try:
            stale.close()
        except OSError:
            pass
    _reserved_sockets[port] = sock


def _reserve_callback_port() -> int:
    """Pick an ephemeral callback port and keep its socket bound.

    Returns the port. The bound (not yet listening) socket is parked in
    ``_reserved_sockets`` so no other process can bind the port before
    ``_wait_for_callback`` adopts it. Adoption (or ``server_close``) owns
    the socket's lifetime from there.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
    except OSError:
        s.close()
        raise
    port = s.getsockname()[1]
    _park_reserved_socket(port, s)
    return port


def _cached_redirect_port(storage: "HermesTokenStorage | None") -> int | None:
    """Return the loopback callback port from cached client registration.

    OAuth providers bind a dynamically-registered ``client_id`` to the exact
    redirect URI that was registered with it. If Hermes restarts and chooses a
    new random callback port while reusing the stored ``client_id``, providers
    such as Summ reject the authorization request with ``redirect_uri does not
    match any registered URIs``. Reusing the cached redirect port keeps the
    authorization request consistent with the stored client registration.
    """
    if storage is None:
        return None

    try:
        data = _read_json(storage._client_info_path())
    except (AttributeError, TypeError, ValueError):
        return None
    if not data:
        return None

    for uri in data.get("redirect_uris") or []:
        try:
            parsed = urlparse(str(uri))
        except (TypeError, ValueError):
            continue
        if (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.path == "/callback"
            and parsed.port is not None
        ):
            return int(parsed.port)
    return None


def _cached_redirect_uri(storage: "HermesTokenStorage | None") -> str | None:
    """Return a cached non-loopback redirect URI, if one was registered."""
    if storage is None:
        return None
    try:
        data = _read_json(storage._client_info_path())
    except (AttributeError, TypeError, ValueError):
        return None
    for uri in (data or {}).get("redirect_uris") or []:
        try:
            parsed = urlparse(str(uri))
        except (TypeError, ValueError):
            continue
        if parsed.scheme == "https" and parsed.netloc:
            return str(uri)
    return None


def _is_interactive() -> bool:
    """Return True if we can reasonably expect to interact with a user."""
    if not _oauth_interactive_enabled.get():
        return False
    if _oauth_interactive_forced.get():
        return True
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _raise_if_non_interactive(lead: str) -> None:
    """Raise ``OAuthNonInteractiveError`` unless an interactive session exists.

    ``lead`` is the boundary-specific first sentence; this helper appends the
    shared, actionable ``hermes mcp login`` next-step so the guidance wording
    lives in one place across every non-interactive OAuth boundary (#57836).
    """
    if not _is_interactive():
        raise OAuthNonInteractiveError(
            f"{lead} "
            "Run `hermes mcp login <server>` interactively to (re)authorize, "
            "then restart or reload the gateway."
        )


@contextmanager
def force_interactive_oauth():
    """Treat the current execution context as interactive despite no TTY.

    For GUI-driven auth (dashboard/desktop REST endpoint): the user IS present
    — just not on stdin. Opens the browser + localhost callback flow that the
    TTY heuristic would otherwise refuse. Same ContextVar propagation story as
    suppress_interactive_oauth() (#35927).
    """
    token = _oauth_interactive_forced.set(True)
    try:
        yield
    finally:
        _oauth_interactive_forced.reset(token)


@contextmanager
def suppress_interactive_oauth():
    """Disable stdin-based OAuth prompts for the current execution context.

    Uses a ContextVar so the suppression propagates from a background-discovery
    thread onto the coroutine scheduled (via run_coroutine_threadsafe) on the
    dedicated MCP event-loop thread — where the OAuth callback actually runs
    (#35927). A threading.local would not cross that thread boundary.
    """
    token = _oauth_interactive_enabled.set(False)
    try:
        yield
    finally:
        _oauth_interactive_enabled.reset(token)


def _can_open_browser() -> bool:
    """Return True if opening a browser is likely to work."""
    # Explicit SSH session → no local display
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        return False
    # macOS and Windows usually have a display
    if os.name == "nt":
        return True
    try:
        if os.uname().sysname == "Darwin":
            return True
    except AttributeError:
        pass
    # Linux/other posix: need DISPLAY or WAYLAND_DISPLAY
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None if it doesn't exist or is invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _write_json(path: Path, data: dict) -> None:
    """Write a dict as JSON with restricted permissions (0o600).

    Uses ``os.open`` with ``O_EXCL`` and an explicit mode so the file is
    created atomically at 0o600. The previous ``write_text`` + post-write
    ``chmod`` opened a TOCTOU window where the temp file briefly inherited
    the process umask (commonly 0o644 = world-readable), exposing OAuth
    tokens to other local users between create and chmod. Mirrors the fix
    in ``agent/google_oauth.py`` (#19673).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tighten parent dir to 0o700 so siblings can't traverse to the creds.
    # No-op on Windows (POSIX mode bits aren't enforced); ignore failures.
    # secure_parent_dir refuses to chmod /, top-level dirs, or the
    # hermes-agent install tree (#25821, #93050).
    secure_parent_dir(path)
    # Per-process random suffix avoids collisions between concurrent
    # writers and stale leftovers from a prior crashed write.
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(
            str(tmp),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# HermesTokenStorage -- persistent token/client-info on disk
# ---------------------------------------------------------------------------


class HermesTokenStorage:
    """Persist OAuth tokens and client registration to JSON files.

    File layout::

        HERMES_HOME/mcp-tokens/<server_name>.json         -- tokens
        HERMES_HOME/mcp-tokens/<server_name>.client.json   -- client info
        HERMES_HOME/mcp-tokens/<server_name>.meta.json     -- oauth server metadata
        HERMES_HOME/mcp-tokens/<server_name>.cimd-off      -- CIMD refused here
    """

    def __init__(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
        configured_scope: str | None = None,
    ):
        self._server_name = _safe_filename(server_name)
        self._hermes_home = Path(hermes_home) if hermes_home is not None else None
        self._configured_scope = configured_scope

    def _tokens_path(self) -> Path:
        return _get_token_dir(self._hermes_home) / f"{self._server_name}.json"

    def _client_info_path(self) -> Path:
        return _get_token_dir(self._hermes_home) / f"{self._server_name}.client.json"

    def _meta_path(self) -> Path:
        return _get_token_dir(self._hermes_home) / f"{self._server_name}.meta.json"

    def _cimd_rejected_path(self) -> Path:
        return _get_token_dir(self._hermes_home) / f"{self._server_name}.cimd-off"

    # -- tokens ------------------------------------------------------------

    async def get_tokens(self) -> "OAuthToken | None":
        data = _read_json(self._tokens_path())
        if data is None:
            return None
        if OAuthToken is None and not _ensure_sdk_loaded():
            return None
        # Hermes records an absolute wall-clock ``expires_at`` alongside the
        # SDK's serialized token (see ``set_tokens``). On read we rewrite
        # ``expires_in`` to the remaining seconds so the SDK's downstream
        # ``update_token_expiry`` computes the correct absolute time and
        # ``is_token_valid()`` correctly reports False for tokens that
        # expired while the process was down.
        #
        # Legacy token files (pre-Fix-A) have ``expires_in`` but no
        # ``expires_at``. We fall back to the file's mtime as a best-effort
        # wall-clock proxy for when the token was written: if (mtime +
        # expires_in) is in the past, clamp ``expires_in`` to zero so the
        # SDK refreshes before the first request. This self-heals one-time
        # on the next successful ``set_tokens``, which writes the new
        # ``expires_at`` field. The stored ``expires_at`` is stripped before
        # model_validate because it's not part of the SDK's OAuthToken schema.
        absolute_expiry = data.pop("expires_at", None)
        if absolute_expiry is not None:
            data["expires_in"] = int(max(absolute_expiry - time.time(), 0))
        elif data.get("expires_in") is not None:
            try:
                file_mtime = self._tokens_path().stat().st_mtime
            except OSError:
                file_mtime = None
            if file_mtime is not None:
                try:
                    implied_expiry = file_mtime + int(data["expires_in"])
                    data["expires_in"] = int(max(implied_expiry - time.time(), 0))
                except (TypeError, ValueError):
                    pass
        try:
            return OAuthToken.model_validate(data)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Corrupt tokens at %s -- ignoring: %s", self._tokens_path(), exc)
            return None

    async def set_tokens(self, tokens: "OAuthToken") -> None:
        payload = tokens.model_dump(mode="json", exclude_none=True)
        # Persist an absolute ``expires_at`` so a process restart can
        # reconstruct the correct remaining TTL. Without this the MCP SDK's
        # ``_initialize`` reloads a relative ``expires_in`` which has no
        # wall-clock reference, leaving ``context.token_expiry_time=None``
        # and ``is_token_valid()`` falsely reporting True. See Fix A in
        # ``mcp-oauth-token-diagnosis`` skill + Claude Code's
        # ``OAuthTokens.expiresAt`` persistence (auth.ts ~180).
        expires_in = payload.get("expires_in")
        if expires_in is not None:
            try:
                payload["expires_at"] = time.time() + int(expires_in)
            except (TypeError, ValueError):
                # Mock tokens or unusual shapes: skip the expires_at write
                # rather than fail persistence.
                pass
        _write_json(self._tokens_path(), payload)
        self._warn_if_scope_not_honoured(payload.get("scope"))
        logger.debug("OAuth tokens saved for %s", self._server_name)

    def _warn_if_scope_not_honoured(self, granted_scope: str | None) -> None:
        """Warn when the granted token exceeds the configured ``oauth.scope``.

        The MCP SDK unconditionally overwrites ``client_metadata.scope`` with
        every scope advertised by the server (WWW-Authenticate scope, then PRM
        ``scopes_supported``, then AS ``scopes_supported``) before client
        registration, so a configured narrow scope is silently discarded and
        the client requests — and is granted — maximum privilege
        (modelcontextprotocol/python-sdk#2317). Until that lands upstream,
        this is the only Hermes-side signal that the knob did nothing.

        ``offline_access`` is excluded from the comparison: the SDK appends it
        itself (SEP-2207) whenever the AS supports it and the client has the
        ``refresh_token`` grant, which Hermes always requests.
        """
        if not self._configured_scope or not granted_scope:
            return
        try:
            configured = set(self._configured_scope.split())
            granted = set(granted_scope.split())
        except AttributeError:
            return
        extras = granted - configured - {"offline_access"}
        if not extras:
            return
        logger.warning(
            "MCP OAuth '%s': configured oauth.scope=%r was NOT honoured — the "
            "granted token additionally carries %r (full grant: %r). The MCP "
            "SDK replaces the configured scope with every scope the server "
            "advertises before registration (modelcontextprotocol/python-sdk#2317). "
            "Check the consent screen actually shown; the token may hold more "
            "privilege than configured.",
            self._server_name,
            self._configured_scope,
            " ".join(sorted(extras)),
            granted_scope,
        )

    # -- client info -------------------------------------------------------

    async def get_client_info(self) -> "OAuthClientInformationFull | None":
        data = _read_json(self._client_info_path())
        if data is None:
            return None
        if OAuthClientInformationFull is None and not _ensure_sdk_loaded():
            return None
        try:
            info = OAuthClientInformationFull.model_validate(data)
            # Some dynamic registration providers (notably Supabase MCP) return
            # a client_secret but omit token_endpoint_auth_method. The MCP SDK
            # defaults that missing field to "none", which causes token exchange
            # to omit client_secret and fail with "Required parameter: client_secret".
            # If a secret is present, use client_secret_post unless the provider
            # explicitly saved a different method.
            if getattr(info, "client_secret", None) and data.get("token_endpoint_auth_method") in (None, "none", ""):
                data["token_endpoint_auth_method"] = "client_secret_post"
                info = OAuthClientInformationFull.model_validate(data)
                _write_json(self._client_info_path(), info.model_dump(mode="json", exclude_none=True))
            return info
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Corrupt client info at %s -- ignoring: %s", self._client_info_path(), exc)
            return None

    async def set_client_info(self, client_info: "OAuthClientInformationFull") -> None:
        data = client_info.model_dump(mode="json", exclude_none=True)
        # Supabase MCP dynamic client registration returns a client_secret but
        # omits token_endpoint_auth_method. The MCP SDK defaults that to
        # "none", which makes token exchange omit client_secret and loops the
        # browser authorization page. Persist the effective method immediately
        # so this flow and subsequent retries use client_secret_post.
        if data.get("client_secret") and data.get("token_endpoint_auth_method") in (None, "none", ""):
            data["token_endpoint_auth_method"] = "client_secret_post"
        _write_json(self._client_info_path(), data)
        logger.debug("OAuth client info saved for %s", self._server_name)

    # -- oauth server metadata --------------------------------------------
    # The MCP SDK keeps discovered ``OAuthMetadata`` (token endpoint URL,
    # etc.) in memory only. Persisting it here lets a restarted process
    # refresh tokens without re-running metadata discovery. Without this,
    # cold-start refresh requests fall back to the SDK's guessed
    # ``{server_url}/token`` which returns 404 on most real providers and
    # forces a full browser re-authorization.

    def save_oauth_metadata(self, metadata: "OAuthMetadata") -> None:
        _write_json(self._meta_path(), metadata.model_dump(exclude_none=True, mode="json"))
        logger.debug("OAuth metadata saved for %s", self._server_name)

    def load_oauth_metadata(self) -> "OAuthMetadata | None":
        data = _read_json(self._meta_path())
        if data is None:
            return None
        if OAuthMetadata is None and not _ensure_sdk_loaded():
            return None
        try:
            return OAuthMetadata.model_validate(data)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Corrupt OAuth metadata at %s -- ignoring: %s", self._meta_path(), exc)
            return None

    # -- CIMD refusal ------------------------------------------------------

    def mark_cimd_rejected(self) -> None:
        """Record that this server refused our Client ID Metadata Document.

        Without a durable marker the in-memory fallback in
        ``mcp_oauth_manager`` only holds for the current process, so every
        restart re-presents a client_id the server has already fetched and
        refused. Cleared by ``remove()``, i.e. by ``hermes mcp login`` /
        ``hermes mcp remove``, so a fixed document gets another chance.
        """
        path = self._cimd_rejected_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        except OSError as exc:  # non-fatal — worst case we retry CIMD later
            logger.debug("Could not record CIMD rejection at %s: %s", path, exc)

    def cimd_rejected(self) -> bool:
        """True when this server has refused our metadata document before."""
        return self._cimd_rejected_path().exists()

    # -- cleanup -----------------------------------------------------------

    def remove(self) -> None:
        """Delete all stored OAuth state for this server."""
        for p in (
            self._tokens_path(),
            self._client_info_path(),
            self._meta_path(),
            self._cimd_rejected_path(),
        ):
            p.unlink(missing_ok=True)

    def snapshot(self) -> dict[str, bytes]:
        """Capture on-disk OAuth state so a failed re-auth can restore it.

        Maps filename -> bytes for whichever of the three state files exist.
        Feed back to ``restore()`` to undo an intervening ``remove()`` when a
        re-authentication attempt fails, so a still-valid token isn't destroyed.
        """
        snap: dict[str, bytes] = {}
        for p in (self._tokens_path(), self._client_info_path(), self._meta_path()):
            try:
                snap[p.name] = p.read_bytes()
            except OSError:
                pass
        return snap

    def restore(self, snapshot: dict[str, bytes], *, only_if_absent: bool = False) -> None:
        """Revert to a snapshot without overwriting a concurrent successful write."""
        if only_if_absent and any(
            path.exists()
            for path in (self._tokens_path(), self._client_info_path(), self._meta_path())
        ):
            logger.info(
                "Skipping OAuth rollback for %s because newer state exists",
                self._server_name,
            )
            return
        self.remove()
        if not snapshot:
            return
        token_dir = _get_token_dir(self._hermes_home)
        token_dir.mkdir(parents=True, exist_ok=True)
        for fname, data in snapshot.items():
            path = token_dir / fname
            try:
                fd = os.open(
                    str(path),
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    stat.S_IRUSR | stat.S_IWUSR,
                )
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            except OSError as exc:
                logger.warning("Failed to restore OAuth state %s: %s", fname, exc)

    def poison_client_registration(self) -> bool:
        """Discard a dead dynamically-registered client so it gets re-created.

        Called when the IdP rejects our cached ``client_id`` with
        ``invalid_client`` on the token endpoint — proof the server-side
        registration is gone (IdP redeploy / DB wipe / rebrand). Deleting
        ``client.json`` makes the MCP SDK's ``async_auth_flow`` take the
        ``if not client_info`` branch and re-run RFC 7591 dynamic client
        registration on the next flow. The stale ``meta.json`` is dropped
        too so discovery re-runs against a freshly fetched document.

        Tokens are intentionally left in place — the subsequent
        re-authorization overwrites them, and keeping them avoids losing a
        still-valid refresh token if the re-registration never completes.

        A single ``.bak`` copy of the client file is kept for recovery.
        Returns True if a client file was present and removed.
        """
        client_path = self._client_info_path()
        if not client_path.exists():
            return False
        backup = client_path.with_name(client_path.name + ".bak")
        try:
            backup.write_bytes(client_path.read_bytes())
        except OSError as exc:  # non-fatal — proceed with the removal anyway
            logger.warning("Could not back up client info at %s: %s", client_path, exc)
        client_path.unlink(missing_ok=True)
        self._meta_path().unlink(missing_ok=True)
        logger.warning(
            "MCP OAuth '%s': cached client registration rejected as invalid_client; "
            "removed client.json + meta.json (backup at %s) to force re-registration",
            self._server_name, backup.name,
        )
        return True

    def has_cached_tokens(self) -> bool:
        """Return True if we have tokens on disk (may be expired)."""
        return self._tokens_path().exists()


# ---------------------------------------------------------------------------
# Callback handler factory -- each invocation gets its own result dict
# ---------------------------------------------------------------------------


def _authorization_code_result(code: str, state: "str | None", iss: "str | None" = None):
    """Package redirect parameters in the shape the installed SDK expects.

    mcp 2.0 changed ``callback_handler``'s contract from a
    ``tuple[str, str | None]`` to an ``AuthorizationCodeResult`` model, and the
    SDK now reads ``result.state`` / ``result.iss`` off it — a tuple raises
    ``AttributeError`` mid-flow. Fall back to the tuple when the model is
    absent so the handler still satisfies an older SDK.
    """
    try:
        from mcp.shared.auth import AuthorizationCodeResult
    except ImportError:  # mcp < 2.0
        return code, state
    return AuthorizationCodeResult(code=code, state=state, iss=iss)


def _make_callback_handler() -> tuple[type, dict]:
    """Create a per-flow callback HTTP handler class with its own result dict.

    Returns ``(HandlerClass, result_dict)`` where *result_dict* is a mutable
    dict that the handler writes ``auth_code`` and ``state`` into when the
    OAuth redirect arrives.  Each call returns a fresh pair so concurrent
    flows don't stomp on each other.
    """
    result: dict[str, Any] = {
        "auth_code": None, "state": None, "error": None, "iss": None,
    }

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            params = parse_qs(urlparse(self.path).query)
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            error = params.get("error", [None])[0]
            # RFC 9207 authorization-response issuer. mcp 2.0 validates it
            # against the discovered metadata and *rejects* a response that
            # omits it when the authorization server advertised
            # `authorization_response_iss_parameter_supported`, so dropping it
            # here would break login against those providers.
            iss = params.get("iss", [None])[0]

            result["auth_code"] = code
            result["state"] = state
            result["error"] = error
            result["iss"] = iss

            body = (
                "<html><body><h2>Authorization Successful</h2>"
                "<p>You can close this tab and return to Hermes.</p></body></html>"
            ) if code else (
                "<html><body><h2>Authorization Failed</h2>"
                f"<p>Error: {error or 'unknown'}</p></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("OAuth callback: %s", fmt % args)

    return _Handler, result


# ---------------------------------------------------------------------------
# Async redirect + callback handlers for OAuthClientProvider
# ---------------------------------------------------------------------------


def _make_redirect_handler(port: int, redirect_uri: str | None = None):
    """Return a redirect handler closure that closes over the given port.

    Using a closure instead of reading the module-level ``_oauth_port`` avoids
    cross-server state pollution when multiple MCP servers run OAuth
    concurrently (fixes #44588).

    ``redirect_uri`` is the configured proxy callback (e.g. a Tailscale Funnel
    URL), or ``None`` for the loopback default. It tailors the remote-session
    hint: a proxied callback reaches this machine on its own, so the loopback
    SSH-tunnel guidance would be misleading.
    """
    async def _redirect_handler(authorization_url: str) -> None:
        """Show the authorization URL to the user.

        Opens the browser automatically when possible; always prints the URL
        as a fallback for headless/SSH/gateway environments.
        """
        from tools.mcp_dashboard_oauth import get_dashboard_oauth_flow

        dashboard_flow = get_dashboard_oauth_flow()
        if dashboard_flow is not None:
            await dashboard_flow.publish_authorization_url(authorization_url)
            return

        # Fail fast at the authorization boundary in non-interactive contexts
        # (systemd gateway, cron, background MCP discovery). A cached-but-unusable
        # token (expired/revoked, refresh rejected) makes the SDK fall through to
        # the authorization-code flow even though build_oauth_auth's token-file
        # guard passed. Without this check we would print a URL and launch a
        # browser flow no operator can complete, then block in _wait_for_callback
        # for the full timeout. Raise before launching so gateway adapters start
        # promptly and the caller can skip this server with an actionable warning.
        # This intentionally re-checks interactivity here rather than trusting the
        # token-file existence guard alone. See #57836.
        _raise_if_non_interactive(
            "MCP OAuth requires browser authorization but no interactive "
            "session is available (non-interactive/background context)."
        )

        msg = (
            f"\n  MCP OAuth: authorization required.\n"
            f"  Open this URL in your browser:\n\n"
            f"    {authorization_url}\n"
        )
        print(msg, file=sys.stderr)

        on_ssh = bool(os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"))
        if on_ssh and redirect_uri:
            # A configured proxy callback (e.g. Tailscale Funnel) forwards the
            # redirect to the listener on this machine, so no tunnel/paste is needed.
            print(
                f"  Remote session detected. After you authorize, the provider redirects to\n"
                f"    {redirect_uri}\n"
                f"  which forwards to the callback listener on this machine — no SSH tunnel needed.\n",
                file=sys.stderr,
            )
        elif on_ssh and port:
            # Loopback default: the provider redirects to
            # http://127.0.0.1:<port>/callback, which reaches the callback server on
            # the *remote* machine — not the user's local machine where the browser
            # opened. Two ways out: paste the redirect URL back (default fallback,
            # offered by _wait_for_callback on interactive TTYs), or set up an SSH
            # port forward so the redirect tunnels through.
            print(
                f"  Remote session detected. After you authorize, the provider redirects to\n"
                f"    http://127.0.0.1:{port}/callback\n"
                f"  which only the listener on THIS machine can receive. Two options:\n"
                f"\n"
                f"    1. Easiest — when your browser shows a connection error after\n"
                f"       authorizing, copy the full URL from the address bar and paste\n"
                f"       it at the prompt below. The pasted ``code=...&state=...`` is\n"
                f"       enough to complete the flow.\n"
                f"\n"
                f"    2. Or forward the port first in a separate terminal:\n"
                f"         ssh -N -L {port}:127.0.0.1:{port} <user>@<this-host>\n"
                f"       then open the URL above and let it redirect normally.\n"
                f"\n"
                f"  See: https://hermes-agent.nousresearch.com/docs/guides/oauth-over-ssh\n",
                file=sys.stderr,
            )

        if _can_open_browser():
            try:
                opened = webbrowser.open(authorization_url)
                if opened:
                    print("  (Browser opened automatically.)\n", file=sys.stderr)
                else:
                    print("  (Could not open browser — please open the URL manually.)\n", file=sys.stderr)
            except Exception:
                print("  (Could not open browser — please open the URL manually.)\n", file=sys.stderr)
        else:
            print("  (Headless environment detected — open the URL manually.)\n", file=sys.stderr)

    return _redirect_handler


async def _wait_for_callback() -> tuple[str, str | None]:
    """Wait for the OAuth callback on the legacy module-level port.

    Kept for backwards compatibility with callers that never went through
    :func:`build_oauth_auth`'s per-flow wiring. New code paths receive a
    per-flow waiter from :func:`_make_callback_waiter` so concurrent OAuth
    flows cannot cross ports (#34260).

    Raises:
        RuntimeError: If ``_oauth_port`` has not been set, which would indicate
            that ``build_oauth_auth`` was skipped — the asserting form below
            was a silent bug when running Python with ``-O``/``-OO``.
    """
    if _oauth_port is None:
        raise RuntimeError(
            "OAuth callback port not set — build_oauth_auth must be called "
            "before _wait_for_oauth_callback"
        )
    return await _make_callback_waiter(_oauth_port)()


def _make_callback_waiter(
    port: int, cimd_url: str | None = None, timeout: float = 300.0
):
    """Return a callback waiter bound to a single OAuth flow's port.

    ``timeout`` bounds how long the waiter polls for the redirect. It used to
    be passed to ``OAuthClientProvider(timeout=...)`` as well, but mcp 2.0
    dropped that constructor argument — the wait happens here, so this is now
    the only place the configured ``oauth.timeout`` takes effect.

    Closing over the port (instead of reading the module-level
    ``_oauth_port``) keeps concurrent OAuth flows isolated: flow A's waiter
    listens on flow A's port even when flow B's ``_configure_callback_port``
    overwrites the legacy global afterwards (#34260, the callback-side
    sibling of the #44588 redirect-handler fix).

    ``cimd_url`` is the Client ID Metadata Document this flow presents, when
    it presents one. It only tailors the timeout message: a server that
    fetches the document and refuses it aborts at the *authorization*
    endpoint (draft section 5.1), so no redirect ever reaches us and a bare
    "timed out" hides the real cause.

    The waiter polls for the redirect without blocking the event loop. On an
    interactive TTY it races the HTTP listener against a stdin paste fallback
    so users without an SSH tunnel can paste the redirect URL (or just the
    ``code=...&state=...`` query string) from a browser on another machine.

    Raises (when awaited):
        OAuthNonInteractiveError: If the callback times out (no user present
            to complete the browser auth), or in non-interactive contexts.
    """

    async def _wait():
        from tools.mcp_dashboard_oauth import get_dashboard_oauth_flow

        dashboard_flow = get_dashboard_oauth_flow()
        if dashboard_flow is not None:
            # The dashboard flow still speaks the legacy tuple; normalize it
            # here so both callback sources hand the SDK one shape.
            dash_code, dash_state = await dashboard_flow.wait_for_callback()
            return _authorization_code_result(dash_code, dash_state)

        # Reject before binding the callback listener in non-interactive
        # contexts. Reaching here means the SDK entered the authorization-code
        # flow (a valid or refreshable token would never call the callback
        # handler), so a cached token file is present but unusable. Binding the
        # listener here would block for the full 300s timeout and — on the next
        # connection retry — collide with the still-bound/TIME_WAIT port,
        # surfacing as ``OSError: [Errno 98] Address already in use``. Failing
        # fast keeps gateway startup independent of an unusable optional MCP
        # server. This guard holds "regardless of whether a token file exists"
        # — the point the build_oauth_auth token-file guard cannot cover.
        # See #57836.
        _raise_if_non_interactive(
            "OAuth callback requires an interactive session but none is "
            "available (non-interactive/background context); skipping browser "
            "authorization without binding a callback listener."
        )

        handler_cls, result = _make_callback_handler()

        # Start a temporary server on this flow's port, adopting the socket
        # reserved at port-selection time when one exists. Holding the bound
        # socket from _reserve_callback_port() until here closes the TOCTOU
        # window where another process could steal the port between selection
        # and bind (#22161). allow_reuse_address is set BEFORE binding (setting
        # it after the constructor has already bound is a no-op) so a lingering
        # TIME_WAIT socket from a previous flow cannot block the next one
        # (#44590).
        try:
            server = HTTPServer(
                ("127.0.0.1", port), handler_cls, bind_and_activate=False
            )
            reserved = _reserved_sockets.pop(port, None)
            if reserved is not None:
                # Adopt the reserved (already bound) socket and start listening.
                server.socket.close()
                server.socket = reserved
                server.server_address = reserved.getsockname()
                server.server_activate()
            else:
                server.allow_reuse_address = True
                server.server_bind()
                server.server_activate()
        except OSError as exc:
            # The loopback callback port is genuinely in use: a concurrent OAuth
            # flow, a leftover listener, or a fixed `oauth.redirect_port` that
            # collided. build_oauth_auth does not start its own callback server,
            # so there is nothing to poll here; surface a clear, actionable error
            # instead of a misleading "timed out".
            raise OAuthNonInteractiveError(
                f"OAuth callback port {port} is already in use ({exc}). "
                "Close any other in-progress login, or set a free `oauth.redirect_port` "
                "in the server config, then retry."
            ) from exc

        server_thread = threading.Thread(target=server.handle_request, daemon=True)
        server_thread.start()

        # Optional paste-fallback thread: only on interactive TTYs. Reads one
        # line from stdin and writes the parsed code/state into the shared
        # result dict. The HTTP listener and this thread race for the result;
        # whichever fills it first wins.
        paste_thread: threading.Thread | None = None
        if _is_interactive():
            print(
                "\n  Or paste the redirect URL here (or the ``?code=...&state=...`` "
                "portion) and press Enter. Type ``skip`` + Enter to continue "
                "without this server:",
                file=sys.stderr,
                flush=True,
            )
            paste_thread = threading.Thread(
                target=_paste_callback_reader, args=(result,), daemon=True
            )
            paste_thread.start()

        poll_interval = 0.5
        elapsed = 0.0
        try:
            while elapsed < timeout:
                if result["auth_code"] is not None or result["error"] is not None:
                    break
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
        finally:
            server.server_close()

        if result["error"] == _USER_SKIPPED_SENTINEL:
            raise OAuthNonInteractiveError("user_skipped")
        if result["error"]:
            raise RuntimeError(f"OAuth authorization failed: {result['error']}")
        if result["auth_code"] is None:
            hint = ""
            if cimd_url:
                hint = (
                    " If the browser showed an invalid-client error instead of "
                    "an approval prompt, the authorization server rejected "
                    f"Hermes' Client ID Metadata Document ({cimd_url}); set "
                    "``cimd: false`` under that server's ``oauth:`` block in "
                    "config.yaml to authorize via dynamic client registration "
                    "instead."
                )
            raise OAuthNonInteractiveError(
                "OAuth callback timed out — no authorization code received. "
                "Ensure you completed the browser authorization flow." + hint
            )

        return _authorization_code_result(
            result["auth_code"], result["state"], result.get("iss")
        )

    return _wait


def _paste_callback_reader(result: dict) -> None:
    """Read one line from stdin, parse it as an OAuth redirect, write to result.

    Accepts any of:
      - Full redirect URL: ``http://127.0.0.1:37949/callback?code=...&state=...``
      - The provider's own callback URL: ``https://mcp.example.com/callback?code=...&state=...``
      - Just the query string: ``?code=...&state=...`` or ``code=...&state=...``
      - A skip token (``skip``, ``cancel``, ``s``, ``n``, ``no``, ``q``, ``quit``)
        — exits the OAuth flow cleanly without auth. Caller raises
        :class:`OAuthNonInteractiveError` so MCP connection setup treats this
        as a non-fatal "user opted out" and continues without that server.

    Failures to parse, EOF, or interrupts are swallowed — this is best-effort
    fallback alongside the HTTP listener, which remains the primary path.
    """
    try:
        line = sys.stdin.readline()
    except (KeyboardInterrupt, OSError, ValueError):
        return
    if not line:
        return  # EOF
    line = line.strip()
    if not line:
        return

    # Skip if HTTP listener already won.
    if result.get("auth_code") is not None or result.get("error") is not None:
        return

    # Skip token: user explicitly opted out of authorization. Mark the
    # result with a sentinel error string that _wait_for_callback maps
    # to OAuthNonInteractiveError (already handled by mcp_tool.py as a
    # non-fatal "skip this server and continue startup" path).
    if line.lower() in _SKIP_TOKENS:
        if result.get("auth_code") is not None or result.get("error") is not None:
            return
        result["error"] = _USER_SKIPPED_SENTINEL
        print(
            "  OAuth skipped. Run `hermes mcp login <server>` later to "
            "authenticate, or set ``enabled: false`` on that server in "
            "config.yaml to disable persistently.",
            file=sys.stderr,
        )
        return

    # Strip a leading "?" if user pasted just a query string.
    query = line
    if "?" in line:
        # Either a full URL or "?code=...". Take everything after the first "?".
        query = line.split("?", 1)[1]
    if query.startswith("?"):
        query = query[1:]

    try:
        params = parse_qs(query)
    except (ValueError, TypeError):
        print(
            "  Could not parse pasted input as an OAuth redirect — ignoring.",
            file=sys.stderr,
        )
        return

    code = params.get("code", [None])[0]
    state = params.get("state", [None])[0]
    error = params.get("error", [None])[0]
    iss = params.get("iss", [None])[0]  # RFC 9207 — see _make_callback_handler

    if not code and not error:
        print(
            "  Pasted input did not contain ``code=`` or ``error=`` — ignoring.",
            file=sys.stderr,
        )
        return

    # One more race-check before writing.
    if result.get("auth_code") is not None or result.get("error") is not None:
        return

    result["auth_code"] = code
    result["state"] = state
    result["error"] = error
    result["iss"] = iss
    if code:
        print("  Got authorization code from paste — completing flow.", file=sys.stderr)


# ---------------------------------------------------------------------------
# OAuth provider compatibility shims
# ---------------------------------------------------------------------------


HermesOAuthClientProvider: Any = None


def _get_hermes_oauth_provider_class() -> type | None:
    global HermesOAuthClientProvider
    if HermesOAuthClientProvider is not None:
        return HermesOAuthClientProvider
    if not _ensure_sdk_loaded():
        return None

    class _HermesOAuthClientProvider(OAuthClientProvider):
        """OAuth provider with pragmatic fixes for real-world MCP providers.

        Supabase MCP dynamic registration returns ``client_secret`` but omits
        ``token_endpoint_auth_method``. The upstream MCP SDK treats the missing
        method as ``none`` and therefore omits ``client_secret`` from the token
        request, causing Supabase to reject the exchange and the browser to show
        the authorization page again. Coerce the in-memory client info right before
        token/refresh requests as well as persisting the fixed shape in storage.

        ``token_user_agent`` (from ``oauth.user_agent``) is stamped onto the
        token-endpoint requests the SDK builds — some authorization servers
        and WAFs reject httpx's default User-Agent there (#75576).
        """

        def __init__(self, *args: Any, token_user_agent: "str | None" = None, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._hermes_token_user_agent = token_user_agent

        def _stamp_token_user_agent(self, request):
            ua = getattr(self, "_hermes_token_user_agent", None)
            if ua:
                request.headers["User-Agent"] = ua
            return request

        def _coerce_client_secret_post(self) -> None:
            info = getattr(self.context, "client_info", None)
            if not info or not getattr(info, "client_secret", None):
                return
            method = getattr(info, "token_endpoint_auth_method", None)
            if method not in (None, "none", ""):
                return
            data = info.model_dump(mode="json", exclude_none=True)
            data["token_endpoint_auth_method"] = "client_secret_post"
            self.context.client_info = OAuthClientInformationFull.model_validate(data)

        async def _exchange_token_authorization_code(self, *args: Any, **kwargs: Any):
            self._coerce_client_secret_post()
            request = await super()._exchange_token_authorization_code(*args, **kwargs)
            return self._stamp_token_user_agent(request)

        async def _refresh_token(self):
            self._coerce_client_secret_post()
            request = await super()._refresh_token()
            return self._stamp_token_user_agent(request)

        async def _handle_token_response(self, response):
            """Accept any 2xx token response and avoid leaking token bodies in errors."""
            if 200 <= response.status_code < 300:
                from mcp.client.auth.utils import handle_token_response_scopes
                from mcp.client.auth.oauth2 import OAuthTokenError
                from httpx import HTTPError

                try:
                    token_response = await handle_token_response_scopes(response)
                except (HTTPError, OAuthTokenError):
                    raise OAuthTokenError("Invalid token response") from None
                self.context.current_tokens = token_response
                self.context.update_token_expiry(token_response)
                await self.context.storage.set_tokens(token_response)
                return

            from mcp.client.auth.oauth2 import OAuthTokenError

            raise OAuthTokenError(f"Token exchange failed ({response.status_code})")

        async def _handle_refresh_response(self, response) -> bool:
            """Accept any 2xx refresh response and avoid logging token bodies."""
            if not (200 <= response.status_code < 300):
                logger.warning("Token refresh failed: %s", response.status_code)
                self.context.clear_tokens()
                return False

            from pydantic import ValidationError
            from httpx import HTTPError

            try:
                content = await response.aread()
                token_response = OAuthToken.model_validate_json(content)
                self.context.current_tokens = token_response
                self.context.update_token_expiry(token_response)
                await self.context.storage.set_tokens(token_response)
                return True
            except (HTTPError, ValidationError):
                logger.warning("Invalid refresh response: %s", response.status_code)
                self.context.clear_tokens()
                return False

    _HermesOAuthClientProvider.__name__ = "HermesOAuthClientProvider"
    _HermesOAuthClientProvider.__qualname__ = "HermesOAuthClientProvider"
    HermesOAuthClientProvider = _HermesOAuthClientProvider
    return HermesOAuthClientProvider


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def remove_oauth_tokens(
    server_name: str,
    *,
    hermes_home: str | Path | None = None,
) -> None:
    """Delete stored OAuth tokens and client info for a server."""
    storage = HermesTokenStorage(server_name, hermes_home=hermes_home)
    storage.remove()
    logger.info("OAuth tokens removed for '%s'", server_name)


# ---------------------------------------------------------------------------
# Extracted helpers (Task 3 of MCP OAuth consolidation)
#
# These compose into ``build_oauth_auth`` below, and are also used by
# ``tools.mcp_oauth_manager.MCPOAuthManager._build_provider`` so the two
# construction paths share one implementation.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CIMD -- OAuth Client ID Metadata Documents
#
# Under CIMD the client_id IS an HTTPS URL that the authorization server
# fetches to learn our app name, logo and permitted redirect URIs, replacing
# the per-install RFC 7591 registration that the MCP spec deprecated in
# 2026-07-28. The SDK does the protocol work; Hermes only decides whether a
# given flow is eligible and hands the URL to ``OAuthClientProvider``.
# ---------------------------------------------------------------------------

# Published from ``website/static/oauth/client-metadata.json`` by the docs
# deploy. The github.io origin is deliberate: an authorization server MUST NOT
# follow HTTP redirects when fetching the document
# (draft-ietf-oauth-client-id-metadata-document section 5), and
# hermes-agent.nousresearch.com/docs/* 301s here.
_CIMD_CLIENT_METADATA_URL = (
    "https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json"
)

# Loopback callback ports declared in that document. The redirect URI in the
# authorization request must be an exact string match against a listed one
# (section 4.2), so a CIMD flow cannot use the ephemeral port Hermes picks
# otherwise. These sit below Linux's 32768 ephemeral floor, so the kernel never
# hands one to an unrelated process. Keep in sync with the document — the
# cross-artifact test in tests/tools/test_mcp_cimd.py enforces that.
_CIMD_PORTS = (27890, 27891, 27892, 27893, 27894)

# Loopback hostnames the document lists alongside each port, so the
# ``oauth.redirect_host: localhost`` WAF workaround still works under CIMD.
_CIMD_REDIRECT_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _is_valid_cimd_url(url: str) -> bool:
    """True when *url* is usable as a CIMD client_id on the installed SDK.

    Delegates to the SDK's own validator so we never hand
    ``OAuthClientProvider`` a URL its constructor would reject outright. An
    ImportError means the SDK predates CIMD, leaving DCR as the only option.

    The SDK checks only the https-scheme and non-root-path halves of
    draft-ietf-oauth-client-id-metadata-document section 3. The rest is
    enforced here because a URL that violates it fails at the authorization
    server, mid-browser-flow, where the user sees an opaque invalid-client
    page instead of a config error.
    """
    try:
        from mcp.client.auth.utils import is_valid_client_metadata_url
    except ImportError:
        return False
    if not is_valid_client_metadata_url(url):
        return False
    try:
        parsed = urlparse(url)
        # Accessing username/password parses the netloc, which can raise.
        has_userinfo = bool(parsed.username or parsed.password)
    except ValueError:
        return False
    if has_userinfo or parsed.fragment:
        return False
    return not any(seg in {".", ".."} for seg in parsed.path.split("/"))


# Pinned ports this process has committed to, in the order they were taken.
# A provider is built once per configured OAuth server and keeps its port for
# the process lifetime, so assignments are never released. Includes a port
# restored from a cached client registration, so a sibling server is never
# handed a port another one is already registered on (#34260).
_assigned_cimd_ports: "list[int]" = []


def _note_assigned_cimd_port(port: int) -> None:
    """Claim *port* for this process when it belongs to the pinned range."""
    if port in _CIMD_PORTS and port not in _assigned_cimd_ports:
        _assigned_cimd_ports.append(port)


def _reserve_cimd_port(port: int) -> bool:
    """Bind *port* and park the socket, or return False if it's taken."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        return False
    _park_reserved_socket(port, sock)
    return True


def _pick_cimd_port() -> int | None:
    """Reserve a pinned CIMD callback port, or None when none is usable.

    Holding the bound socket until ``_wait_for_callback`` adopts it does the
    same job here as ``_reserve_callback_port`` does for ephemeral ports
    (#22161): a fixed port is just as stealable in the minutes between
    selection and the browser redirect arriving. It also makes contention
    cooperative — a second profile mid-login, or a sibling server in this
    process, finds the bind refused and moves down the range instead of
    racing us to the same listener.

    Once every pinned port belongs to this process the range wraps rather
    than falling back to DCR: a reused port only bites if both of its
    servers authorize at the same moment, and ``_wait_for_callback`` reports
    that collision clearly, whereas the DCR fallback would silently use a
    mechanism the server may not support at all.
    """
    for port in _CIMD_PORTS:
        if port in _assigned_cimd_ports:
            continue
        if _reserve_cimd_port(port):
            _assigned_cimd_ports.append(port)
            return port
    return _assigned_cimd_ports[0] if _assigned_cimd_ports else None


def _has_cached_client_info(storage: "HermesTokenStorage | None") -> bool:
    """True when a client registration is already on disk for this server."""
    if storage is None:
        return False
    try:
        return _read_json(storage._client_info_path()) is not None
    except (AttributeError, TypeError, ValueError):
        return False


def _server_declined_cimd(storage: "HermesTokenStorage | None") -> bool:
    """True when cached metadata shows this server doesn't advertise CIMD.

    Pinning a callback port is only needed for a flow that actually ends up
    using CIMD, but the SDK decides that during its 401 branch — long after
    Hermes has to fix the redirect URI. Cached authorization-server metadata
    from an earlier connection closes the gap for every server the user has
    already reached: one that never advertised
    ``client_id_metadata_document_supported`` keeps the reserved ephemeral
    port it has always used, and only a genuinely unknown server pays the
    optimistic pin.
    """
    if storage is None:
        return False
    try:
        metadata = storage.load_oauth_metadata()
    except (AttributeError, TypeError, ValueError):
        return False
    if metadata is None:
        return False
    return getattr(metadata, "client_id_metadata_document_supported", None) is not True


def _maybe_use_cimd(
    cfg: dict,
    storage: "HermesTokenStorage | None" = None,
) -> "tuple[str, int] | None":
    """Return ``(client_id URL, pinned callback port)``, or None to use DCR.

    Every early return below is a case where the redirect URI Hermes would
    send is not one the published document declares, where the client
    identity is already settled, or where the server is known not to want a
    document — DCR remains correct in all of them. Passing a metadata URL
    anyway would make the SDK present a client_id whose registered redirect
    URIs don't match the request, and the authorization server would reject
    the flow.
    """
    if cfg.get("cimd") is False:
        return None

    url = cfg.get("client_metadata_url") or _CIMD_CLIENT_METADATA_URL
    if not _is_valid_cimd_url(url):
        return None

    # A client pinned in config.yaml is the user's explicit choice, and a
    # secret means they want a confidential client — the document forbids
    # shared secrets (draft section 4.1).
    if cfg.get("client_id") or cfg.get("client_secret"):
        return None

    # The document, not the config, supplies the name and auth method the
    # server sees, so a caller that set either is asking for an identity CIMD
    # cannot present. Figma's DCR name allowlist (applied by
    # apply_oauth_provider_defaults) is the in-tree example.
    if cfg.get("client_name"):
        return None
    if (cfg.get("token_endpoint_auth_method") or "none") != "none":
        return None

    # Dashboard/desktop flows redirect to the server's own externally
    # reachable URL (``/api/mcp/oauth/callback/<name>``), which is
    # deployment-specific and can never appear in a static document.
    from tools.mcp_dashboard_oauth import get_dashboard_oauth_flow

    if get_dashboard_oauth_flow() is not None:
        return None

    if cfg.get("redirect_uri") or cfg.get("redirect_port"):
        return None

    if (cfg.get("redirect_host") or "127.0.0.1") not in _CIMD_REDIRECT_HOSTS:
        return None

    # An existing registration is bound to the redirect URI it registered
    # with; swapping in a CIMD client_id now would invalidate stored tokens.
    if _has_cached_client_info(storage):
        return None

    if storage is not None and storage.cimd_rejected():
        return None

    if _server_declined_cimd(storage):
        return None

    port = _pick_cimd_port()
    if port is None:
        return None
    return url, port


def cimd_provider_kwargs(cfg: dict) -> dict[str, Any]:
    """``client_metadata_url=`` for ``OAuthClientProvider``, when CIMD applies.

    Returned as kwargs rather than a plain value so the argument is omitted
    entirely on a DCR flow. An SDK old enough to lack CIMD support — the case
    ``_is_valid_cimd_url`` already refuses to produce a URL for — rejects the
    keyword outright, and that must not take every other OAuth flow with it.
    """
    url = cfg.get("_cimd_url")
    return {"client_metadata_url": url} if url else {}


def token_request_user_agent(cfg: dict) -> str | None:
    """The configured ``oauth.user_agent`` for token-endpoint requests, or None.

    Some authorization servers and network protection layers (WAFs) reject
    the default python-httpx User-Agent on the token endpoint. The value is
    opt-in and per-server; anything that is not a non-empty string is
    treated as unset so a null/empty YAML value never sends a blank header.
    Applied ONLY to authorization-code exchange and refresh-token requests —
    never to MCP traffic or discovery, and no other headers are configurable
    (arbitrary token headers risk secrets landing in config.yaml).
    """
    ua = cfg.get("user_agent")
    if isinstance(ua, str):
        ua = ua.strip()
        if ua:
            return ua
    return None


def _configure_callback_port(
    cfg: dict,
    storage: "HermesTokenStorage | None" = None,
) -> int:
    """Pick or validate the OAuth callback port.

    Stores the resolved port into ``cfg['_resolved_port']`` so sibling
    helpers (and the manager) can read it from the same dict. Returns the
    resolved port.

    Port choice precedence:
    1. explicit ``oauth.redirect_port`` config
    2. cached client registration redirect URI port
    3. a pinned CIMD port, when the flow is CIMD-eligible
    4. newly allocated free port

    A CIMD-eligible flow also records the client_id URL in
    ``cfg['_cimd_url']`` for the provider constructors to forward.

    NOTE: also sets the legacy module-level ``_oauth_port`` so existing
    calls to ``_wait_for_callback`` keep working. The legacy global is
    the root cause of issue #5344 (port collision on concurrent OAuth
    flows); replacing it with a ContextVar is out of scope for this
    consolidation PR.
    """
    global _oauth_port
    from tools.mcp_dashboard_oauth import get_dashboard_oauth_flow

    dashboard_flow = get_dashboard_oauth_flow()
    if dashboard_flow is not None:
        cfg["_resolved_port"] = 0
        cfg["redirect_uri"] = cfg.get("redirect_uri") or dashboard_flow.redirect_uri
        return 0
    cached_redirect_uri = _cached_redirect_uri(storage)
    if not cfg.get("redirect_uri") and cached_redirect_uri:
        cfg["redirect_uri"] = cached_redirect_uri
        cfg["_resolved_port"] = 0
        return 0
    cimd = _maybe_use_cimd(cfg, storage)
    if cimd is not None:
        cfg["_cimd_url"], port = cimd
        cfg["_resolved_port"] = port
        _oauth_port = port
        return port
    requested = int(cfg.get("redirect_port", 0))
    # Precedence: explicit config port → cached client-registration port →
    # fresh ephemeral port. The cached port keeps re-auth consistent with the
    # redirect URI pinned at dynamic client registration (providers reject a
    # mismatched URI). Only a truly fresh ephemeral pick goes through
    # _reserve_callback_port(), which keeps the socket bound until
    # _wait_for_callback adopts it — closing the select→bind TOCTOU race
    # (#22161). Explicit and cached ports are fixed, known values and bind
    # via the reuse_address path instead.
    port = requested or _cached_redirect_port(storage) or _reserve_callback_port()
    # A cached port can be one of the pinned CIMD ports, left behind by an
    # earlier CIMD login for this server. Claim it so a sibling server's
    # _pick_cimd_port doesn't hand the same port out a second time.
    _note_assigned_cimd_port(port)
    cfg["_resolved_port"] = port
    _oauth_port = port  # legacy consumer: _wait_for_callback reads this
    return port


def _resolve_redirect_uri(cfg: dict, port: int) -> str:
    """Resolve the OAuth callback URL: configured ``redirect_uri`` or loopback.

    A configured ``redirect_uri`` lets the callback go through a proxy (e.g. a
    Tailscale Funnel exposing a public HTTPS URL that forwards to localhost);
    otherwise we default to ``http://<redirect_host>:<port>/callback``. An empty
    value is treated as unset. Both the client metadata and any pre-registered
    client info must derive the redirect_uri here so they stay identical — a
    mismatch makes the authorization server reject the callback.

    ``redirect_host`` (default ``127.0.0.1``) tweaks only the hostname of the
    loopback callback. Some providers' WAFs (e.g. Reclaim.ai's AWS API Gateway)
    reject any authorize request whose query string contains a literal
    ``127.0.0.1``, returning ``{"message":"Forbidden"}``; ``redirect_host:
    localhost`` works around that. The callback listener still binds
    ``127.0.0.1`` either way.
    """
    configured = cfg.get("redirect_uri")
    if configured:
        return configured
    host = cfg.get("redirect_host") or "127.0.0.1"
    return f"http://{host}:{port}/callback"


# Figma's remote MCP (https://mcp.figma.com/mcp) implement RFC 7591 DCR as a
# *name allowlist*, not open registration. POST /v1/oauth/mcp/register returns
# 403 Forbidden for any client_name outside a short fixed set. Empirically (as
# of 2026-07, verified by live call against api.figma.com):
#   "Claude Code" → 200
#   "Codex"       → 200
#   "Hermes Agent" / "Hermes" / "Cursor" / "VS Code" / … → 403
# pi-figma-remote-auth and similar tools work around this the same way — register
# under an allowlisted name so the browser flow can start. User can still pin a
# different name via oauth.client_name if Figma ever admits one.
_FIGMA_DCR_CLIENT_NAME = "Claude Code"
_FIGMA_DEFAULT_SCOPE = "mcp:connect"


def _is_figma_remote_mcp(
    server_name: str | None = None,
    server_url: str | None = None,
) -> bool:
    """True when this MCP server is Figma's hosted remote endpoint."""
    url = (server_url or "").lower()
    name = (server_name or "").lower()
    from utils import base_url_host_matches, base_url_hostname
    if base_url_host_matches(url, "mcp.figma.com") or (
        base_url_host_matches(url, "figma.com") and "/mcp" in url
    ):
        return True
    # Name-only match only when the URL isn't some other host called figma-*.
    if "figma" in name and (not url or "figma" in base_url_hostname(url)):
        return True
    return False


def apply_oauth_provider_defaults(
    cfg: dict,
    *,
    server_name: str = "",
    server_url: str | None = None,
) -> dict:
    """Mutate *cfg* with provider-specific OAuth workarounds. Returns *cfg*.

    Call this before :func:`_build_client_metadata` /
    :func:`_maybe_preregister_client`. Only fills keys the user left unset —
    an explicit ``oauth.client_name`` / ``oauth.scope`` always wins.
    """
    if _is_figma_remote_mcp(server_name, server_url):
        if not cfg.get("client_name"):
            cfg["client_name"] = _FIGMA_DCR_CLIENT_NAME
            logger.info(
                "MCP OAuth '%s': Figma DCR allowlist — registering as "
                "client_name=%r (override via oauth.client_name)",
                server_name or server_url,
                _FIGMA_DCR_CLIENT_NAME,
            )
        if not cfg.get("scope"):
            cfg["scope"] = _FIGMA_DEFAULT_SCOPE
        # Figma's register response advertises token_endpoint_auth_method=none
        # *and* returns a client_secret — then the token endpoint rejects the
        # exchange with "Client secret is required". Request confidential-
        # client registration so the SDK includes client_secret on the token
        # POST (auth method client_secret_post).
        if not cfg.get("token_endpoint_auth_method"):
            cfg["token_endpoint_auth_method"] = "client_secret_post"
    return cfg


def _build_client_metadata(cfg: dict) -> "OAuthClientMetadata":
    """Build OAuthClientMetadata from the oauth config dict.

    Requires ``cfg['_resolved_port']`` to have been populated by
    :func:`_configure_callback_port` first.
    """
    port = cfg.get("_resolved_port")
    if port is None:
        raise ValueError(
            "_configure_callback_port() must be called before _build_client_metadata()"
        )
    if OAuthClientMetadata is None:
        _ensure_sdk_loaded()
    client_name = cfg.get("client_name", "Hermes Agent")
    scope = cfg.get("scope")
    redirect_uri = _resolve_redirect_uri(cfg, port)

    # Default public client; confidential only when a secret is already known
    # or the provider (e.g. Figma) needs confidential-style token posts.
    auth_method = cfg.get("token_endpoint_auth_method")
    if not auth_method:
        auth_method = "client_secret_post" if cfg.get("client_secret") else "none"

    metadata_kwargs: dict[str, Any] = {
        "client_name": client_name,
        "redirect_uris": [AnyUrl(redirect_uri)],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": auth_method,
        # SEP-837 (2026-07-28 spec): clients MUST declare an application_type
        # during registration so OIDC-strict authorization servers stop
        # rejecting loopback redirect_uris. Hermes is a CLI/desktop app
        # redirecting to 127.0.0.1/localhost — that is exactly "native".
        # Overridable for the rare hosted-dashboard deployment fronting a
        # real https redirect.
        "application_type": cfg.get("application_type", "native"),
    }
    if scope:
        metadata_kwargs["scope"] = scope

    try:
        return OAuthClientMetadata.model_validate(metadata_kwargs)
    except Exception:
        # mcp 1.x metadata models predate SEP-837 and reject the unknown
        # field — retry without it rather than failing the whole flow.
        metadata_kwargs.pop("application_type", None)
        return OAuthClientMetadata.model_validate(metadata_kwargs)


def _invalidate_tokens_on_client_change(
    storage: "HermesTokenStorage",
    new_client_id: str,
    new_client_secret: str | None,
) -> None:
    """Drop cached tokens when the configured OAuth client identity changes.

    Tokens are minted for a specific ``client_id``: after the user edits
    ``oauth.client_id`` / ``oauth.client_secret`` in config.yaml (or switches
    from dynamic registration to a pre-registered client), the old tokens are
    unusable — the token endpoint rejects their refresh with
    ``invalid_client``. Pre-registered clients are deliberately exempt from
    the ``invalid_client`` auto-poison path (config-supplied identity can't
    be healed by re-registration), so without this check the stale tokens
    wedge every request until the user manually wipes
    ``~/.hermes/mcp-tokens/<server>.*``.

    Compares the on-disk ``client.json`` identity against the incoming
    config identity BEFORE the new client info overwrites it. Matching
    identity is a no-op so live sessions and valid tokens are preserved.
    Port of cline/cline#12983's "invalidate tokens when OAuth client
    changes" invariant.
    """
    existing = _read_json(storage._client_info_path())
    if not isinstance(existing, dict):
        return
    old_client_id = existing.get("client_id")
    if not old_client_id:
        return
    old_client_secret = existing.get("client_secret") or None
    if old_client_id == new_client_id and old_client_secret == (
        new_client_secret or None
    ):
        return
    removed = False
    for path in (storage._tokens_path(), storage._meta_path()):
        try:
            if path.exists():
                path.unlink()
                removed = True
        except OSError as exc:  # non-fatal — stale tokens fail later anyway
            logger.warning(
                "MCP OAuth '%s': could not remove stale %s after client "
                "change: %s", storage._server_name, path.name, exc,
            )
    if removed:
        logger.warning(
            "MCP OAuth '%s': configured OAuth client changed (client_id %r "
            "-> %r); discarded tokens minted under the previous client. "
            "Re-authorize with: hermes mcp login %s",
            storage._server_name, old_client_id, new_client_id,
            storage._server_name,
        )


def _maybe_preregister_client(
    storage: "HermesTokenStorage",
    cfg: dict,
    client_metadata: "OAuthClientMetadata",
) -> None:
    """If cfg has a pre-registered client_id, persist it to storage."""
    client_id = cfg.get("client_id")
    if not client_id:
        return
    if OAuthClientInformationFull is None:
        _ensure_sdk_loaded()
    _invalidate_tokens_on_client_change(
        storage, client_id, cfg.get("client_secret")
    )
    port = cfg["_resolved_port"]
    redirect_uri = _resolve_redirect_uri(cfg, port)

    info_dict: dict[str, Any] = {
        "client_id": client_id,
        "redirect_uris": [redirect_uri],
        "grant_types": client_metadata.grant_types,
        "response_types": client_metadata.response_types,
        "token_endpoint_auth_method": client_metadata.token_endpoint_auth_method,
    }
    if cfg.get("client_secret"):
        info_dict["client_secret"] = cfg["client_secret"]
    if cfg.get("client_name"):
        info_dict["client_name"] = cfg["client_name"]
    if cfg.get("scope"):
        info_dict["scope"] = cfg["scope"]

    client_info = OAuthClientInformationFull.model_validate(info_dict)
    _write_json(storage._client_info_path(), client_info.model_dump(mode="json", exclude_none=True))
    logger.debug("Pre-registered client_id=%s for '%s'", client_id, storage._server_name)


def humanize_oauth_registration_error(
    server_name: str,
    exc: BaseException | str,
    *,
    server_url: str | None = None,
) -> str | None:
    """Turn a Dynamic Client Registration refusal into a useful next step.

    Returns a humanized message when the error is a registration 403/Forbidden,
    else ``None`` so the caller keeps the original exception text.

    Figma's remote MCP gates DCR on exact ``client_name``. Hermes auto-sets
    ``Claude Code`` (known-good); this message fires when the user overrode
    that with something Figma still rejects, or an older Hermes is running.
    """
    msg = str(exc)
    lowered = msg.lower()
    if "403" not in msg and "forbidden" not in lowered:
        return None
    looks_like_registration = (
        "regist" in lowered
        or "client registration" in lowered
        or "dcr" in lowered
        or "dynamic client" in lowered
        or lowered.strip() in {"forbidden", "403 forbidden", "http 403: forbidden"}
        or ("403" in msg and "forbidden" in lowered)
    )
    if not looks_like_registration:
        return None

    if _is_figma_remote_mcp(server_name, server_url):
        return (
            f"'{server_name}' is Figma's remote MCP — DCR is allowlisted by "
            f"exact client_name (\"{_FIGMA_DCR_CLIENT_NAME}\" and \"Codex\" "
            "work; most other names 403). Hermes defaults to "
            f"client_name: {_FIGMA_DCR_CLIENT_NAME!r} automatically. If you "
            "set oauth.client_name yourself, change it to one of those, or "
            "clear it and re-run:\n"
            f"  hermes mcp login {server_name}"
        )

    return (
        f"'{server_name}' only allows pre-approved OAuth clients — it rejected "
        "client registration (403), so no browser flow can start. Options: "
        "set oauth.client_name to a name the provider allowlists, add a "
        "pre-registered client (oauth: {client_id: ..., client_secret: ...}), "
        "or use the provider's stdio / API-key / local server instead."
    )


def build_oauth_auth(
    server_name: str,
    server_url: str,
    oauth_config: dict | None = None,
) -> "OAuthClientProvider | None":
    """Build an ``httpx.Auth``-compatible OAuth handler for an MCP server.

    Public API preserved for backwards compatibility. New code should use
    :func:`tools.mcp_oauth_manager.get_manager` so OAuth state is shared
    across config-time, runtime, and reconnect paths.

    Args:
        server_name: Server key in mcp_servers config (used for storage).
        server_url: MCP server endpoint URL.
        oauth_config: Optional dict from the ``oauth:`` block in config.yaml.

    Returns:
        An ``OAuthClientProvider`` instance, or None if the MCP SDK lacks
        OAuth support.
    """
    if not _OAUTH_AVAILABLE or (
        OAuthClientProvider is None and not _ensure_sdk_loaded()
    ):
        logger.warning(
            "MCP OAuth requested for '%s' but SDK auth types are not available. "
            "Install with: pip install 'mcp>=1.26.0'",
            server_name,
        )
        return None

    cfg = dict(oauth_config or {})  # copy — we mutate _resolved_port
    apply_oauth_provider_defaults(
        cfg, server_name=server_name, server_url=server_url
    )
    storage = HermesTokenStorage(server_name, configured_scope=cfg.get("scope"))

    if not _is_interactive() and not storage.has_cached_tokens():
        raise OAuthNonInteractiveError(
            "MCP OAuth for "
            f"'{server_name}': non-interactive environment and no cached tokens "
            "found. The OAuth flow requires browser authorization. Run "
            f"`hermes mcp login {server_name}` interactively first to complete "
            "initial authorization, then cached tokens will be reused."
        )

    _configure_callback_port(cfg, storage)
    client_metadata = _build_client_metadata(cfg)
    _maybe_preregister_client(storage, cfg, client_metadata)

    # Use closure factories to avoid global state pollution (#44588, #34260).
    resolved_port = cfg.get("_resolved_port", _oauth_port)
    redirect_handler = _make_redirect_handler(
        resolved_port, redirect_uri=cfg.get("redirect_uri") or None
    )
    callback_handler = _make_callback_waiter(
        resolved_port, cfg.get("_cimd_url"), timeout=float(cfg.get("timeout", 300))
    )

    provider_class = _get_hermes_oauth_provider_class()
    if provider_class is None:
        logger.warning(
            "MCP OAuth requested for '%s' but the provider class is unavailable",
            server_name,
        )
        return None

    return provider_class(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        # mcp 2.0 removed the provider's own `timeout` argument; the configured
        # `oauth.timeout` is applied inside the callback waiter above, which is
        # where the browser round-trip is actually awaited.
        callback_handler=callback_handler,
        token_user_agent=token_request_user_agent(cfg),
        **cimd_provider_kwargs(cfg),
    )
