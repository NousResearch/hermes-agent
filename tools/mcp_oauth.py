#!/usr/bin/env python3
"""MCP OAuth 2.1 client support: browser authorization-code flow with PKCE.

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
import contextlib
import contextvars
import errno
import html
import importlib.util as _importlib_util
import json
import logging
import os
import re
import socket
import stat
import sys
import threading
import time
import webbrowser
from functools import partialmethod

# Cross-process advisory file locking for the refresh fence. Mirrors
# cron/jobs.py: fcntl is Unix-only, msvcrt is the Windows fallback.
try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from hermes_constants import secure_parent_dir
from utils import atomic_json_write
from tools.mcp_dashboard_oauth import contextvar_set as _contextvar_set, get_dashboard_oauth_flow

if TYPE_CHECKING:  # annotations only; the SDK is imported lazily at runtime
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata, OAuthToken

logger = logging.getLogger(__name__)

# The refresh fence's critical section spans the token-endpoint POST, so it must
# outlast a slow network round trip. Bounded anyway -- a wedged peer must not
# strand us forever -- but generous enough that a healthy refresh never trips it.
_REFRESH_FENCE_TIMEOUT_SECONDS = 60.0

class RefreshFenceTimeout(RuntimeError):
    """The refresh fence could not be acquired within its bound.

    Raised so the caller FAILS CLOSED. Submitting a refresh token we are not
    certain we own is the whole defect class this fence exists to close: on a
    provider with single-use refresh tokens it burns the credential and logs
    the user out of a working session. Aborting this one refresh attempt is
    strictly cheaper -- the next request retries, and by then the peer that
    held the fence has published its replacement.
    """


# POSIX flock: EWOULDBLOCK/EAGAIN, EACCES on some NFS; msvcrt.locking: EACCES/EDEADLK.
# Same set as cron.scheduler._is_lock_contention_errno (not imported: that module is heavy).
_FENCE_CONTENTION_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES, errno.EDEADLK})


def _refresh_lock_path(path: "Path") -> "Path":
    return path.with_suffix(path.suffix + ".refresh.lock")


async def acquire_refresh_fence(path: "Path", *, timeout: float = _REFRESH_FENCE_TIMEOUT_SECONDS) -> int:
    """Take the fence that owns one refresh generation across read -> POST -> persist.

    Token files are written atomically (``_write_json``), so a reader never
    sees a torn file; the only cross-process hazard is the read-modify-write
    of a single-use refresh token. The damaging interleaving is:

        A: get_tokens() -> R1
        B: get_tokens() -> R1
        A: POST R1              -> 200, receives R2
        B: POST R1              -> 400, credential already burned
        B: clear_tokens()       -> user is logged out of a live session

    No lock scoped to one file read or write can prevent it: the fence must
    be held across the POST. It lives in a ``.refresh.lock`` sibling of the
    token file so the holder can still read/write the tokens normally.

    Entered from the SDK's coroutine-driven auth flow, so the wait is an
    ``asyncio.sleep`` poll on a non-blocking lock: a peer's slow network
    round trip must not freeze every other task on this event loop. No
    in-process lock layer is needed: an advisory lock on a fresh descriptor
    already excludes sibling tasks and threads of the same process.

    Acquisition failure RAISES. Degrading to "proceed unlocked" would
    reintroduce the exact race. Returns the locked descriptor; the caller
    hands it back to ``release_refresh_fence`` from its own exit path (the
    SDK drives the refresh as a generator, so no single ``with`` block can
    span the critical section).
    """
    lock_path = _refresh_lock_path(path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        secure_parent_dir(lock_path)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        # No lock file means no ownership proof. Fail closed: see the
        # class docstring for why proceeding is worse than aborting.
        raise RefreshFenceTimeout(
            f"refresh fence unavailable ({lock_path.name}): {exc}"
        ) from exc

    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:
                    getattr(msvcrt, "locking")(fd, getattr(msvcrt, "LK_NBLCK"), 1)
                else:  # pragma: no cover - no advisory locking primitive
                    raise RefreshFenceTimeout(
                        "refresh fence unsupported: no flock/msvcrt on this platform"
                    )
                return fd
            except OSError as exc:
                if exc.errno not in _FENCE_CONTENTION_ERRNOS:
                    # Not "a peer holds it" but "this filesystem cannot lock"
                    # (e.g. some network mounts). Still fail closed, but say so
                    # now instead of spinning to the deadline and blaming a peer.
                    raise RefreshFenceTimeout(
                        f"refresh fence unavailable on this filesystem: {exc}"
                    ) from exc
                if time.monotonic() >= deadline:
                    raise RefreshFenceTimeout(
                        f"refresh fence held by a peer for {timeout:.0f}s ({lock_path.name})"
                    ) from None
                await asyncio.sleep(0.05)
    except BaseException:
        os.close(fd)
        raise


def release_refresh_fence(fd: int) -> None:
    """Unlock and close a descriptor returned by ``acquire_refresh_fence``. Never raises."""
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt is not None:
            getattr(msvcrt, "locking")(fd, getattr(msvcrt, "LK_UNLCK"), 1)
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


# ---------------------------------------------------------------------------
# Lazy imports -- MCP SDK with OAuth support is optional
# ---------------------------------------------------------------------------

# SDK availability is detected WITHOUT importing mcp (~170 ms); classes bind lazily via _sdk_class().
_OAUTH_AVAILABLE = _importlib_util.find_spec("mcp") is not None
if not _OAUTH_AVAILABLE:
    logger.debug("MCP OAuth types not available -- OAuth MCP auth disabled")

_SDK_CLASS_NAMES = ("OAuthClientProvider", "OAuthClientInformationFull", "OAuthClientMetadata", "OAuthMetadata", "OAuthToken")
_SDK_CLASSES: dict[str, Any] = {}


def _sdk_class(name: str) -> Any:
    """SDK OAuth class *name*, importing the SDK on first call; None when unavailable (a broken SDK is probed once)."""
    global _OAUTH_AVAILABLE
    if not _SDK_CLASSES:
        try:
            from mcp.client import auth as _client_auth
            from mcp.shared import auth as _shared_auth

            _SDK_CLASSES["OAuthClientProvider"] = _client_auth.OAuthClientProvider
            for _name in _SDK_CLASS_NAMES[1:]:
                _SDK_CLASSES[_name] = getattr(_shared_auth, _name)
        except (ImportError, AttributeError):
            _SDK_CLASSES.update(dict.fromkeys(_SDK_CLASS_NAMES))
            _OAUTH_AVAILABLE = False
            logger.debug("MCP OAuth types not available -- OAuth MCP auth disabled")
    return _SDK_CLASSES.get(name)


try:
    from pydantic import AnyUrl
except ImportError:
    AnyUrl = None  # type: ignore[assignment, misc]


class OAuthNonInteractiveError(RuntimeError):
    """Raised when OAuth requires browser interaction in a non-interactive env."""


# Port of the most recent callback-port resolution. Legacy global; per-flow closures are the
# real mechanism (concurrent flows must not share it).
_oauth_port: int | None = None

# Interactivity gates for OAuth stdin prompts. ContextVars, NOT threading.local: background
# discovery sets them on its own thread while connect+OAuth runs on the `mcp-event-loop` thread
# via run_coroutine_threadsafe, which copies the calling context into the coroutine. `forced`
# pushes _is_interactive() past the TTY check for GUI-driven flows (dashboard/desktop REST; the
# paste fallback degrades harmlessly to EOF). Suppression wins — background discovery must never
# start a browser flow.
_oauth_interactive_enabled = contextvars.ContextVar("_oauth_interactive_enabled", default=True)
_oauth_interactive_forced = contextvars.ContextVar("_oauth_interactive_forced", default=False)

# Paste-prompt tokens that exit OAuth without auth; the waiter maps the sentinel to
# OAuthNonInteractiveError("user_skipped") so MCP setup continues without this server.
_SKIP_TOKENS = frozenset({"skip", "cancel", "s", "n", "no", "q", "quit"})
_USER_SKIPPED_SENTINEL = "__hermes_user_skipped__"


def _get_token_dir(hermes_home: str | Path | None = None) -> Path:
    """``HERMES_HOME/mcp-tokens/`` — per-profile token directory."""
    from hermes_constants import get_hermes_home

    return Path(hermes_home if hermes_home is not None else get_hermes_home()) / "mcp-tokens"


def _safe_filename(name: str) -> str:
    """Sanitize a server name for use as a filename (no path separators)."""
    return re.sub(r"[^\w\-]", "_", name).strip("_")[:128] or "default"


# Callback-port reservation: bound-but-not-listening sockets keyed by port, held from selection
# until the waiter adopts them (closes the select→bind TOCTOU window). Bounded so reconnect loops cannot leak fds.
# Holding the socket from port-selection time until _wait_for_callback adopts it closes the TOCTOU window
# where another process could grab the port between _find_free_port() closing its probe socket and
# HTTPServer binding minutes later (#22161).
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
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        if port:
            return None
        raise
    port = s.getsockname()[1]
    _park_reserved_socket(port, s)
    return port


def _reserve_callback_port() -> int:
    """Pick an ephemeral callback port and keep its socket bound (parked)."""
    return _bind_reserved(0)  # type: ignore[return-value]  # port 0 never returns None


def _cached_client_info(storage: "HermesTokenStorage | None") -> dict | None:
    """The on-disk client registration for *storage*, or None."""
    try:
        return _read_json(storage._client_info_path()) if storage is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _cached_redirect(storage: "HermesTokenStorage | None") -> "tuple[str | None, int | None]":
    """``(https proxy URI, loopback callback port)`` from the cached client registration (None when
    absent): a DCR ``client_id`` is bound to its registered redirect URI, so a new random port under
    it gets ``redirect_uri does not match any registered URIs``."""
    uri = port = None
    for raw in (_cached_client_info(storage) or {}).get("redirect_uris") or []:
        try:
            parsed = urlparse(str(raw))
        except (TypeError, ValueError):
            continue
        if uri is None and parsed.scheme == "https" and parsed.netloc:
            uri = str(raw)
        is_loopback_callback = parsed.scheme == "http" and parsed.path == "/callback" and parsed.hostname in {"127.0.0.1", "localhost"}
        if port is None and is_loopback_callback and parsed.port is not None:
            port = int(parsed.port)
    return uri, port


def _stdin_is_console() -> bool:
    """A human can type on stdin. ``isatty()`` alone is wrong on Windows: the CRT reports True for
    a DEVNULL / detached / CREATE_NO_WINDOW stdin (the gateway's), so a background process looked
    interactive and launched browser OAuth flows nobody could finish. Confirm with the console API
    there: ``GetConsoleMode`` fails on anything that is not a real console handle."""
    try:
        if not sys.stdin.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        mode = ctypes.c_ulong()
        return bool(ctypes.windll.kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)))
    except Exception:
        return False


def _is_interactive() -> bool:
    """True if we can reasonably expect to interact with a user."""
    if not _oauth_interactive_enabled.get():
        return False
    if _oauth_interactive_forced.get():
        return True
    return _stdin_is_console()


def _raise_if_non_interactive(lead: str) -> None:
    """Raise ``OAuthNonInteractiveError`` unless interactive; *lead* is the boundary-specific first sentence.

    ``lead`` is the boundary-specific first sentence; this helper appends the shared, actionable ``hermes
    mcp login`` next-step so the guidance wording lives in one place across every non-interactive OAuth
    boundary (#57836).
    """
    if not _is_interactive():
        raise OAuthNonInteractiveError(
            f"{lead} Run `hermes mcp login <server>` interactively to (re)authorize, then restart or reload the gateway."
        )


def force_interactive_oauth():
    """Treat the context as interactive despite no TTY (GUI-driven auth: the user IS present, just not
    on stdin). Crosses the MCP event-loop thread like ``suppress_interactive_oauth``.

    Opens the browser + localhost callback flow that the TTY heuristic would otherwise refuse. Same
    ContextVar propagation story as suppress_interactive_oauth() (#35927).
    """
    return _contextvar_set(_oauth_interactive_forced, True)


def suppress_interactive_oauth():
    """Disable stdin-based OAuth prompts for the current context; ContextVar-based so a
    background-discovery thread's suppression reaches the coroutine on the MCP event-loop thread.

    Uses a ContextVar so the suppression propagates from a background-discovery thread onto the coroutine
    scheduled (via run_coroutine_threadsafe) on the dedicated MCP event-loop thread — where the OAuth
    callback actually runs (#35927). A threading.local would not cross that thread boundary.
    """
    return _contextvar_set(_oauth_interactive_enabled, False)


def _can_open_browser() -> bool:
    """True if opening a browser is likely to work."""
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        return False  # explicit SSH session → no local display
    if os.name == "nt" or (hasattr(os, "uname") and os.uname().sysname == "Darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


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
    """OAuth tokens/client info at 0600 from creation, parent tightened to 0700 (``secure_parent_dir``
    refuses ``/``, top-level dirs and the install tree — #25821, #93050)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tighten parent dir to 0o700 so siblings can't traverse to the creds.
    # No-op on Windows (POSIX mode bits aren't enforced); ignore failures.
    # secure_parent_dir refuses to chmod /, top-level dirs, or the
    # hermes-agent install tree (#25821, #93050).
    secure_parent_dir(path)
    atomic_json_write(path, data, mode=0o600, default=str)


def _model_json(model: Any) -> dict:
    """The on-disk JSON shape of an SDK pydantic model."""
    return model.model_dump(mode="json", exclude_none=True)


class HermesTokenStorage:
    """Persist OAuth tokens and client registration to JSON files.

    File layout::

        HERMES_HOME/mcp-tokens/<server_name>.json         -- tokens
        HERMES_HOME/mcp-tokens/<server_name>.client.json   -- client info
        HERMES_HOME/mcp-tokens/<server_name>.meta.json     -- oauth server metadata
        HERMES_HOME/mcp-tokens/<server_name>.cimd-off      -- CIMD refused here
    """

    def __init__(self, server_name: str, *, hermes_home: str | Path | None = None):
        self._server_name = _safe_filename(server_name)
        self._hermes_home = Path(hermes_home) if hermes_home is not None else None
        # Issuer binding: ``loaded_issuer`` is what the token file on disk recorded (the authorization
        # server that granted the stored refresh token); ``_bound_issuer`` is stamped onto the next
        # ``set_tokens`` write. See ``tools.mcp_oauth_provider.enforce_refresh_token_issuer``.
        self.loaded_issuer: str | None = None
        self._bound_issuer: str | None = None

    def _path(self, suffix: str) -> Path:
        return _get_token_dir(self._hermes_home) / f"{self._server_name}{suffix}"

    _tokens_path = partialmethod(_path, ".json")
    _client_info_path = partialmethod(_path, ".client.json")
    _meta_path = partialmethod(_path, ".meta.json")
    _cimd_rejected_path = partialmethod(_path, ".cimd-off")

    def _state_paths(self) -> tuple[Path, Path, Path]:
        return self._tokens_path(), self._client_info_path(), self._meta_path()

    def _cimd_rejected_path(self) -> Path:
        return _get_token_dir(self._hermes_home) / f"{self._server_name}.cimd-off"

    # -- tokens ------------------------------------------------------------

    async def get_tokens(self) -> "OAuthToken | None":
        data = _read_json(self._tokens_path())
        if data is None:
            return None
        if fixup is not None:
            fixup(data)
        try:
            return cls.model_validate(data)
        except (ValueError, TypeError, KeyError) as exc:
            # A pydantic ValidationError's str() echoes the raw input (the token material); log
            # only which fields failed.
            detail = exc
            if hasattr(exc, "errors"):  # pydantic ValidationError
                detail = "validation failed for " + ", ".join(
                    ".".join(map(str, e.get("loc", ()))) for e in exc.errors(include_input=False))
            logger.warning("Corrupt %s at %s -- ignoring: %s", label, path, detail)
            return None

    def _rebase_expires_in(self, data: dict) -> None:
        """Rewrite ``expires_in`` to seconds remaining from the stored absolute ``expires_at`` (not an
        SDK field, so stripped): a relative value reloaded after restart would make ``is_token_valid()``
        True for tokens that expired while down. Legacy files without it use the file mtime, clamped
        to zero (self-heals on the next ``set_tokens``)."""
        absolute_expiry = data.pop("expires_at", None)
        if absolute_expiry is not None:
            data["expires_in"] = int(max(absolute_expiry - time.time(), 0))
        elif data.get("expires_in") is not None:
            with contextlib.suppress(OSError, TypeError, ValueError):
                implied_expiry = self._tokens_path().stat().st_mtime + int(data["expires_in"])
                data["expires_in"] = int(max(implied_expiry - time.time(), 0))

    def _fixup_loaded_tokens(self, data: dict) -> None:
        # ``hermes_issuer`` is Hermes bookkeeping, not an SDK OAuthToken field: pop before validation.
        self.loaded_issuer = data.pop("hermes_issuer", None)
        self._rebase_expires_in(data)

    async def get_tokens(self) -> "OAuthToken | None":
        self.loaded_issuer = None
        return self._load_model(self._tokens_path(), "OAuthToken", "tokens", self._fixup_loaded_tokens)

    async def set_tokens(self, tokens: "OAuthToken") -> None:
        payload = _model_json(tokens)
        # Absolute ``expires_at``: see _rebase_expires_in.
        if payload.get("expires_in") is not None:
            with contextlib.suppress(TypeError, ValueError):  # mock tokens / odd shapes: skip, don't fail persistence
                payload["expires_at"] = time.time() + int(payload["expires_in"])
        if self._bound_issuer:  # which authorization server granted these tokens (never sent on the wire)
            payload["hermes_issuer"] = self._bound_issuer
            self.loaded_issuer = self._bound_issuer
        _write_json(self._tokens_path(), payload)
        logger.debug("OAuth tokens saved for %s", self._server_name)

    def bind_issuer(self, issuer: str | None) -> None:
        """Set the authorization-server issuer stamped on future token writes."""
        self._bound_issuer = str(issuer) if issuer else None

    def stamp_issuer(self, issuer: str) -> None:
        """Backfill ``hermes_issuer`` onto a pre-binding token file: adopt the currently discovered
        issuer once instead of forcing a re-login, so the *next* read is protected."""
        data = _read_json(self._tokens_path())
        if data is None or data.get("hermes_issuer"):
            return
        data["hermes_issuer"] = str(issuer)
        try:
            _write_json(self._tokens_path(), data)
        except OSError as exc:  # non-fatal — worst case we stamp next time
            logger.debug("Could not stamp issuer on tokens for %s: %s", self._server_name, exc)
            return
        self.loaded_issuer = str(issuer)

    def strip_refresh_token(self) -> None:
        """Drop the refresh token (and its issuer record) from disk, keeping the access token: the
        unexpired access token may still be used, but a refresh token must never go to a different
        issuer than the one that granted it."""
        data = _read_json(self._tokens_path())
        if data is None or not data.get("refresh_token"):
            return
        data.pop("refresh_token", None)
        data.pop("hermes_issuer", None)
        self.loaded_issuer = None
        try:
            _write_json(self._tokens_path(), data)
        except OSError as exc:
            logger.warning("Could not strip refresh token for %s: %s", self._server_name, exc)
            return
        logger.info("Removed issuer-mismatched refresh token for %s (re-authorization will be required "
                    "when the access token expires)", self._server_name)

    @staticmethod
    def _coerce_secret_auth_method(data: dict) -> bool:
        """Set ``client_secret_post`` when a secret is present but no method is: some DCR providers
        (Supabase) omit ``token_endpoint_auth_method``, the SDK defaults it to ``none`` and the
        exchange fails without the secret."""
        if data.get("client_secret") and data.get("token_endpoint_auth_method") in (None, "none", ""):
            data["token_endpoint_auth_method"] = "client_secret_post"
            return True
        return False

    async def get_client_info(self) -> "OAuthClientInformationFull | None":
        coerced: list[bool] = []
        info = self._load_model(
            self._client_info_path(), "OAuthClientInformationFull", "client info",
            lambda data: coerced.append(self._coerce_secret_auth_method(data)))
        if info is not None and coerced[0]:
            _write_json(self._client_info_path(), _model_json(info))  # persist so later flows skip the coercion
        return info

    async def set_client_info(self, client_info: "OAuthClientInformationFull") -> None:
        data = _model_json(client_info)
        self._coerce_secret_auth_method(data)
        _write_json(self._client_info_path(), data)
        logger.debug("OAuth client info saved for %s", self._server_name)

    def save_oauth_metadata(self, metadata: "OAuthMetadata") -> None:
        """Persist server metadata so a restarted process can refresh without re-discovery;
        otherwise the SDK guesses ``{server_url}/token`` (404) and forces re-auth."""
        _write_json(self._meta_path(), _model_json(metadata))
        logger.debug("OAuth metadata saved for %s", self._server_name)

    def load_oauth_metadata(self) -> "OAuthMetadata | None":
        return self._load_model(self._meta_path(), "OAuthMetadata", "OAuth metadata")

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
        """filename -> bytes of the existing state files; ``restore()`` it to undo a ``remove()`` after
        a failed re-auth so a valid token survives."""
        snap: dict[str, bytes] = {}
        for p in self._state_paths():
            with contextlib.suppress(OSError):
                snap[p.name] = p.read_bytes()
        return snap

    def restore(self, snapshot: dict[str, bytes], *, only_if_absent: bool = False) -> None:
        """Revert to a snapshot without overwriting a concurrent successful write."""
        if only_if_absent and any(path.exists() for path in self._state_paths()):
            logger.info("Skipping OAuth rollback for %s because newer state exists", self._server_name)
            return
        self.remove()
        if not snapshot:
            return
        token_dir = _get_token_dir(self._hermes_home)
        token_dir.mkdir(parents=True, exist_ok=True)
        for fname, data in snapshot.items():
            try:
                fd = os.open(str(token_dir / fname), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            except OSError as exc:
                logger.warning("Failed to restore OAuth state %s: %s", fname, exc)

    def poison_client_registration(self) -> bool:
        """Discard a dead DCR client (``invalid_client`` at the token endpoint) plus stale ``meta.json``
        so the SDK re-registers next flow; tokens are kept (a valid refresh token survives if
        re-registration never completes). Keeps one ``.bak``. True if a client file was removed."""
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
            self._server_name, backup.name)
        return True

    def has_cached_tokens(self) -> bool:
        """True if we have tokens on disk (may be expired)."""
        return self._tokens_path().exists()


# Callback capture: the HTTP listener and the stdin paste reader share one result dict.
def _authorization_code_result(code: str, state: "str | None", iss: "str | None" = None):
    """Redirect parameters in the shape the installed SDK expects: mcp 2.0's ``callback_handler``
    returns an ``AuthorizationCodeResult`` (the SDK reads ``.state``/``.iss`` off it); older SDKs take a tuple."""
    try:
        from mcp.shared.auth import AuthorizationCodeResult
    except ImportError:  # mcp < 2.0
        return code, state
    return AuthorizationCodeResult(code=code, state=state, iss=iss)


def _parse_redirect_query(query: str) -> dict[str, Any]:
    """code/state/error/iss from a redirect query string. ``iss`` (RFC 9207 issuer) is kept: mcp 2.0
    rejects a response omitting it when the server advertised ``authorization_response_iss_parameter_supported``."""
    params = parse_qs(query)
    return {k: params.get(k, [None])[0] for k in ("code", "state", "error", "iss")}


def _result_taken(result: dict) -> bool:
    return result.get("auth_code") is not None or result.get("error") is not None


def _make_callback_handler() -> tuple[type, dict]:
    """Fresh ``(HandlerClass, result_dict)`` per flow so concurrent flows don't stomp on each other."""
    result: dict[str, Any] = {"auth_code": None, "state": None, "error": None, "iss": None}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = _parse_redirect_query(urlparse(self.path).query)
            result.update(auth_code=parsed["code"], state=parsed["state"], error=parsed["error"], iss=parsed["iss"])
            body = ("<h2>Authorization Successful</h2><p>You can close this tab and return to Hermes.</p>" if parsed["code"]
                    else f"<h2>Authorization Failed</h2><p>Error: {html.escape(parsed['error'] or 'unknown')}</p>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"<html><body>{body}</body></html>".encode())

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
    """Read one stdin line as an OAuth redirect (full URL, bare query, or a ``_SKIP_TOKENS`` word that
    exits without auth) into *result*. Parse failures, EOF and interrupts are swallowed — best-effort
    fallback racing the HTTP listener, which stays primary."""
    try:
        line = sys.stdin.readline()
    except (KeyboardInterrupt, OSError, ValueError):
        return
    line = (line or "").strip()
    if not line or _result_taken(result):
        return  # EOF / blank, or the HTTP listener already won
    if line.lower() in _SKIP_TOKENS:
        result["error"] = _USER_SKIPPED_SENTINEL
        print(
            "  OAuth skipped. Run `hermes mcp login <server>` later to authenticate, "
            "or set ``enabled: false`` on that server in config.yaml to disable persistently.",
            file=sys.stderr)
        return
    # Full URL or "?code=...": take everything after the first "?".
    query = line.split("?", 1)[1] if "?" in line else line
    try:
        parsed = _parse_redirect_query(query.removeprefix("?"))
    except (ValueError, TypeError):
        print("  Could not parse pasted input as an OAuth redirect — ignoring.", file=sys.stderr)
        return
    if not parsed["code"] and not parsed["error"]:
        print("  Pasted input did not contain ``code=`` or ``error=`` — ignoring.", file=sys.stderr)
        return
    if _result_taken(result):  # one more race-check before writing
        return
    result.update(auth_code=parsed["code"], state=parsed["state"], error=parsed["error"], iss=parsed["iss"])
    if parsed["code"]:
        print("  Got authorization code from paste — completing flow.", file=sys.stderr)


# Remote-session hints printed under the authorization URL: a proxy callback forwards the redirect
# here (no tunnel needed); on loopback it misses this machine, so the user pastes the URL back or SSH-forwards the port.
_SSH_HINT_PROXY = (
    "  Remote session detected. After you authorize, the provider redirects to\n"
    "    {redirect_uri}\n"
    "  which forwards to the callback listener on this machine — no SSH tunnel needed.\n")
_SSH_HINT_LOOPBACK = (
    "  Remote session detected. After you authorize, the provider redirects to\n"
    "    http://127.0.0.1:{port}/callback\n"
    "  which only the listener on THIS machine can receive. Two options:\n"
    "\n"
    "    1. Easiest — when your browser shows a connection error after\n"
    "       authorizing, copy the full URL from the address bar and paste\n"
    "       it at the prompt below. The pasted ``code=...&state=...`` is\n"
    "       enough to complete the flow.\n"
    "\n"
    "    2. Or forward the port first in a separate terminal:\n"
    "         ssh -N -L {port}:127.0.0.1:{port} <user>@<this-host>\n"
    "       then open the URL above and let it redirect normally.\n"
    "\n"
    "  See: https://hermes-agent.nousresearch.com/docs/guides/oauth-over-ssh\n")


def _announce_authorization_url(authorization_url: str, port: int, redirect_uri: str | None) -> None:
    """Print the URL (always, as the fallback) and open the browser when possible."""
    print(f"\n  MCP OAuth: authorization required.\n  Open this URL in your browser:\n\n    {authorization_url}\n", file=sys.stderr)
    if os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"):
        if redirect_uri:
            print(_SSH_HINT_PROXY.format(redirect_uri=redirect_uri), file=sys.stderr)
        elif port:
            print(_SSH_HINT_LOOPBACK.format(port=port), file=sys.stderr)
    if not _can_open_browser():
        note = "Headless environment detected — open the URL manually."
    else:
        opened = False
        with contextlib.suppress(Exception):
            opened = webbrowser.open(authorization_url)
        note = "Browser opened automatically." if opened else "Could not open browser — please open the URL manually."
    print(f"  ({note})\n", file=sys.stderr)


def _make_redirect_handler(port: int, redirect_uri: str | None = None):
    """Redirect handler closing over this flow's port (a closure, not ``_oauth_port``, keeps concurrent
    flows isolated). ``redirect_uri`` is a configured proxy callback (None for loopback) and only tailors the hint.

    Using a closure instead of reading the module-level ``_oauth_port`` avoids cross-server state pollution
    when multiple MCP servers run OAuth concurrently (fixes #44588).
    """
    async def _redirect_handler(authorization_url: str) -> None:
        dashboard_flow = get_dashboard_oauth_flow()
        if dashboard_flow is not None:
            await dashboard_flow.publish_authorization_url(authorization_url)
            return
        # Fail fast when non-interactive: a cached-but-unusable token makes the SDK fall through to the
        # authorization-code flow past the token-file guard, and the waiter would block for the full timeout.
        # Fail fast at the authorization boundary in non-interactive contexts (systemd gateway, cron,
        # background MCP discovery). Without this check we would print a URL and launch a browser flow no
        # operator can complete, then block in _wait_for_callback for the full timeout. Raise before
        # launching so gateway adapters start promptly and the caller can skip this server with an
        # actionable warning. This intentionally re-checks interactivity here rather than trusting the
        # token-file existence guard alone. See #57836.
        _raise_if_non_interactive(
            "MCP OAuth requires browser authorization but no interactive session is available (non-interactive/background context)."
        )
        _announce_authorization_url(authorization_url, port, redirect_uri)

    return _redirect_handler


def _start_callback_server(port: int, handler_cls: type) -> HTTPServer:
    """Bind the callback listener on *port*, adopting a parked reserved socket (closes the select→bind
    TOCTOU window). ``allow_reuse_address`` is set BEFORE binding (a no-op afterwards) so a lingering
    TIME_WAIT socket from a previous flow cannot block the next."""
    try:
        server = HTTPServer(("127.0.0.1", port), handler_cls, bind_and_activate=False)
        reserved = _reserved_sockets.pop(port, None)
        if reserved is not None:
            server.socket.close()
            server.socket, server.server_address = reserved, reserved.getsockname()
        else:
            server.allow_reuse_address = True
            server.server_bind()
        server.server_activate()
    except OSError as exc:  # genuinely in use (concurrent flow / leftover listener / colliding redirect_port): say so, not "timed out"
        raise OAuthNonInteractiveError(
            f"OAuth callback port {port} is already in use ({exc}). Close any other in-progress login, "
            "or set a free `oauth.redirect_port` in the server config, then retry."
        ) from exc
    return server


def _callback_outcome(result: dict, cimd_url: str | None):
    """Turn a filled/empty result dict into the SDK's callback value, or raise."""
    if result["error"] == _USER_SKIPPED_SENTINEL:
        raise OAuthNonInteractiveError("user_skipped")
    if result["error"]:
        raise RuntimeError(f"OAuth authorization failed: {result['error']}")
    if result["auth_code"] is None:
        hint = (
            " If the browser showed an invalid-client error instead of an approval prompt, the authorization "
            f"server rejected Hermes' Client ID Metadata Document ({cimd_url}); set ``cimd: false`` under that "
            "server's ``oauth:`` block in config.yaml to authorize via dynamic client registration instead."
        ) if cimd_url else ""
        raise OAuthNonInteractiveError(
            "OAuth callback timed out — no authorization code received. Ensure you completed the browser authorization flow." + hint
        )
    return _authorization_code_result(result["auth_code"], result["state"], result.get("iss"))


def _make_callback_waiter(port: int, cimd_url: str | None = None, timeout: float = 300.0):
    """Callback waiter bound to one flow's port. ``timeout`` is where ``oauth.timeout`` applies (mcp 2.0
    dropped the provider's own). ``cimd_url`` only tailors the timeout message: a server refusing the
    document aborts at the authorization endpoint, so no redirect arrives and a bare "timed out" would
    hide the cause. Raises ``OAuthNonInteractiveError`` on timeout or when non-interactive.

    Closing over the port (instead of reading the module-level ``_oauth_port``) keeps concurrent OAuth flows
    isolated: flow A's waiter listens on flow A's port even when flow B's ``_configure_callback_port``
    overwrites the legacy global afterwards (#34260, the callback-side sibling of the #44588
    redirect-handler fix).
    """
    async def _wait():
        dashboard_flow = get_dashboard_oauth_flow()
        if dashboard_flow is not None:
            # Dashboard flow speaks the legacy tuple; normalize to one shape.
            return _authorization_code_result(*await dashboard_flow.wait_for_callback())
        # The SDK entered the authorization-code flow, so any cached token is unusable. Reject BEFORE
        # binding: binding would block for the full timeout and collide with the TIME_WAIT port on retry.
        # Reject before binding the callback listener in non-interactive contexts. Reaching here means the
        # SDK entered the authorization-code flow (a valid or refreshable token would never call the
        # callback handler), so a cached token file is present but unusable. Binding the listener here would
        # block for the full 300s timeout and — on the next connection retry — collide with the
        # still-bound/TIME_WAIT port, surfacing as ``OSError: [Errno 98] Address already in use``. Failing
        # fast keeps gateway startup independent of an unusable optional MCP server. This guard holds
        # "regardless of whether a token file exists" — the point the build_oauth_auth token-file guard
        # cannot cover. See #57836.
        _raise_if_non_interactive(
            "OAuth callback requires an interactive session but none is available (non-interactive/background "
            "context); skipping browser authorization without binding a callback listener.")
        handler_cls, result = _make_callback_handler()
        server = _start_callback_server(port, handler_cls)
        threading.Thread(target=server.handle_request, daemon=True).start()
        # Paste fallback races the HTTP listener; whichever fills result first wins.
        if _is_interactive():
            print(
                "\n  Or paste the redirect URL here (or the ``?code=...&state=...`` portion) and press Enter. "
                "Type ``skip`` + Enter to continue without this server:",
                file=sys.stderr, flush=True)
            threading.Thread(target=_paste_callback_reader, args=(result,), daemon=True).start()
        elapsed = 0.0
        try:
            while elapsed < timeout and not _result_taken(result):
                await asyncio.sleep(0.5)
                elapsed += 0.5
        finally:
            server.server_close()
        return _callback_outcome(result, cimd_url)

    return _wait


# Legacy build_oauth_auth provider class, built lazily (SDK) and cached here.
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
    HermesTokenStorage(server_name, hermes_home=hermes_home).remove()
    logger.info("OAuth tokens removed for '%s'", server_name)


# CIMD (OAuth Client ID Metadata Documents): the client_id IS an HTTPS URL the server fetches for our
# name/logo/redirect URIs, replacing per-install DCR. The SDK does the protocol; Hermes only decides
# eligibility. Published from ``website/static/oauth/client-metadata.json``; the github.io origin is
# deliberate — servers MUST NOT follow redirects when fetching it, and hermes-agent.nousresearch.com/docs/* 301s here.
_CIMD_CLIENT_METADATA_URL = "https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json"
# Loopback ports/hosts declared in that document (exact match, so no ephemeral port under CIMD);
# below Linux's 32768 ephemeral floor. tests/tools/test_mcp_cimd.py keeps them in sync.
_CIMD_PORTS = (27890, 27891, 27892, 27893, 27894)
_CIMD_REDIRECT_HOSTS = frozenset({"127.0.0.1", "localhost"})


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


    Port choice precedence:
    1. explicit ``oauth.redirect_port`` config
    2. cached client registration redirect URI port
    3. a pinned CIMD port, when the flow is CIMD-eligible
    4. newly allocated free port

    A CIMD-eligible flow also records the client_id URL in
    ``cfg['_cimd_url']`` for the provider constructors to forward.


def _pick_cimd_port() -> int | None:
    """Reserve a pinned CIMD callback port, or None when none is usable. Holding the bound socket makes
    contention cooperative: a sibling finds the bind refused and moves down the range. Once every pinned
    port belongs to this process the range wraps rather than falling back to DCR — a reused port only
    bites if both servers authorize at the same moment (reported by the waiter); DCR may be unsupported entirely.

    Holding the bound socket until ``_wait_for_callback`` adopts it does the same job here as
    ``_reserve_callback_port`` does for ephemeral ports (#22161): a fixed port is just as stealable in the
    minutes between selection and the browser redirect arriving.
    """
    for port in _CIMD_PORTS:
        if port not in _assigned_cimd_ports and _bind_reserved(port) is not None:
            _assigned_cimd_ports.append(port)
            return port
    return _assigned_cimd_ports[0] if _assigned_cimd_ports else None


def _server_declined_cimd(storage: "HermesTokenStorage | None") -> bool:
    """True when cached metadata shows this server doesn't advertise CIMD. The SDK decides CIMD vs DCR
    in its 401 branch — after Hermes must fix the redirect URI — so cached metadata closes the gap;
    only a genuinely unknown server pays the optimistic pin."""
    try:
        metadata = storage.load_oauth_metadata() if storage is not None else None
    except (AttributeError, TypeError, ValueError):
        return False
    return metadata is not None and getattr(metadata, "client_id_metadata_document_supported", None) is not True


def _maybe_use_cimd(cfg: dict, storage: "HermesTokenStorage | None" = None) -> "tuple[str, int] | None":
    """``(client_id URL, pinned callback port)``, or None to use DCR. Each ineligibility case means the
    redirect URI is not one the document declares, the client identity is already settled, or the
    server is known not to want a document — a metadata URL would be rejected."""
    url = cfg.get("client_metadata_url") or _CIMD_CLIENT_METADATA_URL
    ineligible = (
        cfg.get("cimd") is False
        or not _is_valid_cimd_url(url)
        # pinned client = explicit choice; a secret = confidential client, which the document forbids
        or cfg.get("client_id") or cfg.get("client_secret")
        # the document supplies name + auth method; setting either asks for an identity CIMD can't present
        or cfg.get("client_name") or (cfg.get("token_endpoint_auth_method") or "none") != "none"
        # dashboard/desktop flows redirect to a deployment-specific URL no static document declares
        or get_dashboard_oauth_flow() is not None
        or cfg.get("redirect_uri") or cfg.get("redirect_port")
        or (cfg.get("redirect_host") or "127.0.0.1") not in _CIMD_REDIRECT_HOSTS
        # an existing registration is bound to its redirect URI; swapping client_id would drop tokens
        or _cached_client_info(storage) is not None
        or (storage is not None and storage.cimd_rejected())
        or _server_declined_cimd(storage))
    port = None if ineligible else _pick_cimd_port()
    return None if port is None else (url, port)


def cimd_provider_kwargs(cfg: dict) -> dict[str, Any]:
    """``client_metadata_url=`` kwargs for ``OAuthClientProvider`` when CIMD applies; omitted entirely
    on a DCR flow because an SDK too old for CIMD rejects the keyword outright."""
    url = cfg.get("_cimd_url")
    return {"client_metadata_url": url} if url else {}


def token_request_user_agent(cfg: dict) -> str | None:
    """Configured ``oauth.user_agent`` for token-endpoint requests (exchange + refresh only, never MCP
    traffic or discovery), or None; a null/empty YAML value never sends a blank header. No other
    headers are configurable (secrets would land in config.yaml)."""
    ua = cfg.get("user_agent")
    return ua.strip() if isinstance(ua, str) and ua.strip() else None


def _configure_callback_port(cfg: dict, storage: "HermesTokenStorage | None" = None) -> int:
    """Resolve the callback port into ``cfg['_resolved_port']`` (0 = non-loopback URI). Precedence:
    dashboard flow / cached https redirect URI → CIMD pinned port (sets ``cfg['_cimd_url']``) →
    ``oauth.redirect_port`` → cached registration port → fresh ephemeral port (the only parked one).
    Also sets the legacy ``_oauth_port``.

    NOTE: also sets the legacy module-level ``_oauth_port`` so existing calls to ``_wait_for_callback`` keep
    working. The legacy global is the root cause of issue #5344 (port collision on concurrent OAuth flows);
    replacing it with a ContextVar is out of scope for this consolidation PR.
    """
    global _oauth_port
    dashboard_flow = get_dashboard_oauth_flow()
    if dashboard_flow is not None:
        cfg["_resolved_port"] = 0
        cfg["redirect_uri"] = cfg.get("redirect_uri") or dashboard_flow.redirect_uri
        return 0
    cached_uri, cached_port = _cached_redirect(storage)
    if cached_uri and not cfg.get("redirect_uri"):
        cfg["redirect_uri"] = cached_uri
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
    _oauth_port = port
    return port


def _resolve_redirect_uri(cfg: dict, port: int) -> str:
    """Configured ``redirect_uri`` (proxy) or ``http://<redirect_host>:<port>/callback``; the single
    derivation so client metadata and pre-registered info stay identical. ``redirect_host`` only changes
    the hostname (some WAFs reject a literal ``127.0.0.1``); the listener still binds ``127.0.0.1``."""
    return cfg.get("redirect_uri") or f"http://{cfg.get('redirect_host') or '127.0.0.1'}:{port}/callback"


# Figma's remote MCP allowlists DCR by client_name ("Claude Code"/"Codex" register, others 403);
# register under an allowlisted name so the flow can start. oauth.client_name overrides.
_FIGMA_DCR_CLIENT_NAME = "Claude Code"
_FIGMA_DEFAULT_SCOPE = "mcp:connect"


def _is_figma_remote_mcp(server_name: str | None = None, server_url: str | None = None) -> bool:
    """True when this MCP server is Figma's hosted remote endpoint."""
    from utils import base_url_host_matches, base_url_hostname
    url = (server_url or "").lower()
    if base_url_host_matches(url, "mcp.figma.com") or (base_url_host_matches(url, "figma.com") and "/mcp" in url):
        return True
    # Name-only match only when the URL isn't some other host called figma-*.
    return "figma" in (server_name or "").lower() and (not url or "figma" in base_url_hostname(url))


def apply_oauth_provider_defaults(cfg: dict, *, server_name: str = "", server_url: str | None = None) -> dict:
    """Mutate *cfg* with provider-specific OAuth workarounds (before building client metadata /
    pre-registering); returns *cfg*. Only fills keys the user left unset — explicit values win."""
    if _is_figma_remote_mcp(server_name, server_url):
        if not cfg.get("client_name"):
            cfg["client_name"] = _FIGMA_DCR_CLIENT_NAME
            logger.info(
                "MCP OAuth '%s': Figma DCR allowlist — registering as client_name=%r (override via oauth.client_name)",
                server_name or server_url, _FIGMA_DCR_CLIENT_NAME)
        if not cfg.get("scope"):
            cfg["scope"] = _FIGMA_DEFAULT_SCOPE
        # Figma advertises auth_method=none yet demands the returned client_secret at the token
        # endpoint; request a confidential registration so the SDK posts it.
        cfg["token_endpoint_auth_method"] = cfg.get("token_endpoint_auth_method") or "client_secret_post"
    return cfg


def _build_client_metadata(cfg: dict) -> "OAuthClientMetadata":
    """Build OAuthClientMetadata; requires ``_configure_callback_port`` first."""
    port = cfg.get("_resolved_port")
    if port is None:
        raise ValueError("_configure_callback_port() must be called before _build_client_metadata()")
    metadata_cls = _sdk_class("OAuthClientMetadata")
    # Public client by default; confidential only with a known secret or a provider (Figma) needing confidential-style token posts.
    auth_method = cfg.get("token_endpoint_auth_method") or ("client_secret_post" if cfg.get("client_secret") else "none")
    metadata_kwargs: dict[str, Any] = {
        "client_name": cfg.get("client_name", "Hermes Agent"),
        "redirect_uris": [AnyUrl(_resolve_redirect_uri(cfg, port))],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": auth_method,
        # SEP-837: OIDC-strict servers need application_type to accept loopback redirects; "native"
        # for a CLI/desktop app, overridable for a hosted https dashboard.
        "application_type": cfg.get("application_type", "native")}
    if cfg.get("scope"):
        metadata_kwargs["scope"] = cfg["scope"]
    try:
        return metadata_cls.model_validate(metadata_kwargs)
    except Exception:  # mcp 1.x metadata models predate SEP-837 and reject the unknown field
        metadata_kwargs.pop("application_type", None)
        return metadata_cls.model_validate(metadata_kwargs)


def _invalidate_tokens_on_client_change(
    storage: "HermesTokenStorage", new_client_id: str, new_client_secret: str | None) -> None:
    """Drop cached tokens when the configured client identity changes: tokens minted under the old
    ``client_id`` fail refresh with ``invalid_client``, and pre-registered clients are exempt from
    auto-poison, so stale tokens would wedge every request until a manual wipe. Compares on-disk
    ``client.json`` BEFORE it is overwritten; a matching identity is a no-op.

    Matching identity is a no-op so live sessions and valid tokens are preserved. Port of
    cline/cline#12983's "invalidate tokens when OAuth client changes" invariant.
    """
    existing = _read_json(storage._client_info_path())
    old_client_id = existing.get("client_id") if isinstance(existing, dict) else None
    if not old_client_id or (old_client_id == new_client_id and (existing.get("client_secret") or None) == (new_client_secret or None)):
        return
    removed = False
    for path in (storage._tokens_path(), storage._meta_path()):
        if not path.exists():
            continue
        try:
            path.unlink()
            removed = True
        except OSError as exc:  # non-fatal — stale tokens fail later anyway
            logger.warning("MCP OAuth '%s': could not remove stale %s after client change: %s", storage._server_name, path.name, exc)
    if removed:
        logger.warning(
            "MCP OAuth '%s': configured OAuth client changed (client_id %r -> %r); discarded tokens minted under "
            "the previous client. Re-authorize with: hermes mcp login %s",
            storage._server_name, old_client_id, new_client_id, storage._server_name)


def _maybe_preregister_client(storage: "HermesTokenStorage", cfg: dict, client_metadata: "OAuthClientMetadata") -> None:
    """If cfg has a pre-registered client_id, persist it to storage."""
    client_id = cfg.get("client_id")
    if not client_id:
        return
    info_cls = _sdk_class("OAuthClientInformationFull")
    _invalidate_tokens_on_client_change(storage, client_id, cfg.get("client_secret"))
    info_dict: dict[str, Any] = {
        "client_id": client_id,
        "redirect_uris": [_resolve_redirect_uri(cfg, cfg["_resolved_port"])],
        "grant_types": client_metadata.grant_types,
        "response_types": client_metadata.response_types,
        "token_endpoint_auth_method": client_metadata.token_endpoint_auth_method,
        **{key: cfg[key] for key in ("client_secret", "client_name", "scope") if cfg.get(key)}}
    _write_json(storage._client_info_path(), _model_json(info_cls.model_validate(info_dict)))
    logger.debug("Pre-registered client_id=%s for '%s'", client_id, storage._server_name)


def humanize_oauth_registration_error(
    server_name: str, exc: BaseException | str, *, server_url: str | None = None) -> str | None:
    """Turn a DCR 403/Forbidden into a useful next step; None for anything else so the caller keeps the
    original text. Figma gates DCR on exact ``client_name`` (auto-set to ``Claude Code``), so this fires
    when the user overrode it or an older Hermes is running."""
    msg = str(exc)
    lowered = msg.lower()
    looks_like_registration = ("403" in msg or "forbidden" in lowered) and (
        any(k in lowered for k in ("regist", "dcr", "dynamic client"))
        or lowered.strip() in {"forbidden", "403 forbidden", "http 403: forbidden"}
        or ("403" in msg and "forbidden" in lowered))
    if not looks_like_registration:
        return None
    if _is_figma_remote_mcp(server_name, server_url):
        return (
            f"'{server_name}' is Figma's remote MCP — DCR is allowlisted by exact client_name "
            f"(\"{_FIGMA_DCR_CLIENT_NAME}\" and \"Codex\" work; most other names 403). Hermes defaults to "
            f"client_name: {_FIGMA_DCR_CLIENT_NAME!r} automatically. If you set oauth.client_name yourself, "
            f"change it to one of those, or clear it and re-run:\n  hermes mcp login {server_name}")
    return (
        f"'{server_name}' only allows pre-approved OAuth clients — it rejected client registration (403), so no "
        "browser flow can start. Options: set oauth.client_name to a name the provider allowlists, add a "
        "pre-registered client (oauth: {client_id: ..., client_secret: ...}), or use the provider's stdio / "
        "API-key / local server instead.")


def build_oauth_auth(server_name: str, server_url: str, oauth_config: dict | None = None) -> "OAuthClientProvider | None":
    """``httpx.Auth`` OAuth handler for an MCP server; None if the SDK lacks OAuth. Legacy API — new code
    uses :func:`tools.mcp_oauth_manager.get_manager` so state is shared across config-time, runtime and reconnect paths."""
    global HermesOAuthClientProvider
    if not _OAUTH_AVAILABLE or _sdk_class("OAuthClientProvider") is None:
        logger.warning("MCP OAuth requested for '%s' but SDK auth types are not available. Install with: pip install 'mcp>=1.26.0'", server_name)
        return None
    from tools.mcp_oauth_provider import build_provider_kwargs, prepare_oauth_config

    cfg, storage = prepare_oauth_config(server_name, server_url, oauth_config)
    if not _is_interactive() and not storage.has_cached_tokens():
        raise OAuthNonInteractiveError(
            f"MCP OAuth for '{server_name}': non-interactive environment and no cached tokens found. The OAuth flow "
            f"requires browser authorization. Run `hermes mcp login {server_name}` interactively first to complete "
            "initial authorization, then cached tokens will be reused.")
    kwargs = build_provider_kwargs(cfg, storage, ssh_proxy_hint=True)
    if HermesOAuthClientProvider is None:
        from tools.mcp_oauth_provider import HermesProviderMixin

        HermesOAuthClientProvider = type("HermesOAuthClientProvider", (HermesProviderMixin, _sdk_class("OAuthClientProvider")), {
            "__doc__": "SDK provider plus Hermes' token-endpoint fixes (see ``HermesProviderMixin``).",
            "__module__": __name__, "_hermes_logger": logger})
    return HermesOAuthClientProvider(server_url=server_url, **kwargs)

    # Use closure factories to avoid global state pollution (#44588, #34260).
    resolved_port = cfg.get("_resolved_port", _oauth_port)
    redirect_handler = _make_redirect_handler(
        resolved_port, redirect_uri=cfg.get("redirect_uri") or None
    )
    callback_handler = _make_callback_waiter(
        resolved_port, cfg.get("_cimd_url"), timeout=float(cfg.get("timeout", 300))
    )

# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from contextlib import contextmanager  # noqa: F401,E402

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
