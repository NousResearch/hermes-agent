#!/usr/bin/env python3
"""MCP OAuth 2.1 client support: browser authorization-code flow with PKCE.

The SDK's ``OAuthClientProvider`` does discovery, client identification, PKCE, exchange and
refresh; this module supplies ``HermesTokenStorage`` (on-disk persistence), the localhost callback
listener and ``build_oauth_auth()`` (legacy entry point). client_id is Hermes' Client ID Metadata
Document URL (CIMD) when the server supports it, else RFC 7591 DCR. ``mcp_servers.<name>.oauth`` keys
(all optional): client_id, client_secret, scope, redirect_port, redirect_uri (proxy callback),
redirect_host, client_name, client_metadata_url, cimd, user_agent, timeout."""

import asyncio
import contextlib
import contextvars
import importlib.util as _importlib_util
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
from functools import partialmethod
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from hermes_constants import secure_parent_dir
from tools.mcp_dashboard_oauth import contextvar_set as _contextvar_set, get_dashboard_oauth_flow

if TYPE_CHECKING:  # annotations only; the SDK is imported lazily at runtime
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata, OAuthToken

logger = logging.getLogger(__name__)

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


def _bind_reserved(port: int) -> int | None:
    """Bind ``127.0.0.1:port`` (0 = ephemeral) and park it until the waiter adopts it; None if taken.
    The cap evicts ephemeral parks only: losing a pinned CIMD port (the only ones the published
    document declares) mid-flow would reopen the race."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        if port:
            return None
        raise
    bound = sock.getsockname()[1]
    while len(_reserved_sockets) >= _MAX_RESERVED_SOCKETS:
        stale_port = next((p for p in _reserved_sockets if p not in _CIMD_PORTS), None)
        if stale_port is None:
            break  # only pinned sockets remain — never evict those
        with contextlib.suppress(OSError):
            _reserved_sockets.pop(stale_port).close()
    _reserved_sockets[bound] = sock
    return bound


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


def _is_interactive() -> bool:
    """True if we can reasonably expect to interact with a user."""
    if not _oauth_interactive_enabled.get():
        return False
    if _oauth_interactive_forced.get():
        return True
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


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
    """Atomically write *data* as JSON created at 0o600 (``O_EXCL`` + mode avoids the write-then-chmod
    window where the file inherits a world-readable umask); parent dir tightened to 0o700. The random
    per-process tmp suffix avoids clashes with concurrent writers/crash leftovers.

    The previous ``write_text`` + post-write ``chmod`` opened a TOCTOU window where the temp file briefly
    inherited the process umask (commonly 0o644 = world-readable), exposing OAuth tokens to other local
    users between create and chmod. Mirrors the fix in ``agent/google_oauth.py`` (#19673).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # secure_parent_dir refuses to chmod /, top-level dirs, or the hermes-agent install tree (#25821,
    # #93050).
    # Tighten parent dir to 0o700 so siblings can't traverse to the creds. No-op on Windows (POSIX mode bits
    # aren't enforced); ignore failures. secure_parent_dir refuses to chmod /, top-level dirs, or the
    # hermes-agent install tree (#25821, #93050).
    secure_parent_dir(path)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _model_json(model: Any) -> dict:
    """The on-disk JSON shape of an SDK pydantic model."""
    return model.model_dump(mode="json", exclude_none=True)


class HermesTokenStorage:
    """Persist OAuth state as ``HERMES_HOME/mcp-tokens/<server_name>`` + ``.json`` (tokens),
    ``.client.json`` (client info), ``.meta.json`` (server metadata), ``.cimd-off`` (CIMD refused)."""

    def __init__(self, server_name: str, *, hermes_home: str | Path | None = None):
        self._server_name = _safe_filename(server_name)
        self._hermes_home = Path(hermes_home) if hermes_home is not None else None

    def _path(self, suffix: str) -> Path:
        return _get_token_dir(self._hermes_home) / f"{self._server_name}{suffix}"

    _tokens_path = partialmethod(_path, ".json")
    _client_info_path = partialmethod(_path, ".client.json")
    _meta_path = partialmethod(_path, ".meta.json")
    _cimd_rejected_path = partialmethod(_path, ".cimd-off")

    def _state_paths(self) -> tuple[Path, Path, Path]:
        return self._tokens_path(), self._client_info_path(), self._meta_path()

    @staticmethod
    def _load_model(path: Path, sdk_name: str, label: str, fixup=None):
        """Read *path* into SDK model *sdk_name*; None if absent, no SDK, or corrupt.
        ``fixup(data)`` may rewrite the raw dict before validation."""
        data = _read_json(path)
        cls = _sdk_class(sdk_name) if data is not None else None
        if cls is None:
            return None
        if fixup is not None:
            fixup(data)
        try:
            return cls.model_validate(data)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Corrupt %s at %s -- ignoring: %s", label, path, exc)
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

    async def get_tokens(self) -> "OAuthToken | None":
        return self._load_model(self._tokens_path(), "OAuthToken", "tokens", self._rebase_expires_in)

    async def set_tokens(self, tokens: "OAuthToken") -> None:
        payload = _model_json(tokens)
        # Absolute ``expires_at``: see _rebase_expires_in.
        if payload.get("expires_in") is not None:
            with contextlib.suppress(TypeError, ValueError):  # mock tokens / odd shapes: skip, don't fail persistence
                payload["expires_at"] = time.time() + int(payload["expires_in"])
        _write_json(self._tokens_path(), payload)
        logger.debug("OAuth tokens saved for %s", self._server_name)

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

    def mark_cimd_rejected(self) -> None:
        """Durably record that this server refused our CIMD document so a restart does not re-present
        the refused client_id. Cleared by ``remove()`` so a fixed document gets a retry."""
        path = self._cimd_rejected_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        except OSError as exc:  # non-fatal — worst case we retry CIMD later
            logger.debug("Could not record CIMD rejection at %s: %s", path, exc)

    def cimd_rejected(self) -> bool:
        """True when this server has refused our metadata document before."""
        return self._cimd_rejected_path().exists()

    def remove(self) -> None:
        """Delete all stored OAuth state for this server."""
        for p in (*self._state_paths(), self._cimd_rejected_path()):
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
                    else f"<h2>Authorization Failed</h2><p>Error: {parsed['error'] or 'unknown'}</p>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"<html><body>{body}</body></html>".encode())

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("OAuth callback: %s", fmt % args)

    return _Handler, result


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


def remove_oauth_tokens(server_name: str, *, hermes_home: str | Path | None = None) -> None:
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


def _is_valid_cimd_url(url: str) -> bool:
    """True when *url* is usable as a CIMD client_id on the installed SDK (ImportError = SDK predates
    CIMD → DCR only). The SDK checks only https + non-root path; userinfo, fragments and dot segments
    are rejected here because they fail mid-browser-flow as an opaque invalid-client page."""
    try:
        from mcp.client.auth.utils import is_valid_client_metadata_url
        if not is_valid_client_metadata_url(url):
            return False
        parsed = urlparse(url)
        has_userinfo = bool(parsed.username or parsed.password)  # netloc parse can raise
    except (ImportError, ValueError):
        return False
    return not (has_userinfo or parsed.fragment or any(seg in {".", ".."} for seg in parsed.path.split("/")))


# Pinned ports this process committed to (never released: a provider keeps its port for the process
# lifetime), including ports restored from a cached registration.
# Includes a port restored from a cached client registration, so a sibling server is never handed a port
# another one is already registered on (#34260).
_assigned_cimd_ports: "list[int]" = []


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
    else:
        port = int(cfg.get("redirect_port", 0)) or cached_port or _reserve_callback_port()
        # A cached port may be a pinned CIMD port from an earlier login; claim it from siblings.
        if port in _CIMD_PORTS and port not in _assigned_cimd_ports:
            _assigned_cimd_ports.append(port)
    # Precedence: explicit config port → cached client-registration port → fresh ephemeral port. The cached
    # port keeps re-auth consistent with the redirect URI pinned at dynamic client registration (providers
    # reject a mismatched URI). Only a truly fresh ephemeral pick goes through _reserve_callback_port(),
    # which keeps the socket bound until _wait_for_callback adopts it — closing the select→bind TOCTOU race
    # (#22161). Explicit and cached ports are fixed, known values and bind via the reuse_address path
    # instead.
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from contextlib import contextmanager  # noqa: F401,E402

OAuthClientInformationFull: Any = None

OAuthClientMetadata: Any = None

OAuthClientProvider: Any = None

OAuthMetadata: Any = None

OAuthToken: Any = None
# ---- END PLUGIN-COMPAT ----
