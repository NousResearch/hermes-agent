#!/usr/bin/env python3
"""
MCP OAuth 2.1 Client Support

Implements the browser-based OAuth 2.1 authorization code flow with PKCE
for MCP servers that require OAuth authentication instead of static bearer
tokens.

Uses the MCP Python SDK's ``OAuthClientProvider`` (an ``httpx.Auth`` subclass)
which handles discovery, dynamic client registration, PKCE, token exchange,
refresh, and step-up authorization automatically.

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
          client_name: "My Custom Client"       # default: "Hermes Agent"
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

_OAUTH_AVAILABLE=False
try:
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import (
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthMetadata,
        OAuthToken,
    )

    _OAUTH_AVAILABLE=True
except ImportError:
    logger.debug("MCP OAuth types not available -- OAuth MCP auth disabled")

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

# Interactivity gate for OAuth stdin prompts. A ContextVar (NOT threading.local)
# is required: background MCP discovery sets this on the discovery thread, but
# the actual connect+OAuth runs on the dedicated `mcp-event-loop` thread via
# run_coroutine_threadsafe. asyncio copies the *calling context* into the
# scheduled coroutine, so a ContextVar propagates across that boundary while a
# threading.local would not — see #35927. Default True (interactive allowed).
_oauth_interactive_enabled: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "_oauth_interactive_enabled", default=True
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


def _get_token_dir() -> Path:
    """Return the directory for MCP OAuth token files.

    Uses HERMES_HOME so each profile gets its own OAuth tokens.
    Layout: ``HERMES_HOME/mcp-tokens/``
    """
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except ImportError:
        base = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    return base / "mcp-tokens"


def _safe_filename(name: str) -> str:
    """Sanitize a server name for use as a filename (no path separators)."""
    return re.sub(r"[^\w\-]", "_", name).strip("_")[:128] or "default"


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_interactive() -> bool:
    """Return True if we can reasonably expect to interact with a user."""
    if not _oauth_interactive_enabled.get():
        return False
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


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
    # secure_parent_dir refuses to chmod / or top-level dirs (#25821).
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
    """

    def __init__(self, server_name: str):
        self._server_name = _safe_filename(server_name)

    def _tokens_path(self) -> Path:
        return _get_token_dir() / f"{self._server_name}.json"

    def _client_info_path(self) -> Path:
        return _get_token_dir() / f"{self._server_name}.client.json"

    def _meta_path(self) -> Path:
        return _get_token_dir() / f"{self._server_name}.meta.json"

    # -- tokens ------------------------------------------------------------

    async def get_tokens(self) -> "OAuthToken | None":
        data = _read_json(self._tokens_path())
        if data is None:
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
        logger.debug("OAuth tokens saved for %s", self._server_name)

    # -- client info -------------------------------------------------------

    async def get_client_info(self) -> "OAuthClientInformationFull | None":
        data = _read_json(self._client_info_path())
        if data is None:
            return None
        try:
            return OAuthClientInformationFull.model_validate(data)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Corrupt client info at %s -- ignoring: %s", self._client_info_path(), exc)
            return None

    async def set_client_info(self, client_info: "OAuthClientInformationFull") -> None:
        _write_json(self._client_info_path(), client_info.model_dump(mode="json", exclude_none=True))
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
        try:
            return OAuthMetadata.model_validate(data)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Corrupt OAuth metadata at %s -- ignoring: %s", self._meta_path(), exc)
            return None

    # -- cleanup -----------------------------------------------------------

    def remove(self) -> None:
        """Delete all stored OAuth state for this server."""
        for p in (self._tokens_path(), self._client_info_path(), self._meta_path()):
            p.unlink(missing_ok=True)

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


def _make_callback_handler() -> tuple[type, dict]:
    """Create a per-flow callback HTTP handler class with its own result dict.

    Returns ``(HandlerClass, result_dict)`` where *result_dict* is a mutable
    dict that the handler writes ``auth_code`` and ``state`` into when the
    OAuth redirect arrives.  Each call returns a fresh pair so concurrent
    flows don't stomp on each other.
    """
    result: dict[str, Any] = {"auth_code": None, "state": None, "error": None}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            params = parse_qs(urlparse(self.path).query)
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            error = params.get("error", [None])[0]

            result["auth_code"] = code
            result["state"] = state
            result["error"] = error

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
# OAuthCallbackServer — bind-first persistent callback server
# ---------------------------------------------------------------------------


class OAuthCallbackServer:
    """Persistent localhost HTTP server for OAuth callback capture.

    Binds the server at construction time, eliminating the TOCTOU gap
    between port discovery and server startup (issue #5344, #34260).
    Runs ``handle_request()`` in a loop until the OAuth callback
    arrives or the timeout expires.

    Only processes requests to ``/callback``; all other paths receive
    HTTP 404 so that stray requests (``/favicon.ico``, browser
    preflight) don't consume the single handler slot.

    Attributes:
        port: The actual port the server is bound to.
        _result: Shared dict written by the HTTP handler (and optionally
            by ``_paste_callback_reader`` for SSH paste fallback).
    """

    def __init__(self, port: int = 0, timeout: float = 300.0):
        self._result: dict[str, Any] = {"auth_code": None, "state": None, "error": None}
        self._timeout = timeout
        self._stop_event = threading.Event()
        handler_cls = self._make_handler()
        # bind server at construction time — port consumed immediately
        self._server = HTTPServer(("127.0.0.1", port), handler_cls)
        self._server.timeout = 1.0  # handle_request() polls
        self.port: int = self._server.server_address[1]
        self._thread: threading.Thread | None = None

    def _make_handler(self) -> type:
        """Build a per-instance HTTP handler with path filtering."""
        result = self._result

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                # Only process /callback; ignore favicon, preflight, etc.
                if parsed.path != "/callback":
                    self.send_response(404)
                    self.end_headers()
                    return
                params = parse_qs(parsed.query)
                result["auth_code"] = params.get("code", [None])[0]
                result["state"] = params.get("state", [None])[0]
                result["error"] = params.get("error", [None])[0]
                body = (
                    "<html><body><h2>Authorization Successful</h2>"
                    "<p>You can close this tab and return to Hermes.</p></body></html>"
                ) if result["auth_code"] else (
                    "<html><body><h2>Authorization Failed</h2>"
                    f"<p>Error: {result['error'] or 'unknown'}</p></body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("OAuth callback: %s", fmt % args)

        return _Handler

    def start(self) -> None:
        """Start the server in a daemon thread."""
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _serve_loop(self) -> None:
        """Process requests until callback arrives, timeout, or stop event."""
        deadline = time.time() + self._timeout
        while not self._stop_event.is_set() and time.time() < deadline:
            try:
                self._server.handle_request()
            except (ConnectionError, OSError, ValueError):
                # Client disconnect or socket error — non-fatal
                pass
            if self._result["auth_code"] is not None or self._result["error"] is not None:
                break

    async def wait(self, timeout: float = 300.0) -> tuple[str, str | None]:
        """Async-poll the result until callback arrives or timeout.

        Returns ``(auth_code, state)``.  Raises :class:`RuntimeError`
        on authorization error or :class:`OAuthNonInteractiveError` on
        timeout.
        """
        poll_interval = 0.5
        elapsed = 0.0
        while elapsed < timeout:
            if self._result["auth_code"] is not None or self._result["error"] is not None:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        if self._result["error"]:
            # Let _wait_for_callback handle the user-skip sentinel
            # (maps it to OAuthNonInteractiveError("user_skipped")).
            if self._result["error"] == _USER_SKIPPED_SENTINEL:
                return self._result["auth_code"], self._result["state"]
            raise RuntimeError(f"OAuth authorization failed: {self._result['error']}")
        if self._result["auth_code"] is None:
            raise OAuthNonInteractiveError(
                "OAuth callback timed out — no authorization code received."
            )
        return self._result["auth_code"], self._result["state"]

    def close(self) -> None:
        """Shut down the server and wait for the thread."""
        self._stop_event.set()
        self._server.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Async redirect + callback handlers for OAuthClientProvider
# ---------------------------------------------------------------------------


async def _redirect_handler(authorization_url: str, port: int | None = None) -> None:
    """Show the authorization URL to the user.

    Opens the browser automatically when possible; always prints the URL
    as a fallback for headless/SSH/gateway environments.
    """
    msg = (
        f"\n  MCP OAuth: authorization required.\n"
        f"  Open this URL in your browser:\n\n"
        f"    {authorization_url}\n"
    )
    print(msg, file=sys.stderr)

    # On a remote SSH session the OAuth provider redirects to
    # http://127.0.0.1:<port>/callback, which reaches the callback server on
    # the *remote* machine — not the user's local machine where the browser
    # opened.  Two ways out: paste the redirect URL back (default fallback,
    # offered by _wait_for_callback on interactive TTYs), or set up an SSH
    # port forward so the redirect tunnels through.
    actual_port = port
    if actual_port and (os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY")):
        print(
            f"  Remote session detected. After you authorize, the provider redirects to\n"
            f"    http://127.0.0.1:{actual_port}/callback\n"
            f"  which only the listener on THIS machine can receive. Two options:\n"
            f"\n"
            f"    1. Easiest — when your browser shows a connection error after\n"
            f"       authorizing, copy the full URL from the address bar and paste\n"
            f"       it at the prompt below. The pasted ``code=...&state=...`` is\n"
            f"       enough to complete the flow.\n"
            f"\n"
            f"    2. Or forward the port first in a separate terminal:\n"
            f"         ssh -N -L {actual_port}:127.0.0.1:{actual_port} <user>@<this-host>\n"
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


async def _wait_for_callback(server: "OAuthCallbackServer | None" = None) -> tuple[str, str | None]:
    """Wait for the OAuth callback to arrive.

    Polls the already-bound *server* for the result.  The paste fallback
    writes directly to ``server._result`` so it races naturally with the
    HTTP listener — whichever finishes first wins.

    *server* must be provided via ``functools.partial`` from the caller
    (``build_oauth_auth`` or ``_build_provider``)."""
    # Poll pre-bound server
    if server is not None:
        # Paste fallback: race a stdin reader against the HTTP listener.
        # Both write to `server._result`, so whichever finishes first wins.
        if _is_interactive():
            print(
                "\n  Or paste the redirect URL here (or the ``?code=...&state=...`` "
                "portion) and press Enter. Type ``skip`` + Enter to continue "
                "without this server:",
                file=sys.stderr, flush=True,
            )
            threading.Thread(
                target=_paste_callback_reader, args=(server._result,), daemon=True
            ).start()
        result = await server.wait()
        if result[1] is None and server._result.get("error") == _USER_SKIPPED_SENTINEL:
            raise OAuthNonInteractiveError("user_skipped")
        return result

    raise RuntimeError(
        "OAuth callback server not provided — "
        "caller must pass server via functools.partial"
    )


def _paste_callback_reader(result: dict) -> None:
    """Read lines from stdin, parse as OAuth redirect, write to result.

    Accepts any of:
      - Full redirect URL: ``http://127.0.0.1:37949/callback?code=...&state=...``
      - The provider's own callback URL: ``https://mcp.example.com/callback?code=...&state=...``
      - Just the query string: ``?code=...&state=...`` or ``code=...&state=...``
      - A skip token (``skip``, ``cancel``, ``s``, ``n``, ``no``, ``q``, ``quit``)
        — exits the OAuth flow cleanly without auth.

    Invalid pastes (typos, wrong URL, missing code) print a hint and let
    the user retry.  The loop exits on success, skip, HTTP listener win,
    or after 5 consecutive parse failures.

    Caller raises :class:`OAuthNonInteractiveError` on skip sentinel so
    MCP connection setup treats this as a non-fatal "user opted out" and
    continues without that server.
    """
    _MAX_RETRIES = 5
    failures = 0
    while failures < _MAX_RETRIES:
        # HTTP listener already won — stop polling stdin.
        if result.get("auth_code") is not None or result.get("error") is not None:
            return

        try:
            line = sys.stdin.readline()
        except (KeyboardInterrupt, OSError, ValueError):
            return
        if not line:
            return  # EOF
        line = line.strip()
        if not line:
            continue

        # Skip token: user explicitly opted out.
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
            query = line.split("?", 1)[1]
        if query.startswith("?"):
            query = query[1:]

        try:
            params = parse_qs(query)
        except (ValueError, TypeError):
            print(
                "  Could not parse pasted input as an OAuth redirect — try again.",
                file=sys.stderr,
            )
            failures += 1
            continue

        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        error = params.get("error", [None])[0]

        if not code and not error:
            print(
                "  Pasted input did not contain ``code=`` or ``error=`` — try again.",
                file=sys.stderr,
            )
            failures += 1
            continue

        # One more race-check before writing.
        if result.get("auth_code") is not None or result.get("error") is not None:
            return

        result["auth_code"] = code
        result["state"] = state
        result["error"] = error
        if code:
            print("  Got authorization code from paste — completing flow.", file=sys.stderr)
        return
    if code:
        print("  Got authorization code from paste — completing flow.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def remove_oauth_tokens(server_name: str) -> None:
    """Delete stored OAuth tokens and client info for a server."""
    storage = HermesTokenStorage(server_name)
    storage.remove()
    logger.info("OAuth tokens removed for '%s'", server_name)


# ---------------------------------------------------------------------------
# Extracted helpers (Task 3 of MCP OAuth consolidation)
#
# These compose into ``build_oauth_auth`` below, and are also used by
# ``tools.mcp_oauth_manager.MCPOAuthManager._build_provider`` so the two
# construction paths share one implementation.
# ---------------------------------------------------------------------------


def _configure_callback_port(cfg: dict) -> "OAuthCallbackServer":
    """Resolve the OAuth callback port and bind the callback server.

    Creates and starts an :class:`OAuthCallbackServer` so the port is
    consumed immediately — no TOCTOU gap between port discovery and
    actual server startup (issue #34260).  Stores the server in
    ``cfg['_callback_server']`` and the resolved port in
    ``cfg['_resolved_port']``.
    """
    requested = int(cfg.get("redirect_port", 0))
    server = OAuthCallbackServer(port=requested)
    server.start()
    cfg["_resolved_port"] = server.port
    cfg["_callback_server"] = server
    return server


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
    client_name = cfg.get("client_name", "Hermes Agent")
    scope = cfg.get("scope")
    redirect_uri = f"http://127.0.0.1:{port}/callback"

    metadata_kwargs: dict[str, Any] = {
        "client_name": client_name,
        "redirect_uris": [AnyUrl(redirect_uri)],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    if scope:
        metadata_kwargs["scope"] = scope
    if cfg.get("client_secret"):
        metadata_kwargs["token_endpoint_auth_method"] = "client_secret_post"

    return OAuthClientMetadata.model_validate(metadata_kwargs)


def _maybe_preregister_client(
    storage: "HermesTokenStorage",
    cfg: dict,
    client_metadata: "OAuthClientMetadata",
) -> None:
    """If cfg has a pre-registered client_id, persist it to storage."""
    client_id = cfg.get("client_id")
    if not client_id:
        return
    port = cfg["_resolved_port"]
    redirect_uri = f"http://127.0.0.1:{port}/callback"

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
    if not _OAUTH_AVAILABLE:
        logger.warning(
            "MCP OAuth requested for '%s' but SDK auth types are not available. "
            "Install with: pip install 'mcp>=1.26.0'",
            server_name,
        )
        return None

    cfg = dict(oauth_config or {})  # copy — we mutate _resolved_port
    storage = HermesTokenStorage(server_name)

    if not _is_interactive() and not storage.has_cached_tokens():
        raise OAuthNonInteractiveError(
            "MCP OAuth for "
            f"'{server_name}': non-interactive environment and no cached tokens "
            "found. The OAuth flow requires browser authorization. Run "
            f"`hermes mcp login {server_name}` interactively first to complete "
            "initial authorization, then cached tokens will be reused."
        )

    _configure_callback_port(cfg)
    client_metadata = _build_client_metadata(cfg)
    _maybe_preregister_client(storage, cfg, client_metadata)

    import functools
    callback_server = cfg.get("_callback_server")
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=functools.partial(_redirect_handler, port=callback_server.port),
        callback_handler=functools.partial(_wait_for_callback, server=callback_server),
        timeout=float(cfg.get("timeout", 300)),
    )
