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
          redirect_host: "127.0.0.1"            # loopback host for redirect_uri
                                                #   one of: localhost (binds
                                                #   both 127.0.0.1 and ::1 so
                                                #   dual-stack browsers reach
                                                #   the listener regardless of
                                                #   which family they pick),
                                                #   127.0.0.1, ::1 (or [::1])
          redirect_port: 0                      # 0 = auto-pick free port
          client_name: "My Custom Client"       # default: "Hermes Agent"
"""

import asyncio
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

# Port/host used by the most recent build_oauth_auth() call. Exposed so that
# tests can verify the callback server and the redirect_uri share the same
# listener settings. ``_oauth_bind_host`` is the bare HTTPServer bind address
# (``::1``); ``_oauth_uri_host`` is the URI-authority form (``[::1]``) used in
# the redirect_uri and SSH hint. These globals only back the legacy
# ``_redirect_handler`` / ``_wait_for_callback`` entry points — provider
# construction passes resolved values into per-provider closures instead.
_oauth_port: int | None = None
_oauth_bind_host: str = "127.0.0.1"
_oauth_uri_host: str = "127.0.0.1"


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


# Upper bound on probe retries when picking a port that must be free on
# multiple loopback families simultaneously (only the ``localhost`` path
# triggers a multi-family probe). 50 is generous: each retry is a single
# bind/close pair on the loopback, so even a hostile host with a tight
# port pool resolves quickly.
_MAX_PORT_PROBE_ATTEMPTS = 50


def _listener_specs(bind_host: str) -> list[tuple[int, str]]:
    """Return ``(address_family, bind_address)`` pairs for the callback listener.

    The OAuth ``redirect_uri`` advertises a single host string, but the
    listener may need more than one socket behind it. ``localhost`` is the
    only such case: a dual-stack OS resolves ``localhost`` to either
    ``127.0.0.1`` or ``::1`` depending on platform/order/glibc-tuning, and
    a browser following the OAuth redirect picks whichever the resolver
    returned first. If we listened on only one family while the browser hit
    the other, the callback would never arrive. So we expand ``localhost``
    into BOTH loopback families and bind both. Literal addresses
    (``127.0.0.1`` / ``::1``) keep their single-family behavior.
    """
    if bind_host == "localhost":
        return [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]
    if ":" in bind_host:
        return [(socket.AF_INET6, bind_host)]
    return [(socket.AF_INET, bind_host)]


def _make_loopback_socket(family: int) -> socket.socket:
    """Create a loopback probe socket; force IPV6_V6ONLY=1 on AF_INET6.

    Without ``IPV6_V6ONLY`` the kernel may treat the IPv6 socket as
    dual-stack — accepting IPv4-mapped addresses on the same port. That
    would conflict with the AF_INET sibling we also want to bind for the
    ``localhost`` path. Setting ``IPV6_V6ONLY=1`` keeps the two listeners
    in separate accept queues so they coexist on the same port.

    The socket option is best-effort: on platforms that don't expose it,
    we fall back to default kernel behavior (which on macOS already
    defaults to v6-only).
    """
    s = socket.socket(family, socket.SOCK_STREAM)
    if family == socket.AF_INET6:
        try:
            s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        except (AttributeError, OSError):
            pass
    return s


def _find_free_port(bind_host: str = "127.0.0.1") -> int:
    """Find a TCP port free on every family the listener will bind.

    For literal ``127.0.0.1`` / ``::1`` this is a single-family probe — the
    address family of the probe socket must match the family the listener
    will use, since "free on IPv4 127.0.0.1" tells you nothing about the
    same port number's IPv6 in-use set (different sockets, different
    namespaces).

    For ``localhost`` we expand to both loopback families
    (:func:`_listener_specs`) and only return a port we could
    simultaneously bind on every spec; that simultaneous-bind step also
    gives us a coarse race shield, since holding the head socket bound
    while we probe the rest prevents a competing process from sniping the
    port between probes within the same attempt. After all probes succeed
    we close every socket and return the port — the listener's bind is
    still racy against external processes (no different from the legacy
    single-family probe), but the multi-family ordering is consistent.

    Raises:
        OSError: ``_MAX_PORT_PROBE_ATTEMPTS`` consecutive probes all
            collided on at least one family. Surfaces a system-level
            problem (port pool exhausted, IPv6 disabled, etc.) rather
            than silently looping forever.
    """
    specs = _listener_specs(bind_host)
    last_error: OSError | None = None
    for _ in range(_MAX_PORT_PROBE_ATTEMPTS):
        held: list[socket.socket] = []
        try:
            head_family, head_addr = specs[0]
            head = _make_loopback_socket(head_family)
            held.append(head)
            head.bind((head_addr, 0))
            port = head.getsockname()[1]

            collision: OSError | None = None
            for fam, addr in specs[1:]:
                t = _make_loopback_socket(fam)
                held.append(t)
                try:
                    t.bind((addr, port))
                except OSError as exc:
                    collision = exc
                    break
            if collision is None:
                return port
            last_error = collision
        finally:
            for s in held:
                try:
                    s.close()
                except OSError:
                    pass
    raise OSError(
        f"could not find a port free on all loopback families for "
        f"redirect_host={bind_host!r} after {_MAX_PORT_PROBE_ATTEMPTS} "
        f"attempts (last error: {last_error})"
    )


def _redact_oauth_callback_log_message(message: str) -> str:
    """Redact OAuth callback secrets from HTTP request log lines."""
    return re.sub(r"([?&](?:code|state)=)[^&\s\"']*", r"\1[REDACTED]", message)


def _validate_redirect_host(host: Any) -> tuple[str, str]:
    """Validate ``redirect_host`` and return ``(uri_host, bind_host)``.

    The OAuth callback listener runs on the local machine; allowing the
    user to point the listener at ``0.0.0.0`` or any non-loopback address
    would expose the authorization-code endpoint to other hosts on the
    network. Restrict ``redirect_host`` to the recognised loopback values
    only:

    - ``"localhost"`` — IPv4/IPv6 loopback resolver
    - ``"127.0.0.1"`` — IPv4 loopback literal
    - ``"::1"`` / ``"[::1]"`` — IPv6 loopback literal

    The two return values differ only for IPv6: per RFC 3986 the URI
    authority must bracket the address (``http://[::1]:PORT/...``) but the
    ``HTTPServer`` bind tuple takes the bare address (``"::1"``).

    Raises:
        ValueError: ``host`` is non-string, empty, includes a scheme/path/
            port, or is not one of the allowed loopback values.
    """
    if not isinstance(host, str):
        raise ValueError(
            f"redirect_host must be a string; got {type(host).__name__}"
        )
    raw = host.strip()
    if not raw:
        raise ValueError("redirect_host must be a non-empty string")
    if "://" in raw or "/" in raw:
        raise ValueError(
            f"redirect_host must be a bare hostname (no scheme/path): {host!r}"
        )

    # IPv6 loopback: accept bare ``::1`` or bracketed ``[::1]``.
    if raw == "::1" or raw == "[::1]":
        return "[::1]", "::1"

    # Anything else with a colon is either ``host:port`` (forbidden — the
    # port goes in ``redirect_port``) or a non-loopback IPv6 literal like
    # ``2001:db8::1``. Both are rejected so the only IPv6 we accept is
    # the loopback handled above.
    if ":" in raw or raw.startswith("[") or raw.endswith("]"):
        raise ValueError(
            f"redirect_host must be a loopback host "
            f"(localhost, 127.0.0.1, ::1, [::1]); got {host!r}"
        )

    if raw in {"localhost", "127.0.0.1"}:
        return raw, raw

    raise ValueError(
        f"redirect_host must be a loopback host "
        f"(localhost, 127.0.0.1, ::1, [::1]); got {host!r}"
    )


def _is_interactive() -> bool:
    """Return True if we can reasonably expect to interact with a user."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


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
        payload = client_info.model_dump(mode="json", exclude_none=True)
        payload["hermes_dynamic_client"] = True
        _write_json(self._client_info_path(), payload)
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
            logger.debug("OAuth callback: %s", _redact_oauth_callback_log_message(fmt % args))

    return _Handler, result


# ---------------------------------------------------------------------------
# Async redirect + callback handlers for OAuthClientProvider
# ---------------------------------------------------------------------------


def _format_ssh_tunnel_hint(uri_host: str, port: int) -> str:
    """Build the SSH port-forward hint for a remote (SSH) OAuth session.

    ``uri_host`` is the redirect host in *URI-authority* form: IPv6 literals
    arrive bracketed (``[::1]``), ``localhost`` / ``127.0.0.1`` pass through
    unchanged. That single form drops correctly into both the ``http://``
    callback URL and the ``ssh -L`` forward spec — OpenSSH parses the bracket
    form ``port:[::1]:port`` for IPv6 hosts, so we never have to special-case
    the family here.
    """
    return (
        f"  Remote session detected. The OAuth provider will redirect your browser to\n"
        f"    http://{uri_host}:{port}/callback\n"
        f"  which the callback listener on THIS machine is waiting on. If your browser\n"
        f"  is on a different machine, forward the port first in a separate terminal:\n"
        f"\n"
        f"    ssh -N -L '{port}:{uri_host}:{port}' <user>@<this-host>\n"
        f"\n"
        f"  Then open the URL above. See: https://hermes-agent.nousresearch.com/docs/guides/oauth-over-ssh\n"
    )


async def _redirect_handler_impl(
    authorization_url: str, uri_host: str | None, port: int | None
) -> None:
    """Show the authorization URL to the user (shared redirect-handler body).

    Opens the browser automatically when possible; always prints the URL as a
    fallback for headless/SSH/gateway environments. On an SSH session a
    port-forward hint is printed using the ``uri_host`` / ``port`` resolved
    for the provider that owns this handler — never module globals — so the
    hint stays correct for a configured ``redirect_host`` (e.g. ``[::1]``)
    and for multi-provider setups where each provider has its own port.
    """
    msg = (
        f"\n  MCP OAuth: authorization required.\n"
        f"  Open this URL in your browser:\n\n"
        f"    {authorization_url}\n"
    )
    print(msg, file=sys.stderr)

    # On a remote SSH session the OAuth provider redirects to the callback
    # server on the *remote* machine — not the user's local machine where the
    # browser opened.  Print a port-forward hint so the user knows to tunnel.
    if port and uri_host and (os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY")):
        print(_format_ssh_tunnel_hint(uri_host, port), file=sys.stderr)

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


def _make_redirect_handler(uri_host: str, port: int):
    """Return a per-provider redirect handler closure bound to ``(uri_host, port)``.

    Mirrors :func:`_make_wait_for_callback`: provider construction uses this
    so each provider's ``redirect_handler`` carries the redirect host/port
    resolved for *that* provider. Without the closure the SSH hint would read
    the module-global ``_oauth_port`` / ``_oauth_uri_host``, which a later
    ``build_oauth_auth`` call for a different server overwrites — so the hint
    could advertise the wrong port, or ``127.0.0.1`` for a provider actually
    configured with ``redirect_host: ::1``.
    """
    async def redirect_handler(authorization_url: str) -> None:
        await _redirect_handler_impl(authorization_url, uri_host, port)

    return redirect_handler


async def _redirect_handler(authorization_url: str) -> None:
    """Legacy entry: read module-level globals set by ``build_oauth_auth``.

    Retained for backward compatibility with the no-args ``redirect_handler``
    signature the MCP SDK historically accepted, and with tests that drive the
    handler via globals. New code paths build a provider-local closure via
    :func:`_make_redirect_handler` instead.
    """
    await _redirect_handler_impl(authorization_url, _oauth_uri_host, _oauth_port)


def _make_http_server_cls(family: int) -> type:
    """Return an :class:`HTTPServer` subclass bound to ``family``.

    The IPv6 subclass also forces ``IPV6_V6ONLY=1`` in ``server_bind`` so
    the IPv4 sibling listener (used for ``redirect_host: localhost``) can
    coexist on the same port. Without this, on Linux the IPv6 listener
    might claim IPv4-mapped traffic on that port and the IPv4 bind would
    fail with EADDRINUSE.
    """
    if family == socket.AF_INET:
        return HTTPServer

    class _IPv6HTTPServer(HTTPServer):
        address_family = socket.AF_INET6

        def server_bind(self) -> None:  # type: ignore[override]
            try:
                self.socket.setsockopt(
                    socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1
                )
            except (AttributeError, OSError):
                pass
            super().server_bind()

    return _IPv6HTTPServer


async def _wait_for_callback_impl(
    bind_host: str, port: int
) -> tuple[str, str | None]:
    """Wait for the OAuth callback bound to an explicit ``(bind_host, port)``.

    Starts one listener per spec returned by :func:`_listener_specs`. For
    literal addresses that's a single listener; for ``localhost`` it's two
    (IPv4 + IPv6 loopback) so the callback is reachable regardless of which
    family the browser resolves ``localhost`` to.

    All listeners share one ``(handler_cls, result)`` pair from
    :func:`_make_callback_handler`, so whichever family the browser hits
    writes into the same result dict the polling loop watches.

    Raises:
        OAuthNonInteractiveError: If any required listener fails to bind
            (fail-fast — partial coverage on ``localhost`` would silently
            drop callbacks on the missing family) or if the callback times
            out.
        RuntimeError: If the OAuth server signalled an authorization error.
    """
    handler_cls, result = _make_callback_handler()
    specs = _listener_specs(bind_host)

    servers: list[HTTPServer] = []
    bind_errors: list[tuple[str, OSError]] = []
    for family, addr in specs:
        cls = _make_http_server_cls(family)
        try:
            servers.append(cls((addr, port), handler_cls))
        except OSError as exc:
            bind_errors.append((addr, exc))

    if bind_errors:
        # Either nothing bound at all, or — for ``localhost`` — one of the
        # two families failed. In both cases the listener can't be relied
        # on to receive the callback, so close any partial listeners and
        # surface the bind error promptly. (Requirement: explicit
        # ``redirect_port`` with ``localhost`` must fail fast when either
        # loopback family can't bind.)
        for s in servers:
            try:
                s.server_close()
            except OSError:
                pass
        details = "; ".join(f"{addr}: {exc}" for addr, exc in bind_errors)
        raise OAuthNonInteractiveError(
            f"OAuth callback could not bind {bind_host}:{port} "
            f"({details}). Free the port or change redirect_port and retry."
        )

    threads: list[threading.Thread] = []
    for server in servers:
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()
        threads.append(t)

    timeout = 300.0
    poll_interval = 0.5
    elapsed = 0.0
    try:
        while elapsed < timeout:
            if result["auth_code"] is not None or result["error"] is not None:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
    finally:
        # Close every listener — server_close on a thread-blocked accept
        # unblocks the daemon thread, which then exits. Daemon threads
        # also wouldn't keep the process alive, but we close anyway so
        # the loopback port is released for the next flow.
        for server in servers:
            try:
                server.server_close()
            except OSError:
                pass

    if result["error"]:
        raise RuntimeError(f"OAuth authorization failed: {result['error']}")
    if result["auth_code"] is None:
        raise OAuthNonInteractiveError(
            "OAuth callback timed out — no authorization code received. "
            "Ensure you completed the browser authorization flow."
        )

    return result["auth_code"], result["state"]


def _make_wait_for_callback(bind_host: str, port: int):
    """Return a per-provider callback handler closure bound to ``(bind_host, port)``.

    Provider construction uses this so that each provider's
    ``callback_handler`` carries its own resolved bind host/port. Without
    the closure, all providers would share the module-global
    ``_oauth_port`` / ``_oauth_bind_host`` and provider A could end up
    binding the host/port that was last configured for provider B (the
    cache in :class:`MCPOAuthManager` builds providers lazily, so a later
    ``_configure_callback_port`` for provider B overwrites the globals
    while provider A's advertised ``redirect_uri`` still points at A's
    original values).
    """
    async def callback_handler() -> tuple[str, str | None]:
        return await _wait_for_callback_impl(bind_host, port)

    return callback_handler


async def _wait_for_callback() -> tuple[str, str | None]:
    """Legacy entry: read module-level globals set by ``build_oauth_auth``.

    Retained for backward compatibility with tests that drive the callback
    server via globals, and as the no-args signature the MCP SDK historically
    accepted. New code paths build a provider-local closure via
    :func:`_make_wait_for_callback` instead.

    Raises:
        OAuthNonInteractiveError: If the callback times out (no user present
            to complete the browser auth).
        RuntimeError: If ``_oauth_port`` has not been set, which would indicate
            that ``build_oauth_auth`` was skipped — the asserting form below
            was a silent bug when running Python with ``-O``/``-OO``.
    """
    if _oauth_port is None:
        raise RuntimeError(
            "OAuth callback port not set — build_oauth_auth must be called "
            "before _wait_for_oauth_callback"
        )
    bind_host = _oauth_bind_host or "127.0.0.1"
    return await _wait_for_callback_impl(bind_host, _oauth_port)


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


def _configure_callback_port(cfg: dict) -> int:
    """Pick or validate the OAuth callback port.

    Stores the resolved port into ``cfg['_resolved_port']`` so sibling
    helpers (and the manager) can read it from the same dict. Returns the
    resolved port.

    NOTE: also sets the legacy module-level ``_oauth_port`` so existing
    calls to ``_wait_for_callback`` keep working. The legacy global is
    the root cause of issue #5344 (port collision on concurrent OAuth
    flows); replacing it with a ContextVar is out of scope for this
    consolidation PR.
    """
    global _oauth_port, _oauth_bind_host, _oauth_uri_host

    # Validate redirect_host BEFORE auto-picking a port — the address
    # family of the probe socket must match the family the callback
    # listener will bind, or the port may be in-use on the bind family
    # even though it's free on the probe family (e.g. picking a port
    # via IPv4 127.0.0.1 and then binding ::1 on IPv6).
    raw_host = cfg.get("redirect_host", "127.0.0.1")
    uri_host, bind_host = _validate_redirect_host(raw_host)

    requested = int(cfg.get("redirect_port", 0))
    port = _find_free_port(bind_host) if requested == 0 else requested

    cfg["_resolved_port"] = port
    cfg["_resolved_uri_host"] = uri_host
    cfg["_resolved_bind_host"] = bind_host
    _oauth_port = port  # legacy consumer: _wait_for_callback reads this
    _oauth_bind_host = bind_host
    _oauth_uri_host = uri_host  # legacy consumer: _redirect_handler reads this
    return port


def _build_redirect_uri(cfg: dict) -> str:
    """Build the OAuth ``redirect_uri`` from the resolved port + redirect_host.

    Requires ``cfg['_resolved_port']`` to have been populated by
    :func:`_configure_callback_port` first.
    """
    port = cfg.get("_resolved_port")
    if port is None:
        raise ValueError(
            "_configure_callback_port() must be called before _build_redirect_uri()"
        )
    uri_host = cfg.get("_resolved_uri_host")
    if uri_host is None:
        # Defensive: re-validate if the caller skipped _configure_callback_port
        # (shouldn't happen via the supported entry points).
        uri_host, _ = _validate_redirect_host(cfg.get("redirect_host", "127.0.0.1"))
    return f"http://{uri_host}:{port}/callback"


def _build_client_metadata(cfg: dict) -> "OAuthClientMetadata":
    """Build OAuthClientMetadata from the oauth config dict.

    Requires ``cfg['_resolved_port']`` to have been populated by
    :func:`_configure_callback_port` first.
    """
    client_name = cfg.get("client_name", "Hermes Agent")
    scope = cfg.get("scope")
    redirect_uri = _build_redirect_uri(cfg)

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


def _stored_tokens_have_refresh_token(storage: "HermesTokenStorage") -> bool:
    """Return True when stored tokens can plausibly refresh without browser auth."""
    data = _read_json(storage._tokens_path())
    if not isinstance(data, dict):
        return False
    return bool(data.get("refresh_token"))


def _invalidate_stale_dynamic_client_info(
    storage: "HermesTokenStorage",
    cfg: dict,
    client_metadata: "OAuthClientMetadata",
) -> None:
    """Drop on-disk client_info when it no longer matches cfg.

    When the user edits ``redirect_host`` / ``redirect_port`` in
    ``config.yaml``, ``<server>.client.json`` may still hold the previously
    registered ``client_id`` paired with the old ``redirect_uri``. The MCP
    SDK loads that file via ``storage.get_client_info()`` and would reuse
    the old ``client_id`` against the new ``redirect_uri``, which OAuth
    servers reject (or worse, accept while redirecting back to the stale
    host/port).

    Also handle the inverse of :func:`_maybe_preregister_client`: if the
    config no longer has a pre-registered ``client_id``/``client_secret``
    but the stored file was written by that path (or otherwise has a client
    auth method that no longer matches the current metadata), delete it so
    the SDK falls back to dynamic registration/public-client behavior. This
    prevents a removed client secret from staying active via cached
    ``client_info``. Legacy files written before Hermes added explicit
    dynamic/pre-registered markers are removed once as a conservative
    migration: they could be either dynamic registrations or removed
    pre-registered clients, and re-registering is safer than silently
    retaining a removed configured client_id. Redirect URI changes always
    invalidate stored client_info, even when cached tokens include a refresh
    token: if refresh is skipped or fails, reusing the old dynamic client_id
    with the new redirect_uri makes the next browser flow fail against OAuth
    servers that bind client registrations to exact redirect URIs.

    The active pre-registered ``client_id`` path is intentionally skipped
    here — :func:`_maybe_preregister_client` overwrites the file
    unconditionally on that path, so deletion would be wasted I/O.
    """
    if cfg.get("client_id"):
        return
    path = storage._client_info_path()
    if not path.exists():
        return
    existing = _read_json(path)
    if existing is None:
        return
    stored_uris = [str(u) for u in (existing.get("redirect_uris") or [])]
    new_uris = [str(u) for u in (client_metadata.redirect_uris or [])]
    stored_auth_method = existing.get("token_endpoint_auth_method")
    new_auth_method = client_metadata.token_endpoint_auth_method
    is_dynamic_client = existing.get("hermes_dynamic_client") is True
    has_refresh_token = _stored_tokens_have_refresh_token(storage)
    legacy_confidential_client = (
        not is_dynamic_client
        and (
            bool(existing.get("client_secret"))
            or (
                stored_auth_method is not None
                and stored_auth_method != "none"
                and stored_auth_method != new_auth_method
            )
        )
    )

    stale_reasons: list[str] = []
    if set(stored_uris) != set(new_uris):
        stale_reasons.append(f"redirect_uris {stored_uris} != {new_uris}")
    if existing.get("hermes_preregistered_client") is True:
        stale_reasons.append("stored client_info came from removed pre-registered config")
    elif not is_dynamic_client:
        if legacy_confidential_client:
            stale_reasons.append(
                "legacy unmarked confidential client_info could be removed pre-registered config"
            )
        elif has_refresh_token:
            logger.debug(
                "MCP OAuth '%s': keeping legacy unmarked client_info because "
                "cached tokens include a refresh token",
                storage._server_name,
            )
        else:
            stale_reasons.append("legacy unmarked client_info could be removed pre-registered config")
    elif stored_auth_method is not None and stored_auth_method != new_auth_method:
        stale_reasons.append(
            f"token_endpoint_auth_method {stored_auth_method!r} != {new_auth_method!r}"
        )

    if not stale_reasons:
        return
    try:
        path.unlink()
    except OSError as exc:
        logger.warning(
            "MCP OAuth '%s': failed to remove stale client_info at %s: %s",
            storage._server_name, path, exc,
        )
        return
    logger.info(
        "MCP OAuth '%s': removed stale client_info (%s) so SDK will re-register",
        storage._server_name, "; ".join(stale_reasons),
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
    redirect_uri = _build_redirect_uri(cfg)

    info_dict: dict[str, Any] = {
        "client_id": client_id,
        "redirect_uris": [redirect_uri],
        "grant_types": client_metadata.grant_types,
        "response_types": client_metadata.response_types,
        "token_endpoint_auth_method": client_metadata.token_endpoint_auth_method,
        "hermes_preregistered_client": True,
    }
    if cfg.get("client_secret"):
        info_dict["client_secret"] = cfg["client_secret"]
    if cfg.get("client_name"):
        info_dict["client_name"] = cfg["client_name"]
    if cfg.get("scope"):
        info_dict["scope"] = cfg["scope"]

    client_info = OAuthClientInformationFull.model_validate(info_dict)
    payload = client_info.model_dump(mode="json", exclude_none=True)
    payload["hermes_preregistered_client"] = True
    _write_json(storage._client_info_path(), payload)
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
        logger.warning(
            "MCP OAuth for '%s': non-interactive environment and no cached tokens "
            "found. The OAuth flow requires browser authorization. Run "
            "interactively first to complete the initial authorization, then "
            "cached tokens will be reused.",
            server_name,
        )

    port = _configure_callback_port(cfg)
    client_metadata = _build_client_metadata(cfg)
    _invalidate_stale_dynamic_client_info(storage, cfg, client_metadata)
    _maybe_preregister_client(storage, cfg, client_metadata)

    # Capture the resolved host/port in per-provider closures so a later
    # ``build_oauth_auth`` call for a different server can't retarget this
    # provider's callback listener or SSH hint via the module globals.
    callback_handler = _make_wait_for_callback(cfg["_resolved_bind_host"], port)
    redirect_handler = _make_redirect_handler(cfg["_resolved_uri_host"], port)

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=float(cfg.get("timeout", 300)),
    )
