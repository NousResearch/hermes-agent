"""Slim WebSocket-only gateway server for desktop / headless clients.

Eliminates the FastAPI/uvicorn/dashboard layer from the desktop boot path.
The desktop app spawns this instead of ``hermes serve`` when ``--ws-only`` is
passed; the renderer connects to the same ``handle_ws`` dispatch surface,
gets the same 158 RPC methods, the same event stream — without importing or
initializing a 19.8K-line web framework.

Wire protocol is identical to the dashboard's ``/api/ws`` route: the URL
carries ``?token=<HERMES_DASHBOARD_SESSION_TOKEN>`` for auth, then
newline-delimited JSON-RPC flows in both directions. ``gateway.ready`` is
emitted on connect, identical to the dashboard path.

Why not reuse uvicorn?  The desktop client is a single WS peer on loopback;
spinning up an ASGI server + HTTP router + middleware stack to serve one
WebSocket is the exact "nightmare of layers" this eliminates. The
``websockets`` library gives us a bare TCP listener with per-connection async
handlers — no HTTP framework, no route resolution, no SPA, no CORS.

Auth: constant-time ``hmac.compare_digest`` against ``_SESSION_TOKEN``
(read from ``HERMES_DASHBOARD_SESSION_TOKEN`` or auto-generated, matching
``hermes_cli.web_server._resolve_session_token``). The token is passed as a
``?token=`` query parameter, identical to the dashboard path the desktop
already uses.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse

_log = logging.getLogger(__name__)

# ── Token resolution (mirrors hermes_cli.web_server._resolve_session_token) ──

def _resolve_session_token() -> str:
    """Return the session token for WS auth.

    Matches the dashboard's resolution: an explicit
    ``HERMES_DASHBOARD_SESSION_TOKEN`` env var wins; otherwise we generate
    one so the server is never unauthenticated on loopback. The desktop's
    Electron main process reads the token from the backend's stdout
    sentinel (``HERMES_BACKEND_READY token=<value>``) and passes it to the
    renderer, which includes it in the WS URL as ``?token=<value>``.
    """
    return os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN") or secrets.token_urlsafe(32)


_SESSION_TOKEN = _resolve_session_token()


def get_session_token() -> str:
    """Expose the token for the entrypoint to print as a ready sentinel."""
    return _SESSION_TOKEN


# ── ASGI-compatible WebSocket shim ────────────────────────────────────────
#
# handle_ws expects an object with: accept(), receive_text(), send_text(),
# close(), and a .scope dict. The `websockets` library's ServerConnection
# (protocol v13) provides recv() and send() but NOT accept()/receive_text()/
# send_text() (those are starlette/FastAPI names). This thin adapter wraps
# the websockets connection to expose the interface handle_ws expects.


class _WSAdapter:
    """Adapt a ``websockets`` connection to the interface ``handle_ws`` expects."""

    def __init__(self, ws: Any, path: str = "/", query_string: str = "") -> None:
        self._ws = ws
        # starlette's WebSocket has a .scope dict; handle_ws reads
        # ws.scope["extensions"]["transport"] for TCP_NODELAY + keepalive.
        peer = ws.remote_address if hasattr(ws, "remote_address") else None
        self.scope: dict[str, Any] = {
            "type": "websocket",
            "query_string": query_string,
            "path": path,
            "client": peer,
            "extensions": {},
        }
        # Expose the underlying socket for _disable_nagle's TCP_NODELAY.
        try:
            transport = ws.transport  # websockets >=14
            self.scope["extensions"]["transport"] = transport
        except AttributeError:
            pass

    async def accept(self, subprotocol: str | None = None) -> None:
        # The websockets library auto-accepts on connect; nothing to do.
        # If a subprotocol is requested, it's handled at the handshake level.
        pass

    async def receive_text(self) -> str:
        data = await self._ws.recv()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)

    async def send_text(self, text: str) -> None:
        await self._ws.send(text)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code, reason)


class _WebSocketDisconnect(Exception):
    """Mirrors starlette's WebSocketDisconnect for handle_ws's except clause."""

    def __init__(self, code: int = 1000, reason: str = "") -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"code={code} reason={reason}")


# ── Auth ──────────────────────────────────────────────────────────────────

def _check_token(query_string: str) -> bool:
    """Constant-time token check against the ``?token=`` query parameter."""
    if not _SESSION_TOKEN:
        return False
    params = parse_qs(query_string)
    presented = params.get("token", [None])[0]
    if not presented:
        return False
    return hmac.compare_digest(presented, _SESSION_TOKEN)


# ── Connection handler ───────────────────────────────────────────────────

async def _handle_connection(ws: Any) -> None:
    """Per-connection handler for the websockets server.

    Auth → adapter → ``handle_ws`` (the same function the dashboard's
    ``/api/ws`` route calls). Events flow identically.
    """
    # websockets >=13 passes the path on the ws object, not as a second arg.
    path = getattr(ws, "path", None) or getattr(getattr(ws, "request", None), "path", "/")

    parsed = urlparse(path)
    query_string = parsed.query

    if not _check_token(query_string):
        _log.warning("ws-only: rejecting connection — token mismatch")
        await ws.close(code=4401, reason="unauthorized")
        return

    # Install our disconnect type so handle_ws's except clause catches it.
    import tui_gateway.ws as ws_mod
    ws_mod._WebSocketDisconnect = _WebSocketDisconnect

    adapter = _WSAdapter(ws, path=path, query_string=query_string)
    from tui_gateway.ws import handle_ws

    try:
        await handle_ws(adapter)
    except _WebSocketDisconnect:
        pass
    except Exception:
        _log.exception("ws-only: connection handler crashed")
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ── Server ────────────────────────────────────────────────────────────────

def run(host: str = "127.0.0.1", port: int = 0) -> None:
    """Start the slim WS server. Blocks until interrupted.

    Prints ``HERMES_BACKEND_READY port=<port> token=<token>`` to stdout so
    the desktop's Electron main process can discover the port (when
    ``--port 0``) and the token (when auto-generated), identical to how it
    parses the dashboard's ready sentinel.
    """
    import signal

    # tui_gateway.server does NOT import fastapi/starlette/uvicorn — it's
    # pure stdlib + agent modules. Importing it here is what boots the
    # dispatch surface (session store, thread pool, skin resolver, etc).
    from tui_gateway import server  # noqa: F401 — import side effects initialize the dispatch surface

    # Register all RPC method modules onto the server.
    from tui_gateway.methods_session import register as _reg_session
    _reg_session(server)
    # methods_complete, methods_prompt, etc. are auto-registered via
    # tui_gateway.server's own import-time @method() decorator.

    async def _main() -> None:
        import websockets

        ws_server = await websockets.serve(
            _handle_connection,
            host,
            port,
            # Match the dashboard's keepalive behaviour.
            ping_interval=20,
            ping_timeout=20,
        )

        # Discover the actual port (when port=0, the OS assigns one).
        sockets = ws_server.sockets
        actual_port = sockets[0].getsockname()[1] if sockets else port

        # Ready sentinel — the desktop's main.cjs parses this line.
        print(
            f"HERMES_BACKEND_READY port={actual_port} token={_SESSION_TOKEN}",
            flush=True,
        )
        _log.info("ws-only gateway listening on %s:%d", host, actual_port)

        # Block forever.
        await asyncio.Event().wait()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        _log.info("ws-only gateway shutting down")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
