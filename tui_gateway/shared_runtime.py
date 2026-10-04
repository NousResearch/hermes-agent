"""Local attachment ingress for the owning stdio TUI gateway process.

Desktop/serve advertises its existing listener; stdio TUI exposes the same
handshake and WebSocket dispatcher on a local socket. Session
ownership, prompt acceptance and queue draining remain with the gateway.
"""

from __future__ import annotations

import atexit
import hmac
import secrets
import socket
import sys
import threading
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request, WebSocket

_origin: str | None = None
_server = None
_thread: threading.Thread | None = None
_lifecycle_lock = threading.Lock()


def attach_origin() -> str | None:
    if _origin is not None:
        return _origin
    web = sys.modules.get("hermes_cli.web_server")
    return getattr(web.app.state, "session_attach_origin", None) if web is not None else None


def _attach_reply(session_id: str, lease_id: str, profile_home: str, websocket_url: str) -> dict:
    from tui_gateway import server

    # Match the owner's live record, not the identity echoed by the caller.
    # The lease's registry path is its owning home, including multiplexed sessions.
    with server._session_resume_lock, server._sessions_lock:
        for sid, session in server._sessions.items():
            lease = session.get("active_session_lease")
            if (session.get("_closing") or session.get("_finalized") or lease is None
                    or not lease.enabled or lease.released):
                continue
            home = str(Path(lease.state_path).parent.parent.resolve())
            if (lease.session_id == session_id and lease.lease_id == lease_id and home == profile_home
                    and server._session_lookup_key(session, fallback=sid) == session_id):
                return {"session_id": session_id, "lease_id": lease_id, "profile_home": home,
                        "websocket_url": websocket_url}
    raise HTTPException(status_code=409, detail="Live session owner identity does not match.")


def start_shared_runtime() -> None:
    """Bind once before client RPCs can claim leases; never announce credentials."""
    global _origin, _server, _thread
    with _lifecycle_lock:
        if _server is not None:
            return
        import uvicorn
        from agent.memory_provider import spawn_context_thread
        from tui_gateway.ws import handle_ws

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        authority = f"127.0.0.1:{listener.getsockname()[1]}"
        token = secrets.token_urlsafe(32)
        websocket_url = f"ws://{authority}/api/ws?{urlencode({'token': token})}"
        app = FastAPI()

        @app.get("/api/session-attach")
        def attach(request: Request, session_id: str, lease_id: str, profile_home: str):
            # Native loopback callers only: browser origins and rebinding hostnames
            # cannot fetch the credential, even if they know a session ID.
            if request.headers.get("host") != authority or request.headers.get("origin"):
                raise HTTPException(status_code=403, detail="Local attachment only.")
            return _attach_reply(session_id, lease_id, profile_home, websocket_url)

        @app.websocket("/api/ws")
        async def ws(ws: WebSocket):
            if (ws.headers.get("host") != authority or ws.headers.get("origin")
                    or not hmac.compare_digest(ws.query_params.get("token", "").encode(), token.encode())):
                await ws.close(code=4401)
                return
            await handle_ws(ws)

        runtime_server = uvicorn.Server(uvicorn.Config(
            app, log_level="error", log_config=None, access_log=False, proxy_headers=False,
            ws_ping_interval=None, timeout_graceful_shutdown=1))
        _server = runtime_server
        _thread = spawn_context_thread(lambda: runtime_server.run(sockets=[listener]), name="shared-runtime")
        _thread.start()
        # The socket is already listening. Requests can wait in its backlog until
        # uvicorn completes startup, so first-turn leases can advertise immediately.
        _origin = f"http://{authority}"
        atexit.register(stop_shared_runtime)


def stop_shared_runtime() -> None:
    global _origin, _server, _thread
    with _lifecycle_lock:
        if _server is not None:
            _server.should_exit = True
            _thread.join(timeout=3)
            if _thread.is_alive():
                raise RuntimeError("Shared runtime server thread did not stop.")
            _origin = None
            _server = None
            _thread = None
