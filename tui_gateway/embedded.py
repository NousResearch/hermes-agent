"""In-process transport for hosts that embed the interpreter (a mobile app over JNI, a Swift shell):
frames leave through a host callback and requests arrive as JSON lines, with no stdio or socket.

The host calls :meth:`EmbeddedTransport.submit` from one thread, in arrival order — the same
one-at-a-time contract as the WebSocket reader. Long handlers still run on the dispatch pool and
answer through :meth:`EmbeddedTransport.write`, so ``emit`` is called from several threads. ``emit``
must only hand the frame off: calling ``submit`` from inside it would re-enter the write lock."""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable

from tui_gateway.transport import serialize_frame

logger = logging.getLogger(__name__)


class EmbeddedTransport:
    """``emit(line) -> bool`` receives one serialized frame; ``False`` means the host is gone for good.
    ``connect`` attaches the host as a gateway peer; ``close`` detaches it."""

    __slots__ = ("_closed", "_emit", "_lock")

    def __init__(self, emit: Callable[[str], bool]) -> None:
        self._emit = emit
        # Pool workers write concurrently; a host callback (a JNI upcall) need not be reentrant.
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def connect(cls, emit: Callable[[str], bool]) -> EmbeddedTransport:
        """Attach a host as a gateway peer, as ``ws.handle_ws`` does for a WebSocket client: greet it with
        ``gateway.ready`` and register it for session-less global broadcasts."""
        from tui_gateway import server
        from tui_gateway.event_replay import replay_epoch

        transport = cls(emit)
        if transport.write({"jsonrpc": "2.0", "method": "event", "params": {"type": "gateway.ready", "payload": {
                "skin": server.resolve_skin(), "change_events": True, "replay_epoch": replay_epoch()}}}):
            server._ensure_skin_watcher()
            server._ensure_lease_watcher()
            server.register_live_transport(transport)
        return transport

    def write(self, obj: dict) -> bool:
        line = serialize_frame(obj, "embedded", logger)
        with self._lock:
            if self._closed:
                return False
            if not self._emit(line):
                # The host is gone: latch it like WSTransport, which the session reaper relies on.
                self._closed = True
            return not self._closed

    def close(self) -> None:
        """Detach from the gateway: stop global broadcasts and park or reap this host's sessions."""
        from tui_gateway import server

        with self._lock:
            self._closed = True
        server.unregister_live_transport(self)
        server._close_sessions_for_transport(self, end_reason="embedded_disconnect")

    def submit(self, line: str) -> None:
        """Dispatch one inbound JSON-RPC line; its inline response (if any) goes back through ``write``."""
        from tui_gateway import server

        if self._closed:
            return
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            self.write({"jsonrpc": "2.0", "error": {"code": -32700, "message": "parse error"}, "id": None})
            return
        try:
            resp = server.dispatch(req, self)
        except Exception:
            rid = req.get("id") if isinstance(req, dict) else None
            logger.exception("embedded dispatch crash id=%s", rid)
            resp = {"jsonrpc": "2.0", "error": {"code": -32603, "message": "internal error"}, "id": rid}
        if resp is not None:
            self.write(resp)
