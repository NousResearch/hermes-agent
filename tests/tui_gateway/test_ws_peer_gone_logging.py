"""A client that has already gone away is a normal disconnect, not a send failure worth a warning.

Desktop reloads, a closed laptop lid or a dropped tunnel all end the socket while frames are in
flight. Starlette reports that as ``WebSocketDisconnect`` from ``send_text`` (or, once the close has
been processed, uvicorn's "Unexpected ASGI message 'websocket.send'" ``RuntimeError``). The
``ws closed ... reason=`` line already records it; the send path must not add WARNING/ERROR lines
to errors.log for it, while a send failure on a live socket still warns.
"""

from __future__ import annotations

import asyncio
import logging

from starlette.websockets import WebSocketDisconnect, WebSocketState

from tui_gateway import server
from tui_gateway import ws as ws_mod
from tui_gateway.ws import WSTransport

_AFTER_CLOSE = "Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response already completed."


class _GoneWS:
    """``send_text`` fails the way Starlette/uvicorn fail it for a peer that is already gone."""

    def __init__(self, exc: BaseException, *, client_state: WebSocketState = WebSocketState.CONNECTED) -> None:
        self.exc = exc
        self.client_state = client_state
        self.application_state = WebSocketState.CONNECTED

    async def accept(self) -> None:
        pass

    async def send_text(self, line: str) -> None:
        raise self.exc

    async def receive_text(self) -> str:
        raise WebSocketDisconnect(code=1006)

    async def close(self, code: int = 1000) -> None:
        pass


def _ws_records(caplog, min_level: int = logging.WARNING) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "tui_gateway.ws" and r.levelno >= min_level]


def _send_once(ws) -> tuple[bool, bool]:
    async def _run() -> tuple[bool, bool]:
        transport = WSTransport(ws, asyncio.get_running_loop(), peer="127.0.0.1:1")
        ok = await transport.write_async({"jsonrpc": "2.0", "result": {"ok": True}, "id": 1})
        return ok, transport.closed

    return asyncio.run(_run())


def test_send_to_disconnected_peer_latches_closed_without_warning(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")
    ok, closed = _send_once(_GoneWS(WebSocketDisconnect(code=1006)))
    assert (ok, closed) == (False, True)
    assert _ws_records(caplog) == []


def test_send_after_close_was_processed_latches_closed_without_warning(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")
    ws = _GoneWS(RuntimeError(_AFTER_CLOSE), client_state=WebSocketState.DISCONNECTED)
    ok, closed = _send_once(ws)
    assert (ok, closed) == (False, True)
    assert _ws_records(caplog) == []


def test_send_failure_on_live_socket_still_warns(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")
    ok, closed = _send_once(_GoneWS(RuntimeError("serializer blew up")))
    assert (ok, closed) == (False, True)
    assert [r.levelno for r in _ws_records(caplog)] == [logging.WARNING]


def test_peer_gone_before_ready_frame_is_not_an_error(monkeypatch, caplog) -> None:
    _quiet_server(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")

    asyncio.run(ws_mod.handle_ws(_GoneWS(WebSocketDisconnect(code=1006))))

    assert _ws_records(caplog) == []
    closed = [r.getMessage() for r in _ws_records(caplog, logging.INFO) if r.getMessage().startswith("ws closed")]
    assert len(closed) == 1 and "reason=ready_send_failed" in closed[0] and "send_failures=1" in closed[0]


class _LeavesMidReplyWS(_GoneWS):
    """Takes the ready frame, sends one heartbeat, and is gone by the time the reply goes out."""

    def __init__(self) -> None:
        super().__init__(WebSocketDisconnect(code=1006))
        self._sent = 0
        self._pinged = False

    async def send_text(self, line: str) -> None:
        self._sent += 1
        if self._sent > 1:
            raise self.exc

    async def receive_text(self) -> str:
        if self._pinged:
            raise WebSocketDisconnect(code=1006)
        self._pinged = True
        return '{"jsonrpc": "2.0", "method": "gateway.ping", "id": 7}'


def test_peer_gone_before_reply_is_not_a_warning(monkeypatch, caplog) -> None:
    _quiet_server(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")

    asyncio.run(ws_mod.handle_ws(_LeavesMidReplyWS()))

    assert _ws_records(caplog) == []
    closed = [r.getMessage() for r in _ws_records(caplog, logging.INFO) if r.getMessage().startswith("ws closed")]
    assert len(closed) == 1 and "reason=send_failed_after_heartbeat" in closed[0]


def _quiet_server(monkeypatch) -> None:
    monkeypatch.setattr(server, "resolve_skin", lambda: {})
    monkeypatch.setattr(server, "_start_backend_heartbeat_refresher", lambda: None)
    monkeypatch.setattr(server, "_schedule_startup_orphan_sweep", lambda: None)
    for name in ("_ensure_skin_watcher", "_ensure_lease_watcher", "register_live_transport",
                 "unregister_live_transport", "_release_wake_for_transport"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda *a, **k: (0, 0))
