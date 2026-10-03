"""A client that closes its socket while an RPC is still in flight is an ordinary departure, not an error.

Desktop closes and redials its socket on wake (heartbeat deadline, liveness probe, forced reconnect), often
right after sending a ``ping`` or a status RPC. The reply to that RPC then meets a socket the peer has already
left. Starlette reports this in three shapes, depending on the ASGI server's timing:

* the server already delivered the client's close: ``send`` raises uvicorn's ``RuntimeError("Unexpected ASGI
  message 'websocket.send', after sending 'websocket.close' ...")``;
* the TCP leg is gone: ``send`` raises ``WebSocketDisconnect(1006)`` and latches
  ``application_state = DISCONNECTED``;
* after that latch, the read loop's next ``receive_text()`` raises ``RuntimeError('WebSocket is not
  connected. Need to call "accept" first.')``.

Each used to be logged as a WARNING (twice) or as an ERROR with a traceback in errors.log, and the close was
recorded as ``send_failed_after_response`` / ``receive_failed`` instead of a client disconnect. A phone client
(Hermes Desktop in an Android WebView) produced one such entry on most returns to the foreground, which buried
real errors. A send that fails while the peer is still connected keeps its WARNING.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from starlette.websockets import WebSocketDisconnect, WebSocketState

from tui_gateway import server
from tui_gateway import ws as ws_mod

_AFTER_CLOSE = (
    "Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response already completed."
)
_NOT_CONNECTED = 'WebSocket is not connected. Need to call "accept" first.'


def _rpc(req_id: str, method: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": {}}


@pytest.fixture
def gateway(monkeypatch):
    """Inline dispatch that answers every RPC; session teardown stubbed."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(
        server, "dispatch", lambda req, transport: {"jsonrpc": "2.0", "id": req.get("id"), "result": {"ok": True}}
    )
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))


class _StarletteLikeWS:
    """Scripted socket carrying Starlette's two state fields, as ``tui_gateway.ws`` sees a real one."""

    def __init__(self, frames: list[dict]) -> None:
        self.client_state = WebSocketState.CONNECTED
        self.application_state = WebSocketState.CONNECTED
        self._frames = list(frames)

    async def accept(self) -> None:
        pass

    async def close(self, code: int = 1000) -> None:
        self.application_state = WebSocketState.DISCONNECTED

    async def send_text(self, line: str) -> None:
        pass

    async def receive_text(self) -> str:
        if self.application_state != WebSocketState.CONNECTED:
            raise RuntimeError(_NOT_CONNECTED)
        if self._frames:
            return json.dumps(self._frames.pop(0))
        await asyncio.Event().wait()  # an idle, open socket
        raise AssertionError("unreachable")


def _ws_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "tui_gateway.ws"]


def _closed_reason(caplog) -> str:
    closed = [r.getMessage() for r in _ws_records(caplog) if r.getMessage().startswith("ws closed")]
    assert len(closed) == 1, closed
    return closed[0].split("reason=", 1)[1].split(" messages=", 1)[0]


def test_reply_after_client_close_is_a_client_disconnect(gateway, caplog):
    """The client's close frame is already in: the reply meets uvicorn's 'after websocket.close' RuntimeError."""

    class WS(_StarletteLikeWS):
        async def receive_text(self) -> str:
            if self._frames:
                return await super().receive_text()
            self.client_state = WebSocketState.DISCONNECTED  # Starlette saw websocket.disconnect
            raise WebSocketDisconnect(1005)

        async def send_text(self, line: str) -> None:
            if json.loads(line).get("id") == "r1":
                raise RuntimeError(_AFTER_CLOSE)

    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")
    asyncio.run(asyncio.wait_for(ws_mod.handle_ws(WS([_rpc("r1", "ping")])), 5))

    noisy = [r for r in _ws_records(caplog) if r.levelno >= logging.WARNING]
    assert noisy == [], [r.getMessage() for r in noisy]
    assert _closed_reason(caplog).startswith("client_disconnect"), _closed_reason(caplog)


def test_receive_after_send_saw_the_peer_leave_is_not_an_error(gateway, caplog):
    """The reply hits a dropped TCP leg (WebSocketDisconnect, application_state latched) and the read loop's
    next receive_text() then raises Starlette's 'Need to call "accept" first' RuntimeError."""

    class WS(_StarletteLikeWS):
        latched: asyncio.Event

        async def accept(self) -> None:
            self.latched = asyncio.Event()

        async def receive_text(self) -> str:
            if len(self._frames) == 1:
                # r2 was buffered before the drop; the read loop picks it up while r1's reply is failing.
                await self.latched.wait()
            return await super().receive_text()

        async def send_text(self, line: str) -> None:
            if json.loads(line).get("id") == "r1":
                self.application_state = WebSocketState.DISCONNECTED
                self.latched.set()
                # websockets' ensure_open() awaits the closing handshake before raising, so the read loop
                # runs (and re-enters receive_text) while this send is still unwinding.
                await asyncio.sleep(0.05)
                raise WebSocketDisconnect(1006)

    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")
    asyncio.run(asyncio.wait_for(ws_mod.handle_ws(WS([_rpc("r1", "ping"), _rpc("r2", "ping")])), 5))

    noisy = [r for r in _ws_records(caplog) if r.levelno >= logging.WARNING or r.exc_info]
    assert noisy == [], [(r.levelname, r.getMessage()) for r in noisy]
    assert _closed_reason(caplog).startswith("client_disconnect"), _closed_reason(caplog)


def test_send_failure_on_a_connected_peer_still_warns(gateway, caplog):
    """Only a peer that is known to have left is quiet: a send that fails on a live socket is still a WARNING."""

    class WS(_StarletteLikeWS):
        async def send_text(self, line: str) -> None:
            if json.loads(line).get("id") == "r1":
                raise RuntimeError("transport write failed")

    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")
    asyncio.run(asyncio.wait_for(ws_mod.handle_ws(WS([_rpc("r1", "ping")])), 5))

    warnings = [r.getMessage() for r in _ws_records(caplog) if r.levelno == logging.WARNING]
    assert any(m.startswith("ws send failed") for m in warnings), warnings
    assert _closed_reason(caplog) == "send_failed_after_response"
