"""A client that leaves with a reply in flight is a disconnect, not a server fault.

When a reply hits a socket the client already dropped, Starlette raises ``WebSocketDisconnect(1006)``
and latches ``application_state = DISCONNECTED``; the read loop's next ``receive_text`` then raises
``RuntimeError('WebSocket is not connected. Need to call "accept" first.')``. Both used to be logged
(two WARNINGs plus an ERROR traceback per departure), so one dashboard tab stuck in a reconnect loop
filled ``errors.log`` with ~20k tracebacks an hour. These tests drive ``handle_ws`` over a REAL
Starlette ``WebSocket`` (fake ASGI receive/send) so the state machine under test is Starlette's own.
"""

import asyncio
import json
import logging

from starlette.websockets import WebSocket, WebSocketState

from tui_gateway import server
from tui_gateway import ws as ws_mod


def _rpc(req_id, method="fast"):
    return {"type": "websocket.receive", "text": json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": {}})}


def _harness(monkeypatch, send_error):
    """handle_ws over a real Starlette WebSocket whose reply to ``r1`` raises *send_error*.

    The client sends ``r1`` and, once that reply has failed, one more frame, so the read loop goes back
    to ``receive_text`` on a socket Starlette has already marked DISCONNECTED."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))
    monkeypatch.setattr(
        server, "dispatch",
        lambda req, transport: {"jsonrpc": "2.0", "id": req.get("id"), "result": {}},
    )
    reply_failed = asyncio.Event()
    inbound = [{"type": "websocket.connect"}, _rpc("r1")]
    after_failure = [_rpc("r2")]

    async def receive():
        if inbound:
            return inbound.pop(0)
        await reply_failed.wait()
        if after_failure:
            return after_failure.pop(0)
        return {"type": "websocket.disconnect", "code": 1006}

    async def send(message):
        if message["type"] == "websocket.send" and json.loads(message["text"]).get("id") == "r1":
            reply_failed.set()
            raise send_error

    ws = WebSocket({"type": "websocket", "path": "/api/ws", "headers": [], "client": ("203.0.113.9", 4242)}, receive, send)
    return ws


def _run(ws, caplog):
    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")
    asyncio.run(asyncio.wait_for(ws_mod.handle_ws(ws), 5))
    closed = [r.getMessage() for r in caplog.records if r.getMessage().startswith("ws closed")]
    assert len(closed) == 1, closed
    return closed[0]


def test_reply_to_a_departed_client_is_not_logged_as_a_failure(monkeypatch, caplog):
    """The storm shape: the reply meets a dropped TCP leg, then the reader re-enters receive_text."""
    ws = _harness(monkeypatch, OSError("client disconnected"))

    closed = _run(ws, caplog)

    assert ws.application_state == WebSocketState.DISCONNECTED  # the real Starlette latch was exercised
    noisy = [(r.levelname, r.getMessage()) for r in caplog.records if r.levelno >= logging.WARNING]
    assert noisy == []
    assert "reason=client_disconnect(peer_gone)" in closed
    assert "send_failures=0" in closed


def test_send_failure_on_a_live_peer_still_warns(monkeypatch, caplog):
    """A send that fails while Starlette still holds the socket CONNECTED is a real fault and keeps its WARNING."""
    ws = _harness(monkeypatch, RuntimeError("serializer exploded"))

    closed = _run(ws, caplog)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(m.startswith("ws send failed") for m in warnings), warnings
    assert any(m.startswith("ws response send failed") for m in warnings), warnings
    assert "send_failures=1" in closed


def test_unexpected_receive_failure_on_a_live_socket_is_still_an_error(monkeypatch, caplog):
    """Only a departure is quieted: a receive that blows up on a CONNECTED socket keeps its ERROR traceback."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))
    inbound = [{"type": "websocket.connect"}, {"type": "websocket.receive", "bytes": b"\x00"}]

    async def receive():
        return inbound.pop(0)

    async def send(message):
        pass

    ws = WebSocket({"type": "websocket", "path": "/api/ws", "headers": [], "client": ("203.0.113.9", 4242)}, receive, send)

    closed = _run(ws, caplog)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == ["ws receive failed peer=203.0.113.9:4242"]
    assert errors[0].exc_info is not None
    assert "reason=receive_failed" in closed
