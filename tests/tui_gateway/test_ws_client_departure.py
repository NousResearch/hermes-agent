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
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocket, WebSocketState

from tui_gateway import server
from tui_gateway import ws as ws_mod


@pytest.fixture(autouse=True)
def isolate_gateway_background_work(monkeypatch):
    monkeypatch.setattr(ws_mod, "_note_dashboard_client_activity", lambda **kwargs: None)
    monkeypatch.setattr(ws_mod, "_start_backend_liveness", lambda: None)
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "_ensure_lease_watcher", lambda: None)
    monkeypatch.setattr(server, "register_live_transport", lambda transport: None)
    monkeypatch.setattr(server, "unregister_live_transport", lambda transport: None)
    monkeypatch.setattr(server, "_release_wake_for_transport", lambda transport: None)
    monkeypatch.setattr(server, "resolve_skin", dict)


def _rpc(req_id, method="fast"):
    return {"type": "websocket.receive", "text": json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": {}})}


def _harness(monkeypatch, send_error, req_id="r1"):
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
    inbound = [{"type": "websocket.connect"}, _rpc(req_id, method=req_id)]
    after_failure = [_rpc("r2")]

    async def receive():
        if inbound:
            return inbound.pop(0)
        await reply_failed.wait()
        if after_failure:
            return after_failure.pop(0)
        return {"type": "websocket.disconnect", "code": 1006}

    async def send(message):
        if message["type"] == "websocket.send" and json.loads(message["text"]).get("id") == req_id:
            reply_failed.set()
            raise send_error

    ws = WebSocket({"type": "websocket", "path": "/api/ws", "headers": [], "client": ("203.0.113.9", 4242), "app": SimpleNamespace(state=SimpleNamespace())}, receive, send)
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
    secret = "Bearer synthetic-token-ABC123"
    ws = _harness(monkeypatch, RuntimeError(secret), req_id=secret)

    closed = _run(ws, caplog)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(m.startswith("ws send failed") for m in warnings), warnings
    assert any(m.startswith("ws reply send failed") for m in warnings), warnings
    assert secret not in caplog.text
    assert "send_failures=1" in closed


def test_ready_send_departure_is_quiet(monkeypatch, caplog):
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        if message["type"] == "websocket.send":
            raise OSError("disconnected")

    ws = WebSocket({"type": "websocket", "path": "/api/ws", "headers": [], "client": ("203.0.113.9", 4242), "app": SimpleNamespace(state=SimpleNamespace())}, receive, send)
    closed = _run(ws, caplog)
    assert "reason=client_disconnect(peer_gone)" in closed
    assert "send_failures=0" in closed
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_ready_send_live_failure_is_error_without_exception_text(monkeypatch, caplog):
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        if message["type"] == "websocket.send":
            raise RuntimeError("Bearer synthetic-token-ABC123")

    ws = WebSocket({"type": "websocket", "path": "/api/ws", "headers": [], "client": ("203.0.113.9", 4242), "app": SimpleNamespace(state=SimpleNamespace())}, receive, send)
    closed = _run(ws, caplog)
    assert "reason=ready_send_failed" in closed
    assert "send_failures=1" in closed
    assert any(r.levelno == logging.ERROR and "ready frame send failed" in r.getMessage() for r in caplog.records)
    assert "synthetic-token-ABC123" not in caplog.text


def test_server_closed_socket_is_not_mistaken_for_peer_departure():
    async def run():
        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            pass

        ws = WebSocket({"type": "websocket", "path": "/api/ws", "headers": []}, receive, send)
        await ws.accept()
        await ws.close()
        assert ws.client_state == WebSocketState.CONNECTED
        assert ws.application_state == WebSocketState.DISCONNECTED
        assert not ws_mod.ws_peer_gone(ws, RuntimeError("server closed socket"))

    asyncio.run(run())


def test_latched_receive_failure_is_quiet(caplog):
    ws = SimpleNamespace(client_state=WebSocketState.CONNECTED, application_state=WebSocketState.DISCONNECTED)
    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")
    assert ws_mod._receive_failure_reason(ws, "connected", "203.0.113.9:4242", RuntimeError("synthetic-token-ABC123"), peer_gone=True) == "client_disconnect(peer_gone)"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_client_close_reason_survives_inflight_reply_failure(monkeypatch, caplog):
    """The reader records the close before the dispatcher finishes its in-flight request."""
    original_queue = asyncio.Queue

    async def run():
        dispatch_started = asyncio.Event()
        drain_started = asyncio.Event()

        class NotifyingQueue(original_queue):
            async def put(self, item):
                if not isinstance(item, dict):
                    drain_started.set()
                return await super().put(item)

        monkeypatch.setattr(ws_mod.asyncio, "Queue", NotifyingQueue)
        monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))

        async def dispatch(authority, req, method, transport):
            dispatch_started.set()
            await drain_started.wait()
            return {"jsonrpc": "2.0", "id": req["id"], "result": {}}

        monkeypatch.setattr(ws_mod, "_dispatch_request", dispatch)
        inbound = [{"type": "websocket.connect"}, _rpc("r1")]

        async def receive():
            if inbound:
                return inbound.pop(0)
            await dispatch_started.wait()
            return {"type": "websocket.disconnect", "code": 1001, "reason": "going away"}

        async def send(message):
            if message["type"] == "websocket.send" and json.loads(message["text"]).get("id") == "r1":
                raise OSError("client disconnected")

        ws = WebSocket({"type": "websocket", "path": "/api/ws", "headers": [], "client": ("203.0.113.9", 4242), "app": SimpleNamespace(state=SimpleNamespace())}, receive, send)
        await asyncio.wait_for(ws_mod.handle_ws(ws), 5)

    caplog.set_level(logging.DEBUG, logger="tui_gateway.ws")
    asyncio.run(run())
    closed = [r.getMessage() for r in caplog.records if r.getMessage().startswith("ws closed")]
    assert len(closed) == 1
    assert "reason=client_disconnect(code=1001,reason=going away)" in closed[0]
    assert "send_failures=0" in closed[0]


def test_dynamic_exception_type_is_not_logged(monkeypatch, caplog):
    secret = "synthetic-token-ABC123"
    ws = _harness(monkeypatch, type(secret, (RuntimeError,), {})())
    _run(ws, caplog)
    assert secret not in caplog.text

    ws = SimpleNamespace(client_state=WebSocketState.CONNECTED)
    ws_mod._receive_failure_reason(ws, "connected", "peer", type(secret, (RuntimeError,), {})())
    assert secret not in caplog.text


def test_unexpected_receive_failure_on_a_live_socket_is_still_an_error(monkeypatch, caplog):
    """A live receive failure remains an ERROR without leaking its exception text."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))
    inbound = [{"type": "websocket.connect"}, {"type": "websocket.receive", "bytes": b"\x00"}]

    async def receive():
        if inbound:
            message = inbound.pop(0)
            if message["type"] == "websocket.connect":
                return message
        raise RuntimeError("Bearer synthetic-token-ABC123")

    async def send(message):
        pass

    ws = WebSocket({"type": "websocket", "path": "/api/ws", "headers": [], "client": ("203.0.113.9", 4242), "app": SimpleNamespace(state=SimpleNamespace())}, receive, send)

    closed = _run(ws, caplog)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == ["ws receive failed peer=203.0.113.9:4242"]
    assert errors[0].exc_info is None
    assert "synthetic-token-ABC123" not in caplog.text
    assert "reason=receive_failed" in closed
