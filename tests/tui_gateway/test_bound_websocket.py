"""Ticket-authenticated WS dispatch preserves conditional membership and answer fencing."""
import asyncio
import json
from types import SimpleNamespace

from hermes_cli import web_server_chat, web_server
from hermes_cli.dashboard_auth.ws_tickets import mint_ticket, _reset_for_tests
from tui_gateway import server, server_requests, ws
import pytest

from tests.tui_gateway import test_running_conditional_activation as running


@pytest.fixture
def built(monkeypatch, tmp_path):
    yield from running.built.__wrapped__(monkeypatch, tmp_path)


def test_ticket_identity_reaches_real_ws_dispatch_and_disconnect_fences_operations(built, monkeypatch):
    creator, binding, record, agent, db = built
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(server, "resolve_skin", lambda: {})
    monkeypatch.setattr(server, "_live_transports", set())
    _reset_for_tests()
    class Socket:
        def __init__(self, owner):
            self.client = SimpleNamespace(host="127.0.0.1", port=12345)
            self.url = SimpleNamespace(path="/api/ws")
            self.headers = {}
            self.query_params = {"ticket": mint_ticket(user_id=owner, provider="basic")}
            self.inbound = asyncio.Queue()
            self.outbound = asyncio.Queue()
        async def accept(self):
            pass
        async def receive_text(self):
            frame = await self.inbound.get()
            if frame is None:
                raise ws._WebSocketDisconnect()
            return json.dumps(frame)
        async def send_text(self, line):
            await self.outbound.put(json.loads(line))
        async def close(self):
            pass
    async def run():
        socket = Socket("alice")
        assert web_server_chat._ws_auth_reason(socket)[0] is None
        task = asyncio.create_task(ws.handle_ws(socket, auth_identity=socket._hermes_auth_identity))
        async def call(rid, method, params):
            await socket.inbound.put({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            while True:
                frame = await asyncio.wait_for(socket.outbound.get(), 5)
                if frame.get("id") == rid:
                    return frame
        try:
            result = await call("join", "session.activate_bound", {
                "session_id": binding["session_id"], "expected_binding": binding})
            assert result["result"]["accepted_binding"] == binding
            transport = next(iter(server._live_transports))
            assert transport.auth_identity == {"user_id": "alice", "provider": "basic"}
            assert server._session_transport_contains(record, transport)
            assert (await call("repeat", "session.activate_bound", {
                "session_id": binding["session_id"], "expected_binding": binding}))["result"] == result["result"]
            assert (await call("bypass", "session.close", {"session_id": binding["session_id"]}))["error"]["code"] == 4007
            assert (await call("malformed", "session.invoke_bound", {
                "session_id": binding["session_id"], "expected_binding": binding,
                "operation": {"method": "session.close"}}))["error"]["code"] == 4000
            pending = server_requests.ServerRequest(binding["session_id"], "clarify", {"prompt": "confirm"})
            with server_requests._lock:
                server_requests._open[pending.id] = pending
            try:
                await socket.inbound.put({"jsonrpc": "2.0", "id": pending.id, "result": {"answer": "raw"}})
                assert (await call("ping", "gateway.ping", {}))["result"]["ok"]
                assert not pending.answered
                response = await call("answer", "session.invoke_bound", {
                    "session_id": binding["session_id"], "expected_binding": binding,
                    "operation": {"method": "request.answer", "id": pending.id, "result": {"answer": "accepted"}}})
                assert response["result"]["operation_result"] == {"status": "ok"}
                assert pending.result == {"answer": "accepted"}
            finally:
                with server_requests._lock:
                    server_requests._open.pop(pending.id, None)
        finally:
            await socket.inbound.put(None)
            await asyncio.wait_for(task, 5)
        assert transport not in record["bound_subscribers"]
        assert not server._session_transport_contains(record, transport)
        # A replacement connection with a different server-minted identity refuses.
        other = Socket("bob")
        assert web_server_chat._ws_auth_reason(other)[0] is None
        stranger = ws.WSTransport(other, asyncio.get_running_loop(), auth_identity=other._hermes_auth_identity)
        reply = await asyncio.to_thread(server.dispatch, {
            "id": "foreign", "method": "session.activate_bound", "params": {
                "session_id": binding["session_id"], "expected_binding": binding}}, stranger)
        assert reply["error"]["code"] == 4007
        assert not server._session_transport_contains(record, stranger)
        assert server._session_transport_contains(record, creator)
    try:
        asyncio.run(run())
    finally:
        _reset_for_tests()
