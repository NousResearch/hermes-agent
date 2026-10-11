"""Server-minted ticket identity reaches the real WS pool and child authority cut."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from hermes_cli import web_server_chat, web_server
from hermes_cli.dashboard_auth.ws_tickets import mint_ticket, _reset_for_tests
from tui_gateway import server, ws
from tests.tui_gateway import test_host_conditional as hosts


@pytest.fixture
def hosted(monkeypatch, tmp_path):
    yield from hosts.hosted.__wrapped__(monkeypatch, tmp_path)


class Socket:
    def __init__(self, owner):
        self.client = SimpleNamespace(host="127.0.0.1", port=12345)
        self.url = SimpleNamespace(path="/api/ws")
        self.headers = {}
        self.query_params = {"ticket": mint_ticket(user_id=owner, provider="basic")}
        self.inbound, self.outbound = asyncio.Queue(), asyncio.Queue()

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


def test_ticket_owned_host_activation_and_input_survive_pool_hops_without_foreign_authority(hosted, monkeypatch):
    creator, binding, record, supervisor, _done = hosted
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(server, "resolve_skin", lambda: {})
    monkeypatch.setattr(server, "_live_transports", set())
    _reset_for_tests()

    async def call(socket, rid, method, params):
        await socket.inbound.put({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            frame = await asyncio.wait_for(socket.outbound.get(), 10)
            assert "_conditional_host_boot" not in frame
            if frame.get("id") == rid:
                return frame

    async def run():
        owned, foreign = Socket("alice"), Socket("bob")
        for socket in (owned, foreign):
            assert web_server_chat._ws_auth_reason(socket)[0] is None
        tasks = [asyncio.create_task(ws.handle_ws(socket, auth_identity=socket._hermes_auth_identity))
                 for socket in (owned, foreign)]
        try:
            params = {"session_id": binding["session_id"], "expected_binding": binding}
            assert (await call(owned, "join", "session.activate_bound", params))["result"]["accepted_binding"] == binding
            transport = next(peer for peer in server._live_transports if peer.auth_identity["user_id"] == "alice")
            assert (await call(owned, "repeat", "session.activate_bound", params))["result"]["attached"]
            assert hosts.control(supervisor, binding, "inspect")["members"] == 1
            assert (await call(owned, "prompt", "session.invoke_bound", {
                **params, "operation": {"method": "prompt.submit", "text": "hold"}}))["result"]["operation_result"]["status"] == "streaming"
            for _ in range(150):
                if hosts.control(supervisor, binding, "inspect")["holder"]:
                    break
                await asyncio.sleep(.02)
            assert hosts.control(supervisor, binding, "inspect")["holder"]
            assert (await call(foreign, "foreign", "session.activate_bound", params))["error"]["code"] == 4007
            assert hosts.control(supervisor, binding, "inspect")["members"] == 1
            assert (await call(owned, "bypass", "session.close", {"session_id": binding["session_id"]}))["error"]["code"] == 4007
            assert (await call(owned, "stop", "session.invoke_bound", {
                **params, "operation": {"method": "session.interrupt"}}))["result"]["operation_result"]["status"] == "interrupted"
            for _ in range(150):
                observation = hosts.control(supervisor, binding, "inspect")
                if not observation["running"]:
                    break
                await asyncio.sleep(.02)
            assert not observation["running"] and observation["holder"] is None
            assert observation["calls"] == ["warm", "hold"]
        finally:
            for socket in (owned, foreign):
                await socket.inbound.put(None)
            await asyncio.wait_for(asyncio.gather(*tasks), 10)
        assert transport not in record["host_bound_subscribers"]
        assert not server._session_transport_contains(record, transport)
        assert server._session_transport_contains(record, creator)
    try:
        asyncio.run(run())
    finally:
        _reset_for_tests()
