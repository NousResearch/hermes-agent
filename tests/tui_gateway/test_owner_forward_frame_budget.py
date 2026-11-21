"""Owner forwarding accepts full resume histories while retaining bounded socket reads.

The relay, socket dial, RPC client, authority and admission ledger are real. Only the
private bootstrap ticket issuer is replaced; all sockets and data belong to this test.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_cli import gateway_client
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions
from tui_gateway import owner_forward


@pytest.fixture
def ticket(monkeypatch):
    scopes = []

    def issue(home, endpoint, *, scope=None):
        scopes.append(scope)
        return "test-ticket"

    monkeypatch.setattr(gateway_client, "_session_ticket", issue)
    return scopes


@pytest.mark.asyncio
@pytest.mark.parametrize("history_bytes,accepted", [(2_789_219, True), (9 * 1024 * 1024, False)])
async def test_resume_frame_budget_gates_exactly_one_forwarded_admission(tmp_path, monkeypatch, ticket,
                                                                      history_bytes, accepted):
    db = SessionDB(db_path=tmp_path / "state.db")
    key = "frame-budget-session"
    db.create_session(key, "test")
    db.append_message(key, role="assistant", content="x" * history_bytes)
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="owned", instance_id="gateway",
                                 db=db, epoch=begin_runtime_epoch(db, instance_id="gateway"))
    authority.sessions[key] = LiveSession(SimpleNamespace(platform=None, user_id="desktop"), "route")
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)  # no model runs in this test
    seen, frame_sizes, close_codes, emitted = [], [], [], []

    async def handler(ws):
        loop = asyncio.get_running_loop()
        writes = asyncio.Queue()

        class Transport:
            def write(self, frame):
                loop.call_soon_threadsafe(writes.put_nowait, frame)
                return True

            def close(self):
                return None

        conn = AuthorityConnection(authority, Transport(), {"user_id": "desktop"})
        try:
            async for raw in ws:
                request = json.loads(raw)
                seen.append(request["method"])
                reply = await conn.dispatch(request)
                while not writes.empty():
                    await ws.send(json.dumps(writes.get_nowait()))
                answer = json.dumps(reply)
                frame_sizes.append(len(answer.encode()))
                await ws.send(answer)
                if request["method"] == "prompt.submit":
                    await ws.close()  # admission is durable; relay settles without retrying
        except ConnectionClosed:
            close_codes.append(ws.close_code)
        finally:
            await conn.close()

    try:
        async with serve(handler, "127.0.0.1", 0, subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as listener:
            endpoint = SimpleNamespace(api_origin=f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}")
            relay = owner_forward._Relay("ui", {}, key, tmp_path, endpoint, "new synthetic prompt",
                                         lambda *args: emitted.append(args))
            if accepted:
                await asyncio.wait_for(relay._run(), 10)
                assert relay.admitted.result()["status"] == "queued"
            else:
                with pytest.raises(gateway_client.GatewayClientError):
                    await asyncio.wait_for(relay._run(), 10)
        rows = list_session_admissions(db, session_id=key)
        assert frame_sizes[0] > history_bytes
        assert seen == (["session.resume", "prompt.submit"] if accepted else ["session.resume"])
        assert len(rows) == int(accepted)
        if accepted:
            assert rows[0]["request_id"] == relay.request_id
            assert rows[0]["payload"]["text"] == "new synthetic prompt"
            assert emitted[-1] == ("message.complete", "ui", {"text": ""})
        else:
            assert close_codes == [1009]
        assert ticket == [None]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_attachment_forward_keeps_its_larger_explicit_budget(tmp_path, ticket):
    # Larger than the ordinary resume budget, within the room attachment budget.
    data = "YQ==" * (9 * 1024 * 1024 // 4)

    async def handler(ws):
        request = json.loads(await ws.recv())
        assert request["method"] == "groups.attachment.download"
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"data": data}}))
        await ws.wait_closed()

    async with serve(handler, "127.0.0.1", 0, subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as listener:
        endpoint = SimpleNamespace(api_origin=f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}")
        result = await asyncio.wait_for(asyncio.to_thread(
            owner_forward.forward_rpc, tmp_path, endpoint, "groups.attachment.download", {"attachment_id": "a1"}), 10)
    assert result == {"data": data}
    assert ticket == ["host"]
