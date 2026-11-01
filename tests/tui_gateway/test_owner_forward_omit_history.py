"""#145215: forwarding a prompt needs an attachment, not the owner's large transcript.

Real SQLite history, authority resume/admission/settlement, loopback WebSocket and
GatewayClient. Only ticket issuance and turn execution are replaced; no LLM runs.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from websockets.asyncio.server import serve

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import SessionRef
from gateway.session_controls import AuthorityConnection
from gateway.session_settlement_recovery import commit_and_publish
from hermes_cli import gateway_client
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, claim_session_input, list_session_admissions
from tui_gateway import owner_forward


class _Socket:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        return None

    async def flush(self, ws):
        while self.frames:
            await ws.send(json.dumps(self.frames.pop(0)))


@pytest.mark.asyncio
async def test_long_history_forward_admits_once_and_relays_completion(tmp_path, monkeypatch):
    from gateway import session_local_recovery

    db = SessionDB(db_path=tmp_path / "state.db")
    key, emitted, resumes, history_reads = "long-chat", [], [], []
    db.create_session(key, "test")
    for _ in range(9):
        db.append_message(key, "user", "x" * (1024 * 1024))
    epoch = begin_runtime_epoch(db, instance_id="gateway")
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="owned",
                                 instance_id="gateway", db=db, epoch=epoch)
    authority.sessions[key] = LiveSession(SimpleNamespace(platform=None, user_id="desktop"), "route")
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **kw: "fixture-ticket")
    original_read = session_local_recovery.display_rows

    def read_history(*args):
        history_reads.append(True)
        return original_read(*args)

    monkeypatch.setattr(session_local_recovery, "display_rows", read_history)

    async def handler(ws):
        transport = _Socket()
        conn = AuthorityConnection(authority, transport, {"user_id": "desktop"})
        try:
            async for raw in ws:
                request = json.loads(raw)
                reply = await conn.dispatch(request)
                if request["method"] == "session.resume":
                    resumes.append(reply)
                await ws.send(json.dumps(reply))
                if request["method"] == "prompt.submit" and "result" in reply:
                    row = claim_session_input(db, epoch=epoch, session_id=key)
                    ref, live = SessionRef("owned", key), authority.sessions[key]
                    authority._stamp_claim(ref, live, row)
                    with live.event_stream.lock:
                        authority.executing.pop(key, None)
                    commit_and_publish(authority, live, ref, row, "THE ANSWER", "completed", None, {})
                await transport.flush(ws)
        finally:
            await conn.close()

    try:
        async with serve(handler, "127.0.0.1", 0,
                         subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as listener:
            port = listener.sockets[0].getsockname()[1]
            endpoint = SimpleNamespace(api_origin=f"http://127.0.0.1:{port}")
            relay = owner_forward._Relay("ui", {}, key, tmp_path, endpoint, "my question",
                                         lambda *args: emitted.append(args))
            await asyncio.wait_for(relay._run(), timeout=10)

        assert history_reads == []  # No full-history read hidden behind a small wire reply.
        assert len(resumes) == 1 and len(json.dumps(resumes[0]).encode()) < 64 * 1024
        assert resumes[0]["result"]["messages"] == []
        assert resumes[0]["result"]["messages_omitted"] is True
        assert resumes[0]["result"]["message_count"] == 9
        rows = list_session_admissions(db, session_id=key, pending_only=False)
        assert len(rows) == 1 and rows[0]["status"] == "terminal"
        assert rows[0]["admission_id"] == relay.admitted.result()["admission_id"]
        assert rows[0]["payload"]["text"] == "my question"
        messages = [(event, payload) for event, sid, payload in emitted if event.startswith("message.")]
        assert messages[0][0] == "message.start"
        assert messages[-1] == ("message.complete", {"text": "THE ANSWER", "status": "complete"})
    finally:
        db.close()
