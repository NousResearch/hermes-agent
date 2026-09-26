"""Document the accepted authenticated-client trust boundary for surface claims.

Desktop provenance is not server-attested. The per-conversation opt-in, rather
than the label itself, is the security boundary. An authenticated client can
deliberately claim Desktop; never mistake this test for device attestation.
"""
import asyncio

import pytest

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport
from tui_gateway.ws import WSTransport


class _InlineThread:
    def __init__(self, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


@pytest.fixture
def submit_harness(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *args: False)
    # Capture at the actual submit/row boundary, before a worker could consume it.
    captured = []
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *args: captured.append(
        dict(server._sessions[args[1]]["_submit_user_row"])))
    # The only server-stamped WS identity is the logged-in user, not the app type.
    identity = {"provider": "password", "user_id": "same-user"}
    desktop = WSTransport(object(), loop, peer="desktop", auth_identity=identity)
    tui = WSTransport(object(), loop, peer="tui", auth_identity=identity)
    sessions = []

    def submit(transport):
        token = bind_transport(transport)
        try:
            created = server.handle_request({
                "id": "create", "method": "session.create", "params": {"cols": 96, "source": "desktop"}})
            assert "result" in created, created
            sid = created["result"]["session_id"]
            sessions.append(sid)
            response = server.handle_request({
                "id": "prompt", "method": "prompt.submit",
                "params": {"session_id": sid, "text": "hello", "surface": "desktop"}})
            assert response["result"]["status"] == "streaming", response
            return captured[-1]
        finally:
            reset_transport(token)

    yield submit, desktop, tui
    for sid in sessions:
        server._sessions.pop(sid, None)
    db.close()
    loop.close()


def test_legitimate_desktop_claim_stages_desktop_surface(submit_harness):
    submit, desktop, _tui = submit_harness
    assert submit(desktop)["_client_surface"] == "desktop"


def test_authenticated_non_desktop_client_can_claim_desktop_surface(submit_harness):
    submit, _desktop, tui = submit_harness
    assert submit(tui)["_client_surface"] == "desktop"
