"""The RPC wire accepts display text and persists it separately from model input."""

import pytest

from hermes_state import SessionDB
from tui_gateway import server


@pytest.mark.parametrize("display_text, expected", [
    pytest.param("hello", "hello", id="typed"),
    pytest.param({"text": "spoof"}, None, id="non-string"),
    pytest.param("x" * 100_001, None, id="oversize"),
])
def test_prompt_submit_dispatch_accepts_display_text_and_persists_user_words(monkeypatch, tmp_path, display_text, expected):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    created = server.handle_request({"id": "create", "method": "session.create", "params": {
        "cols": 96, "source": "desktop"}})
    sid = created["result"]["session_id"]
    key = created["result"]["stored_session_id"]
    session = server._sessions[sid]

    class InlineThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *args: False)
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *args: None)
    envelope = '{"protocol":"hermes.browser.turn.v2","human_input":{"text":"hello"},"browser_context":"page"}'
    try:
        response = server.handle_request({"id": "submit", "method": "prompt.submit", "params": {
            "session_id": sid, "text": envelope, "display_text": display_text}})
        assert "result" in response, response
        rows = db.get_messages_as_conversation(key)
        assert rows[0]["content"] == (expected if expected is not None else envelope)
        assert session["inflight_turn"]["user"] == (expected if expected is not None else "hello")
    finally:
        server._sessions.pop(sid, None)
        db.close()
