"""The RPC wire accepts display text and persists it separately from model input."""

from types import SimpleNamespace

import pytest

from agent.turn_context import _stage_turn_user_message, _stamp_api_content_sidecar
from hermes_state import SessionDB
from run_agent import AIAgent
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
        rejected = server.handle_request({"id": "invalid", "method": "prompt.submit", "params": {
            "session_id": sid, "text": envelope, "display_text": "hello", "unexpected_field": True}})
        assert rejected["error"]["code"] == 4000
        assert db.get_messages_as_conversation(key) == []
        response = server.handle_request({"id": "submit", "method": "prompt.submit", "params": {
            "session_id": sid, "text": envelope, "display_text": display_text}})
        assert "result" in response, response
        rows = db.get_messages_as_conversation(key)
        assert rows[0]["content"] == (expected if expected is not None else envelope)
        assert session["inflight_turn"]["user"] == (expected if expected is not None else "hello")
        if expected is not None:
            agent = SimpleNamespace(
                _session_db=db, _session_db_created=True, _persist_disabled=False, session_id=key,
                _session_persist_lock=None, _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
                _last_flushed_db_idx=0, _persist_user_message_idx=None, _persist_user_message_override=expected,
                _persist_user_message_timestamp=None, _pending_cli_user_message=None)
            agent._ensure_db_session = lambda: None
            agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
            agent._flush_messages_to_session_db_unlocked = AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
            server._adopt_submit_user_row(session, agent, expected, envelope, expected)
            user_msg, _ = _stage_turn_user_message(agent, envelope, expected, None, None, None, None)
            assert user_msg["content"] == envelope
            _stamp_api_content_sidecar(agent, [user_msg], 0, "", "", preflight_compressed=False)
            row = db.get_messages_as_conversation(key)[0]
            assert row["content"] == expected
            assert row["api_content"] == envelope
    finally:
        server._sessions.pop(sid, None)
        db.close()
