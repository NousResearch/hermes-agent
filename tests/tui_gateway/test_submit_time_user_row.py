"""prompt.submit writes the user's message at send time, and the turn that follows adopts that row instead of
writing a second one (#111868: a Desktop freeze during a slow first agent build left a session row with no message)."""

from types import SimpleNamespace

from agent.turn_context import _stage_turn_user_message
from hermes_state import SessionDB
from run_agent import AIAgent
from tui_gateway import server


def _desktop_session(monkeypatch, db):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    resp = server.handle_request({"id": "c", "method": "session.create", "params": {"cols": 96, "source": "desktop"}})
    assert "result" in resp, resp
    return resp["result"]["session_id"], resp["result"]["stored_session_id"]


def _flush_agent(db, key):
    """Agent shell owning the real flush (the crash persist at turn start runs this same code)."""
    agent = SimpleNamespace(
        _session_db=db, _session_db_created=True, _persist_disabled=False, session_id=key,
        _session_persist_lock=None, _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0, _persist_user_message_idx=None, _persist_user_message_override=None,
        _persist_user_message_timestamp=None, _pending_cli_user_message=None)
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    agent._flush_messages_to_session_db_unlocked = AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    return agent


def test_user_message_is_durable_at_submit_before_any_agent_turn(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "please refactor the login page")
        assert server._persist_session_row_for_submit("rid", session, "please refactor the login page", None) is None
        # The agent build has not even started: the transcript already resumes with the sent message.
        assert [(r["role"], r["content"]) for r in db.get_messages_as_conversation(key)] == [
            ("user", "please refactor the login page")]
        assert db.get_session(key)["message_count"] == 1
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_submit_ack_binds_the_written_row_even_if_worker_consumes_staging(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]

    class InlineThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *args: False)
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *args: session.pop("_submit_user_row", None))
    try:
        replies = []
        for _ in range(2):
            session["running"] = False
            replies.append(server.handle_request({"id": "p", "method": "prompt.submit", "params": {
                "session_id": sid, "text": "same prompt"}})["result"])
        rows = db.get_messages_as_conversation(key, include_row_ids=True)
        assert [reply.get("user_row_id") for reply in replies] == [row["_row_id"] for row in rows]
        assert replies[0]["user_row_id"] != replies[1]["user_row_id"]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_turn_adopts_the_submit_row_and_writes_no_duplicate(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "look at @notes.md")
        assert server._persist_session_row_for_submit("rid", session, "look at @notes.md", None) is None
        agent = _flush_agent(db, key)
        # The prologue rewrote the persisted prompt (@-expansion): the early row follows it.
        expanded = "look at @notes.md\n\n<file notes.md>todo</file>"
        server._adopt_submit_user_row(session, agent, expanded, "look at @notes.md")
        assert "_submit_user_row" not in session
        user_msg, _pending = _stage_turn_user_message(agent, expanded, expanded, None, None, None, None)
        assert user_msg is agent._pending_cli_user_message  # adopted by identity, not rebuilt
        messages = [user_msg]
        agent._persist_user_message_idx = 0
        agent._flush_messages_to_session_db(messages, [])  # the turn-start crash persist
        agent._flush_messages_to_session_db(messages + [{"role": "assistant", "content": "done"}], [])  # turn end
        rows = db.get_messages_as_conversation(key, include_inactive=True)
        assert [(r["role"], r["content"]) for r in rows] == [("user", expanded), ("assistant", "done")]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_completion_receipt_covers_only_committed_current_turn_rows(monkeypatch, tmp_path):
    from tui_gateway.prompt_turn import _TurnRun

    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    try:
        server._ensure_session_db_row(session)
        agent = _flush_agent(db, key)
        old = [{"role": "user", "content": "again"}, {"role": "assistant", "content": "same"}]
        agent._flush_messages_to_session_db(old, [])
        current = [{"role": "user", "content": "again"}, {"role": "assistant", "content": "same"}]
        messages = old + current
        agent._persist_user_message_idx = len(old)
        agent.context_compressor = SimpleNamespace(compression_count=0)
        st = _TurnRun(agent, None, None, False, history=old, compression_count=0,
                      result={"messages": messages, "final_response": "same"})
        agent._flush_messages_to_session_db(messages, old)
        payload, _, _ = server._complete_turn_payload(session, st, None, 80)
        rows = db.get_messages_as_conversation(key, include_row_ids=True)
        assert payload.get("persisted_turn") == {
            "user_row_id": rows[-2]["_row_id"],
            "row_ids": [row["_row_id"] for row in rows[-2:]],
            "final_assistant_row_id": rows[-1]["_row_id"],
            "complete": True,
        }
        # Compression can discard an already-streamed segment even while the old prefix survives.
        agent.context_compressor = SimpleNamespace(compression_count=1)
        compressed, _, _ = server._complete_turn_payload(session, st, None, 80)
        assert compressed["persisted_turn"]["complete"] is False
        # The row address alone does not assert that a subsequently mutated body was committed.
        current[-1].pop("_db_persisted")
        current[-1]["content"] = "not flushed"
        partial, _, _ = server._complete_turn_payload(session, st, None, 80)
        assert partial["persisted_turn"] == {
            "user_row_id": rows[-2]["_row_id"], "row_ids": [rows[-2]["_row_id"]], "complete": False}
        # No authoritative current-turn anchor: never infer from matching text or positions in old history.
        agent._persist_user_message_idx = None
        missing, _, _ = server._complete_turn_payload(session, st, None, 80)
        assert "persisted_turn" not in missing
        # A preflight failure may return only old history while the agent still carries its old cursor.
        agent._persist_user_message_idx = 0
        st.result["messages"] = old
        stale, _, _ = server._complete_turn_payload(session, st, None, 80)
        assert "persisted_turn" not in stale
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_failed_build_drops_the_staged_row_and_a_later_turn_never_adopts_it(monkeypatch, tmp_path):
    """The submit-time row is the durable record of THAT send only. A turn that ends before the agent runs
    (build failed / bounded wait expired) must drop the staging dict, and a later turn without a matching
    prompt.submit (wake-up, auto-continue, queued drain) must not rewrite the user's row to its own text."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt",
                        lambda _session, _rid, _sid: {"error": {"message": "agent initialization failed"}})
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "please refactor the login page")
        assert server._persist_session_row_for_submit("rid", session, "please refactor the login page", None) is None
        server._run_after_agent_ready("rid", sid, session, "please refactor the login page", None, None, None)
        assert "_submit_user_row" not in session, "staged row survived a turn that never reached the agent"

        # Even if a staged row were still around, a turn whose raw submit differs must leave the DB alone.
        with session["history_lock"]:
            server._start_inflight_turn(session, "please refactor the login page")
        server._persist_submit_user_row(session, "please refactor the login page", None)
        agent = _flush_agent(db, key)
        synthesized = "[subagent finished] result summary"
        server._adopt_submit_user_row(session, agent, synthesized, synthesized)
        assert "_submit_user_row" not in session
        assert agent._pending_cli_user_message is None
        rows = db.get_messages_as_conversation(key, include_inactive=True)
        assert [(r["role"], r["content"]) for r in rows] == [
            ("user", "please refactor the login page"), ("user", "please refactor the login page")]
    finally:
        server._sessions.pop(sid, None)
        db.close()
