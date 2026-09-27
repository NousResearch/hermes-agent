"""A restart discards the in-memory busy-queue, but the accept-time user row it wrote stays
active — the next cold resume's alternation repair glues the never-run prompt into the previous
turn's user message (#125577).

The fix marks the accept-time row (``display_metadata._queued_prompt``) and ``reopen_session``
deactivates still-marked rows: the drain's replacement row is unmarked, so still-marked after a
restart == never drained. Rows are deactivated, never deleted, the same marking the drain uses.
"""

import types

from hermes_state import SessionDB
from tui_gateway import server


def _desktop_session(monkeypatch, db):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    resp = server.handle_request({"id": "c", "method": "session.create", "params": {"cols": 96, "source": "desktop"}})
    assert "result" in resp, resp
    sid = resp["result"]["session_id"]
    server._sessions[sid]["agent"] = types.SimpleNamespace()
    return sid, resp["result"]["stored_session_id"]


def _busy(session, in_flight="prompt A"):
    with session["history_lock"]:
        session["running"] = True
        server._start_inflight_turn(session, in_flight)


def _active_rows(db, key):
    return db.get_messages_as_conversation(key, repair_alternation=True, include_row_ids=True)


def _accept_busy_prompt_a_concludes(db, sid, key, queued_text="queued prompt B RESTART-MARKER"):
    """Turn A runs, B is accepted busy (accept-time row written), turn A concludes — then the
    backend dies BEFORE the queue drains. Raw: [user A, user B(queued), assistant A]."""
    session = server._sessions[sid]
    server._ensure_session_db_row(session)
    db.append_message(key, "user", content="prompt A")
    _busy(session)
    resp = server._handle_busy_submit("r1", sid, session, queued_text, "ws-1", queued=True, display_kind=None)
    assert resp["result"]["status"] == "queued"
    db.append_message(key, "assistant", content="reply A")  # turn A concludes; the drain never runs


def test_restart_retires_the_never_drained_accept_row(monkeypatch, tmp_path):
    """RED for the bug: with the queue gone (restart), nothing retired the accept-time row and the
    repaired projection returned 'prompt A\n\nqueued prompt B' absorbed into turn A's user message."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    try:
        _accept_busy_prompt_a_concludes(db, sid, key)
        # The restart's cold attach: a NEW handle on the same file, then the reopen ritual every
        # resume path (TUI/Desktop hydration, CLI, oneshot) runs before reading the transcript.
        fresh = SessionDB(db_path=tmp_path / "state.db")
        try:
            fresh.reopen_session(key)
            repaired = fresh.get_messages_as_conversation(key, repair_alternation=True, include_row_ids=True)
            assert [(r["role"], r["content"]) for r in repaired] == [
                ("user", "prompt A"), ("assistant", "reply A")]
        finally:
            fresh.close()
        # The never-run prompt is retired (inactive), never deleted: durable history keeps it.
        every = db.get_messages_as_conversation(key, include_inactive=True, include_row_ids=True)
        retired = [r for r in every if "RESTART-MARKER" in str(r["content"])]
        assert len(retired) == 1
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_resume_does_not_retire_drained_or_plain_rows(monkeypatch, tmp_path):
    """The dispatched queued row (re-placed unmarked by the drain) and ordinary user rows survive
    reopen_session — only still-marked never-drained accept rows are retired."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    try:
        server._ensure_session_db_row(server._sessions[sid])
        db.append_message(key, "user", content="plain prompt")
        db.append_message(key, "assistant", content="plain reply")
        db.reopen_session(key)
        assert [(r["role"], r["content"]) for r in _active_rows(db, key)] == [
            ("user", "plain prompt"), ("assistant", "plain reply")]
    finally:
        server._sessions.pop(sid, None)
        db.close()
