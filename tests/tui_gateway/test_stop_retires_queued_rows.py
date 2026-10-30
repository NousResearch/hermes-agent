"""Stop must retire the accept-time rows of the queued prompts it discards.

A prompt sent while a turn runs is queued and made durable at accept time (``_persist_queued_user_row``,
marked ``_queued_prompt``). The drain re-places that row and deactivates the early one; a restart
retires still-marked rows (#125577). Stop purges the in-memory queue but left the row active. The agent
never holds that row, so every in-place compaction treats it as unseen (``_archive_named_rows``) and
clones it after the new compacted set, and ``_adopt_out_of_band_turns`` then hands it to the model
after the newest turn. Measured on a live session: one prompt queued at 16:48 and discarded by a Stop at
17:58 had 17 copies and reached the model after a 21:41 message.
"""

from types import SimpleNamespace

from hermes_state import SessionDB
from tui_gateway import server


def _desktop_session(monkeypatch, db):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    resp = server.handle_request(
        {"id": "c", "method": "session.create", "params": {"cols": 96, "source": "desktop"}})
    assert "result" in resp, resp
    return resp["result"]["session_id"], resp["result"]["stored_session_id"]


def _busy_session_with_queued_prompt(monkeypatch, db):
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    assert server._ensure_session_db_row(session) is not False
    session["agent"] = SimpleNamespace(session_id=key, _session_db=db)
    with session["history_lock"]:
        session["running"] = True
        server._start_inflight_turn(session, "turn A")
    held = [db.append_message(key, "user", content="turn A"),
            db.append_message(key, "assistant", content="working on A")]
    assert server._handle_busy_submit("r1", sid, session, "QUEUED-A", "ws-1",
                                      queued=True, display_kind=None)["result"]["status"] == "queued"
    return sid, key, session, held


def _active(db, key):
    return [str(r["content"]) for r in db.get_messages_as_conversation(key, include_row_ids=True)]


def test_stop_retires_the_discarded_queued_row(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, _held = _busy_session_with_queued_prompt(monkeypatch, db)
    try:
        assert "QUEUED-A" in _active(db, key)
        server._interrupt_session_turn(sid, session)
        assert session.get("queued_prompt") is None
        assert "QUEUED-A" not in _active(db, key)
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_discarded_queued_row_is_not_replayed_after_the_newest_turn(monkeypatch, tmp_path):
    """The live symptom: after Stop and two compactions the stale prompt sat after the newest turn."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, held = _busy_session_with_queued_prompt(monkeypatch, db)
    try:
        server._interrupt_session_turn(sid, session)
        held.append(db.append_message(key, "user", content="turn B"))
        for n in (1, 2):
            # The compressor held the agent's own rows; it never held the queued row.
            db.archive_and_compact(
                key, [{"role": "user", "content": f"[summary {n}]"}, {"role": "user", "content": "turn B"}],
                covered_ids=held)
            held = [r["_row_id"] for r in db.get_messages_as_conversation(key, include_row_ids=True)
                    if r["content"] != "QUEUED-A"]
        assert _active(db, key) == ["[summary 2]", "turn B"]
    finally:
        server._sessions.pop(sid, None)
        db.close()
