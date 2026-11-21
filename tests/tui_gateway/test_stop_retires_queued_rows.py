"""Stop must retire the accept-time rows of the queued prompts it discards (#146968).

A queued prompt is durable from accept time (``_persist_queued_user_row``). The drain and a restart
(#125577) retire that row; Stop left it active, so each in-place compaction cloned it after the new set
and ``_adopt_out_of_band_turns`` handed the cancelled prompt to the model after the newest turn.
"""

from types import SimpleNamespace

from hermes_state import SessionDB
from tools import approval
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


def _busy_session(monkeypatch, db):
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    assert server._ensure_session_db_row(session) is not False
    session["agent"] = SimpleNamespace(session_id=key, _session_db=db)
    with session["history_lock"]:
        session["running"] = True
        server._start_inflight_turn(session, "turn A")
    held = [db.append_message(key, "user", content="turn A"),
            db.append_message(key, "assistant", content="working on A")]
    return sid, key, session, held


def _queue(sid, session, text):
    resp = server._handle_busy_submit("r1", sid, session, text, "ws-1", queued=True, display_kind=None)
    assert resp["result"]["status"] == "queued"


def _active(db, key):
    return [str(r["content"]) for r in db.get_messages_as_conversation(key, include_row_ids=True)]


def test_stop_retires_the_discarded_queued_row(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, _held = _busy_session(monkeypatch, db)
    try:
        _queue(sid, session, "QUEUED-A")
        assert "QUEUED-A" in _active(db, key)
        server._interrupt_session_turn(sid, session)
        assert session.get("queued_prompt") is None
        assert _active(db, key) == ["turn A", "working on A"]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_stop_retires_every_discarded_envelope_including_a_merged_one(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, _held = _busy_session(monkeypatch, db)
    try:
        _queue(sid, session, "M1")
        _queue(sid, session, "M2")  # text-only: merges into the head envelope's row
        session["attached_images"] = [str(tmp_path / "x.png")]
        _queue(sid, session, "IMG")  # image-bearing: its own envelope in queued_prompts
        assert session.get("queued_prompts")
        assert _active(db, key)[-2:] == ["M1\n\nM2", "IMG"]
        server._interrupt_session_turn(sid, session)
        assert _active(db, key) == ["turn A", "working on A"]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_stop_after_a_compaction_retires_the_cloned_row(monkeypatch, tmp_path):
    """A compaction before Stop already re-sequenced the row; Stop must follow the clone, not the dead original."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, held = _busy_session(monkeypatch, db)
    try:
        _queue(sid, session, "QUEUED-A")
        db.archive_and_compact(key, [{"role": "user", "content": "[summary 1]"}], covered_ids=held)
        assert _active(db, key) == ["[summary 1]", "QUEUED-A"]
        server._interrupt_session_turn(sid, session)
        assert _active(db, key) == ["[summary 1]"]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_discarded_prompt_does_not_reach_the_model_after_compactions(monkeypatch, tmp_path):
    """The reported symptom: after Stop and compactions the cancelled prompt followed the newest turn."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, held = _busy_session(monkeypatch, db)
    try:
        _queue(sid, session, "QUEUED-A")
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
        own = [r for r in db.get_messages_as_conversation(key, include_row_ids=True) if r["_row_id"] in held]
        with session["history_lock"]:
            session["history"] = [{"role": r["role"], "content": r["content"], "_row_id": r["_row_id"]} for r in own]
            session["running"] = False
        server._persist_submit_user_row(session, "turn C", None)
        server._adopt_out_of_band_turns(session)
        assert [m["content"] for m in session["history"]] == ["[summary 2]", "turn B"]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_stop_retires_last_and_outside_history_lock(monkeypatch, tmp_path):
    """A busy state.db must not hold up the interrupt or the approval deny: the retire runs after both."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, _held = _busy_session(monkeypatch, db)
    calls, retire = [], server._retire_discarded_queued_rows
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: calls.append("clear_pending"))
    monkeypatch.setattr(approval, "resolve_gateway_approval", lambda *_a, **_k: calls.append("deny"))

    def spy(sess, envelopes):
        calls.append(("retire", sess["history_lock"].locked(), sess.get("running")))
        retire(sess, envelopes)
    monkeypatch.setattr(server, "_retire_discarded_queued_rows", spy)
    try:
        _queue(sid, session, "QUEUED-A")
        server._interrupt_session_turn(sid, session)
        assert calls == ["clear_pending", "deny", ("retire", False, False)]
        assert "QUEUED-A" not in _active(db, key)
    finally:
        server._sessions.pop(sid, None)
        db.close()
