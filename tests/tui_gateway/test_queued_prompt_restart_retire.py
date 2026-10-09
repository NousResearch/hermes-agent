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
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
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
        # Raw row order/roles/active state, not just a count (#125577 review): the accept-time
        # row sits between turn A's rows, exactly where the busy-queue accepted it, and only
        # it went inactive — the restart retired the never-drained prompt, nothing else.
        raw = db.get_messages(key, include_inactive=True)
        assert [(r["role"], r["active"], "RESTART-MARKER" in str(r["content"])) for r in raw] == [
            ("user", 1, False),          # prompt A
            ("user", 0, True),            # queued prompt B — never drained, retired by reopen
            ("assistant", 1, False),      # reply A
        ]
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


def test_restart_between_drains_retires_only_the_later_prompt(monkeypatch, tmp_path):
    """Two prompts accepted busy; the first drains (its row re-placed unmarked), then the backend
    dies before the second drains. The restart retires the SECOND prompt's still-marked row while
    the drained first turn's row — same text family — stays active in its healed position."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        server._ensure_session_db_row(session)
        db.append_message(key, "user", content="prompt A")
        _busy(session)
        session["attached_images"] = ["/tmp/b.png"]  # non-mergeable: two separate envelopes
        assert server._handle_busy_submit("r1", sid, session, "prompt B DRAIN-RESTART-B", "ws-1",
                                          queued=True, display_kind=None)["result"]["status"] == "queued"
        session["attached_images"] = ["/tmp/c.png"]
        assert server._handle_busy_submit("r2", sid, session, "prompt C DRAIN-RESTART-C", "ws-1",
                                          queued=True, display_kind=None)["result"]["status"] == "queued"
        db.append_message(key, "assistant", content="reply A")
        with session["history_lock"]:
            session["running"] = False
            server._clear_inflight_turn(session)
        from tests.tui_gateway.test_queued_prompt_persistence import _run_turn
        monkeypatch.setattr(server, "_run_prompt_submit",
                            lambda rid, s, sess, text, **kw: _run_turn(sess, db, key, text, "reply B"))
        assert server._drain_queued_prompt("r3", sid, session) is True  # drains B only
        # The restart: queue gone; C never drained. The cold reopen retires C's row, not B's turn.
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)
        fresh = SessionDB(db_path=tmp_path / "state.db")
        try:
            fresh.reopen_session(key)
            repaired = fresh.get_messages_as_conversation(key, repair_alternation=True, include_row_ids=True)
            assert [(r["role"], r["content"]) for r in repaired] == [
                ("user", "prompt A"), ("assistant", "reply A"),
                ("user", "prompt B DRAIN-RESTART-B"), ("assistant", "reply B")]
            every = fresh.get_messages_as_conversation(key, include_inactive=True, include_row_ids=True)
            # C never ran: no ACTIVE row carries it. Its copies survive inactive (durable history
            # never deletes — the drain's re-placement left one, the reopen retired the other).
            assert not [r for r in _active_rows(fresh, key) if "DRAIN-RESTART-C" in str(r["content"])]
            assert any("DRAIN-RESTART-C" in str(r["content"])
                       for r in every if "DRAIN-RESTART-B" not in str(r["content"]))
        finally:
            fresh.close()
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_readonly_resume_after_restart_retires_marked_row_without_reopen(
        monkeypatch, tmp_path):
    """RED on the #128508 head: with the reopen gone from the mount paths, a restart + bare
    resume (no submit) left the still-marked accept row ACTIVE — it stayed glued into the
    previous turn's user message by alternation repair until the next send. The mount must
    stay read-only for the session row (ended_at keeps, #85303) while retiring the residue.
    """
    import pytest

    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    try:
        _accept_busy_prompt_a_concludes(db, sid, key)
        # The restart: the queue is gone and no live in-memory session survives (the queue,
        # running flag and inflight turn all lived in it); the row is FINALIZED and carries
        # a never-drained accept row.
        with server._sessions[sid]["history_lock"]:
            server._sessions[sid]["running"] = False
            server._clear_inflight_turn(server._sessions[sid])
        server._sessions.pop(sid, None)
        db.end_session(key, "agent_close")
        assert db.get_session(key)["ended_at"] is not None

        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        monkeypatch.setattr("hermes_state_registry.acquire", lambda db_path=None, **kwargs: db)
        monkeypatch.setattr(server, "_resolve_profile_home",
                            lambda p: (home, server._capture_profile_incarnation(home)) if p else (None, None))
        monkeypatch.setattr(server, "_profile_configured_cwd", lambda _: str(tmp_path))
        monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
        monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
        monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *args: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
        for defer_history in (False, True):  # cold read + deferred hydration paths
            monkeypatch.setattr(server, "_schedule_agent_build", lambda *args: None)
            response = server.handle_request({"id": "resume", "method": "session.resume", "params": {
                "session_id": key, "source": "desktop", "defer_history": defer_history}})
            assert response is not None and "error" not in response, response
            resumed_sid = response["result"]["session_id"]
            try:
                if defer_history:  # the hydration thread owns the retire on this path
                    assert server._sessions[resumed_sid]["resume_history_ready"].wait(5)
                # Read-only mount kept (#85303): the row stays finalized.
                row = db.get_session(key)
                assert row["ended_at"] is not None, "resume must not reopen the finalized row (#85303)"
                assert row["end_reason"] == "agent_close"
                # But the never-drained accept row is retired (#125577): the repaired projection
                # no longer absorbs the queued prompt into turn A's user message.
                repaired = db.get_messages_as_conversation(key, repair_alternation=True,
                                                            include_row_ids=True)
                assert [(r["role"], r["content"]) for r in repaired] == [
                    ("user", "prompt A"), ("assistant", "reply A")]
                raw = db.get_messages(key, include_inactive=True)
                assert [(r["role"], r["active"]) for r in raw] == [
                    ("user", 1), ("user", 0), ("assistant", 1)]
            finally:
                server._sessions.pop(resumed_sid, None)
    finally:
        server._sessions.pop(sid, None)
        db.close()
