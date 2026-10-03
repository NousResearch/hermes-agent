"""Mounting a finalized session is read-only (#85303): session.resume / its deferred
hydration / the lazy watch path must NOT clear ``ended_at``/``end_reason`` — only the
first real turn (prompt.submit) reopens the row, so DB-derived liveness cannot re-light
a finished session just because someone opened it."""

import threading

import pytest

from hermes_state import SessionDB
from tui_gateway import server


def _finalized_db(tmp_path, *, ended_reason="agent_close"):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    db = SessionDB(home / "state.db")
    db.create_session("finalized", source="desktop")
    db.append_message("finalized", "user", "old ask", timestamp=100.0)
    db.append_message("finalized", "assistant", "old answer", timestamp=101.0)
    db.end_session("finalized", ended_reason)
    return db, home


def _mount(monkeypatch, db, home, tmp_path, *, defer_history=False):
    events = []
    built = threading.Event()
    monkeypatch.setattr("hermes_state_registry.acquire", lambda db_path=None, **kwargs: db)
    monkeypatch.setattr(server, "_profile_home", lambda p: home if p else None)
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _: str(tmp_path))
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *args: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: built.set())
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload: events.append((kind, payload)))
    return events, built


@pytest.mark.parametrize("defer_history", [False, True])
def test_resume_mount_keeps_finalized_row_ended(tmp_path, monkeypatch, defer_history):
    db, home = _finalized_db(tmp_path)
    events, built = _mount(monkeypatch, db, home, tmp_path)
    sid = None
    try:
        response = server.handle_request({"id": "resume", "method": "session.resume", "params": {
            "session_id": "finalized", "source": "desktop", "defer_history": defer_history,
        }})
        assert response is not None and "error" not in response, response
        sid = response["result"]["session_id"]
        if defer_history:
            session = server._sessions[sid]
            assert session["resume_history_ready"].wait(5)
        else:
            assert built.wait(5)
        row = db.get_session("finalized")
        assert row["ended_at"] is not None, "mounting a finalized session must not reopen its row (#85303)"
        assert row["end_reason"] == "agent_close"
    finally:
        if sid is not None:
            server._sessions.pop(sid, None)
        db.close()


def test_lazy_watch_mount_keeps_finalized_row_ended(tmp_path, monkeypatch):
    db, home = _finalized_db(tmp_path)
    events, built = _mount(monkeypatch, db, home, tmp_path)
    sid = None
    try:
        response = server.handle_request({"id": "resume", "method": "session.resume", "params": {
            "session_id": "finalized", "source": "desktop", "lazy": True,
        }})
        assert response is not None and "error" not in response, response
        row = db.get_session("finalized")
        assert row["ended_at"] is not None, "the lazy watch mount must not reopen a finalized row"
    finally:
        if sid is not None:
            server._sessions.pop(sid, None)
        db.close()


def test_first_real_turn_reopens_a_finalized_row(tmp_path, monkeypatch):
    """The activity gate: prompt.submit (a real send) is what clears ended_at, not the mount."""
    from tui_gateway import methods_prompt

    db, home = _finalized_db(tmp_path)
    events, built = _mount(monkeypatch, db, home, tmp_path)
    reopened = []
    monkeypatch.setattr(db, "reopen_session", lambda sid: reopened.append(sid) or
                        SessionDB.reopen_session(db, sid))
    reopened_row = db.get_session("finalized")
    assert reopened_row["ended_at"] is not None
    # The submit path's reopen helper: the row is finalized, so a real send reopens it.
    methods_prompt._reopen_if_finalized(db, "finalized")
    assert reopened == ["finalized"]
    row = db.get_session("finalized")
    assert row["ended_at"] is None, "the first real turn must reopen the finalized row"
    db.close()


def test_reopen_if_finalized_leaves_live_rows_untouched(tmp_path):
    db, home = _finalized_db(tmp_path)
    from tui_gateway import methods_prompt

    db.create_session("live", source="desktop")
    db.append_message("live", "user", "ask", timestamp=100.0)
    methods_prompt._reopen_if_finalized(db, "live")
    methods_prompt._reopen_if_finalized(db, "missing-entirely")
    row = db.get_session("live")
    assert row["ended_at"] is None and row["end_reason"] is None
    db.close()
