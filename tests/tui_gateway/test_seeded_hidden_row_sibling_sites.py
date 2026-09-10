"""Sibling sites of the seeded-session hidden-row saga (#107549/#107562): session.active_list's
live sidebar item, session.history, and session.branch all reported a raw ``len(history)``/preview
built from raw history, letting a display_kind="hidden" row (model-facing scaffolding the gateway
never paints) leak into a count or preview text the way session.list/session.create/session.resume
were already fixed not to."""

import threading
from types import SimpleNamespace

from hermes_state import SessionDB
from tui_gateway import server


def _quiet_create(monkeypatch, db):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)


def _create(params: dict) -> dict:
    resp = server.handle_request({"id": "create", "method": "session.create", "params": params})
    assert "result" in resp, resp
    return resp["result"]


def test_active_list_hides_the_seed_row_from_preview_and_count(monkeypatch, tmp_path):
    """A seeded, not-yet-persisted-past-create session is already live in ``_sessions`` and shows up
    in session.active_list (the running-sessions sidebar) before the first prompt — its hidden opening
    row must not leak into the preview text or inflate the count, the same invariant session.list's
    DB-backed query already enforces."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _quiet_create(monkeypatch, db)
    sid = None
    try:
        result = _create({
            "cols": 96, "source": "desktop", "title": "Welcome to Hermes",
            "messages": [
                {"role": "user", "content": "Private setup runbook", "display_kind": "hidden"},
                {"role": "assistant", "content": "Welcome to Hermes"},
            ]})
        sid = result["session_id"]

        active = server.handle_request({"id": "active", "method": "session.active_list", "params": {}})
        row = next(r for r in active["result"]["sessions"] if r["id"] == sid)

        assert row["message_count"] == 1  # hidden row not counted, as session.create/session.resume count it
        assert "Private setup runbook" not in row["preview"]
        assert row["preview"].startswith("Welcome to Hermes")
    finally:
        if sid:
            server._sessions.pop(sid, None)
        db.close()


def test_session_history_count_matches_the_wire(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _quiet_create(monkeypatch, db)
    sid = None
    try:
        result = _create({
            "cols": 96, "source": "desktop", "title": "Welcome",
            "messages": [
                {"role": "user", "content": "Private setup runbook", "display_kind": "hidden"},
                {"role": "assistant", "content": "hi there"},
            ]})
        sid, key = result["session_id"], result["stored_session_id"]

        history = server.handle_request({"id": "h", "method": "session.history", "params": {"session_id": sid}})
        assert history["result"]["count"] == len(history["result"]["messages"]) == 1
    finally:
        if sid:
            server._sessions.pop(sid, None)
        db.close()


def test_session_branch_message_count_matches_the_wire(monkeypatch, tmp_path):
    """A branch child copies its parent's history; a hidden row in that history must not inflate
    session.branch's own message_count past what its own messages array actually carries."""
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_build_branch_agent", lambda *a, **k: SimpleNamespace())
    old_key = "parent-branch-src"
    db.create_session(old_key, source="desktop")
    db.append_message(old_key, "user", "Private setup runbook", display_kind="hidden")
    db.append_message(old_key, "user", "real question")
    db.append_message(old_key, "assistant", "real answer")
    sid = "live-branch-src"
    server._sessions[sid] = {"session_key": old_key, "history": [], "cols": 96, "history_lock": threading.Lock()}
    new_sid = None
    try:
        resp = server.handle_request({"id": "b", "method": "session.branch", "params": {"session_id": sid}})
        assert "result" in resp, resp
        result = resp["result"]
        new_sid = result["session_id"]
        assert result["message_count"] == len(result["messages"])
        assert not any(m.get("text") == "Private setup runbook" for m in result["messages"])
    finally:
        server._sessions.pop(sid, None)
        if new_sid:
            server._sessions.pop(new_sid, None)
        db.close()
