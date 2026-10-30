"""Regression: a model switch on a rotated session must not hit a FOREIGN KEY failure.

``_append_model_switch_marker`` writes a durable pivot row under ``_submit_row_target_key(session)``,
which is the LIVE ``agent.session_id`` once the agent has rotated off ``session_key``. Its INSERT needs
that child row to already exist in the session store. The row is only ever ensured on the ``db is None``
branch, so a rotated session that still carries an agent-owned ``_session_db`` skips the ensure and the
pivot is filed against a session id nothing created — the write dies with
``sqlite3.IntegrityError: FOREIGN KEY constraint failed`` and the model-switch notice is silently lost
(caught and logged at warning as "failed to persist model switch marker").
"""

import sqlite3

import tui_gateway.server as srv
from hermes_state import SessionDB


class _RotatedAgent:
    """An agent whose session has rotated onto a child id the store has never seen."""

    def __init__(self, db: SessionDB, session_id: str) -> None:
        self._session_db = db
        self.session_id = session_id


def test_model_switch_marker_persists_onto_a_rotated_session_id(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        monkeypatch.setattr(srv, "_get_db", lambda: db)
        monkeypatch.setattr(srv, "_schedule_agent_build", lambda sid: None)
        monkeypatch.setattr(srv, "_schedule_session_cap_enforcement", lambda: None)
        created = srv._methods["session.create"](1, {})
        assert "error" not in created, created
        session = srv._sessions[created["result"]["session_id"]]
        try:
            # Only the PARENT row exists; the agent has rotated onto a child id.
            assert srv._ensure_session_db_row(session)
            child_id = f"{session['session_key']}-rotated"
            session["agent"] = _RotatedAgent(db, child_id)

            srv._append_model_switch_marker(session, model="qwen3", provider="local")

            row_id = session["history"][-1].get("_row_id")
            assert row_id, "model-switch marker was not persisted"
            rows = [m for m in db.get_messages(child_id) if m.get("id") == row_id or m.get("row_id") == row_id]
            assert rows, "pivot row was not filed under the rotated child session id"
            assert rows[0]["display_kind"] == "model_switch"
        finally:
            srv._sessions.pop(created["result"]["session_id"], None)


def _new_session(tmp_path, monkeypatch, db):
    monkeypatch.setattr(srv, "_get_db", lambda: db)
    monkeypatch.setattr(srv, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(srv, "_schedule_session_cap_enforcement", lambda: None)
    created = srv._methods["session.create"](1, {})
    assert "error" not in created, created
    return created["result"]["session_id"], srv._sessions[created["result"]["session_id"]]


def _durable_rows(db, sid):
    return [m for m in db.get_messages(sid) if m.get("display_kind") == "model_switch"]


def test_model_switch_marker_persists_when_target_row_was_never_inserted(tmp_path, monkeypatch):
    """Trigger B: no rotation; the agent owns _session_db but nothing ever created the row."""
    with SessionDB(tmp_path / "state.db") as db:
        sid, session = _new_session(tmp_path, monkeypatch, db)
        try:
            assert db.get_session(session["session_key"]) is None
            session["agent"] = _RotatedAgent(db, session["session_key"])
            srv._append_model_switch_marker(session, model="qwen3", provider="local")
            assert session["history"][-1].get("_row_id"), "marker not persisted"
            assert len(_durable_rows(db, session["session_key"])) == 1
        finally:
            srv._sessions.pop(sid, None)


def test_model_switch_marker_ensures_row_in_the_store_it_writes_to(tmp_path, monkeypatch):
    """The ensure must use the agent's db handle, not profile_home/state.db or _get_db()."""
    other_home = tmp_path / "profile"
    other_home.mkdir()
    with SessionDB(tmp_path / "agent.db") as agent_db, SessionDB(tmp_path / "shared.db") as shared_db:
        sid, session = _new_session(tmp_path, monkeypatch, shared_db)
        try:
            session["profile_home"] = str(other_home)
            child_id = f"{session['session_key']}-rotated"
            session["agent"] = _RotatedAgent(agent_db, child_id)
            srv._append_model_switch_marker(session, model="qwen3", provider="local")
            assert session["history"][-1].get("_row_id"), "marker not persisted"
            assert agent_db.get_session(child_id) is not None
            assert len(_durable_rows(agent_db, child_id)) == 1
            assert shared_db.get_session(child_id) is None
            assert not (other_home / "state.db").exists()
        finally:
            srv._sessions.pop(sid, None)


def test_model_switch_marker_ensures_the_live_target_id(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        sid, session = _new_session(tmp_path, monkeypatch, db)
        try:
            child_id = f"{session['session_key']}-rotated"
            session["agent"] = _RotatedAgent(db, child_id)
            calls = []
            real = srv._ensure_session_db_row

            def _spy(sess, *args, **kwargs):
                calls.append(kwargs.get("session_id"))
                return real(sess, *args, **kwargs)

            monkeypatch.setattr(srv, "_ensure_session_db_row", _spy)
            srv._append_model_switch_marker(session, model="qwen3", provider="local")
            assert calls == [child_id]
            assert db.get_session(child_id) is not None
            assert len(_durable_rows(db, child_id)) == 1
        finally:
            srv._sessions.pop(sid, None)


def test_model_switch_marker_survives_a_closed_parent_session(tmp_path, monkeypatch):
    """The pivot must not be lost when the ensure path itself raises (a closed/rotated store)."""
    with SessionDB(tmp_path / "state.db") as db:
        monkeypatch.setattr(srv, "_get_db", lambda: db)
        monkeypatch.setattr(srv, "_schedule_agent_build", lambda sid: None)
        monkeypatch.setattr(srv, "_schedule_session_cap_enforcement", lambda: None)
        created = srv._methods["session.create"](1, {})
        session = srv._sessions[created["result"]["session_id"]]
        try:
            session["agent"] = _RotatedAgent(db, f"{session['session_key']}-rotated")

            def _closed(*args, **kwargs):
                raise sqlite3.OperationalError("database is closed")

            monkeypatch.setattr(srv, "_ensure_session_db_row", _closed)
            srv._append_model_switch_marker(session, model="qwen3", provider="local")

            # In-memory history still carries the marker even though the durable write failed.
            assert srv._is_model_switch_marker(session["history"][-1])
        finally:
            srv._sessions.pop(created["result"]["session_id"], None)