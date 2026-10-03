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