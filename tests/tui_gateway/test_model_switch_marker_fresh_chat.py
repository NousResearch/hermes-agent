"""The model pivot of a chat that has not submitted yet must persist into a row that exists."""

import threading
from types import SimpleNamespace

from agent.context_compressor import _DB_PERSISTED_MARKER
from hermes_state import SessionDB
from tui_gateway import server


def _fresh_chat(tmp_path, monkeypatch):
    """A session the desktop opened but never submitted: the id exists, the ``sessions`` row does not."""
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = SimpleNamespace(_session_db=db, session_id="chat", model="new-model", provider="test-provider")
    session = {"session_key": "chat", "_sid": "marker-test", "agent": agent, "history": [],
               "history_lock": threading.RLock(), "history_version": 0, "test_notices": []}
    monkeypatch.setattr(server, "_emit",
                        lambda event, sid, payload: session["test_notices"].append((event, sid, payload)))
    # The lazily persisted row is written through the launch handle, not the agent's copy of it.
    monkeypatch.setattr(server, "_db", db, raising=False)
    assert db.get_session("chat") is None
    return session, db


def test_first_switch_in_a_fresh_chat_creates_the_row_it_writes_to(tmp_path, monkeypatch):
    """messages.session_id is a hard foreign key: without the row the pivot write is refused outright.

    The live agent always carries a ``_session_db`` handle, so "the handle is there" says nothing about
    the row behind it — the marker has to create the lazily persisted row first, or the switch loses the
    very notice that describes it.
    """
    session, db = _fresh_chat(tmp_path, monkeypatch)
    try:
        server._append_model_switch_marker(session, model="new-model", provider="test-provider")

        assert db.get_session("chat") is not None
        rows = db.get_messages("chat", include_inactive=True)
        assert [(row["role"], row["display_kind"], row["active"]) for row in rows] == [
            ("user", "model_switch", 1)]
        assert session["history"][-1].get(_DB_PERSISTED_MARKER) is True
        assert session["test_notices"] == []
    finally:
        db.close()


def test_switch_after_the_row_exists_does_not_recreate_it(tmp_path, monkeypatch):
    """The existing-row path stays a plain no-op: one row, one marker, no duplicate session."""
    session, db = _fresh_chat(tmp_path, monkeypatch)
    db.create_session("chat", "desktop")
    try:
        before = db.get_session("chat")
        server._append_model_switch_marker(session, model="new-model", provider="test-provider")

        after = db.get_session("chat")
        assert after is not None and before is not None
        assert after["started_at"] == before["started_at"]
        assert [row["display_kind"] for row in db.get_messages("chat", include_inactive=True)] == [
            "model_switch"]
    finally:
        db.close()
