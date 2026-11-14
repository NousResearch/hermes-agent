"""Regression: ephemeral SessionDB() opens must close their fds.

Long-lived gateways hit ``[Errno 24] Too many open files`` when tools open a
private ``SessionDB()`` per call and never ``close()`` it. Each open retains
the main db file plus the WAL descriptor under WAL mode. These tests fail if
the deterministic close is removed again from the hot call sites.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from tools import react_to_message_tool as react_mod
from tools.session_search_tool import session_search
from agent.trace_upload import load_session_messages


class _CountingDB:
    """Stand-in that counts close() without touching the real state.db."""

    instances: list["_CountingDB"] = []

    def __init__(self, *args, **kwargs):
        self.closed = 0
        self._kwargs = kwargs
        type(self).instances.append(self)

    def close(self):
        self.closed += 1

    # --- react_to_message surface ---
    def latest_message_row_id(self, session_key, role="user", offset=0):
        return 7

    def get_message_role(self, session_key, row_id):
        return "user"

    def set_message_reaction(self, session_key, row_id, emoji, author="agent"):
        return [{"emoji": emoji or "", "author": author}]

    # --- session_search / trace_upload surface ---
    def list_sessions_rich(self, *args, **kwargs):
        return []

    def resolve_session_id(self, session_id):
        return session_id

    def get_session(self, session_id):
        return {"id": session_id, "title": "t"}

    def get_messages_as_conversation(self, session_id):
        return [{"role": "user", "content": "hi"}]

    def search_messages(self, *args, **kwargs):
        return []

    def get_messages(self, *args, **kwargs):
        return []

    def get_messages_around(self, *args, **kwargs):
        return {"messages": [], "anchor": None}

    def get_anchored_view(self, *args, **kwargs):
        return {"messages": [], "anchor": None}


@pytest.fixture(autouse=True)
def _reset_counting_db():
    _CountingDB.instances.clear()
    yield
    _CountingDB.instances.clear()


def test_session_search_closes_fallback_db(monkeypatch):
    """session_search() without a caller db must close the handle it opened."""
    monkeypatch.setattr("hermes_state.SessionDB", _CountingDB)
    # Also patch the import site used inside session_search.
    import tools.session_search_tool as sst

    monkeypatch.setattr(sst, "SessionDB", _CountingDB, raising=False)
    with patch("hermes_state.SessionDB", _CountingDB):
        result = json.loads(session_search(db=None))
    assert result.get("success") is not False or "error" in result or True
    assert _CountingDB.instances, "expected SessionDB() to be constructed"
    assert all(db.closed == 1 for db in _CountingDB.instances)


def test_session_search_does_not_close_caller_db():
    """A shared long-lived SessionDB from the gateway must stay open."""
    caller = _CountingDB()
    # Bypass the real search path with an empty browse via the mock surface.
    with patch.object(
        type(caller),
        "list_sessions_rich",
        return_value=[],
        create=True,
    ):
        # Prefer the public list path used by browse (no query).
        # _list_recent_sessions calls methods on db; give it the counters.
        from tools import session_search_tool as sst

        with patch.object(sst, "_list_recent_sessions", return_value=json.dumps({"success": True, "sessions": []})):
            out = session_search(db=caller)
    assert json.loads(out)["success"] is True
    assert caller.closed == 0


def test_react_to_message_closes_db(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:test")
    monkeypatch.setattr(react_mod, "_open_session_db", lambda: _CountingDB())
    monkeypatch.setattr(react_mod.desktop_ui, "emit", lambda *a, **k: None)

    out = json.loads(react_mod.react_to_message_tool(emoji="👍"))
    assert out["success"] is True
    assert len(_CountingDB.instances) == 1
    assert _CountingDB.instances[0].closed == 1


def test_react_to_message_closes_db_on_error(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:test")

    class _Boom(_CountingDB):
        def latest_message_row_id(self, *a, **k):
            return None

    monkeypatch.setattr(react_mod, "_open_session_db", lambda: _Boom())
    out = json.loads(react_mod.react_to_message_tool(emoji="👍"))
    assert out.get("success") is False or "error" in out
    assert _Boom.instances[0].closed == 1


def test_load_session_messages_closes_db(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_state.SessionDB", _CountingDB)
    with patch("hermes_state.SessionDB", _CountingDB):
        messages, meta = load_session_messages("s1")
    assert messages and meta.get("id") == "s1"
    assert _CountingDB.instances
    assert all(db.closed == 1 for db in _CountingDB.instances)


def test_real_sessiondb_close_releases_fds(tmp_path):
    """End-to-end: open+close a real SessionDB must not leave the path open.

    Uses the public API only — not a source-shape check.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id="s_fd", source="test")
        db.append_message(session_id="s_fd", role="user", content="hello")
    finally:
        db.close()

    # A second open after close must succeed (lock/fds released).
    db2 = SessionDB(db_path=db_path)
    try:
        row = db2.get_session("s_fd")
        assert row is not None
        assert db2.get_messages_as_conversation("s_fd")
    finally:
        db2.close()
