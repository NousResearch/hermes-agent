"""RPC contract tests for the ``session.list`` method in ``tui_gateway.server``.

Issue #67122: ``session.list`` projected each session row to ``id``, ``title``,
``preview``, ``started_at``, ``message_count``, ``source`` but dropped the
``last_active`` timestamp that ``list_sessions_rich`` already computes. External
clients (e.g. qelg/hermes-chat) could observe the ordering but could not
display the latest activity timestamp.

These tests exercise the real ``session.list`` handler against a real
``SessionDB`` (no mock recomputation) and assert the ``last_active`` field is
present in the RPC response, reflects the last appended message, and falls back
to ``started_at`` for sessions that have no messages yet.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    """Import once; reload would duplicate the module's atexit hooks."""
    mod = importlib.import_module("tui_gateway.server")
    yield mod
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._db = None


@pytest.fixture()
def db(server, hermes_home, monkeypatch):
    """A SessionDB whose ``list_sessions_rich`` the server reads.

    Patches ``_get_db`` on the live module (the pattern used throughout
    ``tests/test_tui_gateway_server.py``) so the handler resolves this instance
    without auto-initialising one against the real HERMES_HOME. monkeypatch
    restores the original on teardown.
    """
    instance = SessionDB(db_path=hermes_home / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: instance)
    return instance


def _call(server, method, **params):
    return server._methods[method](1, params)


def test_session_list_includes_last_active_reflecting_last_message(server, db):
    """``last_active`` must be present and equal the last message timestamp."""
    db.create_session("sess-with-msgs", source="tui")
    db.append_message("sess-with-msgs", "user", "first", timestamp=1000.0)
    db.append_message("sess-with-msgs", "assistant", "reply", timestamp=2000.0)
    db.append_message("sess-with-msgs", "user", "second", timestamp=5000.0)

    resp = _call(server, "session.list", limit=10)
    sessions = resp["result"]["sessions"]
    assert sessions, "expected at least one session in the list"
    row = next(s for s in sessions if s["id"] == "sess-with-msgs")
    assert "last_active" in row, "session.list must project last_active (#67122)"
    assert row["last_active"] == 5000.0


def test_session_list_last_active_falls_back_to_started_at(server, db):
    """A session with no messages must still surface a usable ``last_active``."""
    db.create_session("sess-empty", source="tui")
    # No append_message calls -> last_active would otherwise be absent/None.

    resp = _call(server, "session.list", limit=10)
    sessions = resp["result"]["sessions"]
    row = next(s for s in sessions if s["id"] == "sess-empty")
    assert "last_active" in row, "last_active key must always be present"
    started = row["started_at"]
    # Falls back to started_at so external clients always get a timestamp.
    assert row["last_active"] == started


def test_session_list_last_active_is_numeric(server, db):
    """``last_active`` must be a number external clients can sort on."""
    db.create_session("sess-num", source="tui")
    db.append_message("sess-num", "user", "hi", timestamp=12345.5)

    resp = _call(server, "session.list", limit=10)
    sessions = resp["result"]["sessions"]
    row = next(s for s in sessions if s["id"] == "sess-num")
    assert isinstance(row["last_active"], (int, float))
