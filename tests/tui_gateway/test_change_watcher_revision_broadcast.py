"""Broadcast semantics on real stores with the write-side sessions revision installed."""
from contextlib import closing
import sqlite3

import pytest

from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def watched_store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])
    for name in ("_sessions_db_sig_cache", "_change_sigs", "_change_checked_at", "_change_broadcast_at"):
        monkeypatch.setattr(server, name, {})
    monkeypatch.setattr(server, "_CHANGE_WATCHES", {
        "sessions.changed": server._CHANGE_WATCHES["sessions.changed"],
    })
    events = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda event, payload: events.append((event, payload)))
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("s1", "cli")
    try:
        assert server._sessions_sig()[0][0] == "revision"
        server._broadcast_watched_changes(now=0.0)
        assert events == []
        yield db, path, events
    finally:
        db.close()


@pytest.mark.parametrize("sql", [
    "UPDATE sessions SET title = 'new' WHERE id = 's1'",
    "UPDATE sessions SET archived = 1 WHERE id = 's1'",
    "UPDATE sessions SET message_count = message_count + 1 WHERE id = 's1'",
    "INSERT INTO sessions (id, source, started_at) VALUES ('s2', 'cli', 1)",
    "DELETE FROM sessions WHERE id = 's1'",
])
def test_committed_session_changes_broadcast_after_window(watched_store, sql):
    _db, path, events = watched_store
    with closing(sqlite3.connect(path)) as writer, writer:
        writer.execute(sql)
    server._broadcast_watched_changes(now=2.0)
    assert events == [("sessions.changed", {})]
    server._broadcast_watched_changes(now=4.0)
    assert len(events) == 1


def test_heartbeat_noop_rollback_and_trailing_edge(watched_store):
    db, path, events = watched_store
    db.touch_session_activity("s1", description="thinking")
    with closing(sqlite3.connect(path)) as writer, writer:
        writer.execute("UPDATE sessions SET title = title WHERE id = 's1'")
        writer.execute("UPDATE sessions SET title = 'rollback' WHERE id = 's1'")
        writer.rollback()
    server._broadcast_watched_changes(now=2.0)
    assert events == []
    db.set_session_title("s1", "first")
    server._broadcast_watched_changes(now=4.0)
    assert events == [("sessions.changed", {})]
    db.set_session_title("s1", "second")
    server._broadcast_watched_changes(now=4.5)
    assert len(events) == 1
    server._broadcast_watched_changes(now=6.0)
    assert events == [("sessions.changed", {}), ("sessions.changed", {})]
