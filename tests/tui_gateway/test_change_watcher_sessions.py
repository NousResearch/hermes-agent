from __future__ import annotations

import sqlite3
import time

from tui_gateway import server


def _seed_store(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            last_activity_at REAL,
            message_count INTEGER,
            archived INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0
        );
        CREATE TABLE gateway_heartbeats (
            backend_id TEXT PRIMARY KEY,
            last_heartbeat REAL
        );
        INSERT INTO sessions(id, title, last_activity_at, message_count)
        VALUES ('session-1', 'Original', 100, 1);
        INSERT INTO gateway_heartbeats(backend_id, last_heartbeat)
        VALUES ('backend-1', 100);
        """
    )
    conn.commit()
    conn.close()


def test_sessions_signature_ignores_gateway_heartbeat_only_writes(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])

    before = server._sessions_sig()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE gateway_heartbeats SET last_heartbeat = 200")
    conn.commit()
    conn.close()

    assert server._sessions_sig() == before


def test_sessions_signature_changes_for_session_metadata_updates(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])

    before = server._sessions_sig()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE sessions SET title = 'Renamed' WHERE id = 'session-1'")
    conn.commit()
    conn.close()

    assert server._sessions_sig() != before


def test_sessions_signature_sees_a_commit_inside_the_same_mtime_tick(tmp_path, monkeypatch):
    """Filesystem timestamps tick coarsely (~4 ms; whole seconds on some filesystems), and SQLite
    commits in place: a commit in the tick of the previous read leaves the DB/WAL mtime unchanged."""
    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])
    frozen_mtime = time.time_ns()
    monkeypatch.setattr(server, "_newest_mtime_ns", lambda _paths: frozen_mtime)

    before = server._sessions_sig()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE sessions SET title = 'Renamed' WHERE id = 'session-1'")
    conn.commit()
    conn.close()

    assert server._sessions_sig() != before


def test_sessions_signature_of_a_current_store_reads_no_session_rows(tmp_path, monkeypatch):
    """A store with the list-revision triggers is signed from one state_meta row, so a write costs the
    watcher the same whatever the history size (#146643); the row digest is only for older stores."""
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("session-1", source="cli")
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])
    digests = []
    monkeypatch.setattr(server, "_session_rows_digest", lambda *args: digests.append(args) or ("digest", b""))
    try:
        before = server._sessions_sig()
        db.set_session_title("session-1", "Renamed")
        after = server._sessions_sig()
    finally:
        db.close()

    assert after != before
    assert after[0][0] == "revision"
    assert digests == []
