from __future__ import annotations

import os
import sqlite3

import tui_gateway.server as server


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

    before = getattr(server, "_sessions_sig")()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE gateway_heartbeats SET last_heartbeat = 200")
    conn.commit()
    conn.close()

    assert getattr(server, "_sessions_sig")() == before


def test_sessions_signature_changes_for_session_metadata_updates(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])

    seeded_mtime = db_path.stat().st_mtime_ns
    before = getattr(server, "_sessions_sig")()
    # The digest is cached behind the DB mtime. On a coarse timestamp clock (~4 ms ticks on the
    # Blacksmith slices) the UPDATE below can land in the same tick as the seed write, so the
    # cache reads it as "no commit yet" and returns the stale digest. Let the clock tick first
    # (the #111105 idiom), so the only thing under test is the content digest.
    while db_path.stat().st_mtime_ns <= seeded_mtime:
        os.utime(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE sessions SET title = 'Renamed' WHERE id = 'session-1'")
    conn.commit()
    conn.close()

    assert getattr(server, "_sessions_sig")() != before
