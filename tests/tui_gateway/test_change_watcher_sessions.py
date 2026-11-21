from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from types import SimpleNamespace

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


def test_sessions_signature_refreshes_after_a_same_tick_wal_commit_during_a_slow_read(tmp_path, monkeypatch):
    """Age the timestamp past the racy window while a read returns its earlier snapshot."""
    from persistence_assembly import open_persistence, profile_scope
    from utils import files

    db_path = tmp_path / "state.db"
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])
    monkeypatch.setattr(server, "_sessions_db_sig_cache", {})
    tick = time.time_ns()
    now = tick
    monkeypatch.setattr(files, "time", SimpleNamespace(time_ns=lambda: now))
    stamp = server._session_db_stamp
    monkeypatch.setattr(server, "_session_db_stamp", lambda db, wal: (stamp(db, wal)[0], tick))

    read = server._session_rows_sig
    reads = 0
    with open_persistence(profile_scope(home=tmp_path), eager=True), closing(sqlite3.connect(db_path)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        writer.execute("INSERT INTO sessions (id, source, started_at, title) VALUES ('s1', 'cli', 1, 'Original')")
        writer.commit()

        def read_then_commit(path, mtime):
            nonlocal now, reads
            prior = read(path, mtime)
            reads += 1
            if reads == 1:
                writer.execute("UPDATE sessions SET title = 'Renamed' WHERE id = 's1'")
                writer.commit()
                assert db_path.with_name("state.db-wal").is_file()
                now += files.RACY_MTIME_WINDOW_NS + 1
            return prior

        monkeypatch.setattr(server, "_session_rows_sig", read_then_commit)
        before = server._sessions_sig()
        refreshed = server._sessions_sig()
        assert refreshed != before
        assert server._sessions_sig() == refreshed
        assert reads == 2  # Once settled, the unchanged mtime is a stat-only hit.


def test_current_store_notifies_on_committed_list_changes_without_scanning_history(tmp_path, monkeypatch):
    from persistence_assembly import open_persistence, profile_scope
    from persistence.sqlite.housekeeping.maintenance import SQLiteSessionMaintenance

    def count_digest(self, **kwargs):
        digests.append(True)
        return original_digest(self, **kwargs)

    original_digest = SQLiteSessionMaintenance.session_projection_digest
    digests = []
    monkeypatch.setattr(SQLiteSessionMaintenance, "session_projection_digest", count_digest)
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])
    with open_persistence(profile_scope(home=tmp_path), eager=True):
        before = server._sessions_sig()
        with closing(sqlite3.connect(tmp_path / "state.db")) as writer:
            writer.execute("INSERT INTO sessions (id, source, started_at) VALUES ('session-1', 'cli', 1)")
            writer.rollback()
            assert server._sessions_sig() == before
            writer.execute("INSERT INTO sessions (id, source, started_at) VALUES ('session-1', 'cli', 1)")
            writer.commit()
            added = server._sessions_sig()
            assert added != before
            writer.execute("UPDATE sessions SET last_activity_at = 200 WHERE id = 'session-1'")
            writer.commit()
            assert server._sessions_sig() == added
            writer.execute("UPDATE sessions SET title = 'Renamed' WHERE id = 'session-1'")
            writer.commit()
            renamed = server._sessions_sig()
            assert renamed != added
            writer.execute("DELETE FROM sessions WHERE id = 'session-1'")
            writer.commit()
            assert server._sessions_sig() != renamed
    assert not digests, 'watcher scanned the full session projection on a current schema'
