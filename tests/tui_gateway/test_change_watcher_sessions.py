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


def test_session_rows_sig_is_chunk_size_invariant_and_handles_blobs(tmp_path):
    from tui_gateway import change_watcher

    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    conn = sqlite3.connect(db_path)
    for idx in range(2, 12):
        conn.execute(
            "INSERT INTO sessions(id, title, last_activity_at, message_count) VALUES (?, ?, ?, ?)",
            (f"session-{idx}", b"binary-\xff-title" if idx == 5 else f"Title {idx}", 100 + idx, idx),
        )
    conn.commit()

    sig_default, is_fb = change_watcher._session_rows_sig(conn, 123)
    assert not is_fb

    class _OneByOneCursor:
        def __init__(self, real_cur):
            self._cur = real_cur

        def fetchmany(self, _size=4096):
            return self._cur.fetchmany(1)

    class _OneByOneConn:
        def __init__(self, real_conn):
            self._conn = real_conn

        def execute(self, sql):
            if sql.startswith("SELECT "):
                return _OneByOneCursor(self._conn.execute(sql))
            return self._conn.execute(sql)

    sig_chunk1, is_fb1 = change_watcher._session_rows_sig(_OneByOneConn(conn), 123)
    conn.close()
    assert not is_fb1
    assert sig_default == sig_chunk1


def test_state_meta_local_policy_partial_indexes_prevent_full_scans(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_common import LOCAL_POLICY_PREFIX, NONDEFAULT_POLICY_WHERE

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        with db._read_ctx() as conn:
            plan_lineage = conn.execute(
                "EXPLAIN QUERY PLAN SELECT substr(key, 25), value FROM state_meta "
                "WHERE key GLOB 'gateway.local_policy.v1:*' AND json_valid(value) "
                "AND json_array_length(value, '$.lineage') > 1"
            ).fetchall()
            plan_special = conn.execute(
                f"EXPLAIN QUERY PLAN SELECT value FROM state_meta WHERE {NONDEFAULT_POLICY_WHERE}"
            ).fetchall()
            plan_union = conn.execute(
                "EXPLAIN QUERY PLAN SELECT key,value FROM state_meta "
                "WHERE key IN (SELECT ? || value FROM json_each(?)) "
                f"UNION SELECT key,value FROM state_meta WHERE {NONDEFAULT_POLICY_WHERE}",
                (LOCAL_POLICY_PREFIX, '["sid-1"]'),
            ).fetchall()
    finally:
        db.close()

    assert any("idx_state_meta_local_policy_lineage" in str(tuple(row)) for row in plan_lineage), [tuple(r) for r in plan_lineage]
    assert any("idx_state_meta_local_policy_special" in str(tuple(row)) for row in plan_special), [tuple(r) for r in plan_special]
    assert any("idx_state_meta_local_policy_special" in str(tuple(row)) for row in plan_union), [tuple(r) for r in plan_union]


