"""Tests for SessionDB WAL checkpoint strategy (issue #45383).

Verifies that ALL checkpoints on the shared state.db use PASSIVE mode:
periodic, close(), and pre-VACUUM. TRUNCATE fires a full WAL reset, and
transient per-cron-run connections closing many times an hour would race
the live gateway writer and corrupt B-tree pages (#45383).
"""

import sqlite3
import logging
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


class TrackingConnection:
    """sqlite3.Connection proxy that records executed SQL strings."""

    def __init__(self, conn):
        self._conn = conn
        self.execute_calls = []

    def execute(self, sql, *args, **kwargs):
        self.execute_calls.append(sql)
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    try:
        session_db.close()
    except Exception:
        pass


class TestTryWalCheckpointPassive:
    """_try_wal_checkpoint() should use PASSIVE mode for periodic use."""

    def test_checkpoint_uses_passive_mode(self, db):
        """PASSIVE checkpoint does not require exclusive lock — safe for large DBs."""
        # Capture the real connection's execute before mocking
        real_conn = db._conn
        execute_calls = []

        def tracking_execute(sql, *args, **kwargs):
            execute_calls.append(sql)
            return real_conn.execute(sql, *args, **kwargs)

        # sqlite3.Connection.execute is read-only (C extension) — replace _conn
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = tracking_execute
        mock_conn.fetchone.return_value = None
        db._conn = mock_conn

        db._try_wal_checkpoint()

        passive_calls = [c for c in execute_calls if "wal_checkpoint(PASSIVE)" in c]
        truncate_calls = [c for c in execute_calls if "wal_checkpoint(TRUNCATE)" in c]
        assert len(passive_calls) == 1, (
            f"Expected 1 PASSIVE checkpoint call, got {len(passive_calls)}"
        )
        assert len(truncate_calls) == 0, (
            "Periodic checkpoint should NOT use TRUNCATE"
        )

    def test_checkpoint_logs_warning_on_failure(self, db, caplog):
        """Failed PASSIVE checkpoint logs a warning instead of silent pass."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("disk I/O error")
        db._conn = mock_conn

        with caplog.at_level(logging.WARNING):
            db._try_wal_checkpoint()

        assert any("WAL checkpoint (PASSIVE) failed" in r.message for r in caplog.records), (
            f"Expected warning log about PASSIVE checkpoint failure, got: {caplog.text}"
        )

    def test_checkpoint_returns_result_on_success(self, db):
        """Successful PASSIVE checkpoint does not raise."""
        db._try_wal_checkpoint()


class TestCloseUsesPassive:
    """close() must use PASSIVE. Transient per-cron-run SessionDB connections
    close many times an hour; a TRUNCATE reset there races the live gateway
    writer on the large WAL DB and corrupts B-tree pages (#45383)."""

    def test_close_uses_passive_mode(self, db):
        """close() checkpoints PASSIVE, never TRUNCATE."""
        real_conn = db._conn
        execute_calls = []

        def tracking_execute(sql, *args, **kwargs):
            execute_calls.append(sql)
            return real_conn.execute(sql, *args, **kwargs)

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = tracking_execute
        db._conn = mock_conn

        db.close()

        truncate_calls = [c for c in execute_calls if "wal_checkpoint(TRUNCATE)" in c]
        passive_calls = [c for c in execute_calls if "wal_checkpoint(PASSIVE)" in c]
        assert len(truncate_calls) == 0, (
            "close() must NOT TRUNCATE (races the live gateway writer, #45383)"
        )
        assert len(passive_calls) == 1, (
            f"Expected 1 PASSIVE checkpoint at close, got {len(passive_calls)}"
        )

    def test_close_logs_debug_on_failure(self, db, caplog):
        """Failed PASSIVE checkpoint at close logs debug (close is best-effort)."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        db._conn = mock_conn

        with caplog.at_level(logging.DEBUG):
            db.close()

        assert any("WAL checkpoint (PASSIVE) at close failed" in r.message for r in caplog.records), (
            f"Expected debug log about PASSIVE failure at close, got: {caplog.text}"
        )


class TestVacuumUsesPassive:
    """Manual vacuum paths must checkpoint PASSIVE, never TRUNCATE."""

    def test_vacuum_uses_passive_before_vacuum(self, db):
        """SessionDB.vacuum() checkpoints PASSIVE before running VACUUM."""
        real_conn = db._conn
        tracking_conn = TrackingConnection(real_conn)
        db._conn = tracking_conn

        db.vacuum()

        checkpoint_calls = [
            c for c in tracking_conn.execute_calls if "wal_checkpoint" in c.lower()
        ]
        truncate_calls = [c for c in checkpoint_calls if "TRUNCATE" in c]
        passive_calls = [c for c in checkpoint_calls if "PASSIVE" in c]
        vacuum_calls = [
            c for c in tracking_conn.execute_calls if c.strip().upper() == "VACUUM"
        ]
        assert truncate_calls == []
        assert passive_calls == ["PRAGMA wal_checkpoint(PASSIVE)"]
        assert vacuum_calls == ["VACUUM"]
        assert tracking_conn.execute_calls.index(
            passive_calls[0]
        ) < tracking_conn.execute_calls.index(vacuum_calls[0])

    def test_optimize_storage_uses_passive_after_vacuum(self, db):
        """optimize_fts_storage() checkpoints PASSIVE after its VACUUM."""
        real_conn = db._conn
        tracking_conn = TrackingConnection(real_conn)
        db._conn = tracking_conn

        result = db.optimize_fts_storage(vacuum=True)

        checkpoint_calls = [
            c for c in tracking_conn.execute_calls if "wal_checkpoint" in c.lower()
        ]
        truncate_calls = [c for c in checkpoint_calls if "TRUNCATE" in c]
        passive_calls = [c for c in checkpoint_calls if "PASSIVE" in c]
        vacuum_calls = [
            c for c in tracking_conn.execute_calls if c.strip().upper() == "VACUUM"
        ]
        assert result["ok"] is True
        assert result["vacuumed"] is True
        assert truncate_calls == []
        assert passive_calls == ["PRAGMA wal_checkpoint(PASSIVE)"]
        assert vacuum_calls == ["VACUUM"]
        assert tracking_conn.execute_calls.index(
            vacuum_calls[0]
        ) < tracking_conn.execute_calls.index(passive_calls[0])


class TestCheckpointFrequency:
    """Checkpoint triggers every N writes."""

    def test_checkpoint_triggers_at_interval(self, db):
        """_try_wal_checkpoint is called every _CHECKPOINT_EVERY_N_WRITES writes."""
        call_count = [0]
        original = db._try_wal_checkpoint

        def counting_checkpoint():
            call_count[0] += 1
            original()

        db._try_wal_checkpoint = counting_checkpoint

        # Write exactly _CHECKPOINT_EVERY_N_WRITES sessions to trigger one checkpoint
        n = db._CHECKPOINT_EVERY_N_WRITES
        import time as _time
        for i in range(n):
            db._execute_write(lambda conn, _i=i: conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                (f"sess_{_i}", "test", _time.time()),
            ))

        assert call_count[0] == 1, (
            f"Expected 1 checkpoint after {n} writes, got {call_count[0]}"
        )
