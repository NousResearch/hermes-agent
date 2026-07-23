"""Tests for SessionDB WAL checkpoint strategy (issue #45383).

Verifies that periodic checkpoints use PASSIVE mode (safe for large DBs)
while close() and pre-VACUUM paths still use TRUNCATE.
"""

import sqlite3
import logging
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


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

    def test_returned_busy_checkpoint_is_logged_at_debug(self, db, caplog):
        """The busy flag returned by SQLite is normal contention, not a failure."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (1, 100, 0)
        db._conn = mock_conn

        with caplog.at_level(logging.DEBUG):
            db._try_wal_checkpoint()

        assert "WAL checkpoint skipped (database busy)" in caplog.text
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)

    def test_busy_checkpoint_exception_is_logged_at_debug(self, db, caplog):
        """A busy exception is expected under concurrency and remains best-effort."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        db._conn = mock_conn

        with caplog.at_level(logging.DEBUG):
            db._try_wal_checkpoint()

        assert "WAL checkpoint skipped (database busy)" in caplog.text
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)

    def test_checkpoint_returns_result_on_success(self, db):
        """Successful PASSIVE checkpoint does not raise."""
        db._try_wal_checkpoint()


class TestCloseUsesTruncate:
    """close() should still use TRUNCATE to shrink WAL on shutdown."""

    def test_close_uses_truncate_mode(self, db):
        """TRUNCATE at close is safe — no concurrent writers during shutdown."""
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
        assert len(truncate_calls) == 1, (
            f"Expected 1 TRUNCATE checkpoint at close, got {len(truncate_calls)}"
        )

    def test_close_logs_debug_on_failure(self, db, caplog):
        """Failed TRUNCATE at close logs debug (not warning — close is best-effort)."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        db._conn = mock_conn

        with caplog.at_level(logging.DEBUG):
            db.close()

        assert any("WAL checkpoint (TRUNCATE) at close failed" in r.message for r in caplog.records), (
            f"Expected debug log about TRUNCATE failure at close, got: {caplog.text}"
        )


class TestVacuumUsesTruncate:
    """vacuum() should surface checkpoint failures and still reclaim space."""

    def test_checkpoint_failure_is_logged_and_vacuum_continues(self, db, caplog):
        """A best-effort checkpoint failure must not prevent VACUUM."""
        execute_calls = []
        mock_conn = MagicMock()

        def execute(sql, *args, **kwargs):
            execute_calls.append(sql)
            if sql == "PRAGMA wal_checkpoint(TRUNCATE)":
                raise sqlite3.OperationalError("disk I/O error")
            return MagicMock()

        mock_conn.execute.side_effect = execute
        db._conn = mock_conn
        db.optimize_fts = MagicMock(return_value=0)

        with caplog.at_level(logging.DEBUG):
            assert db.vacuum() == 0

        assert execute_calls == ["PRAGMA wal_checkpoint(TRUNCATE)", "VACUUM"]
        assert "WAL checkpoint (TRUNCATE) before VACUUM failed" in caplog.text


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
