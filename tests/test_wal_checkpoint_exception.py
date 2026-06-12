"""Tests for WAL checkpoint exception handling in SessionDB.

Covers the fix for #44795: _try_wal_checkpoint silently swallows exceptions
from destructive TRUNCATE checkpoint, potentially corrupting state.db.
"""

import logging
import sqlite3
from unittest.mock import MagicMock, call

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


class TestWalCheckpointExceptionLogging:
    """Verify that WAL checkpoint failures are logged, not silently swallowed."""

    def test_try_wal_checkpoint_logs_warning_on_exception(self, db, caplog):
        """_try_wal_checkpoint should log warning when checkpoint raises."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("disk I/O error")
        db._conn = mock_conn

        with caplog.at_level(logging.WARNING):
            db._try_wal_checkpoint()

        assert any(
            "WAL checkpoint failed" in r.message and "disk I/O error" in r.message
            for r in caplog.records
        ), f"Expected warning log not found in: {[r.message for r in caplog.records]}"

    def test_try_wal_checkpoint_integrity_check_on_failure(self, db):
        """_try_wal_checkpoint should run PRAGMA quick_check after failure."""
        mock_conn = MagicMock()
        call_sequence = []

        def track_execute(sql, *args, **kwargs):
            call_sequence.append(sql)
            if "wal_checkpoint" in sql:
                raise sqlite3.OperationalError("checkpoint failed")
            if "quick_check" in sql:
                return MagicMock(fetchone=lambda: ("ok",))
            return MagicMock(fetchone=lambda: None)

        mock_conn.execute.side_effect = track_execute
        db._conn = mock_conn

        db._try_wal_checkpoint()

        assert any(
            "quick_check" in sql for sql in call_sequence
        ), f"Expected quick_check call, got: {call_sequence}"

    def test_try_wal_checkpoint_logs_error_on_integrity_failure(self, db, caplog):
        """_try_wal_checkpoint should log error when integrity check also fails."""
        mock_conn = MagicMock()

        def track_execute(sql, *args, **kwargs):
            if "wal_checkpoint" in sql:
                raise sqlite3.OperationalError("checkpoint failed")
            if "quick_check" in sql:
                raise sqlite3.OperationalError("corruption detected")
            return MagicMock(fetchone=lambda: None)

        mock_conn.execute.side_effect = track_execute
        db._conn = mock_conn

        with caplog.at_level(logging.ERROR):
            db._try_wal_checkpoint()

        assert any(
            "integrity check failed" in r.message
            for r in caplog.records
        ), f"Expected integrity error log not found in: {[r.message for r in caplog.records]}"

    def test_try_wal_checkpoint_succeeds_silently(self, db, caplog):
        """_try_wal_checkpoint should not warn when checkpoint succeeds."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (0, 0, 0)
        db._conn = mock_conn

        with caplog.at_level(logging.WARNING):
            db._try_wal_checkpoint()

        assert not any(
            "WAL checkpoint failed" in r.message for r in caplog.records
        ), "Unexpected warning on successful checkpoint"

    def test_close_does_not_raise_on_checkpoint_failure(self, db):
        """close() should not raise when final checkpoint fails."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is closed")
        db._conn = mock_conn

        # close() should not raise even if checkpoint fails
        db.close()

        # After close, _conn is None — verify it completed
        assert db._conn is None

    def test_vacuum_continues_on_checkpoint_failure(self, db):
        """vacuum() should continue to VACUUM even if pre-VACUUM checkpoint fails."""
        execute_calls = []

        def track_execute(sql, *args, **kwargs):
            execute_calls.append(sql)
            if "wal_checkpoint" in sql:
                raise sqlite3.OperationalError("checkpoint failed")
            return MagicMock(fetchone=lambda: None)

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = track_execute
        db._conn = mock_conn

        db.vacuum()

        # VACUUM should still be called even if checkpoint fails
        assert any("VACUUM" in sql for sql in execute_calls), \
            f"Expected VACUUM call, got: {execute_calls}"
