"""WAL-keeper regression: the driver's poll loop must not let SQLite delete the
state.db -wal/-shm sidecars on ephemeral open/close cycles.

Root cause (observed in production, Sep 2026): the 5 s idle poll opens and
closes the DB on every cycle. Each close makes SQLite believe it is the last
connection, so it unlinks the sidecars — stranding peer processes holding
long-lived connections on a deleted shm generation (split wal-index). The
driver now holds one persistent keeper connection for the supervisor's
lifetime: while any connection is open, SQLite never deletes the sidecars on
another connection's close.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tui_gateway.hosted_room_driver import HostedRoomRuntime

from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING,
    FakeSessionRPC,
    RecordingTurnLocks,
    _wait_for,
)


def _ephemeral_cycle(db: Path) -> None:
    conn = sqlite3.connect(db, timeout=10)
    try:
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    finally:
        conn.close()


def _wal_path(db: Path) -> Path:
    return db.with_name(db.name + "-wal")


def _set_wal_mode(db: Path) -> None:
    conn = sqlite3.connect(db, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    finally:
        conn.close()


def test_driver_keeps_wal_sidecars_across_ephemeral_cycles(tmp_path: Path):
    db = tmp_path / "state.db"
    _set_wal_mode(db)
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=[BINDING],
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    try:
        runtime.start()
        _wait_for(lambda: runtime._wal_keeper is not None)

        # A writer's WAL must exist, and must STILL exist after ephemeral
        # open/close cycles — exactly what the poll loop does per iteration.
        writer = sqlite3.connect(db, timeout=10)
        try:
            writer.execute("CREATE TABLE IF NOT EXISTS probe (k TEXT)")
            writer.execute("INSERT INTO probe VALUES ('v')")
            writer.commit()
            assert _wal_path(db).exists(), "writer's WAL sidecar missing"
            for _ in range(3):
                _ephemeral_cycle(db)
                assert _wal_path(db).exists(), (
                    "ephemeral close deleted the WAL sidecar while the keeper was open"
                )
        finally:
            writer.close()
            assert _wal_path(db).exists(), (
                "a single close deleted the WAL sidecar while the keeper was open"
            )
    finally:
        runtime.stop(timeout=2.0)
        assert runtime._wal_keeper is None, "keeper must close on stop"


def test_keeper_acquire_retries_after_transient_failure(tmp_path: Path):
    """A failed acquire (e.g. transient SQLITE_BUSY) must not defeat the keeper
    for the runtime's whole life — the worker loop retries on later cycles."""
    db = tmp_path / "state.db"
    _set_wal_mode(db)
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=[BINDING],
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    # Simulate a first failed acquire: raise once, then succeed.
    original_connect = sqlite3.connect
    calls = {"n": 0}

    def flaky_connect(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_connect(*args, **kwargs)

    sqlite3.connect = flaky_connect
    try:
        runtime.start()
        _wait_for(lambda: runtime._wal_keeper is not None, timeout=2.0)
        assert calls["n"] >= 2, "acquire must be retried after a transient failure"
    finally:
        sqlite3.connect = original_connect
        runtime.stop(timeout=2.0)
        assert runtime._wal_keeper is None


def test_failed_acquire_does_not_leak_connection(tmp_path: Path):
    """When the keeper's content probe fails, the opened connection must be
    closed, not leaked."""
    db = tmp_path / "state.db"
    _set_wal_mode(db)
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=[BINDING],
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    leaked = []

    original_connect = sqlite3.connect

    def probe_failing_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        original_close = conn.close

        def failing_close():
            leaked.append(False)  # marker: close called on this connection
            original_close()

        conn.close = failing_close
        return conn

    # Make the content probe fail: monkeypatch the keeper's execute path by
    # pointing the runtime at a database whose sqlite_master read fails.
    # Simplest deterministic route: call _acquire_wal_keeper with a broken
    # in-memory-style path after patching connect to return a connection
    # whose execute raises.
    class _BrokenConn:
        def close(self):
            leaked.append(True)

        def execute(self, *a, **k):
            raise sqlite3.OperationalError("disk I/O error")

    sqlite3.connect = lambda *a, **k: _BrokenConn()
    try:
        runtime._acquire_wal_keeper()
        assert runtime._wal_keeper is None, "keeper must stay unset on failure"
        assert leaked == [True], f"broken connection must be closed exactly once, got {leaked}"
        assert runtime._last_error and "wal keeper unavailable" in runtime._last_error
    finally:
        sqlite3.connect = original_connect