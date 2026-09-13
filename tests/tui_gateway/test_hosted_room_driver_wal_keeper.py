"""WAL-keeper regression for the hosted-room poll lifecycle.

The gateway's idle hosted-room poll opens and closes the shared state.db every
cycle. A persistent content-touched connection must keep the WAL/SHM generation
alive while those ephemeral reads run.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

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


@pytest.mark.linux_only
@pytest.mark.requires_wal
def test_driver_keeps_wal_sidecars_across_ephemeral_cycles(tmp_path: Path):
    """On a fresh database, the keeper must configure WAL mode and prevent
    ephemeral poll cycles from unlinking the WAL sidecar generation."""
    db = tmp_path / "state.db"
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

        writer = sqlite3.connect(db, timeout=10)
        try:
            writer.execute("CREATE TABLE IF NOT EXISTS probe (k TEXT)")
            writer.execute("INSERT INTO probe VALUES ('v')")
            writer.commit()
            assert _wal_path(db).exists(), "writer's WAL sidecar missing"
        finally:
            writer.close()

        for _ in range(3):
            _ephemeral_cycle(db)
        assert _wal_path(db).exists(), (
            "ephemeral close deleted the WAL sidecar while the keeper was open"
        )
    finally:
        runtime.stop(timeout=2.0)
    assert runtime._wal_keeper is None, "keeper must close on stop"


@pytest.mark.requires_wal
def test_fresh_database_keeper_applies_wal_policy(tmp_path: Path):
    """A fresh database must have the canonical WAL policy applied by the keeper
    before retaining it, confirming effective mode is WAL."""
    db = tmp_path / "state.db"
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
        row = runtime._wal_keeper.execute("PRAGMA journal_mode").fetchone()
        assert row is not None and str(row[0]).lower() == "wal"
    finally:
        runtime.stop(timeout=2.0)
    assert runtime._wal_keeper is None


def test_forced_safe_wal_policy_acquires_and_retains_keeper(tmp_path: Path, monkeypatch):
    """When a safe WAL policy is explicitly forced, the keeper is acquired and retained."""
    import hermes_state_wal

    monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: False)
    db = tmp_path / "state.db"
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
        assert runtime._wal_keeper_disabled is False
        row = runtime._wal_keeper.execute("PRAGMA journal_mode").fetchone()
        assert row is not None and str(row[0]).lower() == "wal"
    finally:
        runtime.stop(timeout=2.0)
    assert runtime._wal_keeper is None


@pytest.mark.requires_wal
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_keeper_released_when_lease_cleanup_raises(tmp_path: Path, monkeypatch):
    """When _release_idle_leases() raises sqlite3.OperationalError, the nested
    finally must still release the WAL keeper."""
    db = tmp_path / "state.db"
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=[BINDING],
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    runtime.start()
    _wait_for(lambda: runtime._wal_keeper is not None)
    keeper = runtime._wal_keeper
    assert keeper is not None

    def failing_release_idle_leases():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(runtime, "_release_idle_leases", failing_release_idle_leases)
    stopped = runtime.stop(timeout=2.0)
    assert stopped is True
    assert runtime._wal_keeper is None
    with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
        keeper.execute("SELECT 1")


def test_canonical_delete_mode_is_stable_without_keeper_or_error(tmp_path: Path, monkeypatch):
    """If the canonical journal policy yields a non-WAL mode (e.g. DELETE),
    the runtime treats it as a stable no-keeper state without retrying or recording an error."""
    db = tmp_path / "state.db"
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=[BINDING],
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    import hermes_state_wal

    monkeypatch.setattr(hermes_state_wal, "apply_wal_with_fallback", lambda *a, **k: "delete")

    runtime._acquire_wal_keeper()
    assert runtime._wal_keeper is None
    assert runtime._wal_keeper_disabled is True
    assert runtime.status()["last_error"] is None

    # Subsequent acquire call must be a no-op (stable state, no retry, no error)
    runtime._acquire_wal_keeper()
    assert runtime._wal_keeper is None
    assert runtime.status()["last_error"] is None


@pytest.mark.requires_wal
def test_keeper_acquire_retries_after_transient_failure(tmp_path: Path, monkeypatch):
    db = tmp_path / "state.db"
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=[BINDING],
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    original_connect = sqlite3.connect
    calls = {"n": 0}

    def flaky_connect(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", flaky_connect)
    runtime.start()
    _wait_for(lambda: runtime._wal_keeper is not None, timeout=2.0)
    assert calls["n"] >= 2, "acquire must be retried after a transient failure"
    runtime.stop(timeout=2.0)
    assert runtime._wal_keeper is None


def test_failed_acquire_does_not_leak_connection(tmp_path: Path, monkeypatch):
    db = tmp_path / "state.db"
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=[BINDING],
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    closed = []

    class _BrokenConn:
        def close(self):
            closed.append(True)

        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: _BrokenConn())
    runtime._acquire_wal_keeper()
    assert runtime._wal_keeper is None, "keeper must stay unset on failure"
    assert closed == [True], f"broken connection must close exactly once: {closed}"
    assert runtime._last_error and "wal keeper unavailable" in runtime._last_error
