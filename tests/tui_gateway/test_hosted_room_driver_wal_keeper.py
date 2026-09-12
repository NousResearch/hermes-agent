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
import threading
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
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
    finally:
        conn.close()


def slow_rooms_provider():
    """A rooms provider that blocks from its FIRST call (concurrent with the
    keeper acquire) until the returned ``release`` event is set — isolating
    the fresh-database sequence the review exercised: keeper first, store
    connection second, no room thread racing the WAL flip. Returns
    ``(provider, release)``."""
    gate = threading.Event()

    def _provider():
        gate.wait(timeout=30)
        return [BINDING]

    return _provider, gate


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

        # A writer's WAL must exist while the writer is open.
        writer = sqlite3.connect(db, timeout=10)
        try:
            writer.execute("CREATE TABLE IF NOT EXISTS probe (k TEXT)")
            writer.execute("INSERT INTO probe VALUES ('v')")
            writer.commit()
            assert _wal_path(db).exists(), "writer's WAL sidecar missing"
        finally:
            writer.close()

        # With the last non-keeper connection gone, every ephemeral open/close
        # cycle is a would-be "last close" — exactly what the 5 s poll loop
        # does in production. Without the keeper each close unlinks the
        # sidecars; with it, they must survive.
        for _ in range(3):
            _ephemeral_cycle(db)
        assert _wal_path(db).exists(), (
            "ephemeral close deleted the WAL sidecar while the keeper was open"
        )
    finally:
        runtime.stop(timeout=2.0)
    assert runtime._wal_keeper is None, "keeper must close on stop"


def test_fresh_database_keeper_applies_journal_policy(tmp_path: Path):
    """On a fresh (never-written) database the keeper must itself apply the
    journal policy, not open raw: a raw keeper connects while the file is
    still in DELETE mode, the later store connection flips the header to WAL,
    and the stale keeper never joins the WAL shared-memory index — so every
    ephemeral close remains a "last WAL member" close and deletes the
    sidecars even with the keeper held (review reproduction, PR #103665)."""
    db = tmp_path / "state.db"
    rooms, release = slow_rooms_provider()
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=rooms,
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    try:
        runtime.start()
        _wait_for(lambda: runtime._wal_keeper is not None)

        # The later store connection applies the canonical policy and writes.
        from hermes_state_wal import apply_wal_with_fallback

        store = sqlite3.connect(db, timeout=10)
        assert apply_wal_with_fallback(store) == "wal"
        store.execute("CREATE TABLE IF NOT EXISTS probe (k TEXT)")
        store.execute("INSERT INTO probe VALUES ('v')")
        store.commit()
        assert _wal_path(db).exists(), "writer's WAL sidecar missing"
        store.close()
        assert _wal_path(db).exists(), (
            "store close deleted the WAL sidecar while the keeper was open — "
            "the keeper never joined the WAL index (opened in DELETE mode)"
        )

        _ephemeral_cycle(db)
        assert _wal_path(db).exists(), (
            "ephemeral close deleted the WAL sidecar while the keeper was open"
        )
    finally:
        release.set()
        runtime.stop(timeout=2.0)
    assert runtime._wal_keeper is None, "keeper must close on stop"


def test_configured_delete_mode_leaves_no_wal_sidecars(tmp_path: Path, monkeypatch):
    """With ``database.journal_mode: delete`` the keeper honors the configured
    policy: the fresh DB stays in DELETE and no WAL sidecars appear. (Contract:
    the keeper acquire follows the policy function — the same one
    hosted_rooms_common uses — instead of guessing.)"""
    monkeypatch.setattr("hermes_state_wal.resolve_journal_mode", lambda: "delete")
    db = tmp_path / "state.db"
    rooms, release = slow_rooms_provider()
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=rooms,
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    try:
        runtime.start()
        # The policy now honors journal_mode=delete inside the keeper acquire —
        # the keeper is deliberately DROPPED (a non-WAL keeper guards nothing),
        # so wait for the error record instead of the keeper reference.
        _wait_for(lambda: "wal keeper inapplicable" in (runtime._last_error or ""))
        probe = sqlite3.connect(db, timeout=10)
        mode = probe.execute("PRAGMA journal_mode").fetchone()[0]
        probe.close()
        assert mode.lower() != "wal", (
            f"configured journal_mode=delete must not be flipped to WAL, got {mode!r}"
        )
        assert not _wal_path(db).exists()
    finally:
        release.set()
        runtime.stop(timeout=2.0)


def test_vulnerable_runtime_fresh_db_keeper_and_store_stay_consistent(tmp_path: Path, monkeypatch):
    """On a WAL-reset-vulnerable SQLite (3.7.0–3.51.2) the policy gate refuses
    to enable WAL on non-WAL files. The keeper must take the SAME branch as the
    canonical store connection — both ending in DELETE with no mixed modes and
    no stranded sidecars — instead of a raw keeper opening in DELETE while a
    store connection flips WAL behind its back (the defect-1 shape, one gate
    earlier)."""
    import hermes_state_wal

    monkeypatch.setattr(
        hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: True
    )
    db = tmp_path / "state.db"
    rooms, release = slow_rooms_provider()
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=rooms,
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    try:
        runtime.start()
        # Vulnerable runtime: policy leaves the fresh DB in DELETE, so the
        # keeper is dropped — wait for the inapplicable-mode record instead of
        # a keeper reference, then assert keeper and store agree on the mode.
        _wait_for(lambda: "wal keeper inapplicable" in (runtime._last_error or ""))

        probe = sqlite3.connect(db, timeout=10)
        keeper_mode = probe.execute("PRAGMA journal_mode").fetchone()[0]
        probe.close()
        assert keeper_mode.lower() != "wal", (
            f"vulnerable runtime must not flip fresh DB to WAL, got {keeper_mode!r}"
        )

        from hermes_state_wal import apply_wal_with_fallback

        store = sqlite3.connect(db, timeout=10)
        store_mode = apply_wal_with_fallback(store)
        store.execute("CREATE TABLE IF NOT EXISTS probe (k TEXT)")
        store.execute("INSERT INTO probe VALUES ('v')")
        store.commit()
        store.close()
        assert store_mode == keeper_mode, (
            f"keeper ({keeper_mode!r}) and store ({store_mode!r}) disagree on journal mode"
        )
        assert not _wal_path(db).exists()

        # Data written through the store must be readable after its close.
        probe = sqlite3.connect(db, timeout=10)
        rows = probe.execute("SELECT * FROM probe").fetchall()
        probe.close()
        assert rows == [("v",)]
    finally:
        release.set()
        runtime.stop(timeout=2.0)


def test_indeterminate_mode_probe_drops_keeper(tmp_path: Path, monkeypatch):
    """When the on-disk journal-mode probe cannot decide (fresh DB whose
    PRAGMA is blocked, or the WAL-reset gate's indeterminate `current is None`
    branch), the policy reports 'wal' WITHOUT touching the file and the
    keeper would stay in DELETE — a later connection flipping the header to
    WAL then orphans it (sidecars deleted with _wal_keeper non-null). The
    acquire must therefore trust the file, not the verdict: verify the
    header and DROP the keeper when it is not actually WAL, recording the
    inapplicability and re-probing next cycle instead of holding false
    protection."""
    import hermes_state_wal

    real_probe = hermes_state_wal._on_disk_journal_mode
    # Probe indeterminate ONLY during the keeper's acquire window.
    probe_state = {"indeterminate": True}

    def _probe(conn):
        return None if probe_state["indeterminate"] else real_probe(conn)

    # The worker's acquire imports the helpers at call time via
    # `from hermes_state_wal import ...`, so patch the module attribute and the
    # import inside the driver picks up the mock.
    monkeypatch.setattr(hermes_state_wal, "_on_disk_journal_mode", _probe)

    db = tmp_path / "state.db"
    rooms, release = slow_rooms_provider()
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=rooms,
        rpc=FakeSessionRPC(),
        turn_lock=RecordingTurnLocks(),
        poll_interval_seconds=0.01,
    )
    try:
        runtime.start()
        _wait_for(
            lambda: "wal keeper inapplicable" in (runtime._last_error or "")
        )
        assert runtime._wal_keeper is None, (
            "indeterminate-mode keeper must be dropped, not held"
        )
        # Self-heal: probe becomes decidable; put the file actually into WAL
        # FIRST (while the probe is indeterminate the gate refuses the flip),
        # release the provider so worker cycles complete quickly, then the
        # next cycles must acquire the keeper.
        _set_wal_mode(db)
        probe_state["indeterminate"] = False
        release.set()
        runtime._last_error = None
        _wait_for(lambda: runtime._wal_keeper is not None)
    finally:
        release.set()
        runtime.stop(timeout=2.0)


def test_keeper_released_when_lease_cleanup_fails(tmp_path: Path, monkeypatch):
    """Blocker-2 regression: a non-DriverStateError out of
    _release_idle_leases (e.g. sqlite3.OperationalError) must not exit the
    worker finally block before the keeper is closed — that previously left
    _wal_keeper referenced (restart skips re-acquire) and the connection fd
    open forever."""
    db = tmp_path / "state.db"
    _set_wal_mode(db)
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

    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(runtime, "_release_idle_leases", _boom)
    runtime.stop(timeout=2.0)

    assert runtime._wal_keeper is None, "keeper reference must clear even when lease cleanup raises"
    # The connection object itself must be closed. It belongs to the dead
    # worker thread, so probe it from this fresh thread (only the creating
    # thread may use it — a still-open keeper would answer the SELECT).
    probe_result: list[bool] = []

    def _poke():
        try:
            keeper.execute("SELECT 1").fetchone()
            probe_result.append(True)  # still open — leaked
        except sqlite3.ProgrammingError:
            probe_result.append(False)  # closed as required

    t = threading.Thread(target=_poke)
    t.start()
    t.join(2)
    assert probe_result == [False], "keeper connection must be closed on stop even when lease cleanup raises"


def test_keeper_acquire_retries_after_transient_failure(tmp_path, monkeypatch):
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

    monkeypatch.setattr(sqlite3, "connect", flaky_connect)
    runtime.start()
    _wait_for(lambda: runtime._wal_keeper is not None, timeout=2.0)
    assert calls["n"] >= 2, "acquire must be retried after a transient failure"
    runtime.stop(timeout=2.0)
    assert runtime._wal_keeper is None


def test_failed_acquire_does_not_leak_connection(tmp_path, monkeypatch):
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
    closed = []

    class _BrokenConn:
        def close(self):
            closed.append(True)

        def execute(self, *a, **k):
            raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: _BrokenConn())
    runtime._acquire_wal_keeper()
    assert runtime._wal_keeper is None, "keeper must stay unset on failure"
    assert closed == [True], (
        f"broken connection must be closed exactly once, got {closed}"
    )
    assert runtime._last_error and "wal keeper unavailable" in runtime._last_error
