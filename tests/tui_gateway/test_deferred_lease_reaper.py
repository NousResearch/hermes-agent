"""Recover missed deferred-lease callbacks without revoking a live writer (#62823)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.active_sessions import ActiveSessionLease, try_acquire_active_session
from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor, TurnSettlement


@pytest.fixture(autouse=True)
def isolated_bookkeeping(monkeypatch):
    monkeypatch.setattr(server, "_deferred_active_session_leases", {})
    monkeypatch.setattr(server, "_deferred_active_session_lease_ages", {})
    monkeypatch.setattr(server, "_deferred_active_session_settlements", {})
    monkeypatch.setattr(server, "_deferred_active_session_releasing", set())
    yield
    for lease in server._deferred_active_session_leases.values():
        lease.release()


def _lease(tmp_path: Path, key: str = "stored-session") -> ActiveSessionLease:
    lease, err = try_acquire_active_session(
        session_id=key, surface="desktop", config={}, registry_home=tmp_path, track_liveness=True)
    assert err is None
    assert lease is not None
    return lease


def _defer(lease: ActiveSessionLease, age_seconds: float, settlement=None) -> None:
    lease_id = str(lease.lease_id)
    server._deferred_active_session_leases[lease_id] = lease
    server._deferred_active_session_lease_ages[lease_id] = time.time() - age_seconds
    server._deferred_active_session_settlements[lease_id] = settlement


def test_age_without_settlement_proof_keeps_deferred_lease(tmp_path):
    lease = _lease(tmp_path)
    _defer(lease, age_seconds=server._DEFERRED_ACTIVE_SESSION_LEASE_TTL_SECONDS + 60)

    try:
        assert server._reap_stale_deferred_leases() == 0
        session = {"history_lock": threading.Lock(), "_deferred_active_session_lease": lease}
        server._release_deferred_active_session_lease(session)
        assert lease.released is False
        assert str(lease.lease_id) in server._own_live_lease_ids()
    finally:
        lease.release()


def test_fresh_deferred_lease_is_kept(tmp_path):
    lease = _lease(tmp_path)
    settlement = TurnSettlement()
    _defer(lease, age_seconds=10, settlement=settlement)

    reaped = server._reap_stale_deferred_leases()

    assert reaped == 0
    assert lease.released is False
    assert str(lease.lease_id) in server._deferred_active_session_leases

    # cleanup: settle normally
    settlement.completed.set()
    session = {"history_lock": threading.Lock(), "_deferred_active_session_lease": lease}
    server._release_deferred_active_session_lease(session)
    assert lease.released is True


def test_settlement_clears_the_age_entry(tmp_path):
    lease = _lease(tmp_path)
    settlement = TurnSettlement()
    settlement.completed.set()
    _defer(lease, age_seconds=0, settlement=settlement)
    session = {"history_lock": threading.Lock(), "_deferred_active_session_lease": lease}

    server._release_deferred_active_session_lease(session)

    assert lease.released is True
    assert str(lease.lease_id) not in server._deferred_active_session_lease_ages
    assert str(lease.lease_id) not in server._deferred_active_session_leases


def test_reaped_lease_no_longer_vouched_by_own_live_lease_ids(monkeypatch, tmp_path):
    lease = _lease(tmp_path)
    settlement = TurnSettlement()
    settlement.completed.set()
    _defer(lease, age_seconds=server._DEFERRED_ACTIVE_SESSION_LEASE_TTL_SECONDS + 60, settlement=settlement)

    assert str(lease.lease_id) in server._own_live_lease_ids()
    server._reap_stale_deferred_leases()
    assert str(lease.lease_id) not in server._own_live_lease_ids()


def test_completion_recovers_after_callback_raises(monkeypatch, tmp_path):
    lease = _lease(tmp_path)
    settlement = TurnSettlement()
    supervisor = HostSupervisor(autostart=False)
    monkeypatch.setattr(supervisor, "start", lambda: None)
    monkeypatch.setattr(supervisor, "_send_frame", lambda _frame: None)

    def broken_callback(_frame):
        raise RuntimeError("metadata adoption failed")

    supervisor.submit_turn({"request_id": "r"}, on_complete=broken_callback, settlement=settlement)
    _defer(lease, server._DEFERRED_ACTIVE_SESSION_LEASE_TTL_SECONDS + 60, settlement)
    supervisor._handle_host_frame({"type": "turn.end", "request_id": "r"})
    assert not lease.released
    assert server._reap_stale_deferred_leases() == 1
    assert lease.released


def test_reaper_keeps_live_child_but_recovers_its_exit(tmp_path):
    lease = _lease(tmp_path)
    settlement = TurnSettlement()
    child = SimpleNamespace(returncode=None)
    child.poll = lambda: child.returncode
    settlement.process = child
    _defer(lease, server._DEFERRED_ACTIVE_SESSION_LEASE_TTL_SECONDS + 60, settlement)
    assert server._reap_stale_deferred_leases() == 0
    assert lease.lease_id in server._own_live_lease_ids()
    child.returncode = 1
    assert server._reap_stale_deferred_leases() == 1
    assert lease.released


@pytest.mark.parametrize("via_callback", [False, True])
def test_failed_release_stays_vouched_until_retry(monkeypatch, tmp_path, via_callback):
    lease = _lease(tmp_path)
    settlement = TurnSettlement()
    settlement.completed.set()
    _defer(lease, server._DEFERRED_ACTIVE_SESSION_LEASE_TTL_SECONDS + 60, settlement)
    session = {"history_lock": threading.Lock(), "_deferred_active_session_lease": lease}

    def fail_release():
        raise OSError("registry unavailable")

    with monkeypatch.context() as patcher:
        patcher.setattr(lease, "release", fail_release)
        if via_callback:
            server._release_deferred_active_session_lease(session)
            assert session["_deferred_active_session_lease"] is lease
        else:
            assert server._reap_stale_deferred_leases() == 0
        assert lease.lease_id in server._own_live_lease_ids()
        assert not lease.released
    assert server._reap_stale_deferred_leases() == 1
    assert lease.released
    assert lease.lease_id not in server._own_live_lease_ids()


def test_release_io_does_not_block_bookkeeping_or_double_release(monkeypatch, tmp_path):
    lease = _lease(tmp_path)
    settlement = TurnSettlement()
    settlement.completed.set()
    _defer(lease, server._DEFERRED_ACTIVE_SESSION_LEASE_TTL_SECONDS + 60, settlement)
    entered, finish = threading.Event(), threading.Event()
    release = lease.release
    calls = []

    def blocked_release():
        calls.append(1)
        entered.set()
        assert finish.wait(5)
        release()

    monkeypatch.setattr(lease, "release", blocked_release)
    worker = threading.Thread(target=server._reap_stale_deferred_leases)
    worker.start()
    try:
        assert entered.wait(5)
        assert lease.lease_id in server._own_live_lease_ids()
        assert server._reap_stale_deferred_leases() == 0
        server._release_deferred_active_session_lease(
            {"history_lock": threading.Lock(), "_deferred_active_session_lease": lease})
    finally:
        finish.set()
        worker.join(5)
    assert not worker.is_alive()
    assert calls == [1]
    assert lease.released
