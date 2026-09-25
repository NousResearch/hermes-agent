"""Stale deferred active-session leases are reaped, not held forever (#62823).

When a compute-host turn has not settled by session close, the session's real
active-session lease is parked in ``_deferred_active_session_leases`` and the
turn's completion callback is supposed to release it. If that callback is lost
(supervisor restart, child killed without failing pending turns), the lease
previously sat in the registry FOREVER: ``_own_live_lease_ids`` vouches for it,
the orphan sweep skips it, and the concurrent-session cap treats the dead
session as active — new sessions could not send until a backend restart.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from hermes_cli.active_sessions import ActiveSessionLease, try_acquire_active_session
from tui_gateway import server


def _lease(tmp_path: Path, key: str = "stored-session") -> ActiveSessionLease:
    lease, err = try_acquire_active_session(
        session_id=key, surface="desktop", config={}, registry_home=tmp_path, track_liveness=True)
    assert err is None
    assert lease is not None
    return lease


def _defer(lease: ActiveSessionLease, age_seconds: float) -> None:
    lease_id = str(lease.lease_id)
    server._deferred_active_session_leases[lease_id] = lease
    server._deferred_active_session_lease_ages[lease_id] = time.time() - age_seconds


def test_stale_deferred_lease_is_force_released(tmp_path):
    lease = _lease(tmp_path)
    _defer(lease, age_seconds=server._DEFERRED_ACTIVE_SESSION_LEASE_TTL_SECONDS + 60)

    reaped = server._reap_stale_deferred_leases()

    assert reaped == 1
    assert lease.released is True
    assert str(lease.lease_id) not in server._deferred_active_session_leases
    assert str(lease.lease_id) not in server._deferred_active_session_lease_ages


def test_fresh_deferred_lease_is_kept(tmp_path):
    lease = _lease(tmp_path)
    _defer(lease, age_seconds=10)

    reaped = server._reap_stale_deferred_leases()

    assert reaped == 0
    assert lease.released is False
    assert str(lease.lease_id) in server._deferred_active_session_leases

    # cleanup: settle normally
    session = {"history_lock": threading.Lock(), "_deferred_active_session_lease": lease}
    server._release_deferred_active_session_lease(session)
    assert lease.released is True


def test_settlement_clears_the_age_entry(tmp_path):
    lease = _lease(tmp_path)
    _defer(lease, age_seconds=0)
    session = {"history_lock": threading.Lock(), "_deferred_active_session_lease": lease}

    server._release_deferred_active_session_lease(session)

    assert lease.released is True
    assert str(lease.lease_id) not in server._deferred_active_session_lease_ages
    assert str(lease.lease_id) not in server._deferred_active_session_leases


def test_reaped_lease_no_longer_vouched_by_own_live_lease_ids(monkeypatch, tmp_path):
    lease = _lease(tmp_path)
    _defer(lease, age_seconds=server._DEFERRED_ACTIVE_SESSION_LEASE_TTL_SECONDS + 60)

    assert str(lease.lease_id) in server._own_live_lease_ids()
    server._reap_stale_deferred_leases()
    assert str(lease.lease_id) not in server._own_live_lease_ids()
