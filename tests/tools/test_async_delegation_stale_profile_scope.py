"""Profile-scope regression for async-delegation stale finalization."""

import sqlite3
import threading
import time

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_async_delegation_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    ad._monitor_stop.set()
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _delegation_state(db_path, delegation_id):
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT state FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()
        return None if row is None else row[0]
    finally:
        conn.close()


def test_stale_monitor_force_finalize_updates_origin_profile_ledger(tmp_path, monkeypatch):
    """A bare monitor thread must settle the profile that dispatched the unit.

    Normal async workers carry the dispatching ContextVars via
    ``propagate_context_to_thread``.  The single stale-monitor thread does not:
    Python starts it with an empty Context, so a forced finalization must not
    re-resolve ``get_hermes_home()`` to the process launch profile.
    """
    launch_home = tmp_path / "launch"
    profile_home = launch_home / "profiles" / "secondary"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    # We drive the production monitor loop ourselves below so the test controls
    # exactly when the record becomes expired.
    monkeypatch.setattr(ad, "_ensure_stale_monitor", lambda: None)
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.01)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 0.0)

    gate = threading.Event()

    def runner():
        gate.wait(timeout=30)
        return {"status": "completed", "summary": "late runner result"}

    token = set_hermes_home_override(profile_home)
    try:
        handle = ad.dispatch_async_delegation(
            goal="profile-owned stalled task",
            context=None,
            toolsets=None,
            role="leaf",
            model="test-model",
            session_key="secondary-session",
            runner=runner,
            max_async_children=1,
            progress_fn=lambda: ("frozen", False),
        )
    finally:
        reset_hermes_home_override(token)

    delegation_id = handle["delegation_id"]
    profile_db = profile_home / "state.db"
    launch_db = launch_home / "state.db"
    assert _delegation_state(profile_db, delegation_id) == "running"

    # Put the live record exactly at the monitor's force-finalize boundary.
    with ad._records_lock:
        record = ad._records[delegation_id]
        record.update(
            status="stalling",
            _started=True,
            _interrupted_at=time.time() - 1,
            _stall_quiet_seconds=1.0,
            _stall_threshold_seconds=0.5,
            _stall_in_tool=False,
        )

    # This is the production ownership shape: one daemon monitor thread serves
    # every profile and therefore starts without the dispatching ContextVars.
    ad._monitor_stop.clear()
    monitor = threading.Thread(target=ad._stale_monitor_loop, daemon=True)
    monitor.start()
    monitor.join(timeout=2)
    assert not monitor.is_alive()

    try:
        assert _delegation_state(profile_db, delegation_id) == "stalled"
        assert _delegation_state(launch_db, delegation_id) is None
    finally:
        # The real runner can return after the forced stall; _finalize must then
        # be a no-op for the already-terminal record.
        gate.set()
