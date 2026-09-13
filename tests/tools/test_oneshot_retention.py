"""Live ownership survives terminal-cache pressure (real workers and SQLite)."""
import threading
import os
import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.mark.parametrize("identity_available", [False, True])
def test_recovery_requires_proof_of_owner_loss(monkeypatch, identity_available):
    from gateway import status
    ad._reset_for_tests()
    # Real native PID liveness; only the unavailable identity query is injected.
    record = {"delegation_id": "identity-test", "goal": "offline", "dispatched_at": 1}
    ad._persist_dispatch(record)
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute("UPDATE async_delegations SET owner_started_at=1 WHERE delegation_id=?",
                     (record["delegation_id"],))
    assert status._pid_exists(os.getpid())
    if not identity_available:
        monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)
        assert ad.recover_abandoned_delegations() == 0
    row = ad.get_durable_delegation(record["delegation_id"])
    assert row["state"] == ("unknown" if identity_available else "running")


def test_live_records_survive_retention_and_shutdown_totals(monkeypatch):
    ad._reset_for_tests()
    monkeypatch.setattr(ad, "_MAX_RETAINED_COMPLETED", 1)
    release = threading.Event()
    def runner():
        release.wait(10)
        return {"status": "completed", "summary": "late"}
    ids = []
    try:
        for _ in range(3):
            h = ad.dispatch_async_delegation(goal="live", context=None, toolsets=None,
                role="leaf", model="fixture", session_key="owned", runner=runner,
                interrupt_fn=lambda: None)
            assert h["status"] == "dispatched"
            ids.append(h["delegation_id"])
        # Both transitions still own an eventual terminal write, not cache history.
        with ad._records_lock:
            ad._records[ids[0]]["status"] = "stalling"
            ad._records[ids[1]]["status"] = "finalizing"
            ad._prune_completed_locked()
        assert ad.active_count() == len(ids)
        assert ad.has_live_for_session(session_key="owned")
        with ad._records_lock:
            ad._records[ids[1]]["status"] = "running"
        counts = ad.finalize_for_oneshot_shutdown(grace_seconds=0)
        assert counts == {"signaled": 3, "interrupted": 0, "unknown": 3, "pending": 0}
        assert [ad.get_durable_delegation(i)["state"] for i in ids] == ["unknown"] * 3
        events = [process_registry.completion_queue.get(timeout=3) for _ in ids]
        assert {e["delegation_id"] for e in events} == set(ids)
        assert all(e["status"] == "unknown" for e in events)
    finally:
        release.set()
        ad._executor.shutdown(wait=True)
        ad._reset_for_tests()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
