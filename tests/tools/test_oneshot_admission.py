"""Shutdown admission and dispatch ordering with real workers and SQLite."""
import threading

from tools import async_delegation as ad


def dispatch(runner, interrupt_fn=None):
    return ad.dispatch_async_delegation(
        goal="offline", context=None, toolsets=None, role="leaf", model="fixture",
        session_key="owned", runner=runner, interrupt_fn=interrupt_fn)


def test_interrupt_callback_cannot_admit_unretained_work():
    ad._reset_for_tests()
    release = threading.Event()
    called = threading.Event()
    added = []
    def runner():
        release.wait(5)
        return {"status": "completed"}
    def interrupt():
        added.append(dispatch(runner))
        called.set()
    try:
        original = dispatch(runner, interrupt)
        counts = ad.finalize_for_oneshot_shutdown(grace_seconds=0.02)
        assert called.wait(2)
        repeat = ad.finalize_for_oneshot_shutdown(grace_seconds=0)
        if added[0]["status"] == "dispatched":
            new_id = added[0]["delegation_id"]
            print({"during_shutdown": counts, "repeat": repeat,
                   "new_state": ad.get_durable_delegation(new_id)["state"],
                   "new_in_targets": any(r["delegation_id"] == new_id for r in ad._oneshot_state["targets"])})
        assert added[0]["status"] == "rejected"
        assert "shutdown" in added[0]["error"].lower()
        assert counts == repeat == {"signaled": 1, "interrupted": 0, "unknown": 1, "pending": 0}
        assert set(ad._records) == {original["delegation_id"]}
        with ad._transaction() as conn:
            assert conn.execute("SELECT delegation_id FROM async_delegations").fetchall() == [(original["delegation_id"],)]
    finally:
        release.set()
        if ad._executor:
            ad._executor.shutdown(wait=True)
        ad._reset_for_tests()


def test_shutdown_retains_dispatch_blocked_before_persistence(monkeypatch):
    ad._reset_for_tests()
    entered = threading.Barrier(2)
    persist_release = threading.Event()
    runner_release = threading.Event()
    started = threading.Event()
    handles = []
    persist = ad._persist_dispatch
    def blocked_persist(record):
        entered.wait(timeout=3)
        assert persist_release.wait(3)
        persist(record)
    monkeypatch.setattr(ad, "_persist_dispatch", blocked_persist)
    def runner():
        started.set()
        runner_release.wait(3)
        return {"status": "completed"}
    worker = threading.Thread(target=lambda: handles.append(dispatch(runner)), daemon=True)
    try:
        worker.start()
        entered.wait(timeout=3)
        counts = ad.finalize_for_oneshot_shutdown(grace_seconds=0)
        assert counts["pending"] == 1
        assert counts["unknown"] == 0
        assert not started.is_set()
        rejected = dispatch(runner)
        assert rejected["status"] == "rejected"
        assert len(ad._records) == 1
        persist_release.set()
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert handles[0]["status"] == "dispatched"
        counts = ad.finalize_for_oneshot_shutdown(grace_seconds=1)
        assert counts["unknown"] == 1 and counts["pending"] == 0
        assert ad.get_durable_delegation(handles[0]["delegation_id"])["state"] == "unknown"
    finally:
        persist_release.set()
        runner_release.set()
        worker.join(timeout=3)
        if ad._executor:
            ad._executor.shutdown(wait=True)
        ad._reset_for_tests()
