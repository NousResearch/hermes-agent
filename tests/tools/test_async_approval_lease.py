"""Background children retain an answerable approval after their parent turn ends."""
import threading
import pytest

from tools import approval, async_delegation as ad
from tools.approval_context import set_current_session_key, reset_current_session_key


@pytest.mark.parametrize("choice", ["once", "deny", "timeout", "cancel", "missing"])
def test_child_approval_after_parent_finished(monkeypatch, choice):
    ad._reset_for_tests()
    ready, ask, notified, done = (threading.Event() for _ in range(4))
    decisions = []
    monkeypatch.setattr(approval.approval_context, "_get_approval_timeout", lambda: 0.2 if choice == "timeout" else 5)
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    key = "agent:test:chat-a"

    def notify(data):
        notified.set()

    def child():
        ready.set()
        assert ask.wait(5)
        decisions.append(approval.check_execute_code_guard("print('safe test')", "local"))
        done.set()
        return {"status": "completed"}

    token = set_current_session_key(key)
    if choice != "missing":
        approval.register_gateway_notify(key, notify)
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (None, False, True, False))
    try:
        result = ad.dispatch_async_delegation(goal="approval", context=None, toolsets=None,
            role="leaf", model=None, session_key=key, runner=child)
        assert result["status"] == "dispatched"
        assert ready.wait(5)
        approval.unregister_gateway_notify(key)
        ask.set()
        if choice != "missing":
            assert notified.wait(2), "child approval disappeared after parent unregistered"
        assert approval.resolve_gateway_approval("agent:test:chat-b", "once") == 0
        if choice in ("once", "deny"):
            assert approval.resolve_gateway_approval(key, choice) == 1
            assert approval.resolve_gateway_approval(key, choice) == 0
        elif choice == "cancel":
            assert ad.interrupt_for_session(session_key="other") == 0
            assert approval.has_blocking_approval(key)
            ad.interrupt_for_session(session_key=key)
        assert done.wait(5)
        assert decisions[0]["approved"] is (choice == "once")
        if choice != "once":
            assert decisions[0]["outcome"] == {"deny": "denied", "timeout": "timeout",
                "cancel": "cancelled", "missing": "notify_failed"}[choice]
        assert not approval.has_blocking_approval(key)
    finally:
        ask.set()
        approval.unregister_gateway_notify(key)
        reset_current_session_key(token)
        ad.interrupt_all()
        done.wait(5)
        ad._reset_for_tests()


def test_cancelled_queued_worker_releases_transport(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace
    from tools.approval_notify_lease import current

    ad._reset_for_tests()
    future = Future()
    monkeypatch.setattr(ad, "_get_executor", lambda size: SimpleNamespace(submit=lambda fn: future))
    key = "queued-cancel"
    approval.register_gateway_notify(key, lambda data: None)
    try:
        result = ad.dispatch_async_delegation(goal="queued", context=None, toolsets=None,
            role="leaf", model=None, session_key=key, runner=lambda: {})
        lease = ad._records[result["delegation_id"]]["_approval_lease"]
        assert lease.active
        assert future.cancel()
        assert not lease.active
        assert lease.callback is None
        assert current(key) is None
        assert ad._records[result["delegation_id"]]["status"] == "interrupted"
        lease.release()  # Finalization, interruption and Future completion can all release it.
        assert not lease.active
    finally:
        approval.unregister_gateway_notify(key)
        ad.interrupt_all()
        ad._reset_for_tests()


def test_concurrent_children_have_independent_single_use_consent(monkeypatch):
    from tools.approval_notify_lease import acquire, run
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (None, False, True, False))
    monkeypatch.setattr(approval.approval_context, "_get_approval_timeout", lambda: 5)
    key = "concurrent-children"
    published = threading.Semaphore(0)
    approval.register_gateway_notify(key, lambda data: published.release())
    leases = [acquire(key), acquire(key)]
    approval.unregister_gateway_notify(key)
    results, executions = [], []

    def worker(lease):
        token = set_current_session_key(key)
        try:
            def operation():
                decision = approval.check_execute_code_guard("print(1)", "local")
                results.append(decision)
                if decision["approved"]:
                    executions.append(1)
            run(lease, operation)
        finally:
            reset_current_session_key(token)

    threads = [threading.Thread(target=worker, args=(lease,)) for lease in leases]
    try:
        for thread in threads:
            thread.start()
        assert published.acquire(timeout=5)
        assert published.acquire(timeout=5)
        assert len(approval.list_gateway_approvals(key)) == 2
        approval.unregister_gateway_notify(key)
        assert len(approval.list_gateway_approvals(key)) == 2
        assert approval.resolve_gateway_approval(key, "once") == 1
        assert approval.resolve_gateway_approval(key, "deny") == 1
    finally:
        for lease in leases:
            lease.release()
        for thread in threads:
            thread.join(5)
    assert len(results) == 2
    assert executions == [1]
    assert not approval.has_blocking_approval(key)


def test_cleared_session_revokes_child_transport():
    from tools.approval_notify_lease import acquire

    key = "session-clear-child"
    delivered = []
    approval.register_gateway_notify(key, delivered.append)
    lease = acquire(key)
    try:
        approval.clear_session(key)
        with pytest.raises(RuntimeError, match="unavailable"):
            lease.notify({"command": "print(1)"})
        assert delivered == []
        assert not lease.active
        assert lease.callback is None
    finally:
        lease.release()
        approval.unregister_gateway_notify(key)
