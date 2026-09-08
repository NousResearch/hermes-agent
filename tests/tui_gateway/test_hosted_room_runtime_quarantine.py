"""The scheduler must honor a room fence even with cached runtime ownership."""

from contextlib import contextmanager

from gateway import hosted_room_driver as state
from tests.gateway.test_hosted_room_driver_quarantine import quarantine
from tests.tui_gateway.test_hosted_room_driver_runtime import BINDING, FakeSessionRPC, _admit, _identity, _runtime, db


def test_cached_room_binding_does_not_dispatch_a_quarantined_room(db):
    identity, rpc = _identity(), FakeSessionRPC()
    _admit(db, identity)
    runtime = _runtime(db, rpc)
    runtime._ensure_lease(BINDING)
    quarantine(db)
    runtime._run_room_once(BINDING)
    assert not rpc.calls
    assert state.get_task(db, identity)["status"] == "queued"


def test_quarantine_while_waiting_for_profile_lock_prevents_submit(db):
    identity, rpc = _identity(), FakeSessionRPC()
    _admit(db, identity)

    @contextmanager
    def fenced_lock(_profile):
        quarantine(db)
        yield

    runtime = _runtime(db, rpc, locks=fenced_lock)
    runtime._run_room_once(BINDING)
    assert not any(name == "submit" for name, _ in rpc.calls)
    assert state.get_task(db, identity)["status"] == "running"


def test_completion_after_quarantine_does_not_publish_or_rewrite_task(db):
    identity, rpc = _identity(), FakeSessionRPC(auto_complete=False)
    _admit(db, identity)
    runtime = _runtime(db, rpc)
    lease = runtime._ensure_lease(BINDING)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=runtime.clock)
    before = state.get_task(db, identity)
    quarantine(db)
    runtime._on_terminal(BINDING, attempt, {"status": "settled", "message_id": "late", "text": "late"})
    assert state.get_task(db, identity) == before
    assert not rpc.calls
