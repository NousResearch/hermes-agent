"""A stale session admission (its driver task owns no live attempt) must not wedge stops."""
import time
from pathlib import Path
from typing import Any

from gateway import hosted_room_driver as state
from tests.tui_gateway import test_hosted_room_driver_runtime as runtime_fixtures
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, PROFILE, ROOM_ID, FakeSessionRPC, _admit, _identity, _runtime,
)

db = runtime_fixtures.db
store = runtime_fixtures.store_fixture


class _QueuedBehindStaleTurnRPC(FakeSessionRPC):
    """A stale turn keeps holding the session; newer submits queue behind it unseen."""

    def submit(self, **kwargs):
        self.calls.append(("submit", dict(kwargs)))
        self.submitted.set()
        return {"accepted": True}


def test_deadline_stop_interrupts_a_stale_deferred_admission_and_finishes(db: Path, store):
    """A deferred task's queued admission keeps the session active under its own identity
    forever (#144729), so an exact stop for a newer task retried without end."""
    now = [time.time()]

    def clock():
        return now[0]

    stale_identity = _identity("task-stale")
    _admit(store, stale_identity)
    setup_lease = state.acquire_lease(
        store, placement="profile", room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation="pre-defer", ttl_seconds=0.2, clock=clock)
    state.start_task(store, stale_identity, setup_lease, placement="profile", expected_cancel_generation=0, clock=clock)
    now[0] += 1.0  # the pre-defer lease expires; the next generation takes the room over
    defer_lease = state.acquire_lease(
        store, placement="profile", room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation="defer-process", ttl_seconds=30, clock=clock)
    state.recover_room(store, defer_lease, placement="profile", clock=clock)  # the abandoned attempt goes indeterminate
    state.defer_indeterminate_task(
        store, stale_identity, defer_lease, placement="profile", expected_execution_generation=1,
        expected_cancel_generation=0, reason="member_unavailable", clock=clock)
    state.release_lease(store, defer_lease, placement="profile", clock=clock)

    live_identity = _identity("task-live")
    _admit(store, live_identity)
    published: list[dict[str, Any]] = []
    rpc = _QueuedBehindStaleTurnRPC(auto_complete=False)
    rpc.add_session(active=True, task_id=stale_identity.task_id)
    runtime = _runtime(
        store, rpc, clock=clock, active_poll_interval_seconds=0.01, turn_timeout_seconds=0.05,
        publish_terminal=lambda _binding, task: published.append(task))

    runtime.start()
    runtime_fixtures._wait_for(lambda: state.get_task(store, live_identity, placement="profile")["status"] == "failed")
    assert runtime.stop(timeout=5.0)

    interrupts = [params for method, params in rpc.calls if method == "interrupt"]
    assert interrupts and all(
        params["expected_task_id"] == stale_identity.task_id for params in interrupts)
    assert state.get_task(store, stale_identity, placement="profile")["status"] == "deferred"
    failed = state.get_task(store, live_identity, placement="profile")
    assert failed["result"]["reason_code"] == "turn_deadline_exceeded"


def test_stale_admission_probe_spares_a_live_attempt_holder(db: Path, store):
    """A session turn whose task still owns a live attempt is left to its own lifecycle."""
    holder_identity = _identity("task-holder")
    _admit(store, holder_identity)
    lease = state.acquire_lease(
        store, placement="profile", room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation="holder-process", ttl_seconds=30, clock=time.time)
    state.start_task(store, holder_identity, lease, placement="profile", expected_cancel_generation=0, clock=time.time)
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(active=True, task_id=holder_identity.task_id)
    runtime = _runtime(store, rpc)

    info = {"active": True, "task_id": holder_identity.task_id}
    assert runtime._interrupt_stale_admission(rpc, BINDING, PROFILE, "session-1", info) is False
    # An identity the driver never admitted (a newer turn that owns the session) stays untouched.
    unknown = {"active": True, "task_id": "task-never-admitted"}
    assert runtime._interrupt_stale_admission(rpc, BINDING, PROFILE, "session-1", unknown) is False
    assert not [params for method, params in rpc.calls if method == "interrupt"]
