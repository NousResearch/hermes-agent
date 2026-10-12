"""Cancellation requires proof across admission, lease expiry, and deferred executions."""

import threading
import time

import pytest

from gateway import hosted_room_driver as state
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, ROOM_ID, FakeSessionRPC, _admit, _identity, _runtime, _wait_for,
)
from tests.tui_gateway.test_hosted_room_driver_runtime import db
from tui_gateway.hosted_room_driver import HostedRoomRuntime

__all__ = ["db"]  # imported pytest fixture


class _HookedResolveRPC(FakeSessionRPC):
    """Run one callback inside the slow pre-submit session resolution."""

    def __init__(self, on_resolve, **kwargs):
        super().__init__(**kwargs)
        self.on_resolve = on_resolve
        self.resolved = threading.Event()

    def resolve_exact(self, **kwargs):
        session = super().resolve_exact(**kwargs)
        if not self.resolved.is_set():
            self.resolved.set()
            self.on_resolve()
        return session


def _lease_for(db, runtime, *, ttl_seconds=30.0, clock=None):
    lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation=runtime.process_generation, ttl_seconds=ttl_seconds, clock=clock or time.time)
    runtime._leases[ROOM_ID] = lease
    return lease


def _real_lock_runtime(db, tmp_path, rpc, **kwargs):
    from tools.bot_relay import acquire_turn_lock
    return HostedRoomRuntime(
        db_path=db, rooms=[BINDING], rpc=rpc,
        turn_lock=lambda profile: acquire_turn_lock(tmp_path / "locks", profile, timeout_seconds=2.0),
        **kwargs)


def _active_admitted_attempt(db, runtime, identity, rpc):
    now = time.time()
    lease = _lease_for(db, runtime, ttl_seconds=0.05, clock=lambda: now)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=lambda: now)
    state.fence_task_admission(db, attempt, clock=lambda: now)
    session_id = rpc.add_session(active=True, task_id=identity.task_id)
    rpc.states[session_id]["execution_generation"] = attempt.execution_generation
    return lease, attempt


def test_stop_is_bounded_and_does_not_interrupt_active_turn(db):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc, poll_interval_seconds=0.01)
    runtime.start()
    assert rpc.submitted.wait(5.0)
    started = time.monotonic()
    assert runtime.stop(timeout=5.0) is True
    assert time.monotonic() - started < 2.0
    assert state.get_task(db, identity)["status"] == "running"
    assert not [call for call in rpc.calls if call[0] == "interrupt"]


def test_stop_during_session_resolution_prevents_the_submit_entirely(db):
    identity = _identity()
    _admit(db, identity)
    holder = {}
    rpc = _HookedResolveRPC(
        lambda: holder["runtime"].cancel(identity, cancel_id="stop-in-resolution"), auto_complete=False)
    runtime = _runtime(db, rpc)
    holder["runtime"] = runtime
    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "cancelled")
    # Reach the admission fence before shutdown, so shutdown cannot cause the abort.
    _wait_for(lambda: runtime.status()["last_error"] is not None)
    assert runtime.stop(timeout=5.0)
    task = state.get_task(db, identity)
    assert task["status"] == "cancelled"
    assert task["admitted_at"] is None
    assert [method for method, _ in rpc.calls if method == "submit"] == []
    assert "stopped before admission" in runtime.status()["last_error"]


def test_stop_between_admission_and_submit_stays_provisional_then_interrupts(db):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc)
    original = state.fence_task_admission
    stops = []

    def stop_inside_the_window(db_path, attempt, *, clock, owner_descriptor=None):
        admitted = original(db_path, attempt, clock=clock, owner_descriptor=owner_descriptor)
        state.fence_task_admission = original
        stops.append(runtime.cancel(identity, cancel_id="stop-after-admission"))
        return admitted

    state.fence_task_admission = stop_inside_the_window
    try:
        runtime.start()
        assert rpc.submitted.wait(2.0)
        _wait_for(lambda: state.get_task(db, identity)["status"] == "cancelled")
    finally:
        state.fence_task_admission = original
        assert runtime.stop(timeout=5.0)
    assert [stop["status"] for stop in stops] == ["stopping"]
    assert [method for method, _ in rpc.calls if method == "submit"]
    assert [method for method, _ in rpc.calls if method == "interrupt"]
    assert state.get_task(db, identity)["status"] == "cancelled"


def test_a_paused_owner_past_its_lease_is_never_reported_as_cancelled(db):
    """Lease expiry cannot prove an admitted, paused execution has stopped."""
    identity = _identity()
    _admit(db, identity)
    owner_rpc = FakeSessionRPC(auto_complete=False)
    now = [time.time()]
    owner = _runtime(db, owner_rpc, lease_ttl_seconds=0.05, clock=lambda: now[0])
    observer = _runtime(db, FakeSessionRPC(auto_complete=False), lease_ttl_seconds=30.0, clock=lambda: now[0])
    original = state.fence_task_admission
    admitted, release_owner = threading.Event(), threading.Event()
    observed = {}

    def pause_after_admission(db_path, attempt, *, clock, owner_descriptor=None):
        result = original(db_path, attempt, clock=clock, owner_descriptor=owner_descriptor)
        state.fence_task_admission = original
        admitted.set()
        assert release_owner.wait(5.0)
        return result

    state.fence_task_admission = pause_after_admission
    try:
        owner.start()
        assert admitted.wait(2.0)
        lease = state.get_task(db, identity)
        assert lease["admitted_at"] is not None and lease["status"] == "running"
        now[0] = owner._leases[ROOM_ID].expires_at + 1
        observed["cancel"] = observer.cancel(identity, cancel_id="stop-from-successor")
        observed["after_cancel"] = state.get_task(db, identity)["status"]
    finally:
        state.fence_task_admission = original
        release_owner.set()
        assert owner.stop(timeout=5.0)
    assert observed["cancel"]["status"] != "cancelled"
    assert observed["after_cancel"] == "indeterminate"
    assert [method for method, _ in owner_rpc.calls if method == "submit"]
    assert state.get_task(db, identity)["status"] == "indeterminate"
    observer.rpc = owner_rpc
    resolved = observer.cancel(identity, cancel_id="stop-observed-running")
    assert [method for method, _ in owner_rpc.calls if method == "interrupt"]
    assert resolved["status"] == "cancelled"


def test_a_stop_this_process_owns_is_still_acknowledged_from_an_absent_session(db):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(db, rpc, lease_ttl_seconds=30.0)
    lease = _lease_for(db, runtime)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=time.time)
    state.fence_task_admission(db, attempt, clock=time.time)
    result = runtime.cancel(identity, cancel_id="stop-local-absent-session")
    assert result["status"] == "cancelled"
    assert state.get_task(db, identity)["status"] == "cancelled"


def test_deferred_retry_fails_closed_when_the_exact_attempt_cannot_be_probed(db):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    now = [time.time()]
    clock = lambda: now[0]
    runtime = _runtime(db, rpc, lease_ttl_seconds=30.0, clock=clock)
    crashed = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation="crashed-process", ttl_seconds=0.05, clock=clock)
    attempt = state.start_task(db, identity, crashed, expected_cancel_generation=0, clock=clock)
    state.fence_task_admission(db, attempt, clock=clock)
    now[0] = crashed.expires_at + 1
    lease = _lease_for(db, runtime, clock=clock)
    state.recover_room(db, lease, clock=clock)
    state.defer_indeterminate_task(
        db, identity, lease, expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation, reason="member_unavailable", clock=clock)
    assert state.get_task(db, identity)["status"] == "deferred"

    def unreachable(**_kwargs):
        raise RuntimeError("member is unreachable")

    rpc.resolve_exact = unreachable
    with pytest.raises(state.InvalidTaskTransitionError, match="cannot confirm"):
        runtime.retry_indeterminate(identity)
    assert state.get_task(db, identity)["status"] == "deferred"
    assert [method for method, _ in rpc.calls if method == "submit"] == []


def test_a_live_execution_is_never_deferred_once_its_deadline_passes(db, tmp_path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    owner = _runtime(db, rpc, lease_ttl_seconds=0.05)
    owner_lease, _attempt = _active_admitted_attempt(db, owner, identity, rpc)
    clock = [owner_lease.expires_at + 1.0]
    successor = _real_lock_runtime(
        db, tmp_path, rpc, clock=lambda: clock[0], lease_ttl_seconds=100.0, indeterminate_defer_seconds=5.0)
    lease = _lease_for(db, successor, ttl_seconds=100.0, clock=lambda: clock[0])
    state.recover_room(db, lease, clock=lambda: clock[0])
    assert state.get_task(db, identity)["status"] == "indeterminate"
    assert successor._reconcile_indeterminate(BINDING, lease) is True
    clock[0] += 6.0
    assert successor._reconcile_indeterminate(BINDING, lease) is True
    assert state.get_task(db, identity)["status"] == "indeterminate"
    assert successor.status()["blocked_rooms"] == (ROOM_ID,)
    for session in rpc.states.values():
        session["active"] = False
    assert successor._reconcile_indeterminate(BINDING, lease) is False
    assert state.get_task(db, identity)["status"] == "deferred"


def test_cancelling_an_admitted_deferred_attempt_interrupts_before_acknowledging(db, tmp_path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    owner = _runtime(db, rpc, lease_ttl_seconds=0.05)
    owner_lease, attempt = _active_admitted_attempt(db, owner, identity, rpc)
    clock = [owner_lease.expires_at + 1.0]
    successor = _real_lock_runtime(db, tmp_path, rpc, clock=lambda: clock[0], lease_ttl_seconds=100.0)
    lease = _lease_for(db, successor, ttl_seconds=100.0, clock=lambda: clock[0])
    state.recover_room(db, lease, clock=lambda: clock[0])
    state.defer_indeterminate_task(
        db, identity, lease, expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation, reason="member_unavailable", clock=lambda: clock[0])
    assert state.get_task(db, identity)["status"] == "deferred"
    result = successor.cancel(identity, cancel_id="stop-a-deferred-attempt")
    assert [method for method, _ in rpc.calls if method == "interrupt"]
    assert not any(session["active"] for session in rpc.states.values())
    assert result["status"] == "cancelled"


def test_cancelling_an_unadmitted_deferred_attempt_still_stops_it_outright(db, tmp_path):
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    clock = [time.time()]
    runtime = _real_lock_runtime(db, tmp_path, rpc, clock=lambda: clock[0], lease_ttl_seconds=100.0)
    crashed = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation="crashed-process", ttl_seconds=0.05, clock=lambda: clock[0])
    attempt = state.start_task(db, identity, crashed, expected_cancel_generation=0, clock=lambda: clock[0])
    clock[0] = crashed.expires_at + 1.0
    lease = _lease_for(db, runtime, ttl_seconds=100.0, clock=lambda: clock[0])
    state.recover_room(db, lease, clock=lambda: clock[0])
    state.defer_indeterminate_task(
        db, identity, lease, expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation, reason="member_unavailable", clock=lambda: clock[0])
    task = state.get_task(db, identity)
    assert (task["status"], task["admitted_at"]) == ("deferred", None)
    result = runtime.cancel(identity, cancel_id="stop-an-unavailable-member")
    assert result["status"] == "cancelled"
    assert [method for method, _ in rpc.calls if method == "interrupt"] == []
