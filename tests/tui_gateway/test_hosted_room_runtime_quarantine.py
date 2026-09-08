"""The scheduler must honor a room fence even with cached runtime ownership."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
import time

import pytest

from gateway import hosted_room_driver as state
from tests.gateway.test_hosted_room_driver import FakeClock
from tests.gateway.test_hosted_room_driver_quarantine import quarantine
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, FakeSessionRPC, TerminalPeerClient, _admit, _identity, _peer_resolver, _runtime, db,
)


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


@pytest.mark.parametrize("entry", ["owner", "restart", "waiting", "deadline"])
@pytest.mark.parametrize("lease_state", ["fresh", "renewal_due", "cache_missing"])
def test_exact_stop_survives_quarantine_without_an_execution_lease(db, entry, lease_state):
    identity, clock, rpc = _identity(), FakeClock(), FakeSessionRPC(auto_complete=False)
    _admit(db, identity)
    runtime = _runtime(db, rpc, clock=clock, lease_ttl_seconds=30)
    lease = runtime._ensure_lease(BINDING)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=clock)
    sid = rpc.add_session(active=True, task_id=identity.task_id)
    rpc.states[sid]["execution_generation"] = 1
    if lease_state == "renewal_due":
        clock.advance(16)
    elif lease_state == "cache_missing":
        runtime._leases.clear()
    quarantine(db)
    if entry == "owner":
        runtime.cancel(identity, cancel_id="owner-stop")
    elif entry == "deadline":
        runtime._expire_attempt_deadline(BINDING, state.get_task(db, identity), lease)
    else:
        state.begin_task_cancel(db, identity, cancel_id="owner-stop", expected_cancel_generation=0, clock=clock)
        if entry == "restart":
            runtime._run_room_once(BINDING)
        else:
            runtime._wait_for_terminal(BINDING, profile="ops", session_id=sid, attempt=attempt,
                transport=rpc, deadline_monotonic=time.monotonic() + 1)
    assert state.get_task(db, identity)["status"] == "cancelled"
    interrupts = [params for name, params in rpc.calls if name == "interrupt"]
    assert len(interrupts) == 1
    assert interrupts[0]["expected_task_id"] == identity.task_id
    assert rpc.states[sid]["active"] is False


def test_predispatch_samples_clock_after_waiting_for_writer(db, monkeypatch):
    identity, clock = _identity(), FakeClock()
    _admit(db, identity)
    task = state.get_task(db, identity)
    dispatch_times = []

    class PeerClient(TerminalPeerClient):
        def dispatch(self, **kwargs):
            dispatch_times.append(clock())
            return {"status": "settled", "message_id": "result", "text": "done"}

    runtime = _runtime(db, FakeSessionRPC(), clock=clock, lease_ttl_seconds=30,
        transport_resolver=_peer_resolver(PeerClient(task_id=identity.task_id, execution_generation=1)))
    lease = runtime._ensure_lease(BINDING)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=clock)
    entered, original = threading.Event(), state._transaction

    def observed(path):
        entered.set()
        return original(path)

    monkeypatch.setattr(state, "_transaction", observed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with sqlite3.connect(db) as writer:
            writer.execute("BEGIN IMMEDIATE")
            future = pool.submit(runtime._execute_attempt, BINDING, task, attempt)
            assert entered.wait(timeout=5)
            clock.advance(31)
            writer.rollback()
        future.result(timeout=10)
    assert not dispatch_times
    assert state.get_task(db, identity)["status"] == "running"
