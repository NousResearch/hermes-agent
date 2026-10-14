"""The in-process stop waiter must see a peer cancel before the turn budget."""

import time
from pathlib import Path

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING,
    PROFILE,
    ROOM_ID,
    FakeSessionRPC,
    _admit,
    _identity,
    _peer_resolver,
    _runtime,
)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    hosted_rooms.create_room(
        path,
        room_id=ROOM_ID,
        name="Release room",
        members=[{"profile": PROFILE, "handle": PROFILE}],
        authority_gateway_id=BINDING.gateway_id,
        now=time.time(),
    )
    return path


def test_waiter_reads_a_cancelled_peer_before_the_stop_deadline(db: Path):
    """A peer that already cancelled this generation is not published as stop_deadline."""
    identity = _identity()
    now = [100.0]

    def clock():
        return now[0]

    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    session_id = rpc.add_session(active=True, task_id=identity.task_id)
    runtime = _runtime(db, rpc, clock=clock, lease_ttl_seconds=10_000, turn_timeout_seconds=30)
    lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation=runtime.process_generation, ttl_seconds=10_000, clock=clock)
    runtime._leases[ROOM_ID] = lease
    attempt = state.start_task(
        db, identity, lease, expected_cancel_generation=0, clock=clock)
    state.begin_task_cancel(
        db, identity, cancel_id="cancel-peer-deadline",
        expected_cancel_generation=0, clock=clock)

    class _CancelledPeer:
        def prepare(self, **_kwargs):
            return {"session_id": "peer-session"}

        def status(self, **_kwargs):
            return {
                "active": False, "status": "cancelled", "task_id": identity.task_id,
                "execution_generation": attempt.execution_generation}

        def history(self, **_kwargs):
            return []

    runtime.transport_resolver = _peer_resolver(_CancelledPeer())
    now[0] = 100.0 + runtime.turn_timeout_seconds + 5
    assert runtime._wait_for_terminal(
        BINDING, profile=PROFILE, session_id=session_id, attempt=attempt, transport=rpc,
        deadline_monotonic=time.monotonic() + 5) is None
    task = state.get_task(db, identity)
    assert task["status"] == "cancelled"
    assert (task.get("result") or {}).get("reason") != "stop_deadline"
