"""Runtime tests for the hosted-room session adapter: stopping the runtime."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gateway import hosted_room_driver as state
from tests.tui_gateway import test_hosted_room_driver_runtime as runtime_fixtures
from tests.tui_gateway.test_hosted_room_driver_runtime import FakeSessionRPC, _admit, _identity, _runtime
from persistence_assembly import open_persistence, profile_scope

db = runtime_fixtures.db


@pytest.fixture(name="store")
def store_fixture(tmp_path):
    """A store handle on the test home, closed at teardown."""
    handle = open_persistence(profile_scope(home=tmp_path))
    yield handle
    handle.close()


def test_stop_is_bounded_and_does_not_interrupt_active_turn(db: Path, store):
    identity = _identity()
    _admit(store, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(store, rpc, poll_interval_seconds=0.01)

    runtime.start()
    assert rpc.submitted.wait(5.0)
    started = time.monotonic()
    stopped = runtime.stop(timeout=5.0)

    assert stopped is True
    # Bounded: returns well before its own timeout and without waiting for the active turn (which never completes).
    assert time.monotonic() - started < 2.0
    assert state.get_task(store, identity, placement="profile")["status"] == "running"
    assert not [call for call in rpc.calls if call[0] == "interrupt"]
