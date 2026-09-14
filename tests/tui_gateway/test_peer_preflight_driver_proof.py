"""Only a newly allocated, fenced generation can turn phase evidence into retry."""

import time

import pytest

from gateway import hosted_room_driver as state
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING,
    FakeSessionRPC,
    _admit,
    _identity,
    _runtime,
    db,
)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


class PreflightFailureRPC(FakeSessionRPC):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def submit(self, **_kwargs):
        self.attempts += 1
        failure = PeerRunsHTTPError(
            "refresh refused", status_code=401, error_code="invalid_room_grant"
        )
        failure.dispatch_not_attempted = True
        raise failure


def test_new_generation_can_use_phase_evidence(db):
    identity = _identity()
    _admit(db, identity)
    rpc = PreflightFailureRPC()
    runtime = _runtime(db, rpc)
    runtime._run_cycle()
    task = state.get_task(db, identity)
    assert rpc.attempts == 1
    assert task["status"] == "queued"
    assert task["execution_generation"] == 1


@pytest.mark.parametrize("stale_field", ["status", "generation"])
def test_reused_snapshot_does_not_prove_a_new_generation(db, stale_field):
    identity = _identity()
    _admit(db, identity)
    rpc = PreflightFailureRPC()
    runtime = _runtime(db, rpc)
    snapshot = state.get_task(db, identity)
    lease = runtime._ensure_lease(BINDING)
    attempt = state.start_task(
        db, identity, lease, expected_cancel_generation=0, clock=time.time
    )
    if stale_field == "status":
        snapshot["status"] = "running"
    else:
        snapshot["execution_generation"] = attempt.execution_generation
    runtime._execute_attempt(BINDING, snapshot, attempt)
    task = state.get_task(db, identity)
    assert rpc.attempts == 1
    assert task["status"] == "running"
    assert task["execution_generation"] == 1
    assert task["run_gateway_id"] == BINDING.gateway_id
