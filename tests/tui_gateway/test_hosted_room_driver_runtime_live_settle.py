"""The runtime's live terminal settle past plain expiry of its own room lease (hosted-room driver).

Topical sibling of ``test_hosted_room_driver_runtime.py``, which stays inside its file-line cap.
"""

from __future__ import annotations

from pathlib import Path

from gateway import hosted_room_driver as state
from tests.tui_gateway import test_hosted_room_driver_runtime as runtime_fixtures
from tests.tui_gateway.test_hosted_room_driver_runtime import BINDING, ROOM_ID, FakeSessionRPC, _admit, _identity, _runtime

db = runtime_fixtures.db
store = runtime_fixtures.store_fixture


def _started_attempt(store, runtime):
    identity = _identity()
    _admit(store, identity)
    lease = state.acquire_lease(
        store, placement="profile", room_id=ROOM_ID, gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch, process_generation=runtime.process_generation,
        ttl_seconds=runtime.lease_ttl_seconds, clock=runtime.clock)
    return state.start_task(
        store, identity, lease, placement="profile", expected_cancel_generation=0, clock=runtime.clock)


def test_live_terminal_settle_commits_after_plain_expiry_of_its_own_lease(db: Path, store):
    now = [1_000.0]
    runtime = _runtime(store, FakeSessionRPC(), clock=lambda: now[0])
    attempt = _started_attempt(store, runtime)

    now[0] = attempt.lease.expires_at  # our own lease lapsed, unreleased and not reclaimed
    runtime._on_terminal(BINDING, attempt, {"status": "settled", "text": "Done."})

    task = state.get_task(store, attempt.identity, placement="profile")
    assert task["status"] == "settled"
    assert task["result"]["text"] == "Done."


def test_live_terminal_settle_is_discarded_once_another_process_reclaims_the_lease(db: Path, store):
    now = [1_000.0]
    runtime = _runtime(store, FakeSessionRPC(), clock=lambda: now[0])
    attempt = _started_attempt(store, runtime)

    now[0] = attempt.lease.expires_at
    successor = state.acquire_lease(
        store, placement="profile", room_id=ROOM_ID, gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch, process_generation="process-b", ttl_seconds=60,
        clock=runtime.clock)
    state.recover_room(store, successor, placement="profile", clock=runtime.clock)
    runtime._on_terminal(BINDING, attempt, {"status": "settled", "text": "Done."})

    assert state.get_task(store, attempt.identity, placement="profile")["status"] == "indeterminate"
