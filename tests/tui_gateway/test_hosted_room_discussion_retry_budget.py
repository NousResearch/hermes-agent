"""A retried deferred turn must not stall a discussion configured to its full reply budget."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from gateway import hosted_room_driver as driver
from tui_gateway.hosted_room_service import HostedRoomService


PROFILES = ("research", "build", "review", "ops", "qa", "docs")
ROOM_ID = "retry-budget-room"


def _service(root: Path) -> HostedRoomService:
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=root / "state.db")
    service.local_profiles = lambda: PROFILES
    return service


def test_retried_deferred_turns_do_not_stall_a_full_six_round_discussion(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "hosted_rooms:\n  discussion:\n    max_rounds: 6\n    max_messages: 40\n")
    service = _service(tmp_path)
    service.create_room(
        room_id=ROOM_ID, name="Retry budget",
        members=[{"member_id": p, "profile": p, "handle": p} for p in PROFILES])
    service.send(room_id=ROOM_ID, event_id="kickoff",
                 payload={"text": "Discuss.", "thread_id": "thread-1"})
    binding = service.bindings()[0]
    now = [time.time()]
    clock = lambda: now[0]
    service.runtime.clock = clock

    # The first turn's attempt is abandoned, fenced as deferred, then retried by the user while
    # the rest of the discussion runs: 1 kickoff + 36 replies + 36 settles + 1 deferral = 74.
    first = driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued")[0]
    old = driver.acquire_lease(
        service.db_path, room_id=ROOM_ID, gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation="old", ttl_seconds=1, clock=clock)
    driver.start_task(service.db_path, first["identity"], old, expected_cancel_generation=0, clock=clock)
    now[0] += 2
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(service.db_path, lease, clock=clock)
    driver.defer_indeterminate_task(
        service.db_path, first["identity"], lease, expected_execution_generation=1,
        expected_cancel_generation=0, reason="member_unavailable", clock=clock)
    service.prepare_room(binding)  # publishes turn.deferred and admits the next member
    service.retry_room_task(ROOM_ID, task_id=first["identity"].task_id)

    for index in range(80):
        queued = driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued")
        if not queued:
            break
        attempt = driver.start_task(
            service.db_path, queued[0]["identity"], lease, expected_cancel_generation=0, clock=clock)
        driver.settle_task(service.db_path, attempt, settlement_id=f"reply-{index}", status="settled",
                           result={"text": "@everyone next"}, clock=clock)
        service.prepare_room(binding)

    events = service._events(ROOM_ID)
    assert sum(event["kind"] == "turn.deferred" for event in events) == 1
    assert sum(event["kind"] == "message.member" for event in events) == 36
    assert events[-1]["kind"] == "room.activity"
    assert events[-1]["payload"]["reason_code"] == "max_rounds"
