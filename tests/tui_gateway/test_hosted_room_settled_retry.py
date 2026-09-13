"""Deferred results survive discussion and bounded-transcript cleanup."""

import threading
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.mark.parametrize("later_messages", [0, 25])
def test_settled_discussion_retry_publishes_once(tmp_path, monkeypatch, later_messages):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".hermes" / "profiles" / "reviewer").mkdir(parents=True)
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=tmp_path / ".hermes" / "state.db")
    service.create_room(room_id="room", name="Room", members=[
        {"member_id": "worker", "profile": "default", "handle": "worker"},
        {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
    ])
    service.send(room_id="room", event_id="source", payload={"text": "Review", "thread_id": "thread"})
    binding = service.bindings()[0]
    task = driver.list_tasks(service.db_path, room_id="room", status="queued")[0]
    now = [time.time()]
    clock = lambda: now[0]
    service.runtime.clock = clock
    old = driver.acquire_lease(
        service.db_path, room_id="room", gateway_id=binding.gateway_id, authority_epoch=1,
        process_generation="old", ttl_seconds=1, clock=clock)
    driver.start_task(service.db_path, task["identity"], old, expected_cancel_generation=0, clock=clock)
    now[0] += 2
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(service.db_path, lease, clock=clock)
    driver.defer_indeterminate_task(
        service.db_path, task["identity"], lease, expected_execution_generation=1,
        expected_cancel_generation=0, reason="member_unavailable", clock=clock)
    service.prepare_room(binding)
    reviewer = driver.list_tasks(service.db_path, room_id="room", status="queued")[0]
    attempt = driver.start_task(
        service.db_path, reviewer["identity"], lease, expected_cancel_generation=0, clock=clock)
    driver.settle_task(service.db_path, attempt, settlement_id="pass", status="settled",
                       result={"text": "PASS"}, clock=clock)
    service.prepare_room(binding)
    assert [e["kind"] for e in service._events("room")] == [
        "message.user", "turn.deferred", "turn.settled", "room.activity"]
    for index in range(later_messages):
        hosted_rooms.append_event(
            service.db_path, room_id="room", event_id=f"later-{index}", kind="message.user",
            actor={"kind": "user", "id": "test"},
            payload={"text": "Later", "thread_id": "thread"},
            authority_gateway_id=binding.gateway_id, authority_epoch=1)
    service._policy_snapshot(hosted_rooms.room_state(service.db_path, room_id="room"))
    assert service.retry_room_task("room", task_id=task["identity"].task_id)["status"] == "queued"
    attempt = driver.start_task(
        service.db_path, task["identity"], lease, expected_cancel_generation=0, clock=clock)
    driver.settle_task(service.db_path, attempt, settlement_id="retry", status="settled",
                       result={"text": "Recovered answer"}, clock=clock)
    service.prepare_room(binding)
    restarted = HostedRoomService(server, db_path=service.db_path)
    for _ in range(2):
        restarted.prepare_room(binding)
    events = restarted._events("room")
    assert sum(e["kind"] == "message.member" and e["payload"].get("text") == "Recovered answer"
               for e in events) == (0 if later_messages else 1)
    terminal = [e for e in events if e["kind"] in {"turn.settled", "turn.cancelled"}
                and e["payload"].get("task_id") == task["identity"].task_id]
    assert len(terminal) == 1
    if later_messages:
        assert terminal[0]["payload"]["reason"] == "superseded_by_newer_user_event"
    assert restarted.policy_checkpoint.publication_exists(
        room_id="room", task_id=task["identity"].task_id, status="settled", execution_generation=2)
    for event_id in ("next", "after-stop"):
        assert restarted.send(room_id="room", event_id=event_id,
                              payload={"text": "Continue", "thread_id": "thread"})["event_id"] == event_id
        restarted.stop_room("room", cancel_id=f"stop-{event_id}")
