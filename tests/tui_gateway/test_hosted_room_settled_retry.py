"""Deferred results survive discussion and bounded-transcript cleanup."""

import threading
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from tui_gateway.hosted_room_service import HostedRoomService
from persistence_assembly import open_persistence, profile_scope


@pytest.fixture(name="store")
def store_fixture(tmp_path):
    """A store handle on the test home, closed at teardown."""
    handle = open_persistence(profile_scope(home=tmp_path / ".hermes"))
    yield handle
    handle.close()


@pytest.mark.parametrize("later_messages", [0, 25])
def test_settled_discussion_retry_publishes_once(tmp_path, monkeypatch, later_messages, store):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".hermes" / "profiles" / "reviewer").mkdir(parents=True)
    (tmp_path / ".hermes" / "profiles" / "reviewer" / "config.yaml").write_text("{}\n")  # identity marker
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, store=store, placement="profile")
    service.create_room(room_id="room", name="Room", members=[
        {"member_id": "worker", "profile": "default", "handle": "worker"},
        {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
    ])
    service.send(room_id="room", event_id="source", payload={"text": "Review", "thread_id": "thread"})
    binding = service.bindings()[0]
    task = driver.list_tasks(service.store, placement=service.placement, room_id="room", status="queued")[0]
    now = [time.time()]
    clock = lambda: now[0]
    service.runtime.clock = clock
    old = driver.acquire_lease(
        service.store, placement=service.placement, room_id="room", gateway_id=binding.gateway_id, authority_epoch=1,
        process_generation="old", ttl_seconds=1, clock=clock)
    driver.start_task(service.store, task["identity"], old, placement=service.placement, expected_cancel_generation=0, clock=clock)
    now[0] += 2
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(service.store, lease, placement=service.placement, clock=clock)
    driver.defer_indeterminate_task(
        service.store, task["identity"], lease, placement=service.placement, expected_execution_generation=1,
        expected_cancel_generation=0, reason="member_unavailable", clock=clock)
    service.prepare_room(binding)
    reviewer = driver.list_tasks(service.store, placement=service.placement, room_id="room", status="queued")[0]
    attempt = driver.start_task(
        service.store, reviewer["identity"], lease, placement=service.placement, expected_cancel_generation=0, clock=clock)
    driver.settle_task(service.store, attempt, placement=service.placement, settlement_id="pass", status="settled",
                       result={"text": "PASS"}, clock=clock)
    service.prepare_room(binding)
    assert [e["kind"] for e in service._events("room")] == [
        "message.user", "turn.deferred", "turn.settled", "room.activity"]
    for index in range(later_messages):
        hosted_rooms.append_event(
            service.store, placement=service.placement, room_id="room", event_id=f"later-{index}", kind="message.user",
            actor={"kind": "user", "id": "test"},
            payload={"text": "Later", "thread_id": "thread"},
            authority_gateway_id=binding.gateway_id, authority_epoch=1)
    service._policy_snapshot(hosted_rooms.room_state(service.store, placement=service.placement, room_id="room"))
    assert service.retry_room_task("room", task_id=task["identity"].task_id)["status"] == "queued"
    attempt = driver.start_task(
        service.store, task["identity"], lease, placement=service.placement, expected_cancel_generation=0, clock=clock)
    driver.settle_task(service.store, attempt, placement=service.placement, settlement_id="retry", status="settled",
                       result={"text": "Recovered answer"}, clock=clock)
    service.prepare_room(binding)
    restarted = HostedRoomService(server, store=service.store, placement=service.placement)
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
