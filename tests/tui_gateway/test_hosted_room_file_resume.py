"""Files input must not prevent a later text result from reaching its group."""

import time

from gateway import hosted_room_driver as driver
from tests.tui_gateway.hosted_room_service_fixtures import _server
from tui_gateway.hosted_room_service import HostedRoomService


def test_text_result_publishes_after_stopping_a_file_input(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "ops")
    room_id = "file-resume"
    service.create_room(
        room_id=room_id,
        name="File review",
        members=[
            {"member_id": "planner", "profile": "default", "handle": "planner"},
            {"member_id": "builder", "profile": "ops", "handle": "builder"},
        ],
    )
    stored = service.attachments.put(
        room_id=room_id, upload_id="spec", kind="file", name="SPEC.md",
        mime="text/markdown", data=b"# Synthetic acceptance criteria\n",
    )
    service.send(
        room_id=room_id, event_id="initial",
        payload={
            "text": "@builder review this", "thread_id": "design",
            "attachments": [{key: stored[key] for key in ("attachment_id", "kind", "name", "mime", "size")}],
        },
    )
    assert driver.list_tasks(db, room_id=room_id, status="queued")[0]["payload"]["attachments"]
    assert service.stop_room(room_id, cancel_id="stop-first") == 1
    service.prepare_room(service.bindings()[0])
    service.send(
        room_id=room_id, event_id="followup",
        payload={"text": "@builder finish your review", "thread_id": "design"},
    )
    task = driver.list_tasks(db, room_id=room_id, status="queued")[0]
    assert "attachments" not in task["payload"]
    binding = service.bindings()[0]
    lease = driver.acquire_lease(
        db, room_id=room_id, gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation="file-review",
        ttl_seconds=300, clock=time.time,
    )
    attempt = driver.start_task(
        db, task["identity"], lease,
        expected_cancel_generation=task["cancel_generation"], clock=time.time,
    )
    driver.settle_task(
        db, attempt, settlement_id="finished", status="settled",
        result={"text": "Review complete."}, clock=time.time,
    )
    restarted = HostedRoomService(_server(), db_path=db)
    restarted.local_profiles = lambda: ("default", "ops")
    restarted.prepare_room(restarted.bindings()[0])
    events = restarted._events(room_id)
    replies = [event for event in events if event["kind"] == "message.member"]
    assert len(replies) == 1
    assert replies[0]["payload"]["text"] == "Review complete."
    restarted.prepare_room(restarted.bindings()[0])
    assert restarted._events(room_id) == events
