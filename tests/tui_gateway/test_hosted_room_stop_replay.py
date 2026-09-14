"""A replayed Stop must not cancel work created after its durable fence."""

import time

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_messaging import MessagingRoomBackend
from tui_gateway.hosted_room_service import HostedRoomService

from tests.tui_gateway.hosted_room_service_fixtures import _server


def _room(db):
    return hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "one", "profile": "one", "handle": "one"},
            {"member_id": "two", "profile": "two", "handle": "two"},
        ],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )


def _queue(db, *, task_id, event_id, text):
    room = hosted_rooms.room_state(db, room_id="room-1")
    event = hosted_rooms.append_event(
        db,
        room_id="room-1",
        event_id=event_id,
        kind="message.user",
        actor={"kind": "user", "id": "owner"},
        payload={"text": text, "thread_id": event_id},
        authority_gateway_id=str(room["authority_gateway_id"]),
        authority_epoch=int(room["authority_epoch"]),
    )
    identity = driver.TaskIdentity("room-1", task_id, event_id, f"turn-{task_id}")
    driver.admit_task(
        db,
        identity,
        payload={
            "target_member_id": "one",
            "target_profile": "one",
            "prompt": text,
            "source_event_seq": int(event["seq"]),
        },
        clock=time.time,
    )
    return identity


def test_service_stop_replay_does_not_cancel_later_task(tmp_path):
    db = tmp_path / "state.db"
    _room(db)
    first = _queue(db, task_id="first", event_id="user-1", text="First")
    service = HostedRoomService(_server(), db_path=db)

    assert service.stop_room("room-1", cancel_id="stop-1") == 1
    assert driver.get_task(db, first)["status"] == "cancelled"
    later = _queue(db, task_id="later", event_id="user-2", text="Later")

    assert service.stop_room("room-1", cancel_id="stop-1") == 0
    assert driver.get_task(db, later)["status"] == "queued"


def test_store_only_stop_replay_does_not_cancel_later_task(tmp_path):
    db = tmp_path / "state.db"
    _room(db)
    first = _queue(db, task_id="first", event_id="user-1", text="First")
    backend = MessagingRoomBackend(db_path=db)

    assert backend.stop_room("room-1", cancel_id="stop-1") == 1
    assert driver.get_task(db, first)["status"] == "cancelled"
    later = _queue(db, task_id="later", event_id="user-2", text="Later")

    backend.stop_room("room-1", cancel_id="stop-1")
    assert driver.get_task(db, later)["status"] == "queued"
