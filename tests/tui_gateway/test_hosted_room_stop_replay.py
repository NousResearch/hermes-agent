"""An old Stop must never cancel a user message admitted after its durable fence."""

from types import SimpleNamespace
import threading

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from persistence_assembly import open_persistence, profile_scope
from tui_gateway.hosted_room_service import HostedRoomService


def _service(db):
    service = HostedRoomService(
        SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock()), store=db, placement="profile")
    service.local_profiles = lambda: ("default", "ops")
    return service


@pytest.fixture
def db(tmp_path):
    """The store handle on the test home's state.db, closed at teardown."""
    with open_persistence(profile_scope(home=tmp_path)) as handle:
        yield handle


@pytest.mark.parametrize("restart", [False, True])
def test_replayed_stop_leaves_a_new_message_queued(db, restart):
    service = _service(db)
    service.create_room(room_id="room", name="Stop replay", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"}])
    service.send(room_id="room", event_id="first", payload={"text": "@hermes First", "thread_id": "first"})
    assert service.stop_room("room", cancel_id="old-stop") == 1
    service.send(room_id="room", event_id="next", payload={"text": "@hermes Next", "thread_id": "next"})
    queued = driver.list_tasks(db, placement="profile", room_id="room", status="queued")
    assert len(queued) == 1
    if restart:
        service = _service(db)
    assert service.stop_room("room", cancel_id="old-stop") == 0
    assert driver.list_tasks(db, placement="profile", room_id="room", status="queued")[0]["identity"] == queued[0]["identity"]
    assert service.stop_room("room", cancel_id="new-stop") == 1
    assert driver.list_tasks(db, placement="profile", room_id="room", status="queued") == []


def test_message_admitted_between_stop_fence_and_task_scan_survives(db, monkeypatch):
    service = _service(db)
    service.create_room(room_id="room", name="Concurrent Stop", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"}])
    append_stop = hosted_rooms.request_room_stop

    def append_then_send(*args, **kwargs):
        fence = append_stop(*args, **kwargs)
        service.send(room_id="room", event_id="next", payload={"text": "@hermes Next", "thread_id": "next"})
        return fence

    monkeypatch.setattr(hosted_rooms, "request_room_stop", append_then_send)
    assert service.stop_room("room", cancel_id="stop") == 0
    assert len(driver.list_tasks(db, placement="profile", room_id="room", status="queued")) == 1
