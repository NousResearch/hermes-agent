"""A stalled preparation can be cancelled without repairing its cause (#149660)."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.session_controls import AuthorityConnection
from tests.tui_gateway.test_hosted_room_service import _server
from tui_gateway.hosted_room_prepare_stall import STALL_ATTEMPTS
from tui_gateway.hosted_room_service import HostedRoomService


def group_calls(service, tmp_path, monkeypatch, room_id="stalled"):
    service.authorize_room = lambda *_args: None
    runtime_status = service.runtime.status
    monkeypatch.setattr(service.runtime, "status", lambda: {**runtime_status(), "running": True})
    authority = SimpleNamespace(profile_id=str(tmp_path), instance_id="home", events={}, sessions={},
                                db=SimpleNamespace(db_path=service.db_path), hosted_room_service=service)
    connection = AuthorityConnection(authority, object(), {"user_id": "reader"})

    def call(method, **params):
        answer = asyncio.run(connection.dispatch({
            "id": method, "method": method, "params": {"room_id": room_id, **params}}))
        assert "result" in answer, answer
        return answer["result"]

    return call


def stalled_service(tmp_path, monkeypatch):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(room_id="stalled", name="Stalled group chat", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"}])
    # A seated profile disappears: the real preparation now fails roster validation.
    service.local_profiles = lambda: ("default",)
    call = group_calls(service, tmp_path, monkeypatch)

    def send(event_id):
        call("groups.send", event_id=event_id, payload={"text": "@ops check this", "thread_id": event_id})
        binding, = service.bindings()
        for _ in range(STALL_ATTEMPTS - 1):
            with pytest.raises(ValueError):
                service._prepare_tracked(binding)

    send("first")
    return service, call, send


def test_stop_clears_stalled_preparation_and_later_send_still_works(tmp_path, monkeypatch):
    service, call, send = stalled_service(tmp_path, monkeypatch)
    before = call("groups.state")["driver_status"]
    assert before["working"] is False
    assert before["pending_actions"][0]["task_id"] == "prepare:stalled"
    assert call("groups.log")["events"][-1]["payload"]["reason_code"] == "prepare_failed"
    with pytest.raises(ValueError):
        service.retry_room_task("stalled", task_id="prepare:stalled")
    assert call("groups.state")["driver_status"]["pending_actions"] == before["pending_actions"]
    assert call("groups.stop", cancel_id="stop-first") == {"cancelled": 0}
    after = call("groups.state")["driver_status"]
    assert after["pending_actions"] == []
    assert after["last_error"] is None
    binding, = service.bindings()
    for _ in range(STALL_ATTEMPTS):
        service._prepare_tracked(binding)
    assert call("groups.state")["driver_status"]["pending_actions"] == []
    # Stop cannot repair the missing profile. A new message may stall again, and an old
    # cancellation must not clear that new message's Retry row.
    send("second")
    assert call("groups.stop", cancel_id="stop-first") == {"cancelled": 0}
    assert call("groups.state")["driver_status"]["pending_actions"]
    assert call("groups.stop", cancel_id="stop-second") == {"cancelled": 0}
    service.local_profiles = lambda: ("default", "ops")
    receipt = call("groups.send", event_id="third", payload={"text": "@ops try again", "thread_id": "third"})
    assert receipt["accepted"] is True
    queued, = driver.list_tasks(service.db_path, room_id="stalled", status="queued")
    assert queued["payload"]["source_event_seq"] == receipt["event"]["seq"]
    assert call("groups.state")["driver_status"]["last_error"] is None


def test_stop_clears_the_stall_when_a_rename_lands_right_after_its_fence(tmp_path, monkeypatch):
    service, call, _send = stalled_service(tmp_path, monkeypatch)
    request_room_stop = hosted_rooms.request_room_stop

    def stop_then_rename(*args, **kwargs):
        fence = request_room_stop(*args, **kwargs)
        hosted_rooms.rename_room(service.db_path, room_id="stalled", event_id="rename-after-stop", name="Renamed")
        return fence

    monkeypatch.setattr(hosted_rooms, "request_room_stop", stop_then_rename)
    assert call("groups.state")["driver_status"]["pending_actions"]
    assert call("groups.stop", cancel_id="stop-first") == {"cancelled": 0}
    assert call("groups.log")["events"][-1]["kind"] == "room.renamed"
    assert call("groups.state")["driver_status"]["pending_actions"] == []
