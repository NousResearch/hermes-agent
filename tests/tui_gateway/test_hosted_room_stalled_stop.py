"""A stalled preparation can be cancelled without repairing its cause (#149660)."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.session_controls import AuthorityConnection
from tests.tui_gateway.test_hosted_room_service import _server
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tui_gateway.hosted_room_prepare_stall import STALL_ATTEMPTS
from tui_gateway.hosted_room_service import HostedRoomService


def group_calls(service, tmp_path, monkeypatch, room_id="stalled"):
    service.authorize_room = lambda *_args: None
    runtime_status = service.runtime.status
    monkeypatch.setattr(
        service.runtime, "status", lambda: {**runtime_status(), "running": True}
    )
    authority = SimpleNamespace(
        profile_id=str(tmp_path),
        instance_id="home",
        events={},
        sessions={},
        persistence=service.store,
        hosted_room_service=service,
    )
    connection = AuthorityConnection(authority, object(), {"user_id": "reader"})

    def call(method, *, expect_error=False, **params):
        answer = asyncio.run(
            connection.dispatch(
                {
                    "id": method,
                    "method": method,
                    "params": {"room_id": room_id, **params},
                }
            )
        )
        if expect_error:
            return answer["error"]
        assert "result" in answer, answer
        return answer["result"]

    return call


def stalled_service(tmp_path, monkeypatch):
    service = HostedRoomService(
        _server(), store=storage_handle(tmp_path), placement="profile"
    )
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="stalled",
        name="Stalled group chat",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    # A seated profile disappears: the real preparation now fails roster validation.
    service.local_profiles = lambda: ("default",)
    call = group_calls(service, tmp_path, monkeypatch)

    def send(event_id):
        call(
            "groups.send",
            event_id=event_id,
            payload={"text": "@ops check this", "thread_id": event_id},
        )
        (binding,) = service.bindings()
        for _ in range(STALL_ATTEMPTS - 1):
            with pytest.raises(ValueError):
                service._prepare_tracked(binding)

    send("first")
    return service, call, send


def test_stop_clears_stalled_preparation_and_later_send_still_works(
    tmp_path, monkeypatch
):
    service, call, send = stalled_service(tmp_path, monkeypatch)
    before = call("groups.state")["driver_status"]
    assert before["working"] is False
    assert before["pending_actions"][0]["task_id"] == "prepare:stalled"
    assert (
        call("groups.log")["events"][-1]["payload"]["reason_code"] == "prepare_failed"
    )
    with pytest.raises(ValueError):
        service.retry_room_task("stalled", task_id="prepare:stalled")
    assert (
        call("groups.state")["driver_status"]["pending_actions"]
        == before["pending_actions"]
    )
    assert call("groups.stop", cancel_id="stop-first") == {"cancelled": 0}
    after = call("groups.state")["driver_status"]
    assert after["pending_actions"] == []
    assert after["last_error"] is None
    (binding,) = service.bindings()
    for _ in range(STALL_ATTEMPTS):
        service._prepare_tracked(binding)
    assert call("groups.state")["driver_status"]["pending_actions"] == []
    # Stop cannot repair the missing profile. A new message may stall again, and an old
    # cancellation must not clear that new message's Retry row.
    hosted_rooms.rename_room(
        service.store,
        placement="profile",
        room_id="stalled",
        event_id="rename",
        name="Renamed",
    )
    send("second")
    assert call("groups.stop", cancel_id="stop-first") == {"cancelled": 0}
    assert call("groups.state")["driver_status"]["pending_actions"]
    assert call("groups.stop", cancel_id="stop-second") == {"cancelled": 0}
    service.local_profiles = lambda: ("default", "ops")
    receipt = call(
        "groups.send",
        event_id="third",
        payload={"text": "@ops try again", "thread_id": "third"},
    )
    assert receipt["accepted"] is True
    (queued,) = driver.list_tasks(
        service.store, placement="profile", room_id="stalled", status="queued"
    )
    assert queued["payload"]["source_event_seq"] == receipt["event"]["seq"]
    assert call("groups.state")["driver_status"]["last_error"] is None


def test_stop_clears_the_stall_when_a_rename_lands_right_after_its_fence(
    tmp_path, monkeypatch
):
    service, call, _send = stalled_service(tmp_path, monkeypatch)
    request_room_stop = hosted_rooms.request_room_stop

    def stop_then_rename(*args, **kwargs):
        fence = request_room_stop(*args, **kwargs)
        hosted_rooms.rename_room(
            service.store,
            placement="profile",
            room_id="stalled",
            event_id="rename-after-stop",
            name="Renamed",
        )
        return fence

    monkeypatch.setattr(hosted_rooms, "request_room_stop", stop_then_rename)
    assert call("groups.state")["driver_status"]["pending_actions"]
    assert call("groups.stop", cancel_id="stop-first") == {"cancelled": 0}
    assert call("groups.log")["events"][-1]["kind"] == "room.renamed"
    assert call("groups.state")["driver_status"]["pending_actions"] == []


@pytest.mark.parametrize("route", ["standalone", "authority"])
def test_retry_identifies_failed_preparation_on_both_routes(
    tmp_path, monkeypatch, route
):
    service, call, _ = stalled_service(tmp_path, monkeypatch)
    (action,) = call("groups.state")["driver_status"]["pending_actions"]
    params = {key: value for key, value in action.items() if key != "kind"}
    if route == "standalone":
        monkeypatch.setattr(srv, "get_hosted_room_service", lambda: service)
        reply = srv.handle_request(
            {
                "id": 1,
                "method": "groups.retry",
                "params": {"room_id": "stalled", **params},
            }
        )
        error = reply["error"]
    else:
        error = call("groups.retry", expect_error=True, **params)
    assert error.get("data", {}).get("reason") == "prepare_failed", error


def test_replay_checks_for_new_work_without_reading_event_pages(tmp_path, monkeypatch):
    service, call, send = stalled_service(tmp_path, monkeypatch)
    call("groups.stop", cancel_id="original")
    for i in range(3):
        hosted_rooms.rename_room(
            service.store,
            placement="profile",
            room_id="stalled",
            event_id=f"rename-{i}",
            name=f"Name {i}",
        )
    send("second")
    pages = []
    read = hosted_rooms.read_events

    def count_pages(*args, **kwargs):
        page = read(*args, **kwargs)
        pages.append(page["cursor"])
        return page

    monkeypatch.setattr(hosted_rooms, "MAX_LOG_LIMIT", 1)
    monkeypatch.setattr(hosted_rooms, "read_events", count_pages)
    assert call("groups.stop", cancel_id="original") == {"cancelled": 0}
    assert pages == []
    assert call("groups.state")["driver_status"]["pending_actions"]


def test_stop_cannot_publish_a_second_stall_when_profile_discovery_fails(
    tmp_path, monkeypatch
):
    service, call, _ = stalled_service(tmp_path, monkeypatch)

    def unavailable():
        raise PermissionError("profiles directory is unreadable")

    monkeypatch.setattr(service, "local_profiles", unavailable)
    assert call("groups.stop", cancel_id="stop") == {"cancelled": 0}
    (binding,) = service.bindings()
    for _ in range(STALL_ATTEMPTS):
        with suppress(PermissionError):
            service.runtime.prepare_room(binding)
    assert [event["kind"] for event in call("groups.log")["events"]] == [
        "message.user",
        "member.unavailable",
        "room.stop_requested",
    ]
    assert call("groups.state")["driver_status"]["pending_actions"] == []
