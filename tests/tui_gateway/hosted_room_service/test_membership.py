"""Membership changes, retry receipts and transcript preservation."""

import pytest

from gateway import hosted_rooms
from tests.tui_gateway.test_hosted_room_service import (
    _FakeRPC,
    _append_room_event,
    _server,
    _wait_for,
    store_fixture,
)
from tui_gateway.hosted_room_service import HostedRoomService

store = store_fixture


def test_membership_edit_keeps_old_replies_and_allows_same_thread_after_restart(store):
    def service_for(profiles):
        service = HostedRoomService(_server(), store=store, placement="profile")
        service.rpc = _FakeRPC()
        service.runtime.rpc = service.rpc
        service.local_profiles = lambda: profiles
        return service

    def send(service, event_id, text):
        return service.send(room_id="team", event_id=event_id, payload={"thread_id": "topic", "text": text})

    service = service_for(("default", "helper", "reviewer"))
    created = service.create_room(room_id="team", name="Release", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "helper", "profile": "helper", "handle": "helper"},
    ])
    change = {"room_id": "team", "event_id": "roster", "expected_revision": created["revision"],
              "members": [created["members"][0],
                          {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"}]}
    send(service, "first", "@helper inspect the release")
    with pytest.raises(hosted_rooms.HostedRoomError, match="Finish the current"):
        service.replace_members(**change)
    service.start()
    try:
        _wait_for(lambda: any(event["kind"] == "room.activity" for event in service._events("team")), timeout=5)
    finally:
        assert service.stop()
    before = service._events("team")
    assert any(event["kind"] == "message.member" and event["actor"]["id"] == "helper" for event in before)

    updated = service.replace_members(**change)
    assert updated["room_id"] == created["room_id"]
    assert service._events("team")[:-1] == before

    restarted = service_for(("default", "reviewer"))
    restarted.start()
    try:
        followup = send(restarted, "second", "@reviewer follow up")
        _wait_for(lambda: any(event["kind"] == "room.activity" and
                             event["payload"]["discussion_event_id"] == followup["event_id"]
                             for event in restarted._events("team")), timeout=5)
    finally:
        assert restarted.stop()
    after = restarted._events("team")
    assert after[:len(before)] == before
    assert [event["actor"]["id"] for event in after if event["kind"] == "message.member"] == ["helper", "reviewer"]


def test_membership_retry_survives_a_later_edit_and_profile_retirement(store):
    service = HostedRoomService(_server(), store=store, placement="profile")
    service.local_profiles = lambda: ("default", "helper", "reviewer")
    initial = service.create_room(room_id="team", name="Release", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "helper", "profile": "helper", "handle": "helper"},
    ])
    replace = {"room_id": "team", "event_id": "add-reviewer", "expected_revision": initial["revision"],
               "members": [initial["members"][0],
                           {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"}]}
    edited = service.replace_members(**replace)
    restored = service.replace_members(room_id="team", event_id="restore-helper", expected_revision=edited["revision"],
                                       members=initial["members"])
    service.local_profiles = lambda: ("default", "helper")
    events = service._events("team")

    replay = service.replace_members(**replace)
    assert replay["idempotent"] is True
    assert replay["members"] == restored["members"]
    assert replay["revision"] == restored["revision"]
    with pytest.raises(hosted_rooms.EventConflictError):
        service.replace_members(**{**replace, "members": [initial["members"][0], {
            **replace["members"][1], "handle": "different-handle",
        }]})
    assert service._events("team") == events


def test_membership_edit_refuses_if_a_send_lands_during_its_readiness_check(store, monkeypatch):
    service = HostedRoomService(_server(), store=store, placement="profile")
    service.local_profiles = lambda: ("default", "helper", "reviewer")
    room = service.create_room(room_id="team", name="Release", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "helper", "profile": "helper", "handle": "helper"},
    ])
    snapshot = service._policy_snapshot

    def read_then_send(current):
        result = snapshot(current)
        _append_room_event(
            store, room_id="team", event_id="racing-send", kind="message.user",
            actor={"kind": "user", "id": "desktop"}, payload={"thread_id": "topic", "text": "Please review."})
        return result

    monkeypatch.setattr(service, "_policy_snapshot", read_then_send)
    with pytest.raises(hosted_rooms.HostedRoomError, match="Reopen Manage members"):
        service.replace_members(room_id="team", event_id="racing-edit", expected_revision=room["revision"],
                                members=[room["members"][0], {
                                    "member_id": "reviewer", "profile": "reviewer", "handle": "reviewer",
                                }])
    current = service._room("team")
    assert current["members"] == room["members"]
    assert current["revision"] == room["revision"]
    assert [event["event_id"] for event in service._events("team")] == ["racing-send"]
