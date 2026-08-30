"""Tests for the gateway-hosted ``groups.*`` JSON-RPC contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tui_gateway.server as srv
from tui_gateway import methods_groups


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / ".hermes"
    path.mkdir()
    (path / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(path))
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def _server_authority():
    from gateway.hosted_rooms import local_authority_gateway_id

    return local_authority_gateway_id()


def _create_room():
    return _result(
        srv._methods["groups.create"](
            1,
            {
                "room_id": "room-1",
                "name": "Release room",
                "members": [
                    {
                        "member_id": "default",
                        "profile": "default",
                        "handle": "hermes",
                    },
                    {"member_id": "ops", "profile": "ops", "handle": "ops"},
                ],
                "authority_gateway_id": "gateway-a",
            },
        )
    )["room"]


def test_capabilities_are_honest_about_the_driver_boundary(home):
    methods_groups.stop_hosted_room_service(timeout=1.0)
    result = _result(srv._methods["groups.capabilities"](1, {}))

    assert result["protocol_version"] == 2
    assert result["driver"] is False
    assert result["authority_gateway_id"] == _server_authority()
    assert "authority_epoch" in result["features"]
    assert "coordinator_fencing" in result["features"]
    assert "monotonic_log" in result["features"]
    assert "groups.state" in result["methods"]
    assert "groups.send" in result["methods"]
    assert "groups.send" in srv._LONG_HANDLERS
    assert "groups.retry" in result["methods"]
    assert "groups.approve" in result["methods"]
    advertised = [
        str(value).lower() for value in (*result["features"], *result["methods"])
    ]
    assert not any(
        token in value
        for token in ("attachment", "desktop", "messaging", "peer", "roomlink")
        for value in advertised
    )


def test_create_list_send_and_log_roundtrip(home):
    room = _create_room()
    assert room["idempotent"] is False

    listed = _result(srv._methods["groups.list"](2, {}))
    assert [item["room_id"] for item in listed["rooms"]] == ["room-1"]
    state = _result(srv._methods["groups.state"](3, {"room_id": "room-1"}))
    assert state["room"]["authority_gateway_id"] == _server_authority()
    assert state["room"]["authority_epoch"] == 1
    assert state["room"]["latest_seq"] == 0

    sent = _result(
        srv._methods["groups.send"](
            4,
            {
                "room_id": "room-1",
                "event_id": "event-1",
                "actor": {"kind": "user", "id": "desktop-user"},
                "payload": {"text": "hello", "thread_id": "thread-1"},
            },
        )
    )
    assert sent["accepted"] is True
    assert sent["driver_started"] is True
    assert sent["event"]["seq"] == 1
    assert sent["event"]["kind"] == "message.user"
    assert sent["event"]["actor"] == {"kind": "user", "id": "desktop"}

    replay = _result(
        srv._methods["groups.log"](
            5,
            {"room_id": "room-1", "since_seq": 0},
        )
    )
    assert replay["latest_seq"] == replay["cursor"] == 1
    assert replay["events"][0]["payload"] == {
        "text": "hello",
        "thread_id": "thread-1",
    }


def test_groups_list_returns_bounded_pages(home):
    _create_room()
    _result(
        srv._methods["groups.create"](
            2,
            {
                "room_id": "room-2",
                "name": "Second room",
                "members": [
                    {
                        "member_id": "default",
                        "profile": "default",
                        "handle": "hermes",
                    },
                    {"member_id": "ops", "profile": "ops", "handle": "ops"},
                ],
            },
        )
    )

    first = _result(srv._methods["groups.list"](3, {"limit": 1}))
    second = _result(
        srv._methods["groups.list"](
            4,
            {"limit": 1, "offset": first["next_offset"]},
        )
    )

    assert first["next_offset"] == 1
    assert second["next_offset"] == 2
    assert {first["rooms"][0]["room_id"], second["rooms"][0]["room_id"]} == {
        "room-1",
        "room-2",
    }
    final = _result(srv._methods["groups.list"](5, {"limit": 1, "offset": 2}))
    assert final["rooms"] == []
    assert final["next_offset"] is None


def test_rpc_retry_is_idempotent_and_conflict_is_visible(home):
    _create_room()
    params = {
        "room_id": "room-1",
        "event_id": "event-1",
        "actor": {"kind": "user", "id": "desktop-user"},
        "payload": {"text": "hello", "thread_id": "thread-1"},
    }
    first = _result(srv._methods["groups.send"](2, params))
    repeated = _result(srv._methods["groups.send"](3, params))

    assert first["event"]["seq"] == repeated["event"]["seq"] == 1
    assert first["client_event_id"] == repeated["client_event_id"] == "event-1"
    assert first["event"]["event_id"].startswith("user:")
    assert repeated["event"]["idempotent"] is True

    conflict = srv._methods["groups.send"](
        4,
        {
            **params,
            "payload": {"text": "different", "thread_id": "thread-1"},
        },
    )
    assert conflict["error"]["code"] == 4111
    assert "different content" in conflict["error"]["message"]


def test_foreign_authority_cannot_send_or_disband(home):
    from gateway.hosted_rooms import (
        claim_authority,
        default_db_path,
        list_rooms,
        read_events,
    )

    _create_room()
    claim_authority(
        default_db_path(),
        room_id="room-1",
        expected_gateway_id=_server_authority(),
        expected_epoch=1,
        new_gateway_id="foreign-gateway",
        event_id="claim-foreign",
    )
    before = read_events(default_db_path(), room_id="room-1")

    sent = srv._methods["groups.send"](
        2,
        {
            "room_id": "room-1",
            "event_id": "stale-send",
            "payload": {"text": "must not land", "thread_id": "thread-1"},
        },
    )
    disbanded = srv._methods["groups.disband"](3, {"room_id": "room-1"})

    assert sent["error"]["code"] == 4111
    assert sent["error"]["data"] == {"reason": "authority_conflict"}
    assert disbanded["error"]["code"] == 4113
    assert disbanded["error"]["data"] == {"reason": "authority_conflict"}
    assert read_events(default_db_path(), room_id="room-1") == before
    assert list_rooms(default_db_path())[0]["room_id"] == "room-1"


def test_client_event_id_cannot_squat_disband_receipt(home):
    _create_room()
    sent = _result(
        srv._methods["groups.send"](
            2,
            {
                "room_id": "room-1",
                "event_id": "system:room-disbanded",
                "payload": {"text": "still a user message", "thread_id": "thread-1"},
            },
        )
    )

    assert sent["client_event_id"] == "system:room-disbanded"
    assert sent["event"]["event_id"].startswith("user:")
    assert sent["event"]["event_id"] != "system:room-disbanded"
    first = _result(srv._methods["groups.disband"](3, {"room_id": "room-1"}))
    repeated = _result(srv._methods["groups.disband"](4, {"room_id": "room-1"}))
    assert first["tombstone"]["event"]["event_id"] == "system:room-disbanded"
    assert repeated["tombstone"]["idempotent"] is True

    replay = _result(
        srv._methods["groups.log"](
            5,
            {"room_id": "room-1", "include_disbanded": True},
        )
    )
    assert [event["kind"] for event in replay["events"]] == [
        "message.user",
        "room.stop_requested",
        "room.disbanded",
    ]


def test_send_does_not_trust_client_supplied_actor_identity(home):
    _create_room()
    sent = _result(
        srv._methods["groups.send"](
            2,
            {
                "room_id": "room-1",
                "event_id": "event-1",
                "actor": {"kind": "user", "id": "spoofed-user"},
                "payload": {"text": "hello", "thread_id": "thread-1"},
            },
        )
    )

    assert sent["event"]["actor"] == {"kind": "user", "id": "desktop"}


def test_create_ignores_client_supplied_authority_identity(home):
    members = [
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"},
    ]
    created = _result(
        srv._methods["groups.create"](
            1,
            {"room_id": "legacy-room", "name": "Legacy", "members": members},
        )
    )["room"]
    retried = _result(
        srv._methods["groups.create"](
            2,
            {
                "room_id": "legacy-room",
                "name": "Legacy",
                "members": members,
                "authority_gateway_id": "spoofed-gateway",
            },
        )
    )["room"]

    assert created["authority_gateway_id"] == _server_authority()
    assert retried["authority_gateway_id"] == _server_authority()
    assert retried["idempotent"] is True


def test_legacy_room_adoption_emits_one_lineage_receipt(home):
    from gateway.hosted_rooms import create_room, default_db_path

    members = [
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"},
    ]
    create_room(
        default_db_path(),
        room_id="legacy-room",
        name="Legacy",
        members=members,
        authority_gateway_id="legacy",
        now=1,
    )

    adopted = _result(
        srv._methods["groups.create"](
            2,
            {"room_id": "legacy-room", "name": "Legacy", "members": members},
        )
    )["room"]
    state = _result(srv._methods["groups.state"](3, {"room_id": "legacy-room"}))["room"]

    assert adopted["adopted"] is True
    assert adopted["authority_gateway_id"] == _server_authority()
    assert adopted["authority_epoch"] == 2
    assert adopted["claim_event"]["payload"] == {
        "previous_gateway_id": "legacy",
        "authority_gateway_id": _server_authority(),
        "authority_epoch": 2,
    }
    assert state["authority_claim"]["event_id"] == "system:authority-adopted"
    assert state["latest_seq"] == 1


@pytest.mark.parametrize(
    ("method_name", "params"),
    [
        (
            "groups.create",
            {
                "room_id": "",
                "name": "x",
                "members": [],
                "authority_gateway_id": "gateway-a",
            },
        ),
        (
            "groups.send",
            {
                "room_id": "missing",
                "event_id": "event-1",
                "actor": {"kind": "user", "id": "desktop-user"},
                "payload": {},
            },
        ),
        ("groups.log", {"room_id": "missing", "since_seq": 0}),
    ],
)
def test_invalid_or_unknown_room_returns_contract_error(home, method_name, params):
    result = srv._methods[method_name](1, params)
    assert result["error"]["code"] in {4110, 4111, 4112, 5111, 5112}


def test_retry_and_approval_controls_forward_only_exact_local_coordinates(
    home, monkeypatch
):
    calls = []
    identity = SimpleNamespace(
        room_id="room-1",
        task_id="task-1",
        thread_id="thread-1",
        turn_id="turn-1",
    )
    service = SimpleNamespace(
        retry_room_task=lambda room_id, task_id: (
            calls.append(("retry", room_id, task_id))
            or {
                "identity": identity,
                "status": "queued",
                "execution_generation": 1,
                "cancel_generation": 0,
            }
        ),
        approve_room_task=lambda room_id, **kwargs: (
            calls.append(("approve", room_id, kwargs)) or {"resolved": 1}
        ),
    )
    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: service)

    retried = _result(
        srv._methods["groups.retry"](
            1,
            {"room_id": "room-1", "task_id": "task-1"},
        )
    )
    approved = _result(
        srv._methods["groups.approve"](
            2,
            {
                "room_id": "room-1",
                "member_id": "ops",
                "task_id": "task-1",
                "execution_generation": 1,
                "request_id": "approval-1",
                "choice": "once",
            },
        )
    )

    assert retried["task"] == {
        "room_id": "room-1",
        "task_id": "task-1",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "status": "queued",
        "execution_generation": 1,
        "cancel_generation": 0,
    }
    assert approved == {"approved": True, "result": {"resolved": 1}}
    assert calls == [
        ("retry", "room-1", "task-1"),
        (
            "approve",
            "room-1",
            {
                "member_id": "ops",
                "task_id": "task-1",
                "execution_generation": 1,
                "choice": "once",
                "request_id": "approval-1",
            },
        ),
    ]


@pytest.mark.parametrize(
    ("method_name", "params"),
    [
        ("groups.create", {"room_id": "room-1", "name": "Room", "members": []}),
        ("groups.send", {"room_id": "room-1", "event_id": "event-1", "payload": {}}),
        ("groups.disband", {"room_id": "room-1"}),
        ("groups.stop", {"room_id": "room-1"}),
        ("groups.retry", {"room_id": "room-1", "task_id": "task-1"}),
        (
            "groups.approve",
            {
                "room_id": "room-1",
                "member_id": "ops",
                "task_id": "task-1",
                "execution_generation": 1,
                "request_id": "approval-1",
                "choice": "once",
            },
        ),
    ],
)
def test_mutating_controls_fail_closed_without_a_supervised_worker(
    home, monkeypatch, method_name, params
):
    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: None)

    result = srv._methods[method_name](1, params)

    assert result["error"]["code"] in {4115, 4123}


def test_disband_tombstones_room(home):
    _create_room()
    first = _result(srv._methods["groups.disband"](3, {"room_id": "room-1"}))
    repeated = _result(srv._methods["groups.disband"](4, {"room_id": "room-1"}))
    assert first["tombstone"]["idempotent"] is False
    assert repeated["tombstone"]["idempotent"] is True
    assert _result(srv._methods["groups.list"](5, {}))["rooms"] == []
    deleted = _result(srv._methods["groups.list"](6, {"include_disbanded": True}))[
        "rooms"
    ]
    assert deleted[0]["disbanded_at"] == first["tombstone"]["disbanded_at"]
    replay = _result(
        srv._methods["groups.log"](
            7,
            {"room_id": "room-1", "include_disbanded": True},
        )
    )
    assert [event["kind"] for event in replay["events"]] == [
        "room.stop_requested",
        "room.disbanded",
    ]


def test_pruned_room_send_and_log_report_expired_history(home, monkeypatch):
    from gateway import hosted_rooms

    members = [
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"},
    ]
    _create_room()
    monkeypatch.setattr(hosted_rooms, "MAX_DISBANDED_ROOM_TOMBSTONES", 0)
    _result(srv._methods["groups.disband"](2, {"room_id": "room-1"}))
    repeated = _result(
        srv._methods["groups.disband"](3, {"room_id": "room-1"})
    )["tombstone"]
    assert repeated["idempotent"] is True
    assert repeated["history_expired"] is True

    sent = srv._methods["groups.send"](
        4,
        {
            "room_id": "room-1",
            "event_id": "stale-send",
            "payload": {"text": "stale", "thread_id": "thread-1"},
        },
    )
    logged = srv._methods["groups.log"](
        5,
        {"room_id": "room-1", "include_disbanded": True},
    )

    assert sent["error"]["code"] == 4111, sent
    assert logged["error"]["code"] == 4112, logged
    assert sent["error"]["data"] == {"reason": "room_history_expired"}
    assert logged["error"]["data"] == {"reason": "room_history_expired"}
    assert "permanently retired" in sent["error"]["message"]

    recreated = srv._methods["groups.create"](
        6,
        {"room_id": "room-1", "name": "Replacement", "members": members},
    )
    assert recreated["error"]["code"] == 4110
    created = _result(
        srv._methods["groups.create"](
            7,
            {"room_id": "room-new", "name": "Fresh", "members": members},
        )
    )
    assert created["room"]["room_id"] == "room-new"
