"""Tests for the gateway-hosted ``groups.*`` JSON-RPC contract."""

from __future__ import annotations

import base64

from types import SimpleNamespace

import pytest

import tui_gateway.server as srv
from tui_gateway import methods_groups


@pytest.fixture
def home(tmp_path, monkeypatch):
    class DurableRunStore:
        durable = True

    path = tmp_path / ".hermes"
    path.mkdir()
    (path / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(path))
    monkeypatch.setattr(srv, "_run_idempotency_store", DurableRunStore(), raising=False)
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
    assert "groups.attachment.put" in result["methods"]
    assert "groups.attachment.read" in result["methods"]
    advertised = [
        str(value).lower() for value in (*result["features"], *result["methods"])
    ]
    assert not any(
        token in value
        for token in ("desktop", "messaging")
        for value in advertised
    )
    assert result["room_link"]["enabled"] is True


def test_capabilities_and_invitation_advertise_scoped_roomlink(home, monkeypatch):
    from gateway.platforms import api_server_room_attachments

    monkeypatch.setattr(
        api_server_room_attachments,
        "roomlink_attachments_available",
        lambda: True,
    )
    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    result = _result(srv._methods["groups.capabilities"](1, {}))
    assert result["room_link"]["enabled"] is True
    assert result["room_link"]["profile"] == "reviewer"
    assert result["room_link"]["catalog"]["text"] is True
    assert result["room_link"]["catalog"]["attachments"] is True
    assert "groups.peer.invite" in result["methods"]
    assert "groups.peer.register" in result["methods"]

    invitation = _result(
        srv._methods["groups.peer.invite"](
            2,
            {
                "room_id": "room-1",
                "home_install_id": "install-home",
                "authority_gateway_id": "install-home",
                "authority_epoch": 1,
                "member_id": "member-peer",
                "grant_id": "grant-room-1",
            },
        )
    )
    assert invitation["target_profile"] == "reviewer"
    assert invitation["catalog"] == result["room_link"]["catalog"]
    assert invitation["catalog"]["attachments"] is True
    assert "." in invitation["grant"]
    from gateway import hosted_rooms

    assert hosted_rooms.peer_room_is_reserved(
        hosted_rooms.default_db_path(),
        room_id="room-1",
        target_profile="reviewer",
    )


def test_capabilities_disable_roomlink_when_run_replay_is_not_durable(
    home, monkeypatch
):
    import tui_gateway.methods_groups as groups_methods

    class VolatileRunStore:
        durable = False

    class BoundServer:
        _run_idempotency_store = VolatileRunStore()

    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    monkeypatch.setattr(groups_methods, "_bound_server", BoundServer())
    result = _result(srv._methods["groups.capabilities"](1, {}))
    assert result["room_link"] == {
        "enabled": False,
        "reason": "durable_run_storage_required",
    }
    invitation = srv._methods["groups.peer.invite"](
        2,
        {
            "room_id": "room-volatile",
            "home_install_id": "install-home",
            "authority_gateway_id": "install-home",
            "authority_epoch": 1,
            "member_id": "member-peer",
        },
    )
    assert invitation["error"]["code"] == 4120
    assert "durable run idempotency" in invitation["error"]["message"]


def test_capabilities_open_shared_durable_run_store_without_test_injection(
    home, monkeypatch
):
    """The production dashboard server must not depend on fixture injection."""

    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    monkeypatch.delattr(srv, "_run_idempotency_store", raising=False)

    result = _result(srv._methods["groups.capabilities"](1, {}))
    store = srv._run_idempotency_store
    try:
        assert store.durable is True
        assert result["room_link"]["enabled"] is True
    finally:
        store.close()


def test_app_managed_catalog_and_self_advertised_endpoint_are_consistent(
    home, monkeypatch
):
    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_ROOM_LINK_URL", "https://peer.example.test/hermes")
    capability = _result(srv._methods["groups.capabilities"](1, {}))
    invitation = _result(
        srv._methods["groups.peer.invite"](
            2,
            {
                "room_id": "room-1",
                "home_install_id": "install-home",
                "authority_gateway_id": "install-home",
                "authority_epoch": 1,
                "member_id": "member-peer",
            },
        )
    )
    assert capability["persistent_process"] is False
    assert capability["room_link"]["catalog"] == invitation["catalog"]
    assert capability["room_link"]["endpoint"] == {
        "available": True,
        "url": "https://peer.example.test/hermes",
        "transport_security": "tls",
    }
    assert invitation["endpoint"] == capability["room_link"]["endpoint"]


def test_launch_profile_is_valid_for_roomlink_invitation(home, monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    monkeypatch.setenv("HERMES_PROFILE", "default")

    capability = _result(
        srv._methods["groups.capabilities"](1, {"profile": "default"})
    )
    invitation = _result(
        srv._methods["groups.peer.invite"](
            2,
            {
                "room_id": "room-default",
                "home_install_id": "install-home",
                "authority_gateway_id": "install-home",
                "authority_epoch": 1,
                "member_id": "member-default",
                "profile": "default",
            },
        )
    )

    assert capability["room_link"]["enabled"] is True
    assert capability["room_link"]["profile"] == "default"
    assert invitation["target_profile"] == "default"


def test_roomlink_endpoint_absence_has_machine_reason(home, monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    monkeypatch.delenv("HERMES_ROOM_LINK_URL", raising=False)
    result = _result(srv._methods["groups.capabilities"](1, {}))
    assert result["room_link"]["endpoint"] == {
        "available": False,
        "reason": "not_configured",
    }


def test_multiplexed_invitation_uses_exact_profile_secret(home, monkeypatch):
    from gateway.hosted_room_peer import (
        HostedRoomGrantError,
        decode_room_grant,
        derive_room_grant_secret,
        gateway_room_grant_secret,
    )

    reviewer_home = home / "profiles" / "reviewer"
    reviewer_home.mkdir(parents=True)
    reviewer_key = "reviewer-api-key-1234567890"
    default_key = "default-api-key-1234567890"
    (reviewer_home / ".env").write_text(
        f"API_SERVER_KEY={reviewer_key}\n", encoding="utf-8"
    )
    monkeypatch.setenv("API_SERVER_KEY", default_key)
    invitation = _result(
        srv._methods["groups.peer.invite"](
            3,
            {
                "room_id": "room-1",
                "home_install_id": "install-home",
                "authority_gateway_id": "gateway-home",
                "authority_epoch": 1,
                "member_id": "member-reviewer",
                "profile": "reviewer",
            },
        )
    )
    claims = decode_room_grant(
        gateway_room_grant_secret(home),
        invitation["grant"],
        permission="status",
    )
    assert claims["target_profile"] == "reviewer"
    with pytest.raises(HostedRoomGrantError, match="signature"):
        decode_room_grant(
            derive_room_grant_secret(default_key),
            invitation["grant"],
            permission="status",
        )


def test_named_profile_needs_no_copied_api_key_for_roomlink(home, monkeypatch):
    from gateway.hosted_room_peer import (
        HostedRoomGrantError,
        decode_room_grant,
        derive_room_grant_secret,
        gateway_room_grant_secret,
    )

    reviewer_home = home / "profiles" / "reviewer"
    reviewer_home.mkdir(parents=True)
    gateway_key = "gateway-api-key-1234567890"
    monkeypatch.setenv("API_SERVER_KEY", gateway_key)

    invitation = _result(
        srv._methods["groups.peer.invite"](
            4,
            {
                "room_id": "room-named-bot",
                "home_install_id": "install-home",
                "authority_gateway_id": "gateway-home",
                "authority_epoch": 1,
                "member_id": "member-reviewer",
                "profile": "reviewer",
            },
        )
    )

    claims = decode_room_grant(
        gateway_room_grant_secret(home),
        invitation["grant"],
        permission="status",
    )
    assert claims["target_profile"] == "reviewer"
    with pytest.raises(HostedRoomGrantError, match="signature"):
        decode_room_grant(
            derive_room_grant_secret(gateway_key),
            invitation["grant"],
            permission="status",
        )


def test_register_peer_route_probes_scope_and_persists_via_service(home, monkeypatch):
    from gateway.hosted_room_peer import catalog_mapping
    from gateway.hosted_rooms import local_authority_gateway_id

    catalog = catalog_mapping(
        installation_id="install-peer",
        persistent_process=True,
    )
    captured = {}
    room = _create_room()

    class FakeClient:
        def __init__(self, *, base_url, api_key, **kwargs):
            captured["base_url"] = base_url
            captured["api_key"] = api_key

        def probe(self, *, grant):
            captured["grant"] = grant
            return {
                "room_id": "room-1",
                "home_install_id": local_authority_gateway_id(),
                "authority_gateway_id": room["authority_gateway_id"],
                "authority_epoch": room["authority_epoch"],
                "member_id": "member-peer",
                "target_profile": "reviewer",
                "catalog": catalog,
            }

    class FakeService:
        db_path = home / "state.db"

        def register_peer_route(self, **kwargs):
            captured["registered"] = kwargs

    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: FakeService())
    monkeypatch.setattr(
        "tui_gateway.hosted_room_peer_http.PeerRunsHTTPClient",
        FakeClient,
    )
    result = _result(
        srv._methods["groups.peer.register"](
            3,
            {
                "room_id": "room-1",
                "member_id": "member-peer",
                "target_url": "https://peer.example.test",
                "target_profile": "reviewer",
                "grant": "signed.room.grant",
                "catalog": catalog,
            },
        )
    )
    assert result["registered"] is True
    assert captured["api_key"] == ""
    assert captured["registered"]["target_url"] == ("https://peer.example.test")


def test_register_rejects_plaintext_non_loopback(home, monkeypatch):
    class FakeService:
        db_path = home / "state.db"

    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: FakeService())
    response = srv._methods["groups.peer.register"](
        4,
        {
            "room_id": "room-1",
            "member_id": "member-peer",
            "target_url": "http://peer.example.test:8377",
            "target_profile": "reviewer",
            "grant": "signed.room.grant",
            "catalog": {},
        },
    )
    assert response["error"]["code"] == 5120
    assert "https outside" in response["error"]["message"]


def test_register_requires_roomlink_protocol_v2(home, monkeypatch):
    from gateway.hosted_room_peer import catalog_mapping

    class FakeService:
        db_path = home / "state.db"

    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: FakeService())
    response = srv._methods["groups.peer.register"](
        5,
        {
            "room_id": "room-1",
            "member_id": "member-peer",
            "target_url": "https://peer.example.test",
            "target_profile": "reviewer",
            "grant": "signed.room.grant",
            "catalog": catalog_mapping(
                installation_id="install-peer",
                protocol_versions=(1,),
                persistent_process=True,
            ),
        },
    )
    assert response["error"]["code"] == 5120
    assert "protocol v2" in response["error"]["message"]


def test_create_list_send_and_log_roundtrip(home):
    room = _create_room()
    assert room["idempotent"] is False

    listed = _result(srv._methods["groups.list"](2, {}))
    assert [item["room_id"] for item in listed["rooms"]] == ["room-1"]
    assert listed["next_offset"] is None
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


def test_client_event_id_cannot_squat_disband_receipt(home, monkeypatch):
    from gateway import hosted_rooms

    _create_room()
    service = methods_groups.get_hosted_room_service()
    assert service is not None

    def append_without_starting_work(*, room_id, event_id, payload):
        room = hosted_rooms.room_state(service.db_path, room_id=room_id)
        return hosted_rooms.append_event(
            service.db_path,
            room_id=room_id,
            event_id=event_id,
            kind="message.user",
            actor={"kind": "user", "id": "desktop"},
            payload=payload,
            authority_gateway_id=str(room["authority_gateway_id"]),
            authority_epoch=int(room["authority_epoch"]),
        )

    monkeypatch.setattr(service, "send", append_without_starting_work)
    monkeypatch.setattr(service, "stop_room", lambda *_args, **_kwargs: 0)
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
    kinds = [event["kind"] for event in replay["events"]]
    assert kinds[0] == "message.user"
    assert kinds[-1] == "room.disbanded"
    assert kinds.count("room.disbanded") == 1


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
    renamed = srv._methods["groups.rename"](
        6,
        {"room_id": "room-1", "event_id": "stale-rename", "name": "Stale"},
    )

    assert sent["error"]["code"] == 4111, sent
    assert logged["error"]["code"] == 4112, logged
    assert renamed["error"]["code"] == 4117, renamed
    assert sent["error"]["data"] == {"reason": "room_history_expired"}
    assert logged["error"]["data"] == {"reason": "room_history_expired"}
    assert renamed["error"]["data"] == {"reason": "room_history_expired"}
    assert "permanently retired" in sent["error"]["message"]

    recreated = srv._methods["groups.create"](
        7,
        {"room_id": "room-1", "name": "Replacement", "members": members},
    )
    assert recreated["error"]["code"] == 4110
    created = _result(
        srv._methods["groups.create"](
            8,
            {"room_id": "room-new", "name": "Fresh", "members": members},
        )
    )
    assert created["room"]["room_id"] == "room-new"


def test_disband_stops_and_revokes_before_tombstoning(home, monkeypatch):
    _create_room()
    calls = []

    class FakeService:
        db_path = home / "state.db"
        attachments = SimpleNamespace(
            mark_room_disbanded=lambda room_id: calls.append(("files", room_id)),
            prune=lambda: calls.append(("prune", "room-1")),
        )

        def stop_room(self, room_id, **_kwargs):
            calls.append(("stop", room_id))

        def revoke_room_routes(self, room_id):
            calls.append(("revoke", room_id))

    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: FakeService())
    _result(srv._methods["groups.disband"](9, {"room_id": "room-1"}))

    assert calls == [
        ("stop", "room-1"),
        ("revoke", "room-1"),
        ("files", "room-1"),
        ("prune", "room-1"),
    ]
    assert _result(srv._methods["groups.list"](10, {}))["rooms"] == []


def test_failed_remote_revocation_keeps_room_recoverable(home, monkeypatch):
    _create_room()

    class FakeService:
        db_path = home / "state.db"

        def stop_room(self, _room_id, **_kwargs):
            return 1

        def revoke_room_routes(self, _room_id):
            raise RuntimeError("peer is offline")

    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: FakeService())
    result = srv._methods["groups.disband"](11, {"room_id": "room-1"})

    assert result["error"]["code"] == 5114
    assert [
        room["room_id"]
        for room in _result(srv._methods["groups.list"](12, {}))["rooms"]
    ] == ["room-1"]


def test_disband_does_not_revoke_routes_while_stop_is_unacknowledged(
    home, monkeypatch
):
    _create_room()
    calls = []

    class FakeService:
        db_path = home / "state.db"

        def stop_room(self, _room_id, **kwargs):
            calls.append(("stop", kwargs["require_acknowledged"]))
            raise RuntimeError("room work is still stopping")

        def revoke_room_routes(self, _room_id):
            calls.append(("revoke", True))

    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: FakeService())
    result = srv._methods["groups.disband"](13, {"room_id": "room-1"})

    assert result["error"]["code"] == 5114
    assert calls == [("stop", True)]
    assert [
        room["room_id"]
        for room in _result(srv._methods["groups.list"](14, {}))["rooms"]
    ] == ["room-1"]


def test_approve_routes_one_exact_peer_action(home, monkeypatch):
    captured = {}

    class FakeService:
        def approve_room_task(self, room_id, **kwargs):
            captured["room_id"] = room_id
            captured.update(kwargs)
            return {"resolved": 1}

    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: FakeService())
    result = _result(
        srv._methods["groups.approve"](
            8,
            {
                "room_id": "room-1",
                "member_id": "member-peer",
                "task_id": "task-1",
                "execution_generation": 2,
                "request_id": "approval-1",
                "choice": "once",
            },
        )
    )

    assert result == {"approved": True, "result": {"resolved": 1}}
    assert captured == {
        "room_id": "room-1",
        "member_id": "member-peer",
        "task_id": "task-1",
        "execution_generation": 2,
        "request_id": "approval-1",
        "choice": "once",
    }


def test_attachment_put_send_read_roundtrip_is_recipient_scoped(home, monkeypatch):
    _create_room()
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode("ascii")
    put_params = {
        "room_id": "room-1",
        "upload_id": "upload-1",
        "kind": "image",
        "name": "diagram.png",
        "mime": "image/png",
        "content_base64": encoded,
    }
    stored = _result(
        srv._methods["groups.attachment.put"](1, put_params)
    )["attachment"]
    manifest = {
        key: stored[key]
        for key in ("attachment_id", "kind", "name", "size", "mime")
    }

    sent = _result(
        srv._methods["groups.send"](
            2,
            {
                "room_id": "room-1",
                "event_id": "event-attachment-1",
                "payload": {
                    "text": "Review this image",
                    "thread_id": "thread-1",
                    "attachments": [manifest],
                },
            },
        )
    )
    assert sent["event"]["payload"]["attachments"] == [manifest]
    read = _result(
        srv._methods["groups.attachment.read"](
            3,
            {
                    "room_id": "room-1",
                    "attachment_id": stored["attachment_id"],
                    "purpose": "viewer",
                    "event_id": sent["event"]["event_id"],
                },
        )
    )
    assert base64.b64decode(read["content_base64"]) == b"\x89PNG\r\n\x1a\nimage"
    denied = srv._methods["groups.attachment.read"](
        4,
        {
            "room_id": "room-1",
            "attachment_id": stored["attachment_id"],
            "purpose": "member",
            "event_id": "event-attachment-1",
        },
    )
    assert denied["error"]["code"] == 4141

    service = methods_groups.get_hosted_room_service()
    assert service is not None
    monkeypatch.setattr(service, "stop_room", lambda *_args, **_kwargs: 0)

    _result(
        srv._methods["groups.disband"](
            5,
            {"room_id": "room-1", "cancel_id": "disband-file-room"},
        )
    )
    after_disband = srv._methods["groups.attachment.read"](
        6,
        {
            "room_id": "room-1",
            "attachment_id": stored["attachment_id"],
            "purpose": "viewer",
            "event_id": "event-attachment-1",
        },
    )
    assert after_disband["error"]["code"] == 4141


def test_peer_revoke_discards_the_scoped_attachment_spool(home, monkeypatch):
    from gateway.platforms import api_server_room_attachments

    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    invitation = _result(
        srv._methods["groups.peer.invite"](
            1,
            {
                "room_id": "room-revoke",
                "home_install_id": "install-home",
                "authority_gateway_id": "install-home",
                "authority_epoch": 1,
                "member_id": "member-peer",
                "grant_id": "grant-room-revoke",
            },
        )
    )
    discarded = []

    class FakeSpool:
        def discard_scope(self, claims):
            discarded.append(dict(claims))
            return 1

    monkeypatch.setattr(api_server_room_attachments, "_default_spool", FakeSpool)
    result = _result(
        srv._methods["groups.peer.revoke"](
            2,
            {"grant": invitation["grant"]},
        )
    )

    assert result["revoked"] is True
    assert discarded[0]["room_id"] == "room-revoke"
    assert discarded[0]["member_id"] == "member-peer"
