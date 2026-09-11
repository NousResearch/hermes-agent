"""Changing the coordinator never moves a Bot or substitutes a same-named profile."""

from dataclasses import replace
import sqlite3
from types import SimpleNamespace

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms as rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tui_gateway.hosted_room_peer_transport import PeerHostedRoomTransport, PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService
from tests.tui_gateway.test_hosted_room_service import _FakePeerClient, _server


@pytest.fixture
def recovered(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "gateway-new")
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    rooms.create_room(service.db_path, room_id="same-room", name="Original group", members=[
        {"member_id": "original", "profile": "default", "handle": "original"},
        {"member_id": "remote", "profile": "review", "handle": "review", "target": {
            "kind": "peer", "peer_id": "original-peer", "installation_id": "gateway-new",
            "profile": "review", "capability_digest": "a" * 64}},
        {"member_id": "operations", "profile": "ops", "handle": "ops"},
    ], authority_gateway_id="gateway-old")
    rooms.claim_authority(service.db_path, room_id="same-room", expected_gateway_id="gateway-old",
        expected_epoch=1, new_gateway_id="gateway-new", event_id="operator-claim")
    return service


def task(member="original", profile="default"):
    return {"payload": {"target_member_id": member, "target_profile": profile, "source_event_seq": 3}}


def route(installation="gateway-old", member="original", profile="default"):
    return PeerMemberRoute(home_install_id="gateway-old", target_install_id=installation,
        target_profile=profile, member_id=member, capability_digest="a" * 64,
        cancellation_scope_id="cancel-same-room", trace_id="trace-same-room", grant="existing-grant")


def test_same_named_local_profile_never_replaces_the_original_bot(recovered):
    assert "default" in recovered.local_profiles()
    with pytest.raises(RuntimeError, match="Original-local Bot rejoin is unavailable"):
        recovered._resolve_member_transport(recovered.bindings()[0], task())


def test_original_local_profiles_remain_valid_context_even_if_absent_on_new_host(recovered):
    assert "ops" not in recovered.local_profiles()
    state = recovered._room("same-room")
    checked = discussion.validate_room(state, local_profiles=recovered._discussion_profiles(state))
    assert checked.members[0].target == {"kind": "local", "profile": "default"}
    assert checked.members[1].target["peer_id"] == "original-peer"
    assert checked.members[2].target == {"kind": "local", "profile": "ops"}
    assert recovered._recovered_member_target("same-room", "original") == (
        "gateway-old", "gateway-old", "default", "local")


def test_new_request_keeps_original_roster_but_cannot_admit_unsupported_local_rejoin(recovered):
    from gateway import hosted_room_driver as driver

    with pytest.raises(RuntimeError, match="Original-local Bot rejoin is unavailable"):
        recovered.send(room_id="same-room", event_id="owner-follow-up",
            payload={"text": "@ops Check the latest decision.", "thread_id": "same-thread"})
    with pytest.raises(RuntimeError, match="Original-local Bot rejoin is unavailable"):
        recovered.prepare_room(recovered.bindings()[0])
    events = rooms.read_events(recovered.db_path, room_id="same-room")["events"]
    assert sum(event["event_id"] == "owner-follow-up" for event in events) == 1
    assert driver.list_tasks(recovered.db_path, room_id="same-room", status="queued") == []
    assert driver.list_tasks(recovered.db_path, room_id="same-room", status="running") == []


@pytest.mark.parametrize("corruption", [
    {"home_install_id": "gateway-new"}, {"target_install_id": "gateway-old"},
    {"target_profile": "other"},
])
def test_route_cannot_change_original_bot_or_hidden_session_namespace(recovered, corruption):
    key = ("same-room", "remote")
    recovered.peer_routes[key] = replace(route("gateway-new", "remote", "review"), **corruption)
    recovered.peer_clients[key] = _FakePeerClient()
    with pytest.raises(RuntimeError, match="original host"):
        recovered._resolve_member_transport(recovered.bindings()[0], task("remote", "review"))
    assert recovered.peer_clients[key].dispatches == []


def test_unknown_member_and_changed_profile_fail_before_transport(recovered):
    for request, reason in ((task("unknown"), "recorded membership"), (task(profile="review"), "recorded Bot profile")):
        with pytest.raises(RuntimeError, match=reason):
            recovered._resolve_member_transport(recovered.bindings()[0], request)


def test_self_peer_keeps_existing_peer_transport(recovered):
    key = ("same-room", "remote")
    recovered.peer_routes[key] = route("gateway-new", "remote", "review")
    recovered.peer_clients[key] = _FakePeerClient()
    transport = recovered._resolve_member_transport(recovered.bindings()[0], task("remote", "review"))
    assert isinstance(transport, PeerHostedRoomTransport)
    assert transport.route.home_install_id == "gateway-old"
    assert transport.route.target_install_id == "gateway-new"
    assert recovered.peer_clients[key].dispatches == []


@pytest.mark.parametrize("cached_peer_route", [False, True])
def test_explicit_return_to_original_host_can_use_its_own_local_profile(recovered, monkeypatch, cached_peer_route):
    rooms.claim_authority(recovered.db_path, room_id="same-room", expected_gateway_id="gateway-new",
        expected_epoch=2, new_gateway_id="gateway-old", event_id="operator-return")
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "gateway-old")
    (recovered.root / "profiles" / "ops").mkdir(parents=True)
    if cached_peer_route:
        recovered.peer_routes[("same-room", "original")] = route()
        recovered.peer_clients[("same-room", "original")] = _FakePeerClient()
    assert recovered._resolve_member_transport(recovered.bindings()[0], task()) is recovered.rpc


@pytest.mark.parametrize("lineage", ["missing_claim", "legacy_transfer"])
def test_missing_lineage_cannot_be_used_to_invent_recovery_context(recovered, monkeypatch, lineage):
    if lineage == "missing_claim":
        with sqlite3.connect(recovered.db_path) as conn:
            conn.execute("UPDATE hosted_rooms SET authority_epoch=3 WHERE room_id='same-room'")
        with pytest.raises(rooms.AuthorityConflictError, match="history is unavailable"):
            recovered._room("same-room")
        return
    members = [
        {"member_id": "original", "profile": "default", "handle": "original"},
        {"member_id": "operations", "profile": "ops", "handle": "ops"},
    ]
    rooms.create_room(recovered.db_path, room_id="legacy-room", name="Legacy", members=members,
                      authority_gateway_id="legacy")
    rooms.create_room(recovered.db_path, room_id="legacy-room", name="Legacy", members=members,
                      authority_gateway_id="gateway-old")
    assert recovered._recovered_member_target("legacy-room", "original") is None
    with monkeypatch.context() as patch:
        patch.setattr(rooms, "local_authority_gateway_id", lambda: "gateway-old")
        binding = next(item for item in recovered.bindings() if item.room_id == "legacy-room")
        assert recovered._resolve_member_transport(binding, task()) is recovered.rpc
    rooms.claim_authority(recovered.db_path, room_id="legacy-room", expected_gateway_id="gateway-old",
        expected_epoch=2, new_gateway_id="gateway-new", event_id="legacy-transfer")
    assert "authority_history" not in recovered._room("legacy-room")
    binding = next(item for item in recovered.bindings() if item.room_id == "legacy-room")
    with pytest.raises(RuntimeError, match="no verified authority history"):
        recovered._resolve_member_transport(binding, task())


def test_persisted_route_restart_keeps_original_home_namespace(recovered):
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(installation_id="gateway-new", persistent_process=True))
    valid = replace(route("gateway-new", "remote", "review"), capability_digest=catalog.catalog_digest,
                    execution_policy_digest=catalog.execution_policy.policy_digest)
    recovered.register_peer_route(room_id="same-room", member_id="remote", route=valid,
        client=_FakePeerClient(), target_url="https://new.example.test", catalog=catalog)
    restarted = HostedRoomService(_server(), db_path=recovered.db_path)
    restored = restarted.peer_routes[("same-room", "remote")]
    assert restored.home_install_id == "gateway-old"
    # A stale in-memory route must be rehydrated even if its grant did not change.
    restarted.peer_routes[("same-room", "remote")] = replace(restored, home_install_id="gateway-new")
    hydrated, _ = restarted._hydrate_persisted_peer_route("same-room", "remote")
    assert hydrated.home_install_id == "gateway-old"
    assert hydrated.grant == valid.grant


def test_bad_replacement_route_is_rejected_before_retiring_working_route(recovered):
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(installation_id="gateway-new", persistent_process=True))
    original_client = _FakePeerClient()
    valid = replace(route("gateway-new", "remote", "review"), capability_digest=catalog.catalog_digest,
                    execution_policy_digest=catalog.execution_policy.policy_digest)
    recovered.register_peer_route(room_id="same-room", member_id="remote", route=valid,
        client=original_client, target_url="https://new.example.test", catalog=catalog)
    with pytest.raises(ValueError, match="original host"):
        recovered.register_peer_route(room_id="same-room", member_id="remote",
            route=replace(valid, home_install_id="gateway-new", grant="replacement-grant"),
            client=_FakePeerClient(), target_url="https://new.example.test", catalog=catalog)
    assert original_client.exact_revoked == []
    assert recovered.peer_routes[("same-room", "remote")].grant == valid.grant


@pytest.mark.parametrize("route_available", [False, True])
def test_return_to_origin_keeps_frozen_self_peer_out_of_local_rpc(tmp_path, monkeypatch, route_available):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "gateway-old")
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    (service.root / "profiles" / "review").mkdir(parents=True)
    rooms.create_room(service.db_path, room_id="same-room", name="Self peer", members=[
        {"member_id": "original", "profile": "default", "handle": "original"},
        {"member_id": "remote", "profile": "review", "handle": "review", "target": {
            "kind": "peer", "peer_id": "self-peer", "installation_id": "gateway-old",
            "profile": "review", "capability_digest": "a" * 64}},
    ], authority_gateway_id="gateway-old")
    rooms.claim_authority(service.db_path, room_id="same-room", expected_gateway_id="gateway-old",
        expected_epoch=1, new_gateway_id="gateway-new", event_id="away")
    rooms.claim_authority(service.db_path, room_id="same-room", expected_gateway_id="gateway-new",
        expected_epoch=2, new_gateway_id="gateway-old", event_id="return")
    if route_available:
        service.peer_routes[("same-room", "remote")] = route("gateway-old", "remote", "review")
        service.peer_clients[("same-room", "remote")] = _FakePeerClient()
        transport = service._resolve_member_transport(service.bindings()[0], task("remote", "review"))
        assert isinstance(transport, PeerHostedRoomTransport)
    else:
        with pytest.raises(RuntimeError, match="route is unavailable"):
            service._resolve_member_transport(service.bindings()[0], task("remote", "review"))


@pytest.mark.parametrize("profile,member", [("default", "original"), ("ops", "operations")])
@pytest.mark.parametrize("has_transcript", [False, True])
@pytest.mark.parametrize("operation", ["resolve", "register"])
def test_original_local_rejoin_is_gated_without_touching_target_context(
    recovered, tmp_path, profile, member, has_transcript, operation,
):
    from unittest.mock import Mock

    from gateway import hosted_room_links
    from hermes_state import SessionDB
    from tui_gateway.server import _new_session_key

    target_root = tmp_path / "original-install"
    target_path = target_root / "state.db" if profile == "default" else target_root / "profiles" / profile / "state.db"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=target_path)
    try:
        previous = _new_session_key()
        if has_transcript:
            db.create_session(previous, source="bot_room", profile_name=profile)
            db.set_session_title(previous, "Group: same-room")
            db.set_session_hidden(previous, True)
            db.append_message(previous, "user", "Keep the original local decision.")
            db.append_message(previous, "assistant", "The accepted plan stays here.")
        before = db.get_messages(previous) if has_transcript else []
        client = Mock()
        catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(installation_id="gateway-old", persistent_process=True))
        candidate = replace(route("gateway-old", member, profile), capability_digest=catalog.catalog_digest,
                            execution_policy_digest=catalog.execution_policy.policy_digest)
        with pytest.raises(RuntimeError, match="Original-local Bot rejoin is unavailable"):
            if operation == "resolve":
                recovered.peer_routes[("same-room", member)] = candidate
                recovered.peer_clients[("same-room", member)] = client
                recovered._resolve_member_transport(recovered.bindings()[0], task(member, profile))
            else:
                recovered.register_peer_route(room_id="same-room", member_id=member, route=candidate,
                    client=client, target_url="https://old.example.test", catalog=catalog)
        assert client.mock_calls == []
        assert hosted_room_links.load_room_link(recovered.db_path, room_id="same-room", member_id=member) is None
        assert (db.get_messages(previous) if has_transcript else []) == before
        with sqlite3.connect(target_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == int(has_transcript)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_real_hidden_session_keeps_prior_context_when_coordinator_changes(tmp_path):
    from gateway.platforms.api_server_room_dispatch import _ensure_hosted_member_session
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "target.db")

    async def session_db():
        return db

    target = SimpleNamespace(_ensure_session_db_async=session_db)
    dispatch = SimpleNamespace(home_install_id="gateway-old", room_id="same-room", member_id="remote",
        target_profile="review", authority_gateway_id="gateway-old", authority_epoch=1)
    try:
        previous = await _ensure_hosted_member_session(target, dispatch)
        db.append_message(previous, "user", "Keep the accepted plan.")
        db.append_message(previous, "assistant", "The plan is saved.")
        before = db.get_messages(previous)
        dispatch.authority_gateway_id, dispatch.authority_epoch = "gateway-new", 2
        current = await _ensure_hosted_member_session(target, dispatch)
        assert current == previous
        assert db.get_messages(current) == before
        # Replacing the stable home would create another identity; the target
        # refuses the conflicting title rather than silently losing context.
        dispatch.home_install_id = "gateway-new"
        with pytest.raises(RuntimeError, match="Another group already uses"):
            await _ensure_hosted_member_session(target, dispatch)
        assert db.get_messages(previous) == before
    finally:
        db.close()
