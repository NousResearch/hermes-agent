"""Route-owner contract for the separately delivered Files staging consumer.

No Files imports, runtime construction, network, or coordination: setup's DB
executor and HTTP peer are inert; hydration consumes a real serialized catalog.
"""

from contextlib import nullcontext
from dataclasses import replace
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import hosted_room_links, hosted_room_peer, hosted_rooms, session_group_setup
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from gateway.platforms.api_server_room_grants import _local_room_catalog
from gateway.session_hosted_service import CanonicalHostedRoomService
from tui_gateway import hosted_room_service
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute, build_member_dispatch
from tui_gateway.hosted_room_service import HostedRoomService


def _route(catalog):
    return PeerMemberRoute(
        home_install_id="home", member_id="member", target_install_id="peer",
        target_profile="default", capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id="cancel", trace_id="trace", grant="inert-grant",
    )


def _service(tmp_path):
    # Do not construct HostedRoomService: its constructor owns runtime setup.
    service = HostedRoomService.__new__(HostedRoomService)
    service.db_path = tmp_path / "unused.db"
    service._policy_lock = RLock()
    service.peer_routes, service.peer_clients = {}, {}
    service._peer_route_status = {}
    service._persisted_peer_route_keys = set()
    service._set_route_status = Mock()
    return service


@pytest.mark.parametrize("attachments", [False, True])
@pytest.mark.parametrize("path", ["setup", "load", "hydrate"])
def test_route_preserves_catalog_without_activating_local_target(
    tmp_path, monkeypatch, attachments, path,
):
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: "home")
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
        installation_id="peer", persistent_process=True, text=True,
        attachments=attachments, target_profile="default",
        endpoint={"available": False, "reason": "not_configured"},
    ))
    service = _service(tmp_path)
    if path == "setup":
        authority = SimpleNamespace(profile_id=str(tmp_path), epoch=1)
        canonical = CanonicalHostedRoomService.__new__(CanonicalHostedRoomService)
        canonical.authority, canonical.db_path = authority, service.db_path
        canonical.authorize_room = Mock()
        canonical.register_peer_route = Mock()
        pending = {"binding": {"authority_epoch": 1}, "previous_grant_sha256": ""}
        # Only the constructor seam is under test, not setup receipts/CAS.
        authority.db = SimpleNamespace(_execute_write=Mock(side_effect=[
            (pending, False), (pending, True),
        ]))
        probe = dict(
            catalog=catalog.as_mapping(), room_id="room", member_id="member",
            home_install_id="home", authority_gateway_id="home", authority_epoch=1,
            target_profile="default",
        )
        http = Mock(probe=Mock(return_value=probe))
        factory = Mock(return_value=http)
        monkeypatch.setattr(session_group_setup, "PeerRunsHTTPClient", factory)
        session_group_setup.register_peer(authority, SimpleNamespace(subject="owner"), canonical, {
            "room_id": "room", "member_id": "member", "request_id": "request",
            "target_profile": "default", "target_url": "https://peer.invalid",
            "grant": "inert-grant", "catalog": catalog.as_mapping(),
        })
        canonical.register_peer_route.assert_called_once()
        route = canonical.register_peer_route.call_args.kwargs["route"]
        http.probe.assert_called_once_with(grant="inert-grant")
    else:
        stored = hosted_room_links.StoredRoomLink.from_mapping(dict(
            room_id="room", member_id="member", target_url="https://peer.invalid",
            target_profile="default", grant="inert-grant", catalog=catalog.as_mapping(),
            cancellation_scope_id="cancel", trace_id="trace", updated_at=1,
        ))
        stored = hosted_room_links.StoredRoomLink.from_record(stored.as_record())

        class InertHTTP:
            def __init__(self, *, base_url, **kwargs):
                self.base_url = base_url

        monkeypatch.setattr(hosted_room_service, "PeerRunsHTTPClient", InertHTTP)
        monkeypatch.setattr(hosted_room_links, "load_room_link", lambda *a, **k: stored)
        monkeypatch.setattr(hosted_room_links, "load_room_links_tolerant", lambda *a: ([stored], []))
        if path == "load":
            service._load_stored_links()
            route = service.peer_routes[("room", "member")]
        else:
            route, client = service._hydrate_persisted_peer_route("room", "member")
            assert service._hydrate_persisted_peer_route("room", "member") == (route, client)
    assert getattr(route, "attachments", None) is attachments
    assert route.capability_digest == catalog.catalog_digest
    assert route.execution_policy_digest == catalog.execution_policy.policy_digest
    assert getattr(_route(catalog), "attachments", None) is False

    # The real canonical catalog still refuses text AND attachments.
    adapter = SimpleNamespace(
        gateway_runner=SimpleNamespace(session_authorities=SimpleNamespace(active=lambda: None)),
        _profile_scope=lambda profile: nullcontext(),
    )
    _, unavailable = _local_room_catalog(adapter, "default", "home")
    assert unavailable["text"] is False
    assert unavailable["attachments"] is False


@pytest.mark.parametrize("stale", [False, True])
def test_staging_runs_existing_route_preflight_before_upload(tmp_path, monkeypatch, stale):
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
        installation_id="peer", persistent_process=True, target_profile="default",
        endpoint={"available": False, "reason": "not_configured"},
    ))
    route = _route(catalog)
    dispatch = build_member_dispatch(
        binding=SimpleNamespace(gateway_id="home", authority_epoch=1), route=route,
        room_id="room", task_id="task", target_profile="default", execution_generation=1,
        source_event_seq=1, prompt="inert prompt", trace_id="trace",
    ).as_mapping()
    service = _service(tmp_path)
    service.peer_routes[("room", "member")] = replace(route, trace_id="changed") if stale else route
    events = []

    def retirement(*args, **kwargs):
        events.append("retirement-check")
        return False

    def load(*args, **kwargs):
        events.append("route-check")
        return None

    def fresh(grant):
        events.append("grant-check")
        assert grant == route.grant
        return False

    def stage_attachments(*, dispatch, grant, attachments):
        events.append("upload")
        assert grant == route.grant
        assert dispatch["capability_digest"] == catalog.catalog_digest
        return attachments

    monkeypatch.setattr(hosted_room_service.hosted_room_link_records, "room_link_retirement_started", retirement)
    monkeypatch.setattr(hosted_room_links, "load_room_link", load)
    monkeypatch.setattr(hosted_room_peer, "room_grant_needs_dispatch_refresh", fresh)
    # The real Files consumer supplies this method; R needs no spool or manifest implementation.
    upload = Mock(side_effect=stage_attachments)
    tracked = service._tracked_peer_client("room", "member", SimpleNamespace(stage_attachments=upload), route=route)
    if stale:
        with pytest.raises(RuntimeError, match="route changed before admission"):
            tracked.stage_attachments(dispatch=dispatch, grant=route.grant, attachments=())
        upload.assert_not_called()
        assert events == ["retirement-check", "route-check"]
        service._set_route_status.assert_not_called()
    else:
        assert tracked.stage_attachments(dispatch=dispatch, grant=route.grant, attachments=()) == ()
        upload.assert_called_once_with(dispatch=dispatch, grant=route.grant, attachments=())
        assert events == [
            "retirement-check", "route-check", "grant-check",
            "retirement-check", "route-check", "upload",
        ]
