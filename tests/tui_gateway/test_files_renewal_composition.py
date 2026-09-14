"""Bounded behavior probes for the Files lane's missing renewal consumer."""

import hashlib
import threading
import time
from types import SimpleNamespace

from gateway import hosted_room_driver as driver, hosted_room_links as links, hosted_rooms as rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping, decode_room_grant, issue_room_grant
from tests.tui_gateway.test_groups_methods import home  # noqa: F401
from tui_gateway import server
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


def test_rpc_honors_requested_observation_horizon(home):
    from gateway.hosted_room_peer import gateway_room_grant_secret
    result = server._methods["groups.peer.invite"](1, {
        "room_id": "probe", "home_install_id": "install:home", "authority_gateway_id": "install:home",
        "authority_epoch": 1, "member_id": "ops", "profile": "ops",
        "ttl_seconds": 60, "status_ttl_seconds": 3600,
    })
    assert "error" not in result
    claims = decode_room_grant(gateway_room_grant_secret(home), result["result"]["grant"], permission="status")
    assert claims["status_expires_at"] - claims["issued_at"] == 3600


def test_accepted_observer_adopts_same_scope_rotation_without_redispatch(tmp_path, monkeypatch):
    secret = b"layering-audit-target-secret-only-32"
    target = tmp_path / "target.db"
    source = tmp_path / "source.db"
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "install:home")
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
        installation_id="install:target", target_profile="ops", persistent_process=True, attachments=True))
    now = time.time()
    common = dict(room_id="room", home_install_id="install:home", authority_gateway_id="install:home",
                  authority_epoch=1, member_id="ops", target_install_id="install:target", target_profile="ops",
                  execution_policy_digest=catalog.execution_policy.policy_digest,
                  ttl_seconds=600, status_ttl_seconds=3600)
    old = issue_room_grant(secret, grant_id="old", issued_at=now - 2, **common)
    new = issue_room_grant(secret, grant_id="new", issued_at=now - 1, **common)
    old_claims = decode_room_grant(secret, old, permission="status")
    rooms.reserve_peer_room(target, claims=old_claims, expires_at=old_claims["status_expires_at"])
    observations = []
    def request(_self, path, *, method="GET", body=None, room_grant=None, **kwargs):
        claims = decode_room_grant(secret, room_grant, permission="status")
        if path.endswith("/revoke-exact"):
            rooms.revoke_room_grant_id(target, claims=claims, expires_at=claims["status_expires_at"])
            return {"revoked": True}
        assert path == "/v1/runs/accepted" and method == "GET"
        observations.append(claims["grant_id"])
        if rooms.room_grant_is_revoked(target, claims=claims):
            raise PeerRunsHTTPError("retired grant", status_code=403, error_code="room_reauthorization_required")
        return {"run_id": "accepted", "status": "completed", "output": "Accepted result"}
    monkeypatch.setattr(PeerRunsHTTPClient, "_request", request)
    service = HostedRoomService(
        SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock()), db_path=source)
    try:
        service.create_room(room_id="room", name="Probe", members=[
            {"member_id": "default", "profile": "default", "handle": "home"},
            {"member_id": "ops", "profile": "ops", "handle": "ops", "target": {
                "kind": "peer", "peer_id": "target", "installation_id": "install:target",
                "profile": "ops", "capability_digest": catalog.catalog_digest}},
        ])
        client = PeerRunsHTTPClient(base_url="https://peer.invalid", api_key="", target_profile="ops")
        route = PeerMemberRoute(home_install_id="install:home", member_id="ops", target_install_id="install:target",
                                target_profile="ops", capability_digest=catalog.catalog_digest,
                                execution_policy_digest=catalog.execution_policy.policy_digest,
                                cancellation_scope_id="cancel", trace_id="trace", grant=old)
        service.register_peer_route(room_id="room", member_id="ops", route=route, client=client,
                                    target_url=client.base_url, catalog=catalog)
        task = {"identity": driver.TaskIdentity("room", "task", "thread", "turn"),
                "execution_generation": 1, "status": "running",
                "payload": {"target_member_id": "ops", "target_profile": "ops", "source_event_seq": 1}}
        transport = service._resolve_member_transport(service.bindings()[0], task)
        client._runs[("task", 1)] = {"room_id": "room", "target_profile": "ops", "session_id": "session",
                                     "task_id": "task", "execution_generation": 1, "run_id": "accepted"}
        client.bind_observation(task_id="task", execution_generation=1)
        service._rotate_route_grant("room", "ops", new,
                                    expected_grant_sha256=hashlib.sha256(old.encode()).hexdigest())
        assert rooms.room_grant_is_revoked(target, claims=old_claims)
        assert links.load_room_link(source, room_id="room", member_id="ops").grant == new
        assert transport.info(profile="ops", session_id="session", source="bot_room")["status"] == "completed"
        assert observations == ["new"]
    finally:
        assert service.stop(timeout=2)
