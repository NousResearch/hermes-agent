"""Idle worker renewal uses real signed grants, target handlers and route storage."""

import asyncio
import json
import sqlite3
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import hosted_room_driver as driver, hosted_room_links, hosted_room_link_records, hosted_rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, decode_room_grant, gateway_room_grant_secret
from gateway.platforms import api_server, api_server_room_grants
from gateway.run import _profile_runtime_scope
from tests.tui_gateway.test_hosted_room_service import _server
from tests.tui_gateway.test_groups_grant_lifetimes import invite
from tests.tui_gateway.test_groups_methods import home, _result
from tui_gateway import methods_groups
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture
def renewal(home, monkeypatch):
    methods_groups.stop_hosted_room_service(timeout=1)
    authority = hosted_rooms.local_authority_gateway_id()
    issued = _result(invite(ttl_seconds=3600, status_ttl_seconds=2592000,
                            home_install_id=authority, authority_gateway_id=authority))
    secret = gateway_room_grant_secret(home)
    claims = decode_room_grant(secret, issued["grant"], permission="dispatch")
    clock = [claims["issued_at"]]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._profile_scope = lambda _profile: _profile_runtime_scope(home / "profiles" / "ops")

    class Peer(PeerRunsHTTPClient):
        offline = False
        on_refresh = None

        def __init__(self):
            super().__init__(base_url="https://peer.example.test", api_key="", target_profile="ops")
            self.refreshes = []
            self.issued = []
            self.failures = []

        def refresh_grant(self, **kwargs):
            try:
                return super().refresh_grant(**kwargs)
            except PeerRunsHTTPError as exc:
                self.failures.append((str(exc), exc.error_code))
                raise

        def _request(self, path, *, method="GET", body=None, room_grant=None, **kwargs):
            if path.endswith("/refresh"):
                self.refreshes.append(clock[0])
            if self.offline:
                raise PeerRunsHTTPError("offline", retryable=True, not_admitted=True)

            async def call():
                token = api_server._api_request_profile.set("ops")
                adapter._read_json_body = AsyncMock(return_value=(body or {}, None))
                request = SimpleNamespace(headers={"Authorization": f"HermesRoom {room_grant}"})
                try:
                    if path.endswith("/refresh"):
                        response = await adapter._handle_room_member_grant_refresh(request)
                    elif path.endswith("/capabilities"):
                        response = await adapter._handle_room_member_capabilities(request)
                    elif path.endswith("/revoke-exact"):
                        response = await api_server_room_grants._handle_room_member_grant_revoke_exact(
                            adapter, request, _openai_error=api_server._openai_error,
                            _api_request_profile=api_server._api_request_profile)
                    else:
                        raise AssertionError(path)
                    payload = json.loads(response.text)
                    if response.status >= 400:
                        raise PeerRunsHTTPError("target refused", status_code=response.status,
                                                error_code=payload["error"]["code"], not_admitted=True)
                    return payload
                finally:
                    api_server._api_request_profile.reset(token)

            payload = asyncio.run(call())
            if path.endswith("/refresh"):
                self.issued.append(payload["grant"])
                if self.on_refresh:
                    callback, self.on_refresh = self.on_refresh, None
                    callback()
            return payload

    peer = Peer()
    # Rotation constructs a fresh exact-revocation client; keep that path in-process too.
    monkeypatch.setattr(PeerRunsHTTPClient, "_request", lambda _client, *a, **kw: peer._request(*a, **kw))
    service = HostedRoomService(_server(), db_path=home / "state.db")
    service.runtime.clock = lambda: clock[0]
    catalog = GatewayRoomCatalog.from_mapping(issued["catalog"])
    route = PeerMemberRoute(home_install_id=authority, member_id="ops", target_install_id=authority,
                            target_profile="ops", capability_digest=catalog.catalog_digest,
                            execution_policy_digest=catalog.execution_policy.policy_digest,
                            cancellation_scope_id="cancel-renewal", trace_id="trace-renewal", grant=issued["grant"])
    service.register_peer_route(room_id="renewal-room", member_id="ops", route=route, client=peer,
                                target_url=peer.base_url, catalog=catalog)
    service.create_room(room_id="renewal-room", name="Renewal", members=[
        {"member_id": "local", "profile": "default", "handle": "local"},
        {"member_id": "ops", "profile": "ops", "handle": "ops", "target": {
            "kind": "peer", "installation_id": authority, "peer_id": authority,
            "profile": "ops", "capability_digest": catalog.catalog_digest}},
    ])
    return SimpleNamespace(service=service, peer=peer, clock=clock, claims=claims, secret=secret,
                            home=home, old=issued["grant"])


def test_idle_cycles_renew_with_bounded_cadence_and_offline_backoff(renewal):
    r = renewal
    route = r.service.peer_routes[("renewal-room", "ops")]
    catalog = hosted_room_links.load_room_link(r.service.db_path, room_id="renewal-room", member_id="ops").catalog
    r.service.register_peer_route(room_id="corrupt-room", member_id="other", route=replace(route, member_id="other"),
                                  client=r.peer, target_url=r.peer.base_url, catalog=catalog)
    with sqlite3.connect(r.service.db_path) as conn:
        conn.execute("UPDATE hosted_room_links SET target_url='http://invalid.example.test' WHERE room_id='corrupt-room'")
    r.service.runtime._run_cycle()
    assert not r.peer.refreshes
    # No turns or LLM calls: idle clock advances beyond multiple original one-hour lifetimes.
    for _ in range(3):
        r.clock[0] += 3300
        r.service.runtime._run_cycle()
        current = r.service.peer_routes[("renewal-room", "ops")].grant
        claims = decode_room_grant(r.secret, current, permission="dispatch")
        assert claims["expires_at"] == r.clock[0] + 3600, r.peer.failures
        assert claims["status_expires_at"] == r.claims["status_expires_at"]
        assert claims["permissions"] == r.claims["permissions"]
        calls = len(r.peer.refreshes)
        r.service.runtime._run_cycle()
        assert len(r.peer.refreshes) == calls
    assert len(r.peer.refreshes) == 3
    assert not driver.list_tasks(r.service.db_path, room_id="renewal-room")
    r.clock[0] += 3300
    r.peer.offline = True
    r.service.runtime._run_cycle()
    calls = len(r.peer.refreshes)
    r.clock[0] += 29
    r.service.runtime._run_cycle()
    assert len(r.peer.refreshes) == calls
    r.clock[0] += 1
    r.service.runtime._run_cycle()
    assert len(r.peer.refreshes) == calls + 1
    r.peer.offline = False
    r.clock[0] += 60
    r.service.runtime._run_cycle()
    assert len(r.peer.refreshes) == calls + 2
    assert r.service._route_statuses("renewal-room")[0]["status"] == "ready"


@pytest.mark.parametrize("change", ["expired", "revoked", "policy", "disband", "epoch", "lease_lost", "takeover"])
def test_idle_renewal_cannot_cross_authority_or_grant_fences(renewal, change):
    r = renewal
    r.clock[0] += 3300
    if change == "expired":
        r.clock[0] = r.claims["expires_at"] + 1
    elif change == "revoked":
        hosted_rooms.revoke_room_grant_scope(r.service.db_path, claims=r.claims,
                                             expires_at=r.claims["status_expires_at"])
    elif change == "policy":
        (r.home / "profiles" / "ops" / "config.yaml").write_text("agent:\n  max_turns: 7\n", encoding="utf-8")
    elif change == "disband":
        r.peer.on_refresh = lambda: hosted_room_link_records.begin_room_link_retirement(
            r.service.db_path, room_id="renewal-room",
            authority_gateway_id=r.claims["authority_gateway_id"], authority_epoch=1)
    elif change == "epoch":
        def change_epoch():
            with sqlite3.connect(r.service.db_path) as conn:
                conn.execute("UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id='renewal-room'")
        r.peer.on_refresh = change_epoch
    elif change == "lease_lost":
        r.peer.on_refresh = lambda: driver.release_lease(
            r.service.db_path, r.service.runtime._leases["renewal-room"], clock=lambda: r.clock[0])
    else:
        competitor = HostedRoomService(_server(), db_path=r.service.db_path)
        competitor.runtime.clock = lambda: r.clock[0]
        competitor.peer_clients[("renewal-room", "ops")] = r.peer

        def takeover():
            # The second worker cannot renew while the first still owns its exact lease.
            calls = len(r.peer.refreshes)
            competitor.runtime._run_cycle()
            assert len(r.peer.refreshes) == calls
            driver.release_lease(r.service.db_path, r.service.runtime._leases["renewal-room"], clock=lambda: r.clock[0])
            competitor.runtime._run_cycle()
        r.peer.on_refresh = takeover
    r.service.runtime._run_cycle()
    stored = hosted_room_links.load_room_link(r.service.db_path, room_id="renewal-room", member_id="ops")
    if change == "takeover":
        assert len(r.peer.issued) == 2
        assert stored.grant == r.peer.issued[1]
    else:
        assert stored.grant == r.old
    if change in {"expired", "revoked", "policy"}:
        assert not r.peer.issued
        assert stored.status == "needs_reauthorization"
        calls = len(r.peer.refreshes)
        r.clock[0] += 120
        r.service.runtime._run_cycle()
        assert len(r.peer.refreshes) == calls
    else:
        claims = decode_room_grant(r.secret, r.peer.issued[0], permission="status")
        assert hosted_rooms.room_grant_is_revoked(r.service.db_path, claims=claims)
        if change == "lease_lost":
            assert stored.status == "ready"
            r.clock[0] += 60
            r.service.runtime._run_cycle()
            assert r.service.peer_routes[("renewal-room", "ops")].grant != r.old
