"""Reconnect capabilities, RPC fields and legacy-peer cleanup stay source-local."""

import hashlib
import time
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv
from gateway import hosted_room_link_records
from gateway import hosted_room_links, hosted_room_peer, hosted_rooms
from tests.tui_gateway.test_hosted_room_grant_fingerprint import peers as peers
from tui_gateway import methods_groups
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


def test_capabilities_and_state_expose_fingerprint_without_bearer(peers, monkeypatch):
    first, _second, _register = peers
    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: first)
    monkeypatch.setattr(hosted_rooms, "default_db_path", lambda: first.db_path)
    capability = srv._methods["groups.capabilities"](1, {})["result"]
    assert "peer_route_grant_fingerprint" in capability["features"]
    assert "groups.peer.revoke_exact" in capability["methods"]
    assert "groups.peer.revoke_exact" in srv._LONG_HANDLERS
    assert "attachment_ids" in capability["features"]
    state = srv._methods["groups.state"](2, {"room_id": "room-1"})["result"]
    assert (
        state["driver_status"]["peer_routes"][0]["grant_sha256"]
        == hashlib.sha256(b"signed.room.grant").hexdigest()
    )
    assert "signed.room.grant" not in repr(state)


@pytest.mark.parametrize("mode", ["match", "stale", "omitted", "invalid", "empty"])
def test_registration_rpc_carries_optional_expected_grant_to_persisted_cas(
    peers, monkeypatch, mode
):
    first, _second, _register = peers
    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: first)
    stored = hosted_room_links.load_room_link(
        first.db_path, room_id="room-1", member_id="member-peer"
    )
    room = hosted_rooms.room_state(first.db_path, room_id="room-1")
    monkeypatch.setattr(
        PeerRunsHTTPClient,
        "probe",
        lambda self, **kwargs: {
            "catalog": stored.catalog_mapping(),
            "room_id": "room-1",
            "home_install_id": room["authority_gateway_id"],
            "authority_gateway_id": room["authority_gateway_id"],
            "authority_epoch": 1,
            "member_id": "member-peer",
            "target_profile": "reviewer",
        },
    )
    params = {
        "room_id": "room-1",
        "member_id": "member-peer",
        "target_profile": "reviewer",
        "target_url": stored.target_url,
        "catalog": stored.catalog_mapping(),
        "grant": "replacement.room.grant",
    }
    if mode != "omitted":
        params["expected_grant_sha256"] = {
            "match": hashlib.sha256(stored.grant.encode()).hexdigest(),
            "stale": "0" * 64,
            "invalid": "invalid",
            "empty": "",
        }[mode]
    result = srv._methods["groups.peer.register"](1, params)
    row = hosted_room_link_records.room_link_record(
        first.db_path, room_id="room-1", member_id="member-peer"
    )
    if mode in {"match", "omitted"}:
        assert "error" not in result, result
        assert result["result"]["registered"] is True
        assert row["grant"] == params["grant"]
        assert (
            srv._methods["groups.peer.register"](2, params)["result"]["registered"]
            is True
        )
    else:
        assert "error" in result
        assert row["grant"] == stored.grant


def test_exact_rpc_uses_requested_profile_and_preserves_sibling_grant(
    tmp_path, monkeypatch
):
    root = tmp_path / "home"
    profile = root / "profiles" / "reviewer"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(
        methods_groups,
        "_bound_server",
        SimpleNamespace(
            _current_profile_name=lambda: "default",
            _profile_home=lambda name: profile if name == "reviewer" else None,
            _response_profile_name=lambda name: name,
        ),
    )
    secret = b"source-rpc-exact-test-secret-only"
    monkeypatch.setattr(hosted_room_peer, "gateway_room_grant_secret", lambda: secret)
    monkeypatch.setattr(
        hosted_rooms, "local_authority_gateway_id", lambda: "install:peer"
    )
    kwargs = dict(
        room_id="rpc-room",
        home_install_id="install:home",
        authority_gateway_id="install:home",
        authority_epoch=1,
        member_id="member",
        target_install_id="install:peer",
        target_profile="reviewer",
        issued_at=time.time() - 1,
        ttl_seconds=3600,
    )
    losing = hosted_room_peer.issue_room_grant(secret, grant_id="losing", **kwargs)
    winning = hosted_room_peer.issue_room_grant(secret, grant_id="winning", **kwargs)
    method = srv._methods["groups.peer.revoke_exact"]
    for _ in range(2):
        assert (
            method(1, {"profile": "reviewer", "grant": losing})["result"]["revoked"]
            is True
        )
    assert "error" in method(1, {"profile": "default", "grant": winning})
    for db in (root / "state.db", profile / "state.db"):
        assert hosted_rooms.room_grant_is_revoked(
            db,
            claims=hosted_room_peer.decode_room_grant(
                secret, losing, permission="status"
            ),
        )
        assert not hosted_rooms.room_grant_is_revoked(
            db,
            claims=hosted_room_peer.decode_room_grant(
                secret, winning, permission="status"
            ),
        )


def test_old_gateway_without_exact_cleanup_never_falls_back_to_scope_revoke():
    client = PeerRunsHTTPClient(
        base_url="https://peer.example.test", api_key="", target_profile="reviewer"
    )
    catalog = hosted_room_peer.catalog_mapping(
        installation_id="install:peer",
        persistent_process=True,
        execution_policy=hosted_room_peer.execution_policy_mapping(
            target_profile="reviewer"
        ),
    )
    paths = []

    def request(path, **kwargs):
        paths.append(path)
        if path.endswith("/refresh"):
            return {"grant": "unpublished.grant"}
        if path.endswith("/capabilities"):
            return {"catalog": catalog}
        assert path.endswith("/revoke-exact")
        raise PeerRunsHTTPError("old peer", status_code=404, error_code="not_found")

    client._request = request
    with pytest.raises(PeerRunsHTTPError) as caught:
        client.refresh_grant(
            grant="old.grant",
            capability_digest="0" * 64,
            execution_policy_digest=catalog["execution_policy"]["policy_digest"],
        )
    assert caught.value.error_code == "room_capability_catalog_changed"
    assert paths == [
        "/v1/room-members/grants/refresh",
        "/v1/room-members/capabilities",
        "/v1/room-members/grants/revoke-exact",
    ]
