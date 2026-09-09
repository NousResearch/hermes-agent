"""RPC invitations preserve HTTP lifetime bounds and signed renewal fences."""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret
from gateway.platforms import api_server
from gateway.run import _profile_runtime_scope
from tests.tui_gateway.test_groups_methods import home, _result
from tui_gateway import server


def invite(**extra):
    return server._methods["groups.peer.invite"](1, {
        "room_id": "renewal-room", "home_install_id": "install-home",
        "authority_gateway_id": "install-home", "authority_epoch": 1,
        "member_id": "ops", "profile": "ops", **extra,
    })


@pytest.mark.parametrize("lifetime", [
    {}, {"ttl_seconds": 3600, "status_ttl_seconds": 2592000},
    {"ttl_seconds": 60, "status_ttl_seconds": 60},
    {"ttl_seconds": 86400, "status_ttl_seconds": 2592000},
    {"ttl_seconds": 59}, {"ttl_seconds": 86401}, {"ttl_seconds": float("nan")},
    {"ttl_seconds": float("inf")}, {"status_ttl_seconds": 3599},
    {"status_ttl_seconds": 2592001}, {"status_ttl_seconds": float("nan")},
    {"status_ttl_seconds": float("inf")}, {"status_ttl_seconds": None},
])
def test_rpc_invitation_reports_only_bounded_signed_lifetimes(home, lifetime):
    result = invite(**lifetime)
    ttl = lifetime.get("ttl_seconds", 3600)
    status_ttl = lifetime.get("status_ttl_seconds", ttl)
    valid = status_ttl is not None and 60 <= ttl <= 86400 and ttl <= status_ttl <= 2592000
    if not valid:
        assert result["error"]["code"] == 4120
        assert not hosted_rooms.peer_room_is_reserved(home / "state.db", room_id="renewal-room", target_profile="ops")
        return
    result = _result(result)
    claims = decode_room_grant(gateway_room_grant_secret(home), result["grant"], permission="dispatch")
    assert claims["expires_at"] - claims["issued_at"] == ttl
    assert claims["status_expires_at"] - claims["issued_at"] == status_ttl
    assert result["expires_at"] == claims["expires_at"]
    assert result["status_expires_at"] == claims["status_expires_at"]
    assert "peer_grant_renewal" in _result(server._methods["groups.capabilities"](0, {}))["features"]
    for db in (home / "state.db", home / "profiles" / "ops" / "state.db"):
        assert hosted_rooms.peer_room_is_reserved(db, room_id="renewal-room", target_profile="ops",
                                                  now=claims["status_expires_at"] - 1)


@pytest.mark.asyncio
async def test_rpc_grant_renews_via_real_http_handler_without_reviving_or_widening(home, monkeypatch):
    issued = _result(invite(ttl_seconds=3600, status_ttl_seconds=2592000))
    secret = gateway_room_grant_secret(home)
    claims = decode_room_grant(secret, issued["grant"], permission="dispatch")
    now = [claims["issued_at"] + 3300]
    monkeypatch.setattr(time, "time", lambda: now[0])
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._read_json_body = AsyncMock(return_value=({"ttl_seconds": 3600}, None))
    adapter._profile_scope = lambda _profile: _profile_runtime_scope(home / "profiles" / "ops")
    profile_token = api_server._api_request_profile.set("ops")

    async def refresh(grant):
        response = await adapter._handle_room_member_grant_refresh(
            SimpleNamespace(headers={"Authorization": f"HermesRoom {grant}"}))
        return response.status, json.loads(response.text)

    try:
        status, renewed = await refresh(issued["grant"])
        assert status == 200
        replacement = decode_room_grant(secret, renewed["grant"], permission="dispatch")
        assert replacement["status_expires_at"] == claims["status_expires_at"]
        assert replacement["expires_at"] == now[0] + 3600
        for field in ("room_id", "member_id", "authority_epoch", "permissions", "execution_policy_digest"):
            assert replacement[field] == claims[field]
        now[0] = claims["expires_at"] + 1
        assert (await refresh(issued["grant"]))[0] == 401
        # A policy edit is not silently approved by a still-live renewal bearer.
        config = home / "profiles" / "ops" / "config.yaml"
        config.write_text("agent:\n  max_turns: 7\n", encoding="utf-8")
        assert (await refresh(renewed["grant"]))[0] == 403
        config.unlink()
        _result(server._methods["groups.peer.revoke"](2, {"profile": "ops", "grant": renewed["grant"]}))
        assert (await refresh(renewed["grant"]))[0] == 403
        now[0] = claims["status_expires_at"] + 1
        assert (await refresh(renewed["grant"]))[0] == 401
    finally:
        api_server._api_request_profile.reset(profile_token)
