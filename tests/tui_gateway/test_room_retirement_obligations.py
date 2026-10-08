"""Ended rooms retain exact cleanup custody until a target owner authorizes retirement."""
import asyncio
import hashlib
import time

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_peer import decode_room_grant
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.tui_gateway.test_hosted_room_two_gateway_scoped import _linked_home, _server_module
from tui_gateway import server as rpc_server
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_service import HostedRoomService


async def _rpc(method, **params):
    return await asyncio.to_thread(rpc_server._methods[method], 1, params)


async def _cancel_many(client, grant, claims):
    probe = await asyncio.to_thread(client.probe, grant=grant)
    body = {"input": "cancelled work", "hosted_room_dispatch": {
        "protocol_version": 2,
        **{name: claims[name] for name in ("room_id", "home_install_id", "authority_gateway_id",
          "authority_epoch", "member_id", "target_install_id", "target_profile", "execution_policy_digest")},
        "execution_generation": 1, "source_event_seq": 1, "cancellation_scope_id": "cancel-room-1",
        "prompt": "cancelled work", "prompt_digest": hashlib.sha256(b"cancelled work").hexdigest(),
        "capability_digest": probe["catalog"]["catalog_digest"], "trace_id": "trace-room-1"}}
    for index in range(64):
        body["hosted_room_dispatch"]["task_id"] = f"never-sent-{index}"
        reply = await asyncio.to_thread(client._request, "/v1/runs/stop", method="POST", body=body,
            headers={"Idempotency-Key": f"room:never-sent-{index}:1"}, room_grant=grant)
        assert reply["admission_cancelled"] is True
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_reply", [False, True])
async def test_expired_end_retains_recoverable_retirement_after_restart_and_history_prune(tmp_path, monkeypatch, lost_reply):
    target, server, home = await _linked_home(tmp_path)
    client = home.peer_clients[("room-1", "member-peer")]
    route = home.peer_routes[("room-1", "member-peer")]
    claims = decode_room_grant(target._room_grant_secret(), route.grant, permission="retire")
    monkeypatch.setattr(rpc_server, "get_hosted_room_service", lambda: home)
    store = target._run_idempotency_store
    try:
        body = await _cancel_many(client, route.grant, claims)
        if lost_reply:
            original = client.revoke_grant
            def lose_reply(**kwargs):
                original(**kwargs)
                raise PeerRunsHTTPError("retirement reply lost", ambiguous=True, retryable=True)
            with monkeypatch.context() as loss:
                loss.setattr(client, "revoke_grant", lose_reply)
                assert "error" in await _rpc("groups.disband", room_id="room-1")
        future = claims["status_expires_at"] + 86400
        monkeypatch.setattr(time, "time", lambda: future)
        home.stop(timeout=5)
        home = HostedRoomService(_server_module(), db_path=tmp_path / "home-state.db")
        ended = await _rpc("groups.disband", room_id="room-1")
        assert "error" not in ended, ended
        pending, = ended["result"]["retirements"]
        assert pending["status"] == "needs_reauthorization"
        assert pending["room_id"] == "room-1" and pending["member_id"] == "member-peer"
        assert "grant" not in pending and route.grant not in str(ended)
        assert hosted_rooms.list_room_link_records(home.db_path) == []
        assert home.peer_routes == {}
        future += hosted_rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1
        hosted_rooms.prune_disbanded_rooms(home.db_path, now=future)
        home.stop(timeout=5)
        home = HostedRoomService(_server_module(), db_path=tmp_path / "home-state.db")
        observed = await _rpc("groups.peer.retirements", room_id="room-1")
        assert observed["result"]["retirements"] == [pending]
        store.close()
        store = target._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "target-runs.db"))
        expected = 0 if lost_reply else 64
        assert store._conn.execute("SELECT COUNT(*) FROM run_idempotency").fetchone()[0] == expected
        fresh = await asyncio.to_thread(client.issue_invitation, retirement_only=True,
            **{name: claims[name] for name in ("room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id")},
            grant_id="fresh-retirement-only")
        permissions = decode_room_grant(target._room_grant_secret(), fresh["grant"], permission="retire")["permissions"]
        assert set(permissions) == {"status", "retire"}
        settled = await _rpc("groups.peer.retire", room_id="room-1", retirement_id=pending["retirement_id"], grant=fresh["grant"])
        assert settled == {"jsonrpc": "2.0", "id": 1, "result": {"retirements": []}}, settled
        assert store._conn.execute("SELECT COUNT(*) FROM run_idempotency").fetchone()[0] == 0
        assert home.peer_routes == {} and home.bindings() == ()
        assert "error" in await _rpc("groups.send", room_id="room-1", event_id="late", payload={"text": "must remain ended"})
        for token in (route.grant, fresh["grant"]):
            with pytest.raises(PeerRunsHTTPError) as refused:
                await asyncio.to_thread(client._request, "/v1/runs", method="POST", body=body,
                    headers={"Idempotency-Key": "room:never-sent-63:1"}, room_grant=token)
            assert refused.value.status_code in {401, 403, 409}
    finally:
        home.stop(timeout=5)
        await server.close()
        store.close()


@pytest.mark.asyncio
async def test_ordinary_revocation_reply_keeps_distinct_retirement_and_rejects_other_scope(tmp_path, monkeypatch):
    target, server, home = await _linked_home(tmp_path)
    client = home.peer_clients[("room-1", "member-peer")]
    route = home.peer_routes[("room-1", "member-peer")]
    claims = decode_room_grant(target._room_grant_secret(), route.grant, permission="retire")
    monkeypatch.setattr(rpc_server, "get_hosted_room_service", lambda: home)
    try:
        await _cancel_many(client, route.grant, claims)
        original = client.revoke_grant
        with monkeypatch.context() as old_endpoint:
            old_endpoint.setattr(client, "revoke_grant", lambda **kwargs: original(grant=kwargs["grant"]))
            ended = await _rpc("groups.disband", room_id="room-1")
        pending, = ended["result"]["retirements"]
        assert pending["status"] == "needs_reauthorization"
        assert target._run_idempotency_store._conn.execute("SELECT COUNT(*) FROM run_idempotency").fetchone()[0] == 64
        other = await asyncio.to_thread(client.issue_invitation, room_id="unrelated-room",
            home_install_id=claims["home_install_id"], authority_gateway_id=claims["authority_gateway_id"],
            authority_epoch=1, member_id=claims["member_id"], grant_id="unrelated-grant")
        refused = await _rpc("groups.peer.retire", room_id="room-1", retirement_id=pending["retirement_id"], grant=other["grant"])
        assert "error" in refused
        assert (await _rpc("groups.peer.retirements", room_id="room-1"))["result"]["retirements"] == [pending]
        assert (await asyncio.to_thread(client.probe, grant=other["grant"]))["room_id"] == "unrelated-room"
        assert home.peer_routes == {}
    finally:
        home.stop(timeout=5)
        await server.close()
        target._run_idempotency_store.close()
