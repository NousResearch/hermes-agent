"""Reconnect identity remains safe across gateway workers sharing one store."""

import hashlib
import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_room_link_records
from gateway import hosted_rooms
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    catalog_mapping,
    execution_policy_mapping,
)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


def _hash(grant):
    return hashlib.sha256(grant.encode()).hexdigest()


@pytest.fixture
def peers(tmp_path, monkeypatch):
    monkeypatch.setattr(PeerRunsHTTPClient, "revoke_grant_exact", lambda self, **kwargs: {"revoked": True})
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
            execution_policy=execution_policy_mapping(target_profile="reviewer"),
        )
    )

    def register(service, grant, *, expected=None):
        route = PeerMemberRoute(
            home_install_id=hosted_rooms.local_authority_gateway_id(),
            member_id="member-peer",
            target_install_id="install-peer",
            target_profile="reviewer",
            capability_digest=catalog.catalog_digest,
            execution_policy_digest=catalog.execution_policy.policy_digest,
            cancellation_scope_id="cancel-room",
            trace_id="trace-room",
            grant=grant,
        )
        client = PeerRunsHTTPClient(
            base_url="https://peer.example.test",
            api_key="",
            target_profile="reviewer",
            receipt_db_path=db,
        )
        service.register_peer_route(
            room_id="room-1",
            member_id="member-peer",
            route=route,
            client=client,
            target_url="https://peer.example.test",
            catalog=catalog,
            expected_grant_sha256=expected,
        )

    first = HostedRoomService(server, db_path=db)
    register(first, "signed.room.grant")
    first.create_room(
        room_id="room-1",
        name="Reconnect",
        members=[
            {"member_id": "default", "profile": "default", "handle": "local"},
            {
                "member_id": "member-peer",
                "profile": "reviewer",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "peer_id": "install-peer",
                    "installation_id": "install-peer",
                    "profile": "reviewer",
                    "capability_digest": catalog.catalog_digest,
                },
            },
        ],
    )
    second = HostedRoomService(server, db_path=db)
    return first, second, register


def test_route_status_exposes_only_the_grant_fingerprint(peers):
    first, _second, _register = peers
    status = first.status_with_grant_fingerprints("room-1")
    assert status["peer_routes"][0]["grant_sha256"] == _hash("signed.room.grant")
    assert "signed.room.grant" not in repr(status)


def test_status_refreshes_a_route_changed_by_another_worker(peers):
    first, second, register = peers
    register(first, "winner.room.grant", expected=_hash("signed.room.grant"))
    status = second.status_with_grant_fingerprints("room-1")
    assert status["peer_routes"][0]["grant_sha256"] == _hash("winner.room.grant")
    assert "winner.room.grant" not in repr(status)


def test_peer_registration_compares_persisted_grant_and_retries_idempotently(peers):
    first, second, register = peers
    old_hash = _hash("signed.room.grant")
    register(first, "winner.room.grant", expected=old_hash)
    with pytest.raises(hosted_rooms.HostedRoomError, match="changed during reconnect"):
        register(second, "loser.room.grant", expected=old_hash)
    register(second, "winner.room.grant", expected=old_hash)
    record = hosted_room_link_records.room_link_record(
        first.db_path, room_id="room-1", member_id="member-peer"
    )
    assert record["grant"] == "winner.room.grant"


def test_late_refresh_cannot_replace_a_reenrolled_route(peers):
    first, second, register = peers
    register(first, "winner.room.grant", expected=_hash("signed.room.grant"))
    with pytest.raises(hosted_rooms.HostedRoomError, match="changed during reconnect"):
        second._rotate_route_grant("room-1", "member-peer", "stale.refresh.grant")
    record = hosted_room_link_records.room_link_record(
        first.db_path, room_id="room-1", member_id="member-peer"
    )
    assert record["grant"] == "winner.room.grant"


def test_concurrent_workers_allow_only_one_reconnect_winner(peers):
    first, second, register = peers
    barrier = threading.Barrier(2)
    outcomes = []

    def reconnect(service, grant):
        barrier.wait(timeout=2)
        try:
            register(service, grant, expected=_hash("signed.room.grant"))
            outcomes.append("accepted")
        except hosted_rooms.HostedRoomError:
            outcomes.append("stale")

    threads = [
        threading.Thread(target=reconnect, args=(first, "first.room.grant")),
        threading.Thread(target=reconnect, args=(second, "second.room.grant")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert sorted(outcomes) == ["accepted", "stale"]
