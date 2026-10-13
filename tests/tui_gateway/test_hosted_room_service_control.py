"""Approval identity and peer recovery preserve the admitted room execution."""

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tests.tui_gateway.test_hosted_room_service import _FakeRPC, _RecoveringPeerClient, _server
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


def test_local_room_approval_uses_the_exact_hidden_session(tmp_path):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    service._set_pending_action("room-1", "local", {
        "kind": "approval", "task_id": "task-local-1", "execution_generation": 1,
        "session_id": "local-session", "request_id": "approval-local-1",
        "approval": {"description": "Run focused tests", "command": "pytest -q tests/focused",
                     "choices": ["once", "deny"]},
    })
    assert service.approve_room_task(
        "room-1", member_id="local", task_id="task-local-1", execution_generation=1,
        choice="once", request_id="approval-local-1") == {"resolved": 1}
    assert rpc.approvals == [{
        "session_id": "local-session", "request_id": "approval-local-1", "choice": "once"}]
    assert service.status("room-1")["pending_actions"] == []


def test_stale_local_approval_cannot_resolve_replacement_request(tmp_path):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    action = {
        "kind": "approval", "task_id": "task-local-1", "execution_generation": 1,
        "session_id": "local-session", "approval": {"choices": ["once", "deny"]},
    }
    service._set_pending_action("room-1", "local", {**action, "request_id": "approval-A"})
    service._set_pending_action("room-1", "local", {**action, "request_id": "approval-B"})
    with pytest.raises(RuntimeError, match="no longer pending"):
        service.approve_room_task(
            "room-1", member_id="local", task_id="task-local-1", execution_generation=1,
            choice="once", request_id="approval-A")
    assert rpc.approvals == []
    assert service.status("room-1")["pending_actions"][0]["request_id"] == "approval-B"


def test_peer_recovery_replays_the_same_execution_generation(tmp_path):
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(target_profile="default", installation_id="install-peer", persistent_process=True))
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(), member_id="member-peer",
        target_install_id="install-peer", target_profile="reviewer", capability_digest=catalog.catalog_digest,
        cancellation_scope_id="cancel-room-1", trace_id="trace-room-1", grant="signed.room.grant")
    peer = _RecoveringPeerClient()
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    service.register_peer_route(
        room_id="room-1", member_id="member-peer", route=route, client=peer,
        target_url="https://peer.example.test", catalog=catalog)
    service.create_room(room_id="room-1", name="Peer room", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "member-peer", "profile": "reviewer", "handle": "reviewer", "target": {
            "kind": "peer", "peer_id": "peer-review", "installation_id": "install-peer",
            "profile": "reviewer", "capability_digest": catalog.catalog_digest}},
    ])
    identity = driver.TaskIdentity("room-1", "task-1", "thread-1", "turn-1")
    service._resolve_member_transport(service.bindings()[0], {
        "identity": identity, "status": "indeterminate", "execution_generation": 1,
        "payload": {"target_member_id": "member-peer", "target_profile": "reviewer",
                    "source_event_seq": 9, "prompt": "Recover the accepted review."},
    })
    assert len(peer.recoveries) == 1
    recovered = peer.recoveries[0]["dispatch"]
    assert recovered["task_id"] == "task-1"
    assert recovered["execution_generation"] == 1
    assert recovered["prompt"] == "Recover the accepted review."
