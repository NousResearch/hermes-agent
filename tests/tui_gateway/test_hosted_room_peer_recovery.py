"""Focused RoomLink peer-recovery regressions."""

from pathlib import Path

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService

from tests.tui_gateway.test_hosted_room_service import (
    _FakePeerClient,
    _server,
)


class _RecoveringPeerClient(_FakePeerClient):
    def __init__(self) -> None:
        super().__init__()
        self.recoveries = []

    def recover_dispatch(self, **kwargs):
        dispatch = dict(kwargs["dispatch"])
        self.recoveries.append({**kwargs, "dispatch": dispatch})
        self.dispatches.append(dispatch)
        return {
            "status": "accepted",
            "task_id": dispatch["task_id"],
            "execution_generation": dispatch["execution_generation"],
            "run_id": "run-recovered",
        }


class _UnreachablePeerClient(_RecoveringPeerClient):
    def recover_dispatch(self, **kwargs):
        self.recoveries.append({**kwargs, "dispatch": dict(kwargs["dispatch"])})
        raise PeerRunsHTTPError(
            "peer RoomLink endpoint is unreachable", retryable=True, not_admitted=True)


def _peer_room(db: Path, peer: _FakePeerClient) -> HostedRoomService:
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(target_profile="default", installation_id="install-peer", persistent_process=True)
    )
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=catalog.catalog_digest,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed.room.grant",
    )
    service = HostedRoomService(_server(), db_path=db)
    service.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=route,
        client=peer,
        target_url="https://peer.example.test",
        catalog=catalog,
    )
    service.create_room(
        room_id="room-1",
        name="Peer room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {
                "member_id": "member-peer",
                "profile": "reviewer",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "peer_id": "peer-review",
                    "installation_id": "install-peer",
                    "profile": "reviewer",
                    "capability_digest": catalog.catalog_digest,
                },
            },
        ],
    )
    return service


def test_peer_recovery_replays_only_indeterminate_generation(tmp_path: Path):
    peer = _RecoveringPeerClient()
    service = _peer_room(tmp_path / "state.db", peer)
    identity = driver.TaskIdentity("room-1", "task-1", "thread-1", "turn-1")
    task = {
        "identity": identity,
        "execution_generation": 1,
        "payload": {
            "target_member_id": "member-peer",
            "target_profile": "reviewer",
            "source_event_seq": 9,
            "prompt": "Recover the accepted review.",
        },
    }

    service._resolve_member_transport(
        service.bindings()[0],
        {**task, "status": "running"},
    )
    assert peer.recoveries == []

    service._resolve_member_transport(
        service.bindings()[0],
        {**task, "status": "indeterminate"},
    )

    assert len(peer.recoveries) == 1
    recovered = peer.recoveries[0]["dispatch"]
    assert recovered["task_id"] == "task-1"
    assert recovered["execution_generation"] == 1
    assert recovered["prompt"] == "Recover the accepted review."


def test_uncertain_peer_turn_is_deferred_while_its_peer_stays_unreachable(tmp_path: Path):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _UnreachablePeerClient()
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 30
    service.runtime.indeterminate_defer_seconds = 5
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@reviewer check this", "thread_id": "thread-1"},
    )
    queued = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    assert queued["payload"]["target_member_id"] == "member-peer"
    binding = service.bindings()[0]
    # A home process admitted the turn and exited before it learned the outcome.
    crashed = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="crashed-home",
        ttl_seconds=1,
        clock=clock,
    )
    driver.start_task(db, queued["identity"], crashed, expected_cancel_generation=0, clock=clock)

    now[0] = 102.0
    service.runtime._run_room_once(binding)
    assert driver.get_task(db, queued["identity"])["status"] == "indeterminate"

    now[0] = 108.0
    service.runtime._run_room_once(binding)

    deferred = driver.get_task(db, queued["identity"])
    assert deferred["status"] == "deferred"
    assert deferred["execution_generation"] == 1
    assert deferred["result"] == {"reason": "member_unavailable", "retryable": True}
    assert peer.recoveries
    assert {r["dispatch"]["execution_generation"] for r in peer.recoveries} == {1}
