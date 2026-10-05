"""Focused RoomLink peer-recovery regressions."""

import time
from pathlib import Path

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService

from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING,
    PROFILE,
    ROOM_ID,
    FakeSessionRPC,
    _admit,
    _identity,
    _runtime,
)
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
    """Unreachable until ``reachable`` flips; then it returns the run it already holds."""

    def __init__(self) -> None:
        super().__init__()
        self.reachable = False

    def recover_dispatch(self, **kwargs):
        if self.reachable:
            return super().recover_dispatch(**kwargs)
        self.recoveries.append({**kwargs, "dispatch": dict(kwargs["dispatch"])})
        raise PeerRunsHTTPError(
            "peer RoomLink endpoint is unreachable", retryable=True, not_admitted=True)

    def prepare(self, **kwargs):
        return self.session if self.reachable else super().prepare(**kwargs)


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


def test_unreachable_peer_turn_is_deferred_and_retried_on_its_own_generation(tmp_path: Path):
    """A peer turn whose same-generation recovery keeps failing is deferred, so the room moves
    on. Retry then recovers that same generation first: it fails while the peer is down, and once
    the peer is back it settles generation 1 and never sends generation 2."""
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
    assert {r["dispatch"]["execution_generation"] for r in peer.recoveries} == {1}

    # Retry while the peer is still down keeps the turn deferred at generation 1.
    with pytest.raises(PeerRunsHTTPError):
        service.retry_room_task("room-1", task_id=queued["identity"].task_id)
    assert driver.get_task(db, queued["identity"])["status"] == "deferred"

    # Once the peer is back, Retry recovers generation 1 and never sends generation 2.
    peer.reachable = True
    retried = service.retry_room_task("room-1", task_id=queued["identity"].task_id)
    service.runtime._run_room_once(binding)

    assert (retried["status"], retried["execution_generation"]) == ("settled", 1)
    assert driver.get_task(db, queued["identity"])["status"] == "settled"
    assert {r["dispatch"]["execution_generation"] for r in peer.recoveries} == {1}
    assert {d["execution_generation"] for d in peer.dispatches} == {1}


def _driver_room(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Release room", members=[{"profile": PROFILE, "handle": PROFILE}],
        authority_gateway_id=BINDING.gateway_id, now=time.time())
    return db


def test_contradictory_admission_flags_keep_the_same_attempt_at_lease_expiry(tmp_path: Path):
    db = _driver_room(tmp_path)
    now = [100.0]
    identity = _identity()
    _admit(db, identity)

    class UncertainRPC(FakeSessionRPC):
        def submit(self, **kwargs):
            super().submit(**kwargs)
            raise PeerRunsHTTPError("mixed admission evidence", retryable=True, ambiguous=True, not_admitted=True)

    rpc = UncertainRPC(auto_complete=False)
    runtime = _runtime(db, rpc, clock=lambda: now[0], lease_ttl_seconds=1)
    runtime._run_cycle()
    first = driver.get_task(db, identity)
    assert first["status"] == "running" and first["execution_generation"] == 1
    runtime._run_cycle()
    now[0] += 2
    runtime._run_cycle()
    recovered = driver.get_task(db, identity)
    assert recovered["status"] == "indeterminate" and recovered["execution_generation"] == 1
    assert [params["execution_generation"] for method, params in rpc.calls if method == "submit"] == [1]
