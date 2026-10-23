"""Focused RoomLink peer-recovery regressions."""

from tests.tui_gateway.test_hosted_room_driver_runtime import (
    db as db_fixture,
    FakeSessionRPC,
    _admit,
    _identity,
    _runtime,
)
from tests.tui_gateway.test_hosted_room_service import (
    _FakePeerClient,
    _peer_service,
)
from pathlib import Path

from gateway import hosted_room_driver as driver
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_service import HostedRoomService



db = db_fixture


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


def _peer_room(db: Path, peer: _FakePeerClient) -> HostedRoomService:
    service = _peer_service(db, peer)
    route = service.peer_routes[("room-1", "member-peer")]
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
                    "capability_digest": route.capability_digest,
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


def test_contradictory_admission_flags_keep_the_same_attempt_at_lease_expiry(db):
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
