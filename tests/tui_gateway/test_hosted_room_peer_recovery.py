"""Focused RoomLink peer-recovery regressions."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_service import HostedRoomService

from tests.tui_gateway.test_hosted_room_driver_runtime import (
    db as db_fixture,
    FakeSessionRPC,
    _admit,
    _identity,
    _runtime,
)
from gateway import hosted_rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tests.tui_gateway.test_hosted_room_service import _FakePeerClient, _server
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute


def _peer_service(db: Path, peer: _FakePeerClient, *, grant="signed.room.grant") -> HostedRoomService:
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(target_profile="default", installation_id="install-peer", persistent_process=True))
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(), member_id="member-peer",
        target_install_id="install-peer", target_profile="reviewer", capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id="cancel-room-1", trace_id="trace-room-1", grant=grant)
    service = HostedRoomService(_server(), db_path=db)
    service.register_peer_route(
        room_id="room-1", member_id="member-peer", route=route, client=peer,
        target_url="https://peer.example.test", catalog=catalog)
    return service


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


def _crashed_turn(tmp_path: Path, peer: _FakePeerClient, *, text: str = "@reviewer check this",
                  recover: bool = True, **runtime):
    """A room turn that a crashed home admitted; ``now[0]`` drives the clock (102 on return). With
    ``recover``, this home has taken the room over and found that turn uncertain."""
    now = [100.0]
    db = tmp_path / "state.db"
    service = _peer_room(db, peer)
    service.runtime.clock = lambda: now[0]
    for name, value in {"lease_ttl_seconds": 1000, "indeterminate_defer_seconds": 5, **runtime}.items():
        setattr(service.runtime, name, value)
    service.send(room_id="room-1", event_id="user-1", payload={"text": text, "thread_id": "thread-1"})
    task, = driver.list_tasks(db, room_id="room-1", status="queued")
    binding = service.bindings()[0]
    crashed = driver.acquire_lease(db, room_id="room-1", gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation="crashed-home", ttl_seconds=1,
        clock=lambda: now[0])
    driver.start_task(db, task["identity"], crashed, expected_cancel_generation=0, clock=lambda: now[0])
    now[0], lease = 102.0, None
    if recover:
        lease = service.runtime._ensure_lease(binding)
        driver.recover_room(db, lease, clock=lambda: now[0])
    return SimpleNamespace(service=service, db=db, task=task, identity=task["identity"], binding=binding,
                           lease=lease, now=now)


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
    peer = _UnreachablePeerClient()
    turn = _crashed_turn(tmp_path, peer, recover=False, lease_ttl_seconds=30)
    service, db, binding, now, queued = turn.service, turn.db, turn.binding, turn.now, turn.task
    assert queued["payload"]["target_member_id"] == "member-peer"
    service.runtime._run_room_once(binding)
    assert driver.get_task(db, queued["identity"])["status"] == "indeterminate"
    # Each cycle replays the uncertain admission once, not once per route lookup.
    replays = len(peer.recoveries)
    service.runtime._reconcile_indeterminate(binding, service.runtime._ensure_lease(binding))
    assert len(peer.recoveries) == replays + 1

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



def _deferred_peer_turn(tmp_path: Path, peer: _FakePeerClient):
    """A peer turn that a crashed home admitted and recovery then deferred."""
    turn = _crashed_turn(tmp_path, peer)
    recovered = driver.get_task(turn.db, turn.identity)
    turn.service.runtime._fenced(driver.defer_indeterminate_task, None, recovered, turn.lease, reason="member_unavailable")
    return turn.service, turn.db, turn.identity


def test_stop_of_a_deferred_peer_turn_whose_gateway_is_down_frees_the_room(tmp_path: Path):
    """Deferral already let the room move on. Stopping that turn while its member stays
    unreachable reports nothing stopped (the run may still be going there), never replays its
    admission, and does not hold the next message. Disband waits for that member's Stop."""
    peer = _UnreachablePeerClient()
    service, db, identity = _deferred_peer_turn(tmp_path, peer)
    peer.reachable, replays = True, len(peer.recoveries)

    assert service.stop_room("room-1", cancel_id="stop-deferred") == 0
    with pytest.raises(RuntimeError, match="still stopping"):
        service.stop_room("room-1", cancel_id="room-disbanded", require_acknowledged=True)
    assert driver.get_task(db, identity)["status"] == "deferred"
    assert len(peer.recoveries) == replays and peer.dispatches == []
    service.send(room_id="room-1", event_id="user-2",
                 payload={"text": "@hermes next question", "thread_id": "thread-2"})
    queued = driver.list_tasks(db, room_id="room-1", status="queued")
    assert [t["payload"]["target_member_id"] for t in queued] == ["default"]


def test_stop_of_a_deferred_peer_turn_is_reported_once_its_gateway_confirms(tmp_path: Path):
    """With the run's receipt in hand, Stop asks the member's gateway to stop that exact run, and
    the turn is cancelled only when that gateway answers the run has ended."""
    stops, answer = [], {"status": "running"}

    class Stoppable(_RecoveringPeerClient):
        def stop_receipt(self, **kwargs):
            stops.append((kwargs["task_id"], kwargs["execution_generation"]))
            return dict(answer)

    service, db, identity = _deferred_peer_turn(tmp_path, Stoppable())
    assert service.stop_room("room-1", cancel_id="stop-1") == 0
    assert driver.get_task(db, identity)["status"] == "deferred"
    answer["status"] = "cancelled"
    assert service.stop_room("room-1", cancel_id="stop-2") == 1
    assert driver.get_task(db, identity)["status"] == "cancelled"
    assert stops == [(identity.task_id, 1)] * 2


def test_an_unconfirmed_peer_stop_holds_the_room_only_for_the_deferral_window(tmp_path: Path):
    """Stop of an uncertain peer turn waits for its gateway's acknowledgement, but a gateway that
    stays unreachable holds the room only for the deferral window. The turn is then deferred, never
    reported cancelled, and Disband keeps waiting for that member's Stop."""
    turn = _crashed_turn(tmp_path, _UnreachablePeerClient())
    service, db, task, binding, lease, now = turn.service, turn.db, turn.task, turn.binding, turn.lease, turn.now

    # Disband waits for the member gateway's Stop, as for this gateway's own work.
    with pytest.raises(RuntimeError, match="still stopping"):
        service.stop_room("room-1", cancel_id="stop-uncertain", require_acknowledged=True)
    assert driver.get_task(db, task["identity"])["status"] == "stopping"
    assert service.runtime._retry_stopping_tasks(binding, lease) is True

    now[0] = 110.0
    assert service.runtime._retry_stopping_tasks(binding, lease) is False
    released = driver.get_task(db, task["identity"])
    assert (released["status"], released["execution_generation"]) == ("deferred", 1)
    assert released["result"] == {"reason": "member_unavailable", "retryable": True}
    with pytest.raises(RuntimeError, match="still stopping"):
        service.stop_room("room-1", cancel_id="room-disbanded", require_acknowledged=True)

@pytest.mark.parametrize("elapsed", [5, 61], ids=["before-deadline", "after-deadline"])
def test_peer_completion_is_observed_after_the_first_recovery_probe(tmp_path, monkeypatch, elapsed):
    """A peer may finish between probes, before the deferral window expires."""
    completed, peer = [False], _RecoveringPeerClient()
    turn = _crashed_turn(tmp_path, peer, indeterminate_defer_seconds=60)
    service, db, task, binding, lease, now = turn.service, turn.db, turn.task, turn.binding, turn.lease, turn.now
    monkeypatch.setattr(peer, "prepare", lambda **_: peer.session)
    monkeypatch.setattr(peer, "history", lambda **_: [{
        "role": "assistant", "task_id": task["identity"].task_id, "execution_generation": 1,
        "status": "settled", "message_id": "peer-complete", "content": "Recovered review",
    }] if completed[0] else [])
    monkeypatch.setattr(peer, "status", lambda **_: {
        "active": not completed[0], "status": "completed" if completed[0] else "running",
        "task_id": task["identity"].task_id, "execution_generation": 1,
    })

    assert service.runtime._reconcile_indeterminate(binding, lease)
    completed[0] = True
    now[0] += elapsed
    service.runtime._reconcile_indeterminate(binding, lease)
    settled = driver.get_task(db, task["identity"])
    assert (settled["status"], settled["execution_generation"]) == ("settled", 1)
    assert settled["result"]["text"] == "Recovered review"
    assert {entry["dispatch"]["execution_generation"] for entry in peer.recoveries} == {1}


def test_a_recovered_peer_turn_still_running_is_stopped_at_its_execution_deadline(tmp_path, monkeypatch):
    """A peer run that keeps going holds the room only until the turn's own time limit, measured
    from when it started; then it is stopped through the exact Stop protocol and fails."""
    stops, peer = [], _RecoveringPeerClient()
    turn = _crashed_turn(tmp_path, peer, indeterminate_defer_seconds=60, turn_timeout_seconds=300)
    service, db, identity, binding, lease, now = (
        turn.service, turn.db, turn.identity, turn.binding, turn.lease, turn.now)
    monkeypatch.setattr(peer, "prepare", lambda **_: peer.session)
    monkeypatch.setattr(peer, "history", lambda **_: [])
    monkeypatch.setattr(peer, "status", lambda **_: {
        "active": not stops, "status": "cancelled" if stops else "running",
        "task_id": identity.task_id, "execution_generation": 1})
    peer.stop_receipt = lambda **kwargs: stops.append(kwargs) or {"status": "cancelled"}

    # Past the deferral window but inside the turn's time limit, a running peer keeps the room waiting.
    now[0] = 170.0
    assert service.runtime._reconcile_indeterminate(binding, lease) is True
    assert driver.get_task(db, identity)["status"] == "indeterminate" and not stops

    now[0] = 401.0
    service.runtime._reconcile_indeterminate(binding, lease)
    stopped = driver.get_task(db, identity)
    assert [(s["task_id"], s["execution_generation"]) for s in stops] == [(identity.task_id, 1)]
    assert (stopped["status"], stopped["execution_generation"]) == ("failed", 1)
    assert stopped["result"]["reason_code"] == "turn_deadline_exceeded"
    assert "room-1" not in service.runtime.status()["blocked_rooms"]


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


def test_deferred_turn_on_an_in_process_route_retries_as_before(tmp_path: Path):
    """Only a member the room's roster places on another gateway replays its generation on Retry.
    A local member reached through its own session transport (the gateway's canonical member
    route) is requeued at once, as before."""
    turn = _crashed_turn(tmp_path, _RecoveringPeerClient(), text="@hermes check this", lease_ttl_seconds=30)
    service, db, task, binding, lease, now = turn.service, turn.db, turn.task, turn.binding, turn.lease, turn.now
    peer_route, member_route = service.runtime.transport_resolver, FakeSessionRPC(auto_complete=False)
    service.runtime.transport_resolver = lambda binding, task: (
        member_route if task["payload"].get("target_member_id") == "default" else peer_route(binding, task))
    assert task["payload"]["target_member_id"] == "default"
    assert service.runtime._reconcile_indeterminate(binding, lease) is True
    now[0] = 108.0
    assert service.runtime._reconcile_indeterminate(binding, lease) is False
    assert driver.get_task(db, task["identity"])["status"] == "deferred"

    retried = service.runtime.retry_indeterminate(task["identity"])

    assert retried["status"] == "queued"


def test_a_failing_route_lookup_still_lets_an_uncertain_turn_reach_its_deadline(tmp_path: Path):
    """Resolving a member's route can fail during recovery (here a member on this gateway whose
    session route is gone). The failure is recorded and the turn still reaches its deferral
    deadline, instead of aborting every recovery cycle."""
    turn = _crashed_turn(tmp_path, _RecoveringPeerClient(), text="@hermes check this")
    runtime = turn.service.runtime

    def missing_route(binding, task):
        raise RuntimeError("member session route is gone")

    runtime.transport_resolver = missing_route
    assert runtime._reconcile_indeterminate(turn.binding, turn.lease) is True
    turn.now[0] = 108.0
    assert runtime._reconcile_indeterminate(turn.binding, turn.lease) is False
    deferred = driver.get_task(turn.db, turn.identity)
    assert (deferred["status"], deferred["execution_generation"]) == ("deferred", 1)
    assert "route is gone" in str(runtime.status()["last_error"])


def test_a_retried_deferred_peer_turn_gets_its_own_time_limit(tmp_path, monkeypatch):
    """Retry of a turn deferred long ago follows its run on the member's gateway; that attempt's
    time limit counts from the retry, so it is not stopped at once as past the first start's."""
    stops, peer = [], _RecoveringPeerClient()
    service, db, identity = _deferred_peer_turn(tmp_path, peer)
    service.runtime.turn_timeout_seconds = 300
    monkeypatch.setattr(peer, "prepare", lambda **_: peer.session)
    monkeypatch.setattr(peer, "history", lambda **_: [])
    monkeypatch.setattr(peer, "status", lambda **_: {
        "active": True, "status": "running", "task_id": identity.task_id, "execution_generation": 1})
    peer.stop_receipt = lambda **kwargs: stops.append(kwargs) or {"status": "cancelled"}
    now = [100.0 + 7200]  # two hours after the first start
    service.runtime.clock = lambda: now[0]

    retried = service.retry_room_task("room-1", task_id=identity.task_id)
    assert (retried["status"], retried["execution_generation"]) == ("indeterminate", 1)
    binding = service.bindings()[0]
    lease = service.runtime._ensure_lease(binding)
    now[0] += 60
    assert service.runtime._reconcile_indeterminate(binding, lease) is True
    assert driver.get_task(db, identity)["status"] == "indeterminate" and not stops
    now[0] += 300
    service.runtime._reconcile_indeterminate(binding, lease)
    assert [(s["task_id"], s["execution_generation"]) for s in stops] == [(identity.task_id, 1)]
