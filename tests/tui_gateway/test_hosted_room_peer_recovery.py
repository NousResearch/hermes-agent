"""Focused RoomLink peer-recovery regressions."""

import time
from pathlib import Path

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tui_gateway.hosted_room_driver import _stop_route_missing
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


@pytest.mark.parametrize("elapsed", [5, 61], ids=["before-deadline", "after-deadline"])
def test_peer_completion_is_observed_after_the_first_recovery_probe(tmp_path, monkeypatch, elapsed):
    """A peer may finish between probes, before the deferral window expires."""
    now, completed = [100.0], [False]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    service.send(room_id="room-1", event_id="user-1",
                 payload={"text": "@reviewer inspect", "thread_id": "thread-1"})
    task, = driver.list_tasks(db, room_id="room-1", status="queued")
    binding = service.bindings()[0]
    crashed = driver.acquire_lease(db, room_id="room-1", gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation="crashed", ttl_seconds=1, clock=clock)
    driver.start_task(db, task["identity"], crashed, expected_cancel_generation=0, clock=clock)
    now[0] += 2
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(db, lease, clock=clock)
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


def _crash_to_indeterminate(service, db, clock, now):
    service.send(
        room_id="room-1", event_id="user-1",
        payload={"text": "@reviewer check this", "thread_id": "thread-1"})
    queued = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    binding = service.bindings()[0]
    crashed = driver.acquire_lease(
        db, room_id="room-1", gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch,
        process_generation="crashed-home", ttl_seconds=1, clock=clock)
    attempt = driver.start_task(
        db, queued["identity"], crashed, expected_cancel_generation=0, clock=clock)
    now[0] = 102.0
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(db, lease, clock=clock)
    return queued, binding, lease, attempt


def test_deferred_retry_fences_the_room_before_peer_io(tmp_path: Path):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _UnreachablePeerClient()
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 30
    service.runtime.indeterminate_defer_seconds = 5
    queued, binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    now[0] = 108.0
    service.runtime._run_room_once(binding)
    assert driver.get_task(db, queued["identity"])["status"] == "deferred"
    successor = driver.TaskIdentity("room-1", "task-next", "thread-2", "turn-next")
    driver.admit_task(
        db, successor, payload={
            "target_profile": "reviewer", "target_member_id": "member-peer",
            "prompt": "Next turn.", "source_event_seq": 50}, clock=clock)
    seen = []

    def recover_dispatch(**kwargs):
        current = driver.get_task(db, queued["identity"])
        seen.append(current["status"])
        with pytest.raises(driver.InvalidTaskTransitionError, match="prior task"):
            driver.start_task(
                db, successor, service.runtime._leases["room-1"],
                expected_cancel_generation=0, clock=clock)
        raise PeerRunsHTTPError(
            "peer RoomLink endpoint is unreachable", retryable=True, not_admitted=True)

    peer.recover_dispatch = recover_dispatch
    with pytest.raises(PeerRunsHTTPError):
        service.retry_room_task("room-1", task_id=queued["identity"].task_id)
    assert seen == ["indeterminate"]
    assert driver.get_task(db, queued["identity"])["status"] == "deferred"
    assert driver.get_task(db, successor)["status"] == "queued"


def test_recovered_peer_is_not_deferred_when_the_status_probe_fails(tmp_path, monkeypatch):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    service.runtime.indeterminate_defer_seconds = 5
    service.runtime.turn_timeout_seconds = 30
    queued, binding, lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    monkeypatch.setattr(peer, "prepare", lambda **_: peer.session)
    monkeypatch.setattr(peer, "history", lambda **_: [])

    def status(**_kwargs):
        raise PeerRunsHTTPError("status timed out", retryable=True, ambiguous=True)

    monkeypatch.setattr(peer, "status", status)
    now[0] = 108.0
    assert service.runtime._reconcile_indeterminate(binding, lease) is True
    held = driver.get_task(db, queued["identity"])
    assert held["status"] == "indeterminate"
    assert held["execution_generation"] == 1
    assert (held.get("result") or {}).get("reason") != "member_unavailable"


@pytest.mark.parametrize("error_code", ["run_history_retired", "idempotency_key_conflict"])
def test_refused_recovery_settles_instead_of_deferring(tmp_path: Path, error_code: str):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def recover_dispatch(**_kwargs):
        raise PeerRunsHTTPError(
            "peer refused this generation", retryable=True, ambiguous=True, status_code=409,
            error_code=error_code)

    peer.recover_dispatch = recover_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    service.runtime.indeterminate_defer_seconds = 5
    queued, binding, lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    now[0] = 108.0
    assert service.runtime._reconcile_indeterminate(binding, lease) is False
    failed = driver.get_task(db, queued["identity"])
    assert failed["status"] == "failed"
    assert failed["execution_generation"] == 1
    assert failed["result"]["reason"] == error_code
    assert failed["result"]["retryable"] is False


@pytest.mark.parametrize("status_code", [404, 405])
def test_missing_stop_route_releases_the_same_generation(tmp_path: Path, status_code: int):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def cancel_dispatch(**_kwargs):
        raise PeerRunsHTTPError(
            "peer cannot cancel an uncertain admission", ambiguous=True, status_code=status_code)

    peer.cancel_dispatch = cancel_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    queued, _binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    released = service.runtime.cancel(queued["identity"], cancel_id="stop-1")
    assert released["status"] == "deferred"
    assert released["execution_generation"] == 1
    assert released["result"] == {"reason": "peer_stop_unsupported", "retryable": False}


def test_stop_room_reports_unacknowledged_rows_as_stopping(tmp_path: Path):
    db = tmp_path / "state.db"
    service = _peer_room(db, _RecoveringPeerClient())
    service.send(
        room_id="room-1", event_id="user-1",
        payload={"text": "@reviewer check this", "thread_id": "thread-1"})

    def cancel(identity, *, cancel_id):
        task = driver.get_task(db, identity)
        return {**task, "status": "stopping", "cancel_id": cancel_id}

    service.runtime.cancel = cancel
    assert service.stop_room("room-1", cancel_id="stop-1") == (0, 1)


def test_running_waiter_completes_a_cancelled_peer_generation(tmp_path: Path):
    db = _driver_room(tmp_path)
    identity = _identity()
    _admit(db, identity)
    now = [time.time()]
    rpc = FakeSessionRPC(auto_complete=False)
    session_id = rpc.add_session(active=True, task_id=identity.task_id)

    def info(**_kwargs):
        return {
            "active": False, "status": "cancelled", "task_id": identity.task_id,
            "execution_generation": 1}

    rpc.info = info
    runtime = _runtime(
        db, rpc, clock=lambda: now[0], lease_ttl_seconds=30,
        active_poll_interval_seconds=0.05, turn_timeout_seconds=1830)
    lease = driver.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation=runtime.process_generation, ttl_seconds=30, clock=lambda: now[0])
    attempt = driver.start_task(
        db, identity, lease, expected_cancel_generation=0, clock=lambda: now[0])
    started = time.monotonic()
    assert runtime._wait_for_terminal(
        BINDING, profile=PROFILE, session_id=session_id, attempt=attempt, transport=rpc,
        deadline_monotonic=time.monotonic() + 30) is None
    task = driver.get_task(db, identity)
    assert task["status"] == "cancelled"
    assert task["execution_generation"] == 1
    assert time.monotonic() - started < 2


def _next_turn(db, clock):
    successor = driver.TaskIdentity("room-1", "task-next", "thread-2", "turn-next")
    driver.admit_task(
        db, successor, payload={
            "target_profile": "reviewer", "target_member_id": "member-peer",
            "prompt": "Next turn.", "source_event_seq": 50}, clock=clock)
    return successor


def test_explicit_retry_holds_a_generation_whose_recovery_returned(tmp_path, monkeypatch):
    """A status error after recover_dispatch returned must not free the room."""
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    service.runtime.indeterminate_defer_seconds = 5
    service.runtime.turn_timeout_seconds = 30
    queued, _binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    now[0] = 108.0
    service.runtime._run_room_once(service.bindings()[0])
    assert driver.get_task(db, queued["identity"])["status"] == "deferred"
    before = len(peer.recoveries)
    monkeypatch.setattr(peer, "prepare", lambda **_: peer.session)
    monkeypatch.setattr(peer, "history", lambda **_: [])

    def status(**_kwargs):
        raise PeerRunsHTTPError("status timed out", retryable=True, ambiguous=True)

    monkeypatch.setattr(peer, "status", status)
    with pytest.raises(PeerRunsHTTPError):
        service.retry_room_task("room-1", task_id=queued["identity"].task_id)
    assert len(peer.recoveries) > before
    held = driver.get_task(db, queued["identity"])
    assert held["status"] == "indeterminate"
    assert held["execution_generation"] == 1
    successor = _next_turn(db, clock)
    with pytest.raises(driver.InvalidTaskTransitionError, match="prior task"):
        driver.start_task(
            db, successor, service.runtime._leases["room-1"],
            expected_cancel_generation=0, clock=clock)


@pytest.mark.parametrize("error_code", ["run_history_retired", "idempotency_key_conflict"])
def test_explicit_retry_settles_a_refused_generation(tmp_path: Path, error_code: str):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    service.runtime.indeterminate_defer_seconds = 5
    queued, _binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    now[0] = 108.0
    service.runtime._run_room_once(service.bindings()[0])
    assert driver.get_task(db, queued["identity"])["status"] == "deferred"

    def recover_dispatch(**_kwargs):
        raise PeerRunsHTTPError(
            "peer refused this generation", retryable=True, ambiguous=True, status_code=409,
            error_code=error_code)

    peer.recover_dispatch = recover_dispatch
    with pytest.raises(PeerRunsHTTPError):
        service.retry_room_task("room-1", task_id=queued["identity"].task_id)
    failed = driver.get_task(db, queued["identity"])
    assert failed["status"] == "failed"
    assert failed["execution_generation"] == 1
    assert failed["result"]["reason"] == error_code
    assert failed["result"]["retryable"] is False


def test_run_not_found_does_not_release_a_stop(tmp_path: Path):
    """A 404 from the run-id route is not 'this peer has no stop route'."""
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def cancel_dispatch(**_kwargs):
        raise PeerRunsHTTPError(
            "Run not found", status_code=404, error_code="run_not_found")

    peer.cancel_dispatch = cancel_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    queued, _binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    assert _stop_route_missing(PeerRunsHTTPError(
        "missing", status_code=404, error_code="run_not_found")) is False
    stuck = service.runtime.cancel(queued["identity"], cancel_id="stop-1")
    assert stuck["status"] == "stopping"
    assert stuck["execution_generation"] == 1
    successor = _next_turn(db, clock)
    with pytest.raises(driver.InvalidTaskTransitionError, match="prior task"):
        driver.start_task(
            db, successor, service.runtime._leases["room-1"],
            expected_cancel_generation=0, clock=clock)


def test_run_not_active_after_restart_settles_the_stop(tmp_path: Path):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def cancel_dispatch(**_kwargs):
        raise PeerRunsHTTPError(
            "Run is not active in this gateway process", status_code=409, error_code="run_not_active")

    peer.cancel_dispatch = cancel_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    queued, _binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    settled = service.runtime.cancel(queued["identity"], cancel_id="stop-1")
    assert settled["status"] == "failed"
    assert settled["execution_generation"] == 1
    assert settled["result"]["reason"] == "run_not_active"
    assert settled["result"]["retryable"] is False
    successor = _next_turn(db, clock)
    driver.start_task(
        db, successor, service.runtime._leases["room-1"],
        expected_cancel_generation=0, clock=clock)
    assert driver.get_task(db, successor)["status"] == "running"


def test_missing_stop_route_is_not_counted_as_cancelled(tmp_path: Path):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def cancel_dispatch(**_kwargs):
        raise PeerRunsHTTPError(
            "peer cannot cancel an uncertain admission", ambiguous=True, status_code=404)

    peer.cancel_dispatch = cancel_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    queued, _binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    cancelled, stopping = service.stop_room("room-1", cancel_id="stop-1", require_acknowledged=True)
    task = driver.get_task(db, queued["identity"])
    assert task["status"] == "deferred"
    assert task["result"] == {"reason": "peer_stop_unsupported", "retryable": False}
    assert (cancelled, stopping) == (0, 0)
    with pytest.raises(driver.InvalidTaskTransitionError, match="no retryable"):
        service.retry_room_task("room-1", task_id=queued["identity"].task_id)
    actions = service.status("room-1")["pending_actions"]
    assert all(action.get("task_id") != queued["identity"].task_id for action in actions)


def test_recovery_hold_expiry_stops_instead_of_deferring(tmp_path, monkeypatch):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 1000
    service.runtime.indeterminate_defer_seconds = 5
    service.runtime.turn_timeout_seconds = 30
    queued, binding, lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    monkeypatch.setattr(peer, "prepare", lambda **_: peer.session)
    monkeypatch.setattr(peer, "history", lambda **_: [])
    monkeypatch.setattr(
        peer, "status", lambda **_: (_ for _ in ()).throw(
            PeerRunsHTTPError("status timed out", retryable=True, ambiguous=True)))
    now[0] = 108.0
    assert service.runtime._reconcile_indeterminate(binding, lease) is True
    assert driver.get_task(db, queued["identity"])["status"] == "indeterminate"
    now[0] = 140.0
    assert service.runtime._reconcile_indeterminate(binding, lease) is True
    held = driver.get_task(db, queued["identity"])
    assert held["status"] == "stopping"
    assert str(held["cancel_id"]).startswith("deadline:")
    successor = _next_turn(db, clock)
    with pytest.raises(driver.InvalidTaskTransitionError, match="prior task"):
        driver.start_task(
            db, successor, service.runtime._leases["room-1"],
            expected_cancel_generation=0, clock=clock)


def test_ambiguous_recovery_failure_is_held_until_stop(tmp_path: Path):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def recover_dispatch(**_kwargs):
        raise PeerRunsHTTPError(
            "peer timed out after the request was sent", retryable=True, ambiguous=True,
            not_admitted=False)

    peer.recover_dispatch = recover_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 1000
    service.runtime.indeterminate_defer_seconds = 5
    service.runtime.turn_timeout_seconds = 30
    queued, binding, lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    now[0] = 108.0
    assert service.runtime._reconcile_indeterminate(binding, lease) is True
    assert driver.get_task(db, queued["identity"])["status"] == "indeterminate"
    now[0] = 140.0
    assert service.runtime._reconcile_indeterminate(binding, lease) is True
    held = driver.get_task(db, queued["identity"])
    assert held["status"] == "stopping"
    successor = _next_turn(db, clock)
    with pytest.raises(driver.InvalidTaskTransitionError, match="prior task"):
        driver.start_task(
            db, successor, service.runtime._leases["room-1"],
            expected_cancel_generation=0, clock=clock)


def test_retired_stop_settles_instead_of_staying_stopping(tmp_path: Path):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def cancel_dispatch(**_kwargs):
        raise PeerRunsHTTPError(
            "Room authority has advanced.", status_code=409, error_code="run_history_retired")

    peer.cancel_dispatch = cancel_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    queued, _binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    settled = service.runtime.cancel(queued["identity"], cancel_id="stop-1")
    assert settled["status"] == "failed"
    assert settled["result"]["reason"] == "run_history_retired"
    assert settled["result"]["retryable"] is False
    successor = _next_turn(db, clock)
    driver.start_task(
        db, successor, service.runtime._leases["room-1"],
        expected_cancel_generation=0, clock=clock)
    assert driver.get_task(db, successor)["status"] == "running"


def test_run_not_found_stop_settles_after_the_turn_budget(tmp_path: Path):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def cancel_dispatch(**_kwargs):
        raise PeerRunsHTTPError("Run not found", status_code=404, error_code="run_not_found")

    peer.cancel_dispatch = cancel_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 1000
    service.runtime.turn_timeout_seconds = 30
    queued, binding, lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    stuck = service.runtime.cancel(queued["identity"], cancel_id="stop-1")
    assert stuck["status"] == "stopping"
    successor = _next_turn(db, clock)
    with pytest.raises(driver.InvalidTaskTransitionError, match="prior task"):
        driver.start_task(
            db, successor, service.runtime._leases["room-1"],
            expected_cancel_generation=0, clock=clock)
    now[0] = 102.0 + 30.0 + 1.0
    assert service.runtime._retry_stopping_tasks(binding, lease) is False
    settled = driver.get_task(db, queued["identity"])
    assert settled["status"] == "failed"
    assert settled["result"]["reason"] == "run_not_found"
    assert settled["result"]["retryable"] is False
    driver.start_task(
        db, successor, service.runtime._leases["room-1"],
        expected_cancel_generation=0, clock=clock)
    assert driver.get_task(db, successor)["status"] == "running"


def test_run_id_404_without_an_error_code_does_not_release(tmp_path: Path):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def cancel_dispatch(**_kwargs):
        exc = PeerRunsHTTPError("<html>not found</html>", status_code=404)
        exc.run_id_route = True
        raise exc

    peer.cancel_dispatch = cancel_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    queued, _binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    stuck = service.runtime.cancel(queued["identity"], cancel_id="stop-1")
    assert stuck["status"] == "stopping"
    successor = _next_turn(db, clock)
    with pytest.raises(driver.InvalidTaskTransitionError, match="prior task"):
        driver.start_task(
            db, successor, service.runtime._leases["room-1"],
            expected_cancel_generation=0, clock=clock)


def test_failed_stop_is_counted(tmp_path: Path):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    peer = _RecoveringPeerClient()

    def cancel_dispatch(**_kwargs):
        raise PeerRunsHTTPError(
            "Run is not active in this gateway process", status_code=409, error_code="run_not_active")

    peer.cancel_dispatch = cancel_dispatch
    service = _peer_room(db, peer)
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 120
    queued, _binding, _lease, _attempt = _crash_to_indeterminate(service, db, clock, now)
    cancelled, stopping = service.stop_room("room-1", cancel_id="stop-1")
    task = driver.get_task(db, queued["identity"])
    assert task["status"] == "failed"
    assert task["result"]["retryable"] is False
    assert (cancelled, stopping) == (1, 0)
