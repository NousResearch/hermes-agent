"""Approval projections must follow the durable attempt, including Stop without another poll."""

from pathlib import Path
import time

import pytest

from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService
from tests.tui_gateway.test_hosted_room_service import _ApprovalPeerClient, _FakeRPC, _server

from gateway import hosted_room_driver as driver, hosted_rooms
from tests.tui_gateway.test_hosted_room_stop_replay import _service


@pytest.fixture
def pending_approval(tmp_path):
    db = tmp_path / "state.db"
    service = _service(db)
    service.create_room(room_id="room", name="Approval lifetime", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"}])
    service.send(room_id="room", event_id="ask", payload={"text": "@ops Inspect", "thread_id": "ask"})
    task, = driver.list_tasks(db, room_id="room", status="queued")
    binding, = service.bindings()
    lease = driver.acquire_lease(
        db, room_id="room", gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch,
        process_generation="worker", ttl_seconds=30, clock=time.time)
    driver.start_task(db, task["identity"], lease, expected_cancel_generation=0, clock=time.time)
    task = driver.get_task(db, task["identity"])
    service.runtime._report_pending_action(task, session_id="ops-session", info={"pending_approval": {
        "request_id": "approval", "command": "echo example", "choices": ["once", "deny"]}})
    assert service.status("room")["pending_actions"][0]["request_id"] == "approval"
    return service, task, lease


def test_stop_withdraws_cached_approval_before_and_after_acknowledgement(pending_approval):
    service, task, _lease = pending_approval
    db = service.db_path
    stopping = driver.begin_task_cancel(
        db, task["identity"], cancel_id="stop", expected_cancel_generation=0, clock=time.time)
    assert service.status("room")["pending_actions"] == []
    driver.complete_task_cancel(
        db, task["identity"], cancel_id="stop",
        expected_cancel_generation=stopping["cancel_generation"], clock=time.time)
    # No new RPC info poll clears the cached action after cancellation.
    assert service.status("room")["pending_actions"] == []


def test_deferred_attempt_keeps_its_live_approval(pending_approval):
    service, task, lease = pending_approval
    db = service.db_path
    def clock():
        return lease.expires_at + 1
    recovered = driver.acquire_lease(
        db, room_id="room", gateway_id=lease.gateway_id, authority_epoch=lease.authority_epoch,
        process_generation="recovered", ttl_seconds=30, clock=clock)
    driver.recover_room(db, recovered, clock=clock)
    assert driver.get_task(db, task["identity"])["status"] == "indeterminate"
    assert any(a.get("request_id") == "approval" for a in service.status("room")["pending_actions"])
    driver.defer_indeterminate_task(
        db, task["identity"], recovered, expected_execution_generation=task["execution_generation"],
        expected_cancel_generation=0, reason="observation_lost", clock=clock)
    assert driver.get_task(db, task["identity"])["status"] == "deferred"
    assert any(a.get("request_id") == "approval" for a in service.status("room")["pending_actions"])
    service.rpc = rpc = _FakeRPC()
    assert service.approve_room_task(
        "room", member_id="ops", task_id=task["identity"].task_id,
        execution_generation=task["execution_generation"], request_id="approval", choice="once") == {"resolved": 1}
    assert rpc.approvals == [{"session_id": "ops-session", "request_id": "approval", "choice": "once"}]


@pytest.mark.parametrize("peer", [False, True])
def test_stopped_approval_cannot_dispatch_and_is_retired(pending_approval, monkeypatch, peer):
    from types import SimpleNamespace
    from unittest.mock import Mock

    service, task, _lease = pending_approval
    rpc = service.rpc = service.runtime.rpc = _FakeRPC()
    peer_client = SimpleNamespace(approve_receipt=Mock(return_value={"resolved": 1}))
    if peer:
        service.peer_routes[("room", "ops")] = SimpleNamespace(grant="grant")
        service.peer_clients[("room", "ops")] = peer_client
    # The transport has not acknowledged interruption: the original prompt can still be open.
    for hook in ("_interrupt_stopping_task", "_settle_stopping_completion", "_peer_stop_acknowledged"):
        monkeypatch.setattr(service.runtime, hook, lambda *_args: False)
    service.stop_room("room", cancel_id="stop")
    assert driver.get_task(service.db_path, task["identity"])["status"] == "stopping"
    with pytest.raises(RuntimeError, match="room approval is no longer pending"):
        service.approve_room_task(
            "room", member_id="ops", task_id=task["identity"].task_id,
            execution_generation=task["execution_generation"], request_id="approval", choice="once")
    assert rpc.approvals == []
    peer_client.approve_receipt.assert_not_called()
    assert ("room", "ops") not in service._pending_actions



def test_peer_approval_is_scoped_visible_and_resolvable(tmp_path: Path, monkeypatch):
    db = tmp_path / "state.db"
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
    peer = _ApprovalPeerClient()
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
            {
                "member_id": "default",
                "profile": "default",
                "handle": "hermes",
            },
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
            }
        ],
    )
    identity = driver.TaskIdentity("room-1", "task-1", "thread-1", "turn-1")
    monkeypatch.setattr(driver, "list_tasks", lambda *_a, **_kw: [{
        "identity": identity, "execution_generation": 2, "status": "running",
        "payload": {"target_member_id": "member-peer", "target_profile": "reviewer"}}])
    transport = service._resolve_member_transport(
        service.bindings()[0],
        {
            "identity": identity,
            "execution_generation": 2,
            "payload": {
                "target_member_id": "member-peer",
                "target_profile": "reviewer",
                "source_event_seq": 1,
            },
        },
    )

    status = transport.info(
        profile="reviewer",
        session_id="peer-group-session",
        source="bot_room",
    )
    assert status["status"] == "waiting_for_approval"
    service._set_pending_action(
        "room-1",
        "member-peer",
        {
            "kind": "approval",
            "task_id": status["task_id"],
            "execution_generation": status["execution_generation"],
            "run_id": status["run_id"],
            "session_id": "peer-group-session",
            "request_id": "req-peer-1",
            "approval": status["approval"],
        },
    )
    pending = service.status("room-1")["pending_actions"]
    assert pending == [
        {
            "kind": "approval",
            "task_id": "task-1",
            "execution_generation": 2,
            "run_id": "run-peer-1",
            "session_id": "peer-group-session",
            "request_id": "req-peer-1",
            "approval": {
                "description": "Run the focused tests",
                "command": "pytest -q tests/focused",
                "choices": ["once", "deny"],
            },
            "member_id": "member-peer",
        }
    ]

    assert service.approve_room_task(
        "room-1",
        member_id="member-peer",
        task_id="task-1",
        execution_generation=2,
        choice="once",
        request_id="req-peer-1",
    ) == {"resolved": 1}
    assert peer.approvals == [
        {
            "task_id": "task-1",
            "execution_generation": 2,
            "request_id": "req-peer-1",
            "choice": "once",
            "grant": "signed.room.grant",
        }
    ]
    assert service.status("room-1")["pending_actions"] == []



def test_stale_local_approval_cannot_resolve_replacement_request(tmp_path: Path, monkeypatch):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    monkeypatch.setattr(driver, "list_tasks", lambda *_a, **_kw: [{
        "identity": driver.TaskIdentity("room-1", "task-local-1", "thread-1", "turn-1"),
        "execution_generation": 1, "status": "running", "payload": {"target_profile": "local"}}])
    action = {
        "kind": "approval",
        "task_id": "task-local-1",
        "execution_generation": 1,
        "session_id": "local-session",
        "approval": {"choices": ["once", "deny"]},
    }
    service._set_pending_action(
        "room-1", "local", {**action, "request_id": "approval-A"}
    )
    service._set_pending_action(
        "room-1", "local", {**action, "request_id": "approval-B"}
    )

    with pytest.raises(RuntimeError, match="no longer pending"):
        service.approve_room_task(
            "room-1",
            member_id="local",
            task_id="task-local-1",
            execution_generation=1,
            choice="once",
            request_id="approval-A",
        )

    assert rpc.approvals == []
    assert service.status("room-1")["pending_actions"][0]["request_id"] == (
        "approval-B"
    )


def test_allow_paused_before_dispatch_cannot_answer_after_a_concurrent_stop(pending_approval, monkeypatch):
    """An Allow already validated for dispatch is ordered with another window's Stop."""
    import threading

    service, task, _lease = pending_approval
    statuses_at_dispatch = []

    class _RecordingRPC(_FakeRPC):
        def approve(self, **kwargs):
            statuses_at_dispatch.append(driver.get_task(service.db_path, task["identity"])["status"])
            return super().approve(**kwargs)

    service.rpc = service.runtime.rpc = _RecordingRPC()
    for hook in ("_interrupt_stopping_task", "_settle_stopping_completion", "_peer_stop_acknowledged"):
        monkeypatch.setattr(service.runtime, hook, lambda *_args: False)
    read, stop_progress = threading.Event(), threading.Event()
    approver = threading.current_thread()
    dispatch = service._dispatch_approval

    def paused_dispatch(*args, **kwargs):
        read.set()
        assert stop_progress.wait(10), "Stop neither finished nor waited for this Allow"
        return dispatch(*args, **kwargs)

    class _SignallingLock:
        """Stop waiting on this lock is as far as it can get while the Allow holds it."""
        def __init__(self, lock):
            self._lock = lock

        def __enter__(self):
            if not self._lock.acquire(blocking=False):
                if threading.current_thread() is not approver:
                    stop_progress.set()
                self._lock.acquire()
            return self

        def __exit__(self, *exc):
            self._lock.release()

    monkeypatch.setattr(service, "_dispatch_approval", paused_dispatch)
    signalling = service._approval_stop_locks["room"] = _SignallingLock(threading.RLock())
    stopper = threading.Thread(target=lambda: (read.wait(10), service.stop_room("room", cancel_id="stop"),
                                               stop_progress.set()))
    stopper.start()
    try:
        result = service.approve_room_task(
            "room", member_id="ops", task_id=task["identity"].task_id,
            execution_generation=task["execution_generation"], request_id="approval", choice="once")
    except RuntimeError as exc:
        result = str(exc)
    stopper.join(10)
    assert not stopper.is_alive()
    # Either the member got its answer while the attempt still ran, or the Allow was refused.
    assert statuses_at_dispatch in ([], ["running"]), (statuses_at_dispatch, result)
    assert driver.get_task(service.db_path, task["identity"])["status"] == "stopping"
    assert ("room", "ops") not in service._pending_actions
    # Both sides took the planted room lock (the registry holds it only while referenced).
    assert service._approval_stop_locks.get("room") is signalling


def test_every_spelling_of_a_room_shares_its_stop_lock_and_cached_approvals(pending_approval):
    """Storage strips room ids, so a padded id names the same room: same lock, same approvals."""
    import threading

    service, _task, _lease = pending_approval
    held, done = service._approval_stop_lock("room"), threading.Event()
    with held:
        stopper = threading.Thread(target=lambda: (service.stop_room("  room ", cancel_id="stop"), done.set()))
        stopper.start()
        assert not done.wait(0.5), "a padded room id took another lock"
    stopper.join(10)
    assert done.is_set()
    assert ("room", "ops") not in service._pending_actions
    del held
    assert "room" not in service._approval_stop_locks, "an idle room keeps no lock"
