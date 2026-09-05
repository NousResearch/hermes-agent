"""Cross-process approval recovery for hosted Group Chat messaging."""

from __future__ import annotations

import time

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_rooms
from tests.tui_gateway.hosted_room_service_fixtures import _FakeRPC, _server
from tui_gateway.hosted_room_service import HostedRoomService


def _create_local_room(service: HostedRoomService) -> None:
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Approval room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )


def test_cross_process_messaging_approval_is_consumed_by_room_worker(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    action = {
        "kind": "approval",
        "task_id": "task-approval-1",
        "execution_generation": 2,
        "session_id": "ops-session",
        "request_id": "request-approval-1",
        "approval": {
            "description": "Run focused tests",
            "command": "pytest -q tests/focused",
            "choices": ["once", "deny"],
        },
    }
    service._set_pending_action("room-1", "ops", action)
    pending = approvals.list_pending_approvals(db, room_id="room-1")[0]
    approvals.submit_approval(
        db,
        service=None,
        command_id="messaging-approval-1",
        pending=pending,
        choice="once",
    )

    recovered = HostedRoomService(_server(), db_path=db)
    recovered_rpc = _FakeRPC()
    recovered.rpc = recovered_rpc
    recovered.runtime.rpc = recovered_rpc
    recovered._apply_pending_control_approvals(recovered.bindings()[0])

    assert recovered_rpc.approvals == [
        {
            "session_id": "ops-session",
            "request_id": "request-approval-1",
            "choice": "once",
        }
    ]
    assert approvals.list_pending_approval_commands(db, room_id="room-1") == []
    assert approvals.list_pending_approvals(db, room_id="room-1") == []


def test_zero_resolution_keeps_messaging_approval_pending(tmp_path):
    class ZeroResolutionRPC(_FakeRPC):
        def approve(self, **_kwargs):
            return {"resolved": 0}

    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = ZeroResolutionRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    _create_local_room(service)
    service._set_pending_action(
        "room-1",
        "ops",
        {
            "kind": "approval",
            "task_id": "task-1",
            "execution_generation": 2,
            "session_id": "member-session",
            "request_id": "request-1",
            "approval": {
                "description": "Run focused tests",
                "command": "pytest -q tests/focused",
                "choices": ["once", "deny"],
            },
        },
    )

    with pytest.raises(RuntimeError, match="did not resolve"):
        service.approve_room_task(
            "room-1",
            member_id="ops",
            task_id="task-1",
            execution_generation=2,
            choice="once",
            request_id="request-1",
        )

    assert service.status("room-1")["pending_actions"][0]["request_id"] == "request-1"
    assert (
        approvals.list_pending_approvals(db, room_id="room-1")[0]["request_id"]
        == "request-1"
    )


def test_approval_is_fenced_to_the_authority_epoch(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Authority fence",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service._set_pending_action(
        "room-1",
        "ops",
        {
            "kind": "approval",
            "task_id": "task-1",
            "execution_generation": 1,
            "session_id": "ops-session",
            "request_id": "request-1",
            "approval": {"choices": ["once", "deny"]},
        },
    )
    room = hosted_rooms.room_state(db, room_id="room-1")
    hosted_rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id=str(room["authority_gateway_id"]),
        expected_epoch=int(room["authority_epoch"]),
        new_gateway_id="install:new-authority",
        event_id="authority-transfer-1",
    )

    with pytest.raises(
        approvals.MessagingApprovalTerminalError, match="authority changed"
    ):
        service.approve_room_task(
            "room-1",
            member_id="ops",
            task_id="task-1",
            execution_generation=1,
            choice="once",
            request_id="request-1",
        )

    assert rpc.approvals == []


def test_room_disappearance_terminally_completes_queued_decision(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    _create_local_room(service)
    service._set_pending_action(
        "room-1",
        "ops",
        {
            "kind": "approval",
            "task_id": "task-1",
            "execution_generation": 1,
            "session_id": "ops-session",
            "request_id": "request-1",
            "approval": {"choices": ["once", "deny"]},
        },
    )
    pending = approvals.list_pending_approvals(db, room_id="room-1")[0]
    result = approvals.submit_approval(
        db,
        service=None,
        command_id="approval-command-1",
        pending=pending,
        choice="once",
    )
    assert result["queued"] is True
    room = hosted_rooms.room_state(db, room_id="room-1")
    hosted_rooms.disband_room(
        db,
        room_id="room-1",
        expected_gateway_id=str(room["authority_gateway_id"]),
        expected_epoch=int(room["authority_epoch"]),
    )

    service.bindings()
    result = approvals.approval_command(
        db,
        command_id="approval-command-1",
    )

    assert result["state"] == "completed"
    assert "no longer available" in result["result_text"]
    assert approvals.list_pending_approval_commands(db, room_id="room-1") == []


def test_late_pending_callback_keeps_its_original_authority_epoch(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    _create_local_room(service)
    original = service.bindings()[0]
    room = hosted_rooms.room_state(db, room_id="room-1")
    hosted_rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id=str(room["authority_gateway_id"]),
        expected_epoch=int(room["authority_epoch"]),
        new_gateway_id="install:new-authority",
        event_id="authority-transfer-1",
    )
    task = {
        "identity": driver.TaskIdentity(
            "room-1",
            "task-1",
            "thread-1",
            "turn-1",
        ),
        "execution_generation": 1,
        "payload": {"target_member_id": "ops", "target_profile": "ops"},
    }

    service.runtime._report_pending_action(
        original,
        task,
        session_id="ops-session",
        info={
            "pending_approval": {
                "request_id": "request-1",
                "choices": ["once", "deny"],
            }
        },
    )

    assert service.status("room-1")["pending_actions"] == []
    assert approvals.list_pending_approvals(db, room_id="room-1") == []


def test_old_epoch_empty_callback_cannot_clear_newer_approval(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    _create_local_room(service)
    original = service.bindings()[0]
    task = {
        "identity": driver.TaskIdentity(
            "room-1",
            "task-1",
            "thread-1",
            "turn-1",
        ),
        "execution_generation": 1,
        "payload": {"target_member_id": "ops", "target_profile": "ops"},
    }
    room = hosted_rooms.room_state(db, room_id="room-1")
    hosted_rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id=str(room["authority_gateway_id"]),
        expected_epoch=int(room["authority_epoch"]),
        new_gateway_id=str(room["authority_gateway_id"]),
        event_id="authority-transfer-1",
    )
    current = service.bindings()[0]
    current_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=current.gateway_id,
        authority_epoch=current.authority_epoch,
        process_generation=service.runtime.process_generation,
        ttl_seconds=30,
        clock=time.time,
    )
    service.runtime._leases["room-1"] = current_lease
    service.runtime._report_pending_action(
        current,
        task,
        session_id="ops-session",
        info={
            "pending_approval": {
                "request_id": "request-new",
                "choices": ["once", "deny"],
            }
        },
    )

    service.runtime._report_pending_action(
        original,
        task,
        session_id="ops-session",
        info={},
    )

    assert service.status("room-1")["pending_actions"][0]["request_id"] == (
        "request-new"
    )
    assert (
        approvals.list_pending_approvals(db, room_id="room-1")[0]["request_id"]
        == "request-new"
    )


def test_old_process_empty_callback_cannot_clear_replacement_observation(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    _create_local_room(service)
    binding = service.bindings()[0]
    task = {
        "identity": driver.TaskIdentity(
            "room-1",
            "task-1",
            "thread-1",
            "turn-1",
        ),
        "execution_generation": 1,
        "payload": {"target_member_id": "ops", "target_profile": "ops"},
    }
    old_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="worker-old",
        ttl_seconds=30,
        clock=time.time,
    )
    service.runtime._leases["room-1"] = old_lease
    service.runtime.process_generation = "worker-old"
    service.runtime._report_pending_action(
        binding,
        task,
        session_id="ops-session",
        info={
            "pending_approval": {
                "request_id": "request-1",
                "choices": ["once", "deny"],
            }
        },
    )
    driver.release_lease(db, old_lease, clock=time.time)
    new_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="worker-new",
        ttl_seconds=30,
        clock=time.time,
    )
    service.runtime._leases["room-1"] = new_lease
    service.runtime.process_generation = "worker-new"
    service.runtime._report_pending_action(
        binding,
        task,
        session_id="ops-session",
        info={
            "pending_approval": {
                "request_id": "request-1",
                "choices": ["once", "deny"],
            }
        },
    )

    service._set_pending_action(
        "room-1",
        "ops",
        {
            "kind": "approval",
            "authority_gateway_id": binding.gateway_id,
            "authority_epoch": binding.authority_epoch,
            "task_id": "task-1",
            "execution_generation": 1,
            "session_id": "ops-session",
            "observer_generation": "worker-old",
            "observer_lease_generation": old_lease.lease_generation,
            "request_id": "request-old-late",
            "approval": {"choices": ["once", "deny"]},
        },
    )
    with pytest.raises(approvals.MessagingApprovalObservationStale):
        approvals.persist_pending_approval(
            db,
            room_id="room-1",
            member_id="ops",
            action={
                "kind": "approval",
                "authority_gateway_id": binding.gateway_id,
                "authority_epoch": binding.authority_epoch,
                "task_id": "task-1",
                "execution_generation": 1,
                "session_id": "ops-session",
                "observer_generation": "worker-old",
                "observer_lease_generation": old_lease.lease_generation,
                "request_id": "request-old-atomic",
                "approval": {"choices": ["once", "deny"]},
            },
        )
    service._set_pending_action(
        "room-1",
        "ops",
        {
            "kind": "approval_clear",
            "authority_gateway_id": binding.gateway_id,
            "authority_epoch": binding.authority_epoch,
            "task_id": "task-1",
            "execution_generation": 1,
            "session_id": "ops-session",
            "observer_generation": "worker-old",
        },
    )

    assert service._pending_actions[("room-1", "ops")]["observer_generation"] == (
        "worker-new"
    )
    assert (
        approvals.list_pending_approvals(db, room_id="room-1")[0]["observer_generation"]
        == "worker-new"
    )


def test_new_worker_clear_retires_hydrated_old_observation(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    _create_local_room(service)
    binding = service.bindings()[0]
    task = {
        "identity": driver.TaskIdentity(
            "room-1",
            "task-1",
            "thread-1",
            "turn-1",
        ),
        "execution_generation": 1,
        "payload": {"target_member_id": "ops", "target_profile": "ops"},
    }
    old_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="worker-old",
        ttl_seconds=30,
        clock=time.time,
    )
    service.runtime._leases["room-1"] = old_lease
    service.runtime.process_generation = "worker-old"
    service.runtime._report_pending_action(
        binding,
        task,
        session_id="ops-session",
        info={
            "pending_approval": {
                "request_id": "request-1",
                "choices": ["once", "deny"],
            }
        },
    )
    driver.release_lease(db, old_lease, clock=time.time)
    new_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="worker-new",
        ttl_seconds=30,
        clock=time.time,
    )
    service.runtime._leases["room-1"] = new_lease
    service.runtime.process_generation = "worker-new"

    service.runtime._report_pending_action(
        binding,
        task,
        session_id="ops-session",
        info={},
    )

    assert service.status("room-1")["pending_actions"] == []
    assert approvals.list_pending_approvals(db, room_id="room-1") == []


def test_transient_durable_clear_failure_is_retried_before_memory_cleanup(
    tmp_path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Cleanup retry",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service._set_pending_action(
        "room-1",
        "ops",
        {
            "kind": "approval",
            "task_id": "task-1",
            "execution_generation": 1,
            "session_id": "ops-session",
            "request_id": "request-1",
            "approval": {"choices": ["once", "deny"]},
        },
    )
    real_clear = approvals.clear_pending_approval
    calls = []

    def clear_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("transient state.db failure")
        return real_clear(*args, **kwargs)

    monkeypatch.setattr(approvals, "clear_pending_approval", clear_once)

    assert service.approve_room_task(
        "room-1",
        member_id="ops",
        task_id="task-1",
        execution_generation=1,
        choice="once",
        request_id="request-1",
    ) == {"resolved": 1}
    assert len(calls) == 2
    assert service.status("room-1")["pending_actions"] == []
    assert approvals.list_pending_approvals(db, room_id="room-1") == []


def test_pending_callback_keeps_memory_until_durable_clear_succeeds(
    tmp_path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    _create_local_room(service)
    service._set_pending_action(
        "room-1",
        "ops",
        {
            "kind": "approval",
            "task_id": "task-1",
            "execution_generation": 1,
            "session_id": "member-session",
            "request_id": "request-1",
            "approval": {"choices": ["once", "deny"]},
        },
    )
    real_clear = approvals.clear_pending_approval
    monkeypatch.setattr(
        approvals,
        "clear_pending_approval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk busy")),
    )

    with pytest.raises(OSError, match="disk busy"):
        service._set_pending_action("room-1", "ops", None)

    assert service.status("room-1")["pending_actions"][0]["request_id"] == "request-1"
    monkeypatch.setattr(approvals, "clear_pending_approval", real_clear)
    service._set_pending_action("room-1", "ops", None)
    assert service.status("room-1")["pending_actions"] == []


def test_failed_first_persist_rolls_back_memory_and_retries(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    _create_local_room(service)
    action = {
        "kind": "approval",
        "task_id": "task-1",
        "execution_generation": 1,
        "session_id": "ops-session",
        "request_id": "request-1",
        "approval": {"choices": ["once", "deny"]},
    }
    real_persist = approvals.persist_pending_approval
    calls = []

    def persist_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("state.db busy")
        return real_persist(*args, **kwargs)

    monkeypatch.setattr(approvals, "persist_pending_approval", persist_once)

    with pytest.raises(OSError, match="busy"):
        service._set_pending_action("room-1", "ops", action)
    assert service.status("room-1")["pending_actions"] == []

    service._set_pending_action("room-1", "ops", action)
    assert len(calls) == 2
    assert service.status("room-1")["pending_actions"][0]["request_id"] == "request-1"
    assert (
        approvals.list_pending_approvals(db, room_id="room-1")[0]["request_id"]
        == "request-1"
    )


def test_missing_room_retires_stale_approval_without_contacting_target(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    pending = approvals.persist_pending_approval(
        db,
        room_id="missing-room",
        member_id="ops",
        action={
            "kind": "approval",
            "authority_gateway_id": hosted_rooms.local_authority_gateway_id(),
            "authority_epoch": 1,
            "task_id": "task-1",
            "execution_generation": 1,
            "session_id": "ops-session",
            "request_id": "request-1",
            "approval": {"choices": ["once", "deny"]},
        },
    )
    service._pending_actions[("missing-room", "ops")] = pending

    with pytest.raises(RuntimeError, match="no longer available"):
        service.approve_room_task(
            "missing-room",
            member_id="ops",
            task_id="task-1",
            execution_generation=1,
            choice="once",
            request_id="request-1",
        )

    assert rpc.approvals == []
    assert approvals.list_pending_approvals(db, room_id="missing-room") == []


def test_local_pending_approval_requires_exact_task_generation_and_request(tmp_path):
    class ApprovalRPC(_FakeRPC):
        def approve(self, *, session_id, request_id, choice):
            self.approvals.append((session_id, request_id, choice))
            return {"resolved": 1}

    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = ApprovalRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    _create_local_room(service)
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )
    task = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    binding = service.bindings()[0]
    service.runtime.process_generation = "worker"
    lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="worker",
        ttl_seconds=30,
        clock=time.time,
    )
    service.runtime._leases["room-1"] = lease
    driver.start_task(
        db,
        task["identity"],
        lease,
        expected_cancel_generation=0,
        clock=time.time,
    )
    task = driver.get_task(db, task["identity"])
    service.runtime._report_pending_action(
        binding,
        task,
        session_id="ops-session",
        info={
            "pending_approval": {
                "request_id": "approval-1",
                "choices": ["once", "always", "deny"],
            }
        },
    )

    action = service.status("room-1")["pending_actions"][0]
    assert action["member_id"] == "ops"
    assert action["approval"]["choices"] == ["once", "deny"]
    with pytest.raises(RuntimeError, match="no longer pending"):
        service.approve_room_task(
            "room-1",
            member_id="ops",
            task_id=task["identity"].task_id,
            execution_generation=1,
            choice="once",
            request_id="wrong-request",
        )

    assert service.approve_room_task(
        "room-1",
        member_id="ops",
        task_id=task["identity"].task_id,
        execution_generation=1,
        choice="once",
        request_id="approval-1",
    ) == {"resolved": 1}
    assert rpc.approvals == [("ops-session", "approval-1", "once")]
    assert service.status("room-1")["pending_actions"] == []


def test_local_room_approval_uses_the_exact_hidden_session(tmp_path):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    _create_local_room(service)
    service._set_pending_action(
        "room-1",
        "ops",
        {
            "kind": "approval",
            "task_id": "task-local-1",
            "execution_generation": 1,
            "session_id": "local-session",
            "request_id": "approval-local-1",
            "approval": {
                "description": "Run focused tests",
                "command": "pytest -q tests/focused",
                "choices": ["once", "deny"],
            },
        },
    )

    assert service.approve_room_task(
        "room-1",
        member_id="ops",
        task_id="task-local-1",
        execution_generation=1,
        choice="once",
        request_id="approval-local-1",
    ) == {"resolved": 1}
    assert rpc.approvals == [
        {
            "session_id": "local-session",
            "request_id": "approval-local-1",
            "choice": "once",
        }
    ]
    assert service.status("room-1")["pending_actions"] == []


def test_stale_local_approval_cannot_resolve_replacement_request(tmp_path):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    _create_local_room(service)
    action = {
        "kind": "approval",
        "task_id": "task-local-1",
        "execution_generation": 1,
        "session_id": "local-session",
        "approval": {"choices": ["once", "deny"]},
    }
    service._set_pending_action("room-1", "ops", {**action, "request_id": "approval-A"})
    service._set_pending_action("room-1", "ops", {**action, "request_id": "approval-B"})

    with pytest.raises(RuntimeError, match="no longer pending"):
        service.approve_room_task(
            "room-1",
            member_id="ops",
            task_id="task-local-1",
            execution_generation=1,
            choice="once",
            request_id="approval-A",
        )

    assert rpc.approvals == []
    assert service.status("room-1")["pending_actions"][0]["request_id"] == (
        "approval-B"
    )
