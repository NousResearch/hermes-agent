"""Disband cannot falsify a decision already being applied by another worker."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_rooms
from tests.tui_gateway.hosted_room_service_fixtures import _FakeRPC, _server
from tui_gateway.hosted_room_service import HostedRoomService
from tests.gateway.test_hosted_room_messaging_approvals import _action


def test_expiring_untouched_decision_preserves_a_new_task_with_the_same_request_id(
    tmp_path,
):
    db = tmp_path / "state.db"
    pending = approvals.persist_pending_approval(
        db, room_id="room-1", member_id="member-1", action=_action()
    )
    approvals.begin_approval_command(
        db, command_id="old-command", pending=pending, choice="once"
    )
    approvals.persist_pending_approval(
        db,
        room_id="room-1",
        member_id="member-1",
        action=_action(task_id="new-task", execution_generation=3),
    )
    assert approvals.expire_unstarted_approval_command(
        db, command_id="old-command", result="Expired"
    )
    remaining = approvals.list_pending_approvals(db, room_id="room-1")
    assert len(remaining) == 1
    assert remaining[0]["task_id"] == "new-task"
    assert remaining[0]["execution_generation"] == 3


def test_existing_command_table_adds_application_boundary_column(tmp_path):
    db = tmp_path / "state.db"
    pending = approvals.persist_pending_approval(
        db, room_id="room-1", member_id="member-1", action=_action()
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "ALTER TABLE hosted_room_messaging_approval_commands DROP COLUMN application_started_at"
        )
    approvals.begin_approval_command(
        db, command_id="old-command", pending=pending, choice="once"
    )
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT application_started_at FROM hosted_room_messaging_approval_commands WHERE command_id='old-command'"
        ).fetchone()
    assert row == (None,)


@pytest.mark.parametrize("completed", [False, True])
def test_migration_preserves_existing_approval_outcome_uncertainty(tmp_path, completed):
    db = tmp_path / "state.db"
    pending = approvals.persist_pending_approval(
        db, room_id="room-1", member_id="member-1", action=_action()
    )
    approvals.begin_approval_command(
        db, command_id="legacy-command", pending=pending, choice="once"
    )
    if completed:
        approvals.complete_approval_command(
            db, command_id="legacy-command", result="Approved once."
        )
    # Old commands exist before migration and carry no application evidence.
    with sqlite3.connect(db) as conn:
        conn.execute(
            "ALTER TABLE hosted_room_messaging_approval_commands "
            "DROP COLUMN application_started_at"
        )
    assert not approvals.expire_unstarted_approval_command(
        db, command_id="legacy-command", result="Expired"
    )
    receipt = approvals.approval_command(db, command_id="legacy-command")
    replay = approvals.submit_approval(
        db, service=None, command_id="legacy-command", pending=pending, choice="once"
    )
    if completed:
        assert receipt["result_text"] == "Approved once."
        assert replay["applied"] is True
    else:
        assert receipt["application_started_at"] is not None
        assert receipt["state"] == "pending"
        assert replay["queued"] is True
        assert "applied" not in replay


@pytest.mark.parametrize("response_lost", [False, True])
def test_disband_cleanup_preserves_an_inflight_success_receipt(tmp_path, response_lost):
    entered = threading.Event()
    release = threading.Event()

    class AppliedButUnacknowledgedRPC(_FakeRPC):
        def approve(self, *, session_id, request_id, choice):
            self.approvals.append((session_id, request_id, choice))
            entered.set()
            assert release.wait(10), "review test failed to release target reply"
            if response_lost:
                raise TimeoutError("approval response lost")
            return {"resolved": 1}

    db = tmp_path / "state.db"
    owner = HostedRoomService(_server(), db_path=db)
    owner.local_profiles = lambda: ("default", "ops")
    owner.create_room(
        room_id="room-1",
        name="Approval race",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    rpc = AppliedButUnacknowledgedRPC()
    owner.rpc = owner.runtime.rpc = rpc
    owner.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )
    binding = owner.bindings()[0]
    task = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    lease = driver.acquire_lease(
        db,
        room_id=binding.room_id,
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation=owner.runtime.process_generation,
        ttl_seconds=30,
        clock=owner.runtime.clock,
    )
    owner.runtime._leases[binding.room_id] = lease
    driver.start_task(
        db,
        task["identity"],
        lease,
        expected_cancel_generation=task["cancel_generation"],
        clock=owner.runtime.clock,
    )
    task = driver.get_task(db, task["identity"])
    session = owner.runtime._resolve_or_create(rpc, "ops", binding.room_id)
    owner.runtime._report_pending_action(
        binding,
        task,
        session_id=session["session_id"],
        info={
            "pending_approval": {
                "request_id": "approval-1",
                "choices": ["once", "deny"],
            }
        },
    )
    pending = approvals.list_pending_approvals(db, room_id="room-1")[0]
    assert pending["observer_generation"] == owner.runtime.process_generation
    approvals.submit_approval(
        db, service=None, command_id="command-1", pending=pending, choice="once"
    )
    other = HostedRoomService(_server(), db_path=db)
    driver.require_active_lease(db, lease, clock=owner.runtime.clock)

    with ThreadPoolExecutor(max_workers=1) as executor:
        applying = executor.submit(owner._apply_pending_controls, binding, lease)
        try:
            assert entered.wait(10), "lease owner did not reach exact approval target"
            hosted_rooms.disband_room(
                db,
                room_id=binding.room_id,
                expected_gateway_id=binding.gateway_id,
                expected_epoch=binding.authority_epoch,
            )
            other.bindings()
        finally:
            release.set()
        applying.result(timeout=10)

    assert rpc.approvals == [(session["session_id"], "approval-1", "once")]
    receipt = approvals.approval_command(db, command_id="command-1")
    replayed = approvals.submit_approval(
        db, service=None, command_id="command-1", pending=pending, choice="once"
    )
    if response_lost:
        assert receipt["state"] == "pending"
        assert replayed["queued"] is True
        assert "applied" not in replayed
    else:
        assert receipt["state"] == "completed"
        assert (receipt["result_text"], replayed["applied"]) == ("Approved once.", True)
