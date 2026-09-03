"""Cross-process hosted Group Chat retry ownership tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gateway import hosted_room_controls, hosted_rooms
from gateway import hosted_room_driver as driver
from tests.tui_gateway.test_hosted_room_service import _FakeRPC, _server
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.mark.parametrize(
    "mode",
    ["normal", "lease_takeover", "stopped", "already_cancelled"],
)
def test_active_worker_applies_retry_queued_by_another_process(
    tmp_path: Path,
    monkeypatch,
    mode: str,
):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.rpc = _FakeRPC()
    service.runtime.rpc = service.rpc
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 30
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Cross-process retry",
        members=[
            {"member_id": "default", "profile": "default", "handle": "default"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "Retry this", "thread_id": "thread-1"},
    )
    task = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    old_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=service.bindings()[0].gateway_id,
        authority_epoch=1,
        process_generation="old-process",
        ttl_seconds=1,
        clock=clock,
    )
    attempt = driver.start_task(
        db,
        task["identity"],
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    now[0] = 102.0
    owner_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=service.bindings()[0].gateway_id,
        authority_epoch=1,
        process_generation=service.runtime.process_generation,
        ttl_seconds=30,
        clock=clock,
    )
    driver.recover_room(db, owner_lease, clock=clock)
    driver.defer_indeterminate_task(
        db,
        task["identity"],
        owner_lease,
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation,
        reason="member_unavailable",
        clock=clock,
    )
    hosted_room_controls.begin_control_retry(
        db,
        command_id="retry-command-1",
        room_id="room-1",
        member_id="messaging-owner",
        task_ids=[task["identity"].task_id],
        now=now[0],
    )
    service.runtime._leases["room-1"] = owner_lease

    if mode in {"stopped", "already_cancelled"}:
        room = hosted_rooms.room_state(db, room_id="room-1")
        hosted_rooms.request_room_stop(
            db,
            room_id="room-1",
            cancel_id="stop-before-retry",
            expected_gateway_id=str(room["authority_gateway_id"]),
            expected_epoch=int(room["authority_epoch"]),
        )
        if mode == "already_cancelled":
            service.runtime.cancel(
                task["identity"],
                cancel_id="stop-before-retry",
            )

    if mode == "lease_takeover":
        real_complete_control_retry = hosted_room_controls.complete_control_retry
        expire_once = [True]

        def complete_after_takeover(*args, **kwargs):
            if expire_once[0]:
                expire_once[0] = False
                now[0] = owner_lease.expires_at
                driver.acquire_lease(
                    db,
                    room_id="room-1",
                    gateway_id=service.bindings()[0].gateway_id,
                    authority_epoch=1,
                    process_generation="takeover-worker",
                    ttl_seconds=30,
                    clock=clock,
                )
            return real_complete_control_retry(*args, **kwargs)

        monkeypatch.setattr(
            hosted_room_controls,
            "complete_control_retry",
            complete_after_takeover,
        )

    if mode == "lease_takeover":
        with pytest.raises(driver.StaleLeaseError):
            service.runtime._process_room(service.bindings()[0])
        assert (
            len(
                hosted_room_controls.load_pending_control_retries(
                    db,
                    room_id="room-1",
                )
            )
            == 1
        )
        service.runtime.process_generation = "takeover-worker"
        service.runtime._leases.clear()
        service.runtime._process_room(service.bindings()[0])
    else:
        service.runtime._process_room(service.bindings()[0])

    completed = hosted_room_controls.begin_control_retry(
        db,
        command_id="retry-command-1",
        room_id="room-1",
        member_id="messaging-owner",
        task_ids=[task["identity"].task_id],
        now=now[0] + 1,
    )
    assert completed.result == {"action": "retry", "processed": 1}
    assert driver.get_task(db, task["identity"])["status"] == (
        "cancelled" if mode in {"stopped", "already_cancelled"} else "settled"
    )
    if mode not in {"stopped", "already_cancelled"}:
        assert driver.retry_receipt_exists(
            db,
            room_id="room-1",
            task_id=task["identity"].task_id,
            retry_id=hosted_room_controls.control_retry_attempt_id(
                "retry-command-1",
                task["identity"].task_id,
            ),
        )
    if mode in {"stopped", "already_cancelled"}:
        assert not any(
            event["kind"] == "message.member" for event in service._events("room-1")
        )


def test_worker_retry_redelivery_does_not_requeue_a_later_generation(
    tmp_path: Path,
    monkeypatch,
):
    now = [100.0]

    def clock():
        return now[0]

    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.rpc = _FakeRPC()
    service.runtime.rpc = service.rpc
    service.runtime.clock = clock
    service.runtime.lease_ttl_seconds = 30
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Retry redelivery",
        members=[
            {"member_id": "default", "profile": "default", "handle": "default"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "Retry this", "thread_id": "thread-1"},
    )
    task = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    expired_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=service.bindings()[0].gateway_id,
        authority_epoch=1,
        process_generation="expired-process",
        ttl_seconds=1,
        clock=clock,
    )
    first_attempt = driver.start_task(
        db,
        task["identity"],
        expired_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    now[0] = 102.0
    owner_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=service.bindings()[0].gateway_id,
        authority_epoch=1,
        process_generation="owner-process",
        ttl_seconds=30,
        clock=clock,
    )
    driver.recover_room(db, owner_lease, clock=clock)
    driver.defer_indeterminate_task(
        db,
        task["identity"],
        owner_lease,
        expected_execution_generation=first_attempt.execution_generation,
        expected_cancel_generation=first_attempt.cancel_generation,
        reason="member_unavailable",
        clock=clock,
    )
    hosted_room_controls.begin_control_retry(
        db,
        command_id="retry-command-1",
        room_id="room-1",
        member_id="messaging-owner",
        task_ids=[task["identity"].task_id],
        now=now[0],
    )

    real_complete = hosted_room_controls.complete_control_retry
    lose_first_completion = [True]

    def complete_after_lost_response(*args, **kwargs):
        if lose_first_completion[0]:
            lose_first_completion[0] = False
            raise driver.StaleLeaseError("simulated lost completion")
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(
        hosted_room_controls,
        "complete_control_retry",
        complete_after_lost_response,
    )
    service.runtime._leases["room-1"] = owner_lease
    service._apply_pending_control_retries(service.bindings()[0], owner_lease)

    retry_id = hosted_room_controls.control_retry_attempt_id(
        "retry-command-1",
        task["identity"].task_id,
    )
    assert driver.retry_receipt_exists(
        db,
        room_id="room-1",
        task_id=task["identity"].task_id,
        retry_id=retry_id,
    )
    assert driver.get_task(db, task["identity"])["status"] == "queued"
    assert (
        len(hosted_room_controls.load_pending_control_retries(db, room_id="room-1"))
        == 1
    )

    second_attempt = driver.start_task(
        db,
        task["identity"],
        owner_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    now[0] = owner_lease.expires_at
    next_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=service.bindings()[0].gateway_id,
        authority_epoch=1,
        process_generation="next-process",
        ttl_seconds=30,
        clock=clock,
    )
    driver.recover_room(db, next_lease, clock=clock)
    driver.defer_indeterminate_task(
        db,
        task["identity"],
        next_lease,
        expected_execution_generation=second_attempt.execution_generation,
        expected_cancel_generation=second_attempt.cancel_generation,
        reason="member_unavailable_again",
        clock=clock,
    )

    service.runtime._leases["room-1"] = next_lease
    service._apply_pending_control_retries(service.bindings()[0], next_lease)

    assert driver.get_task(db, task["identity"])["status"] == "deferred"
    assert (
        hosted_room_controls.load_pending_control_retries(
            db,
            room_id="room-1",
        )
        == ()
    )
