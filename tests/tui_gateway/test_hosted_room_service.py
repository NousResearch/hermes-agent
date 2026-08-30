"""Integration tests for the hosted Discussion coordinator."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import MAX_ACTIVE_POLICY_EVENTS
from tui_gateway.hosted_room_service import HostedRoomService


def _append_room_event(db, **kwargs):
    if kwargs.get("kind") == "message.user":
        room = hosted_rooms.room_state(db, room_id=kwargs["room_id"])
        kwargs.setdefault(
            "authority_gateway_id", str(room["authority_gateway_id"])
        )
        kwargs.setdefault("authority_epoch", int(room["authority_epoch"]))
    return hosted_rooms.append_event(db, **kwargs)


class _FakeRPC:
    def __init__(self) -> None:
        self.sessions = {}

    def resolve_exact(self, *, profile, title, source):
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        session = {"session_id": f"{profile}-session", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def submit(
        self,
        *,
        profile,
        session_id,
        prompt,
        source,
        task,
        execution_generation,
        on_terminal,
    ):
        on_terminal({"status": "settled", "text": f"reply from {profile}"})
        return {"accepted": True}

    def history(self, *, profile, session_id, source):
        return []

    def info(self, *, profile, session_id, source):
        return {"active": False, "task_id": None}

    def interrupt(self, *, profile, session_id, source, expected_task_id):
        return {"interrupted": True}


class _PromptRecordingRPC(_FakeRPC):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[tuple[str, str]] = []

    def submit(
        self,
        *,
        profile,
        session_id,
        prompt,
        source,
        task,
        execution_generation,
        on_terminal,
    ):
        self.prompts.append((profile, prompt))
        on_terminal({"status": "settled", "text": f"reply from {profile}"})
        return {"accepted": True}


class _BlockingFirstRPC(_PromptRecordingRPC):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def submit(self, **kwargs):
        self.prompts.append((kwargs["profile"], kwargs["prompt"]))
        if len(self.prompts) == 1:
            self.first_started.set()
            assert self.release_first.wait(timeout=2)
        kwargs["on_terminal"](
            {"status": "settled", "text": f"reply from {kwargs['profile']}"}
        )
        return {"accepted": True}


def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def test_create_send_drive_publish_and_replay_without_client_transport(tmp_path: Path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.rpc = _FakeRPC()
    service.runtime.rpc = service.rpc
    service.local_profiles = lambda: ("default", "ops")
    room = service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    assert room["room_id"] == "room-1"

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops inspect the release", "thread_id": "thread-1"},
    )
    _wait_for(
        lambda: any(
            event["kind"] == "message.member" for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)

    events = service._events("room-1")
    assert [event["kind"] for event in events][:3] == [
        "message.user",
        "message.member",
        "turn.settled",
    ]
    assert events[1]["payload"]["text"] == "reply from ops"
    assert service.status("room-1")["working"] is False


def test_restart_republishes_terminal_task_before_admitting_more(tmp_path: Path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    event = _append_room_event(
        db,
        room_id="room-1",
        event_id="user-1",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )
    binding = service.bindings()[0]
    service.prepare_room(binding)
    task = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="crashed",
        ttl_seconds=30,
        clock=time.time,
    )
    attempt = driver.start_task(
        db,
        task["identity"],
        lease,
        expected_cancel_generation=0,
        clock=time.time,
    )
    driver.settle_task(
        db,
        attempt,
        settlement_id="reply-1",
        status="settled",
        result={"text": "done"},
        clock=time.time,
    )

    service.prepare_room(binding)
    events = service._events("room-1")
    assert event["seq"] == 1
    assert sum(row["kind"] == "message.member" for row in events) == 1
    assert sum(row["kind"] == "turn.settled" for row in events) == 1
    service.prepare_room(binding)
    replayed = service._events("room-1")
    assert replayed == events


def test_policy_checkpoint_bounds_replay_after_completed_room_history(
    tmp_path: Path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "ops")
    room = service.create_room(
        room_id="room-1",
        name="Long-running room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "default"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    authority = str(room["authority_gateway_id"])
    rows = []
    for index in range(200):
        user_seq = index * 2 + 1
        activity_seq = user_seq + 1
        thread_id = f"thread-{index}"
        event_id = f"user-{index}"
        rows.extend((
            (
                "room-1",
                user_seq,
                event_id,
                "message.user",
                json.dumps({"kind": "user", "id": "load-test"}),
                None,
                json.dumps({"text": "done", "thread_id": thread_id}),
                float(user_seq),
            ),
            (
                "room-1",
                activity_seq,
                f"activity-{index}",
                "room.activity",
                json.dumps({"kind": "gateway", "id": authority}),
                1,
                json.dumps({
                    "status": "settled",
                    "reason_code": "silent_round",
                    "thread_id": thread_id,
                    "discussion_event_id": event_id,
                }),
                float(activity_seq),
            ),
        ))
    with sqlite3.connect(db) as conn:
        conn.executemany(
            """INSERT INTO hosted_room_events(
                   room_id, seq, event_id, kind, actor_json,
                   authority_epoch, payload_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.execute(
            """UPDATE hosted_rooms
               SET next_seq=401, revision=revision+400, updated_at=400
               WHERE room_id='room-1'"""
        )
    _append_room_event(
        db,
        room_id="room-1",
        event_id="user-active",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "Review this", "thread_id": "thread-active"},
        now=401,
    )

    original_read_events = hosted_rooms.read_events
    reads = {"calls": 0, "rows": 0}

    def counted_read_events(*args, **kwargs):
        page = original_read_events(*args, **kwargs)
        reads["calls"] += 1
        reads["rows"] += len(page["events"])
        return page

    monkeypatch.setattr(hosted_rooms, "read_events", counted_read_events)
    binding = service.bindings()[0]
    service.prepare_room(binding)
    assert reads["rows"] == 401
    snapshot = service._policy_snapshot(hosted_rooms.room_state(db, room_id="room-1"))
    assert len(snapshot.events) == 1
    assert len(snapshot.events) <= MAX_ACTIVE_POLICY_EVENTS
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM hosted_room_policy_events").fetchone()[0]
            == 1
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM hosted_room_policy_threads").fetchone()[
                0
            ]
            == 1
        )

    reads.update(calls=0, rows=0)
    service.prepare_room(binding)
    assert reads == {"calls": 0, "rows": 0}


def test_same_thread_followup_migrates_and_delivers_committed_peer_reply(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.rpc = _PromptRecordingRPC()
    service.runtime.rpc = service.rpc
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Shared context room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops provide the marker", "thread_id": "thread-1"},
    )
    _wait_for(lambda: len(service.rpc.prompts) == 1)
    _wait_for(
        lambda: any(
            event["kind"] == "room.activity"
            and event["payload"]["discussion_event_id"] == "user-1"
            for event in service._events("room-1")
        )
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            """SELECT COUNT(*) FROM hosted_room_policy_transcript
               WHERE room_id='room-1' AND thread_id='thread-1'"""
        ).fetchone()[0] == 2
        conn.execute("DELETE FROM hosted_room_policy_transcript")
        conn.execute(
            """DELETE FROM hosted_room_policy_transcript_state
               WHERE room_id='room-1'"""
        )
    service.send(
        room_id="room-1",
        event_id="user-2",
        payload={"text": "@hermes continue", "thread_id": "thread-1"},
    )
    _wait_for(lambda: len(service.rpc.prompts) == 2)
    assert service.stop(timeout=1.0)

    profile, prompt = service.rpc.prompts[1]
    assert profile == "default"
    assert "@ops: reply from ops" in prompt
    assert "User (user): @hermes continue" in prompt


def test_active_same_thread_followup_waits_for_current_task(tmp_path: Path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.rpc = _BlockingFirstRPC()
    service.runtime.rpc = service.rpc
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Serialized room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops start", "thread_id": "thread-1"},
    )
    assert service.rpc.first_started.wait(timeout=2)
    service.send(
        room_id="room-1",
        event_id="user-2",
        payload={"text": "@hermes follow up", "thread_id": "thread-1"},
    )
    assert len(service.rpc.prompts) == 1
    service.rpc.release_first.set()
    _wait_for(lambda: len(service.rpc.prompts) == 2)
    _wait_for(
        lambda: any(
            event["kind"] == "room.activity"
            and event["payload"]["discussion_event_id"] == "user-2"
            for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)
    assert "User (user): @hermes follow up" in service.rpc.prompts[1][1]


def test_thread_transcript_prunes_committed_message_and_settlement_together(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.rpc = _FakeRPC()
    service.runtime.rpc = service.rpc
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Bounded room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service.start()
    service.send(
        room_id="room-1",
        event_id="user-first",
        payload={"text": "@ops old", "thread_id": "thread-1"},
    )
    _wait_for(
        lambda: any(
            event["kind"] == "room.activity"
            for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)
    for index in range(24):
        _append_room_event(
            db,
            room_id="room-1",
            event_id=f"user-tail-{index}",
            kind="message.user",
            actor={"kind": "user", "id": "desktop"},
            payload={"text": f"tail {index}", "thread_id": "thread-1"},
        )

    room = hosted_rooms.room_state(db, room_id="room-1")
    snapshot = service._policy_snapshot(room)
    assert len(snapshot.events) == 24
    assert {event["kind"] for event in snapshot.events} == {"message.user"}
    discussion.plan_next_task(
        room,
        snapshot.events,
        local_profiles=service.local_profiles(),
        initial_watermarks=snapshot.watermarks,
    )


def test_service_uses_low_idle_poll_with_immediate_wakeup(tmp_path: Path):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")

    assert service.runtime.poll_interval_seconds == 5.0
    assert service.runtime.active_poll_interval_seconds == 0.25
    assert service.runtime.turn_timeout_seconds == 1830.0
    service.runtime._wake.clear()
    service.wakeup()
    assert service.runtime._wake.is_set()


def test_service_derives_room_deadline_from_agent_timeout(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "90")

    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")

    assert service.runtime.turn_timeout_seconds == 120.0


def test_service_publishes_deferred_turn_continues_and_retries_new_generation(
    tmp_path: Path,
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
    service.runtime.indeterminate_defer_seconds = 5
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Resilient room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "default"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service.send(
        room_id="room-1",
        event_id="user-resilience",
        payload={"text": "Check this", "thread_id": "thread-1"},
    )
    first = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    old_lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=service.bindings()[0].gateway_id,
        authority_epoch=1,
        process_generation="offline-member",
        ttl_seconds=1,
        clock=clock,
    )
    old_attempt = driver.start_task(
        db,
        first["identity"],
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )

    now[0] = 102.0
    binding = service.bindings()[0]
    service.runtime._process_room(binding)
    now[0] = 108.0
    service.runtime._process_room(binding)

    events = service._events("room-1")
    deferred = next(event for event in events if event["kind"] == "turn.deferred")
    assert deferred["payload"]["task_id"] == first["identity"].task_id
    assert deferred["payload"]["execution_generation"] == 1
    assert any(
        event["kind"] == "message.member" and event["payload"]["member_id"] == "ops"
        for event in events
    )

    requeued = service.retry_room_task(
        "room-1",
        task_id=first["identity"].task_id,
    )
    assert requeued["status"] == "queued"
    lease = service.runtime._leases["room-1"]
    retried = driver.start_task(
        db,
        first["identity"],
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    assert retried.execution_generation == old_attempt.execution_generation + 1


def test_stop_fence_prevents_the_next_room_member_from_starting(
    tmp_path: Path, monkeypatch
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    monkeypatch.setattr(service, "local_profiles", lambda: ("default", "ops"))
    service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "Inspect the release", "thread_id": "thread-1"},
    )
    assert len(driver.list_tasks(db, room_id="room-1")) == 1

    assert service.stop_room("room-1", cancel_id="stop-1") == 1
    service.prepare_room(service.bindings()[0])

    tasks = driver.list_tasks(db, room_id="room-1")
    assert len(tasks) == 1
    assert tasks[0]["status"] == "cancelled"
    assert any(
        event["kind"] == "room.stop_requested" for event in service._events("room-1")
    )


def test_acknowledged_stop_refuses_to_disband_while_exact_turn_is_still_running(
    tmp_path: Path,
):
    class PendingStopRPC(_FakeRPC):
        def __init__(self) -> None:
            super().__init__()
            self.active_task_id = None

        def info(self, *, profile, session_id, source):
            return {"active": True, "task_id": self.active_task_id}

        def interrupt(self, *, profile, session_id, source, expected_task_id):
            return None

    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = PendingStopRPC()
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
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )
    task = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    binding = service.bindings()[0]
    lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="worker",
        ttl_seconds=30,
        clock=time.time,
    )
    driver.start_task(
        db,
        task["identity"],
        lease,
        expected_cancel_generation=0,
        clock=time.time,
    )
    rpc.sessions[("ops", "Group: room-1")] = {"session_id": "ops-session"}
    rpc.active_task_id = task["identity"].task_id

    with pytest.raises(RuntimeError, match="still stopping"):
        service.stop_room(
            "room-1",
            cancel_id="stop-1",
            require_acknowledged=True,
        )

    stopping = driver.get_task(db, task["identity"])
    assert stopping["status"] == "stopping"
    assert stopping["cancel_id"] == "stop-1"


def test_local_pending_approval_requires_exact_task_generation_and_request(
    tmp_path: Path,
):
    class ApprovalRPC(_FakeRPC):
        def __init__(self) -> None:
            super().__init__()
            self.approvals = []

        def approve(self, *, session_id, request_id, choice):
            self.approvals.append((session_id, request_id, choice))
            return {"resolved": 1}

    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = ApprovalRPC()
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
    service.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )
    task = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    binding = service.bindings()[0]
    lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="worker",
        ttl_seconds=30,
        clock=time.time,
    )
    driver.start_task(
        db,
        task["identity"],
        lease,
        expected_cancel_generation=0,
        clock=time.time,
    )
    task = driver.get_task(db, task["identity"])
    service.runtime._report_pending_action(
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
