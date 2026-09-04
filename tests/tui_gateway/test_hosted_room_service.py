"""Integration tests for the hosted Discussion coordinator."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import MAX_ACTIVE_POLICY_EVENTS
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    HostedMemberDispatch,
    PROTOCOL_VERSION,
    catalog_mapping,
    issue_room_grant,
)
from tui_gateway.hosted_room_peer_status import _RouteStatusPeerClient
from tui_gateway.hosted_room_service import (
    HostedRoomService,
    _grant_revoke_is_terminal,
)
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


from tests.tui_gateway.hosted_room_service_fixtures import (
    _append_room_event,
    _FakeRPC,
    _FakePeerClient,
    _UnavailablePeerClient,
    _NotAdmittedPeerClient,
    _ExpiredGrantPeerClient,
    _UnavailableRevokePeerClient,
    _ExpiredRevokePeerClient,
    _RefreshingPeerClient,
    _ApprovalPeerClient,
    _RecoveringPeerClient,
    _PromptRecordingRPC,
    _BlockingFirstRPC,
    _server,
    _wait_for,
)


@pytest.mark.parametrize(
    ("status_code", "error_code", "terminal"),
    [
        (401, "invalid_room_grant", True),
        (403, "invalid_room_grant", True),
        (403, "room_reauthorization_required", True),
        (400, "invalid_room_grant", False),
        (403, "room_execution_policy_changed", False),
        (500, "invalid_room_grant", False),
    ],
)
def test_grant_revoke_terminal_classification_uses_structured_fields(
    status_code,
    error_code,
    terminal,
):
    exc = PeerRunsHTTPError(
        "opaque peer error",
        status_code=status_code,
        error_code=error_code,
    )

    assert _grant_revoke_is_terminal(exc) is terminal


def test_stop_room_snapshots_tasks_before_status_transitions(monkeypatch, tmp_path):
    """One running task must not be counted again after it becomes stopping."""

    identity = driver.TaskIdentity("room-1", "task-1", "thread-1", "turn-1")
    task = {
        "identity": identity,
        "payload": {"source_event_seq": 1},
        "status": "running",
        "cancel_id": None,
    }
    calls = []

    def listed(_db, *, room_id, status):
        assert room_id == "room-1"
        return [dict(task)] if task["status"] == status else []

    def cancel(_identity, *, cancel_id):
        calls.append(cancel_id)
        task["status"] = "stopping"
        task["cancel_id"] = cancel_id
        return dict(task)

    monkeypatch.setattr(driver, "list_tasks", listed)
    monkeypatch.setattr(
        hosted_rooms,
        "request_room_stop",
        lambda _db, *, room_id, cancel_id, **_authority: {
            "room_id": room_id,
            "cancel_id": cancel_id,
            "seq": 2,
        },
    )
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    hosted_rooms.create_room(
        service.db_path,
        room_id="room-1",
        name="Stop room",
        members=[],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )
    service.runtime = SimpleNamespace(cancel=cancel, wakeup=lambda: None)

    assert service.stop_room("room-1", cancel_id="stop-1") == 1
    assert calls == ["stop-1"]


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
        retry_id="retry-1",
    )
    assert requeued["status"] == "queued"
    replayed = service.retry_room_task(
        "room-1",
        task_id=first["identity"].task_id,
        retry_id="retry-1",
    )
    assert replayed["status"] == "queued"
    assert replayed["idempotent"] is True
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE hosted_room_driver_tasks
                  SET status='deferred', execution_generation=2
                WHERE room_id='room-1' AND task_id=?""",
            (first["identity"].task_id,),
        )
        conn.commit()
    second_retry = service.retry_room_task(
        "room-1",
        task_id=first["identity"].task_id,
        retry_id="retry-2",
    )
    assert second_retry["status"] == "queued"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE hosted_room_driver_tasks
                  SET status='deferred', execution_generation=3
                WHERE room_id='room-1' AND task_id=?""",
            (first["identity"].task_id,),
        )
        conn.commit()
    delayed_first = service.retry_room_task(
        "room-1",
        task_id=first["identity"].task_id,
        retry_id="retry-1",
    )
    assert delayed_first["status"] == "deferred"
    assert delayed_first["idempotent"] is True
    third_retry = service.retry_room_task(
        "room-1",
        task_id=first["identity"].task_id,
        retry_id="retry-3",
    )
    assert third_retry["status"] == "queued"
    lease = service.runtime._leases["room-1"]
    retried = driver.start_task(
        db,
        first["identity"],
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    assert retried.execution_generation == third_retry["execution_generation"] + 1


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


def test_registered_peer_route_rehydrates_after_service_restart(tmp_path: Path):
    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
        )
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
    first = HostedRoomService(_server(), db_path=db)
    first.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=route,
        client=_FakePeerClient(),
        target_url="https://peer.example.test",
        catalog=catalog,
    )

    restarted = HostedRoomService(_server(), db_path=db)
    restored = restarted.peer_routes[("room-1", "member-peer")]
    assert restored.target_install_id == "install-peer"
    assert restored.target_profile == "reviewer"
    assert restored.grant == "signed.room.grant"
    assert ("room-1", "member-peer") in restarted.peer_clients


def test_one_corrupt_stored_route_does_not_hide_healthy_peers(tmp_path: Path):
    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
    )
    service = HostedRoomService(_server(), db_path=db)
    for room_id, member_id in (("room-good", "member-good"), ("room-bad", "member-bad")):
        route = PeerMemberRoute(
            home_install_id=hosted_rooms.local_authority_gateway_id(),
            member_id=member_id,
            target_install_id="install-peer",
            target_profile="reviewer",
            capability_digest=catalog.catalog_digest,
            cancellation_scope_id=f"cancel-{room_id}",
            trace_id=f"trace-{room_id}",
            grant=f"grant-{room_id}",
        )
        service.register_peer_route(
            room_id=room_id,
            member_id=member_id,
            route=route,
            client=_FakePeerClient(),
            target_url="https://peer.example.test",
            catalog=catalog,
        )

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_links SET target_url=? WHERE room_id=?",
            ("http://public-plaintext.example.test", "room-bad"),
        )

    restarted = HostedRoomService(_server(), db_path=db)

    assert ("room-good", "member-good") in restarted.peer_routes
    assert ("room-bad", "member-bad") not in restarted.peer_routes
    assert restarted.status()["link_load_error"] == "room-bad:member-bad:invalid"


def test_unpublished_roomlink_v1_route_is_quarantined_for_reinvitation(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    legacy_catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install-peer",
            protocol_versions=(1,),
            persistent_process=True,
        )
    )
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=legacy_catalog.catalog_digest,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="legacy-v1-grant",
    )
    service = HostedRoomService(_server(), db_path=db)
    service.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=route,
        client=_FakePeerClient(),
        target_url="https://peer.example.test",
        catalog=legacy_catalog,
    )

    restarted = HostedRoomService(_server(), db_path=db)

    assert restarted.peer_routes == {}
    assert restarted.status()["link_load_error"] == (
        "room-1:member-peer:protocol-upgrade-required"
    )


def test_peer_member_without_route_fails_closed_instead_of_running_locally(
    tmp_path: Path,
):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    service.create_room(
        room_id="room-1",
        name="Peer room",
        members=[
            {"member_id": "local", "profile": "default", "handle": "local"},
            {
                "member_id": "member-peer",
                "profile": "reviewer",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "peer_id": "peer-review",
                    "installation_id": "install-peer",
                    "profile": "reviewer",
                    "capability_digest": "a" * 64,
                },
            },
        ],
    )
    with pytest.raises(RuntimeError, match="route is unavailable"):
        service._resolve_member_transport(
            service.bindings()[0],
            {
                "payload": {
                    "target_member_id": "member-peer",
                    "target_profile": "reviewer",
                    "source_event_seq": 9,
                }
            },
        )


def test_registration_disk_failure_does_not_publish_live_route(
    tmp_path: Path, monkeypatch
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
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
    monkeypatch.setattr(
        "gateway.hosted_room_links.save_room_link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        service.register_peer_route(
            room_id="room-1",
            member_id="member-peer",
            route=route,
            client=_FakePeerClient(),
            target_url="https://peer.example.test",
            catalog=catalog,
        )
    assert ("room-1", "member-peer") not in service.peer_routes
    assert "install-peer" not in service.peer_clients


def test_room_route_revocation_is_remote_first_and_removes_local_state(
    tmp_path: Path,
):
    from gateway import hosted_room_links

    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
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
    peer = _FakePeerClient()
    service = HostedRoomService(_server(), db_path=db)
    service.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=route,
        client=peer,
        target_url="https://peer.example.test",
        catalog=catalog,
    )

    assert service.revoke_room_routes("room-1") == 1
    assert peer.revoked == ["signed.room.grant"]
    assert ("room-1", "member-peer") not in service.peer_routes
    assert hosted_room_links.load_room_links(db) == ()


def test_failed_remote_revocation_preserves_route_for_retry(tmp_path: Path):
    from gateway import hosted_room_links

    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
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
        client=_UnavailableRevokePeerClient(),
        target_url="https://peer.example.test",
        catalog=catalog,
    )

    with pytest.raises(RuntimeError, match="offline during revocation"):
        service.revoke_room_routes("room-1")
    assert ("room-1", "member-peer") in service.peer_routes
    assert len(hosted_room_links.load_room_links(db)) == 1


def test_expired_remote_grant_no_longer_blocks_room_cleanup(tmp_path: Path):
    from gateway import hosted_room_links

    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
    )
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="expired.room.grant",
    )
    service = HostedRoomService(_server(), db_path=db)
    service.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=route,
        client=_ExpiredRevokePeerClient(),
        target_url="https://peer.example.test",
        catalog=catalog,
    )

    assert service.revoke_room_routes("room-1") == 1
    assert ("room-1", "member-peer") not in service.peer_routes
    assert hosted_room_links.load_room_links(db) == ()


def test_expired_grant_surfaces_needs_reauthorization_without_secret(
    tmp_path: Path, monkeypatch,
):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient

    monkeypatch.setattr(PeerRunsHTTPClient, "revoke_grant_exact", lambda self, **kwargs: {"revoked": True})
    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
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
        client=_ExpiredGrantPeerClient(),
        target_url="https://peer.example.test",
        catalog=catalog,
    )
    service.create_room(
        room_id="room-1",
        name="Peer room",
        members=[
            {"member_id": "local", "profile": "default", "handle": "local"},
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
    transport = service._resolve_member_transport(
        service.bindings()[0],
        {
            "payload": {
                "target_member_id": "member-peer",
                "target_profile": "reviewer",
                "source_event_seq": 3,
            }
        },
    )
    with pytest.raises(PeerRunsHTTPError):
        transport.resolve_exact(
            profile="reviewer",
            title="Group: room-1",
            source="bot_room",
        )
    status = service.status("room-1")
    assert status["peer_routes"] == [
        {
            "room_id": "room-1",
            "member_id": "member-peer",
            "status": "needs_reauthorization",
        }
    ]
    assert "signed.room.grant" not in repr(status)
    restarted = HostedRoomService(_server(), db_path=db)
    assert restarted.status("room-1")["peer_routes"] == status["peer_routes"]

    rotated = PeerMemberRoute(
        home_install_id=route.home_install_id,
        member_id=route.member_id,
        target_install_id=route.target_install_id,
        target_profile=route.target_profile,
        capability_digest=route.capability_digest,
        cancellation_scope_id=route.cancellation_scope_id,
        trace_id="trace-room-rotated",
        grant="rotated.room.grant",
    )
    restarted.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=rotated,
        client=_FakePeerClient(),
        target_url="https://peer.example.test",
        catalog=catalog,
    )
    assert restarted.status("room-1")["peer_routes"][0]["status"] == "ready"
    after_rotation = HostedRoomService(_server(), db_path=db)
    assert after_rotation.peer_routes[("room-1", "member-peer")].grant == (
        "rotated.room.grant"
    )


def test_not_admitted_dispatch_persists_unavailable_route_until_success(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
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
    peer = _NotAdmittedPeerClient()
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
            {"member_id": "local", "profile": "default", "handle": "local"},
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
    binding = service.bindings()[0]
    task = {
        "identity": driver.TaskIdentity(
            "room-1", "task-peer", "thread-1", "turn-1"
        ),
        "execution_generation": 1,
        "payload": {
            "target_member_id": "member-peer",
            "target_profile": "reviewer",
            "source_event_seq": 3,
        },
    }
    transport = service._resolve_member_transport(binding, task)
    session = transport.create(
        profile="reviewer",
        title="Group: room-1",
        source="bot_room",
    )

    with pytest.raises(PeerRunsHTTPError) as caught:
        transport.submit(
            profile="reviewer",
            session_id=session["session_id"],
            prompt="Review the queued task.",
            source="bot_room",
            task=task["identity"],
            execution_generation=1,
            on_terminal=lambda _receipt: None,
        )

    assert caught.value.not_admitted is True
    assert service.status("room-1")["peer_routes"][0]["status"] == "unavailable"
    restarted = HostedRoomService(_server(), db_path=db)
    assert restarted.status("room-1")["peer_routes"][0]["status"] == "unavailable"

    peer.offline = False
    transport.submit(
        profile="reviewer",
        session_id=session["session_id"],
        prompt="Review the queued task.",
        source="bot_room",
        task=task["identity"],
        execution_generation=2,
        on_terminal=lambda _receipt: None,
    )
    assert service.status("room-1")["peer_routes"][0]["status"] == "ready"


def test_dispatch_refresh_persists_before_remote_admission(tmp_path: Path):
    now = time.time()
    secret = b"s" * 32
    old_grant = issue_room_grant(
        secret,
        grant_id="grant-old",
        room_id="room-1",
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        issued_at=now - 3700,
        ttl_seconds=3600,
        status_expires_at=now + 10_000,
    )
    new_grant = issue_room_grant(
        secret,
        grant_id="grant-new",
        room_id="room-1",
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        issued_at=now,
        ttl_seconds=3600,
        status_expires_at=now + 10_000,
    )
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
    )
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=catalog.catalog_digest,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant=old_grant,
    )
    peer = _RefreshingPeerClient(new_grant)
    db = tmp_path / "state.db"
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
            {"member_id": "local", "profile": "default", "handle": "local"},
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
    identity = driver.TaskIdentity("room-1", "task-1", "thread-1", "turn-1")
    transport = service._resolve_member_transport(
        service.bindings()[0],
        {
            "identity": identity,
            "execution_generation": 1,
            "payload": {
                "target_member_id": "member-peer",
                "target_profile": "reviewer",
                "source_event_seq": 11,
            },
        },
    )
    session = transport.create(
        profile="reviewer", title="Group: room-1", source="bot_room"
    )
    transport.submit(
        profile="reviewer",
        session_id=session["session_id"],
        prompt="Review this",
        source="bot_room",
        task=identity,
        execution_generation=1,
        on_terminal=lambda _receipt: None,
    )
    assert peer.refreshed == [old_grant]
    assert peer.exact_revoked == [old_grant]
    assert peer.refresh_arguments == [
        {
            "grant": old_grant,
            "capability_digest": catalog.catalog_digest,
            "execution_policy_digest": catalog.execution_policy.policy_digest,
        }
    ]
    assert peer.dispatched_grants == [new_grant]
    assert peer.dispatches[0]["capability_digest"] == catalog.catalog_digest
    assert (
        peer.dispatches[0]["execution_policy_digest"]
        == catalog.execution_policy.policy_digest
    )
    assert HostedRoomService(_server(), db_path=db).peer_routes[
        ("room-1", "member-peer")
    ].grant == new_grant


@pytest.mark.parametrize(
    ("capability_changed", "policy_changed"),
    [(False, True), (True, False), (True, True)],
)
def test_dispatch_refresh_marks_route_for_reauthorization_on_drift(
    capability_changed,
    policy_changed,
):
    from gateway.hosted_room_execution_policy import execution_policy_mapping

    base_policy = execution_policy_mapping(
        target_profile="reviewer",
        config={"agent": {"max_turns": 20}},
    )
    changed_policy = execution_policy_mapping(
        target_profile="reviewer",
        config={"agent": {"max_turns": 21}},
    )
    base_catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
            execution_policy=base_policy,
        )
    )
    refreshed_catalog = catalog_mapping(
        installation_id="install-peer",
        persistent_process=True,
        attachments=capability_changed,
        execution_policy=changed_policy if policy_changed else base_policy,
    )
    now = time.time()
    grant = issue_room_grant(
        b"s" * 32,
        grant_id="grant-old",
        room_id="room-1",
        home_install_id="install-home",
        authority_gateway_id="install-home",
        authority_epoch=1,
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        execution_policy_digest=base_catalog.execution_policy.policy_digest,
        issued_at=now - 3500,
        ttl_seconds=3600,
        status_expires_at=now + 10_000,
    )
    prompt = "Review this"
    dispatch = HostedMemberDispatch(
        protocol_version=PROTOCOL_VERSION,
        room_id="room-1",
        home_install_id="install-home",
        authority_gateway_id="install-home",
        authority_epoch=1,
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        task_id="task-1",
        execution_generation=1,
        source_event_seq=1,
        cancellation_scope_id="cancel-room-1",
        prompt=prompt,
        prompt_digest=hashlib.sha256(prompt.encode()).hexdigest(),
        capability_digest=base_catalog.catalog_digest,
        execution_policy_digest=base_catalog.execution_policy.policy_digest,
        trace_id="trace-room-1",
    )
    peer = _RefreshingPeerClient(
        "replacement.room.grant",
        catalog=refreshed_catalog,
    )
    reauthorization = []
    refreshed = []
    tracked = _RouteStatusPeerClient(
        peer,
        on_ready=lambda: None,
        on_reauthorization=lambda: reauthorization.append(True),
        on_unavailable=lambda: None,
        on_refreshed=lambda *args: refreshed.append(args),
    )

    with pytest.raises(PeerRunsHTTPError) as caught:
        tracked.dispatch(dispatch=dispatch.as_mapping(), grant=grant)

    assert caught.value.needs_reauthorization is True
    assert reauthorization == [True]
    assert refreshed == []
    assert peer.dispatches == []


def test_peer_approval_is_scoped_visible_and_resolvable(tmp_path: Path):
    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
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


def test_peer_recovery_replays_the_same_execution_generation(tmp_path: Path):
    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
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
    peer = _RecoveringPeerClient()
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
    identity = driver.TaskIdentity("room-1", "task-1", "thread-1", "turn-1")

    service._resolve_member_transport(
        service.bindings()[0],
        {
            "identity": identity,
            "status": "indeterminate",
            "execution_generation": 1,
            "payload": {
                "target_member_id": "member-peer",
                "target_profile": "reviewer",
                "source_event_seq": 9,
                "prompt": "Recover the accepted review.",
            },
        },
    )

    assert len(peer.recoveries) == 1
    recovered = peer.recoveries[0]["dispatch"]
    assert recovered["task_id"] == "task-1"
    assert recovered["execution_generation"] == 1
    assert recovered["prompt"] == "Recover the accepted review."
