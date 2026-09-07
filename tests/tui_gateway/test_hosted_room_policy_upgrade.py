"""Late-receipt indexing upgrades derived caches, never accepted inputs or history."""

import sqlite3
import time
from types import SimpleNamespace
import threading

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
from tui_gateway.hosted_room_service import HostedRoomService

PROFILES = ("writer", "reviewer")


def log(db):
    return rooms.read_events(db, room_id="upgrade-room")["events"]


def append(db, event_id, kind, payload, actor_kind="user"):
    return rooms.append_event(db, room_id="upgrade-room", event_id=event_id, kind=kind,
        actor={"kind": actor_kind, "id": "home"}, authority_gateway_id="home", authority_epoch=1, payload=payload)


def legacy_cache(db, cache_version):
    room = rooms.create_room(db, room_id="upgrade-room", name="Upgrade", authority_gateway_id="home",
        members=[{"member_id": profile, "profile": profile, "handle": profile} for profile in PROFILES])
    append(db, "source", "message.user", {"text": "@writer Prepare the plan.", "thread_id": "thread"})
    original = discussion.plan_next_task(room, log(db), local_profiles=PROFILES, freeze_input_context=True).task
    deferred = discussion.plan_publication(room, log(db), original, status="deferred", execution_generation=1,
        result={"reason": "member_unavailable"}, local_profiles=PROFILES)
    for event in deferred.events:
        rooms.append_event(db, **event.append_kwargs("upgrade-room"))
    append(db, "silent", "room.activity", {"status": "settled", "reason_code": "silent_round",
        "thread_id": "thread", "discussion_event_id": "source"}, "gateway")
    completed = discussion.plan_publication(room, log(db), original, status="settled",
        result={"text": "The committed plan is ready."}, local_profiles=PROFILES)
    rooms.append_event(db, **completed.events[0].append_kwargs("upgrade-room"))
    append(db, "newer", "message.user", {"text": "@reviewer Use the earlier plan.", "thread_id": "thread"})
    checkpoint = HostedRoomPolicyCheckpoint(db)
    before = checkpoint.snapshot(room_id="upgrade-room", latest_seq=log(db)[-1]["seq"])
    late = rooms.append_event(db, **completed.events[-1].append_kwargs("upgrade-room"))
    # Versions 2/3 consumed a late receipt without indexing it once newer
    # input prevented reopening the old discussion. Reproduce its persisted state.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_policy_cursors SET through_seq=? WHERE room_id='upgrade-room'", (late["seq"],))
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=? WHERE room_id='upgrade-room'",
                     (cache_version,))
    assert not checkpoint.publication_exists(room_id="upgrade-room", task_id=original.identity.task_id,
                                             status="settled", execution_generation=2)
    planned = discussion.plan_next_task(room, before.events, local_profiles=PROFILES,
                                       initial_watermarks=before.watermarks, freeze_input_context=True).task
    assert "The committed plan is ready." not in planned.payload["prompt"]
    return room, original, planned


@pytest.mark.parametrize("cache_version", [2, 3])
def test_old_cache_rebuilds_late_receipts_and_context_without_changing_history(tmp_path, cache_version):
    db = tmp_path / "state.db"
    room, original, old_plan = legacy_cache(db, cache_version)
    history = log(db)
    checkpoint = HostedRoomPolicyCheckpoint(db)
    updated = checkpoint.snapshot(room_id="upgrade-room", latest_seq=history[-1]["seq"])
    assert checkpoint.publication_exists(room_id="upgrade-room", task_id=original.identity.task_id,
                                         status="settled", execution_generation=2)
    planned = discussion.plan_next_task(room, updated.events, local_profiles=PROFILES,
        initial_watermarks=updated.watermarks, freeze_input_context=True).task
    assert "The committed plan is ready." in planned.payload["prompt"]
    assert planned.identity.task_id != old_plan.identity.task_id
    assert log(db) == history
    assert HostedRoomPolicyCheckpoint(db).snapshot(room_id="upgrade-room", latest_seq=history[-1]["seq"]) == updated


@pytest.mark.parametrize("status", ["queued", "running", "indeterminate"])
@pytest.mark.parametrize("cache_version", [2, 3])
def test_upgrade_preserves_an_already_admitted_frozen_turn(tmp_path, monkeypatch, status, cache_version):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "home")
    db = tmp_path / "state.db"
    _, _, old_plan = legacy_cache(db, cache_version)
    clock = [time.time()]
    admitted = driver.admit_task(db, old_plan.identity, payload=old_plan.payload, clock=lambda: clock[0])
    if status != "queued":
        lease = driver.acquire_lease(db, room_id="upgrade-room", gateway_id="home", authority_epoch=1,
            process_generation="prior", ttl_seconds=1, clock=lambda: clock[0])
        driver.start_task(db, old_plan.identity, lease, expected_cancel_generation=0, clock=lambda: clock[0])
        if status == "indeterminate":
            clock[0] += 2
            current = driver.acquire_lease(db, room_id="upgrade-room", gateway_id="home", authority_epoch=1,
                process_generation="current", ttl_seconds=60, clock=lambda: clock[0])
            driver.recover_room(db, current, clock=lambda: clock[0])
    history = log(db)
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=db)
    service.local_profiles = lambda: PROFILES
    service.runtime.clock = lambda: clock[0]
    service.prepare_room(service.bindings()[0])
    existing = driver.get_task_for_turn(db, old_plan.identity)
    assert existing["status"] == status
    assert existing["identity"] == admitted["identity"]
    assert existing["payload"] == admitted["payload"]
    assert log(db) == history
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_driver_tasks WHERE room_id='upgrade-room'").fetchone()[0] == 1
