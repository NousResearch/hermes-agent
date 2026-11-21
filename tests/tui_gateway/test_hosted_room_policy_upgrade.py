"""Late-receipt indexing upgrades derived caches, never accepted inputs or history."""

import json
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
from persistence_assembly import open_persistence, profile_scope

PROFILES = ("writer", "reviewer")


@pytest.fixture(name="store")
def store_fixture(tmp_path):
    """A store handle on the test home, closed at teardown."""
    handle = open_persistence(profile_scope(home=tmp_path))
    yield handle
    handle.close()


def log(store):
    return rooms.read_events(store, placement="profile", room_id="upgrade-room")["events"]


def append(store, event_id, kind, payload, actor_kind="user"):
    return rooms.append_event(store, placement="profile", room_id="upgrade-room", event_id=event_id, kind=kind,
        actor={"kind": actor_kind, "id": "home"}, authority_gateway_id="home", authority_epoch=1, payload=payload)


def legacy_cache(db, store, cache_version):
    room = rooms.create_room(store, placement="profile", room_id="upgrade-room", name="Upgrade", authority_gateway_id="home",
        members=[{"member_id": profile, "profile": profile, "handle": profile} for profile in PROFILES])
    append(store, "source", "message.user", {"text": "@writer Prepare the plan.", "thread_id": "thread"})
    original = discussion.plan_next_task(room, log(store), local_profiles=PROFILES, freeze_input_context=True).task
    deferred = discussion.plan_publication(room, log(store), original, status="deferred", execution_generation=1,
        result={"reason": "member_unavailable"}, local_profiles=PROFILES)
    for event in deferred.events:
        rooms.append_event(store, placement="profile", **event.append_kwargs("upgrade-room"))
    append(store, "silent", "room.activity", {"status": "settled", "reason_code": "silent_round",
        "thread_id": "thread", "discussion_event_id": "source"}, "gateway")
    completed = discussion.plan_publication(room, log(store), original, status="settled",
        result={"text": "The committed plan is ready."}, local_profiles=PROFILES)
    rooms.append_event(store, placement="profile", **completed.events[0].append_kwargs("upgrade-room"))
    append(store, "newer", "message.user", {"text": "@reviewer Use the earlier plan.", "thread_id": "thread"})
    checkpoint = HostedRoomPolicyCheckpoint(store, placement="profile")
    before = checkpoint.snapshot(room_id="upgrade-room", latest_seq=log(store)[-1]["seq"])
    late = rooms.append_event(store, placement="profile", **completed.events[-1].append_kwargs("upgrade-room"))
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
def test_old_cache_rebuilds_late_receipts_and_context_without_changing_history(tmp_path, cache_version, store):
    db = tmp_path / "state.db"
    room, original, old_plan = legacy_cache(db, store, cache_version)
    history = log(store)
    checkpoint = HostedRoomPolicyCheckpoint(store, placement="profile")
    updated = checkpoint.snapshot(room_id="upgrade-room", latest_seq=history[-1]["seq"])
    assert checkpoint.publication_exists(room_id="upgrade-room", task_id=original.identity.task_id,
                                         status="settled", execution_generation=2)
    planned = discussion.plan_next_task(room, updated.events, local_profiles=PROFILES,
        initial_watermarks=updated.watermarks, freeze_input_context=True).task
    assert "The committed plan is ready." in planned.payload["prompt"]
    assert planned.identity.task_id != old_plan.identity.task_id
    assert log(store) == history
    assert HostedRoomPolicyCheckpoint(store, placement="profile").snapshot(room_id="upgrade-room", latest_seq=history[-1]["seq"]) == updated


def bounded_historical_settlement(store, *, with_other_thread_source=False):
    room = rooms.create_room(store, placement="profile", room_id="upgrade-room", name="Upgrade", authority_gateway_id="home",
        members=[{"member_id": profile, "profile": profile, "handle": profile} for profile in PROFILES])
    if with_other_thread_source:
        append(store, "other-source", "message.user", {"text": "@reviewer Separate work.", "thread_id": "other"})
        append(store, "complete-other", "room.activity", {"status": "settled", "reason_code": "silent_round",
            "thread_id": "other", "discussion_event_id": "other-source"}, "gateway")
    append(store, "source", "message.user", {"text": "@writer Prepare the plan.", "thread_id": "thread"})
    initial = discussion.plan_next_task(room, log(store), local_profiles=PROFILES, freeze_input_context=True).task
    assert initial is not None
    publication = discussion.plan_publication(room, log(store), initial, status="settled",
        result={"text": "The committed plan is ready."}, local_profiles=PROFILES)
    for event in publication.events:
        rooms.append_event(store, placement="profile", **event.append_kwargs("upgrade-room"))
    append(store, "complete-source", "room.activity", {"status": "settled", "reason_code": "silent_round",
        "thread_id": "thread", "discussion_event_id": "source"}, "gateway")
    for index in range(23):
        append(store, f"later-{index}", "message.user", {"text": "@reviewer Continue.", "thread_id": "thread"})
    return room


def test_snapshot_does_not_import_settlement_source_from_other_thread(tmp_path, store):
    """A malformed historical settlement cannot borrow another thread's durable user."""
    db = tmp_path / "state.db"
    room = bounded_historical_settlement(store, with_other_thread_source=True)

    # Keep the legitimate member reply and terminal, but corrupt their source link
    # to name a real user event in another thread of the same room.
    with sqlite3.connect(db) as conn:
        for kind in ("message.member", "turn.settled"):
            seq, encoded = conn.execute(
                "SELECT seq, payload_json FROM hosted_room_events "
                "WHERE room_id='upgrade-room' AND kind=? ORDER BY seq LIMIT 1", (kind,)
            ).fetchone()
            payload = json.loads(encoded)
            payload["discussion_event_id"] = "other-source"
            conn.execute(
                "UPDATE hosted_room_events SET payload_json=? WHERE room_id='upgrade-room' AND seq=?",
                (json.dumps(payload), seq))

    checkpoint = HostedRoomPolicyCheckpoint(store, placement="profile")
    snapshot = checkpoint.snapshot(room_id="upgrade-room", latest_seq=log(store)[-1]["seq"])
    assert "other-source" not in {event["event_id"] for event in snapshot.events}
    with pytest.raises(discussion.DiscussionValidationError, match="no matching user discussion"):
        discussion.plan_next_task(
            room, snapshot.events, local_profiles=PROFILES, initial_watermarks=snapshot.watermarks)


def test_snapshot_keeps_source_user_for_retained_non_pass_settlement(store):
    room = bounded_historical_settlement(store)

    checkpoint = HostedRoomPolicyCheckpoint(store, placement="profile")
    snapshot = checkpoint.snapshot(room_id="upgrade-room", latest_seq=log(store)[-1]["seq"])

    assert any(event["event_id"] == "source" for event in snapshot.events)
    assert discussion.plan_next_task(room, snapshot.events, local_profiles=PROFILES,
                                     initial_watermarks=snapshot.watermarks).status == "task"


def test_snapshot_leaves_missing_settlement_source_invalid(tmp_path, store):
    db = tmp_path / "state.db"
    room = bounded_historical_settlement(store)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM hosted_room_events WHERE room_id='upgrade-room' AND event_id='source'")

    checkpoint = HostedRoomPolicyCheckpoint(store, placement="profile")
    snapshot = checkpoint.snapshot(room_id="upgrade-room", latest_seq=log(store)[-1]["seq"])

    with pytest.raises(discussion.DiscussionValidationError, match="no matching user discussion"):
        discussion.plan_next_task(room, snapshot.events, local_profiles=PROFILES,
                                  initial_watermarks=snapshot.watermarks)


@pytest.mark.parametrize("status", ["queued", "running", "indeterminate"])
@pytest.mark.parametrize("cache_version", [2, 3])
def test_upgrade_preserves_an_already_admitted_frozen_turn(tmp_path, monkeypatch, status, cache_version, store):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "home")
    db = tmp_path / "state.db"
    _, _, old_plan = legacy_cache(db, store, cache_version)
    clock = [time.time()]
    admitted = driver.admit_task(store, old_plan.identity, placement="profile", payload=old_plan.payload, clock=lambda: clock[0])
    if status != "queued":
        lease = driver.acquire_lease(store, placement="profile", room_id="upgrade-room", gateway_id="home", authority_epoch=1,
            process_generation="prior", ttl_seconds=1, clock=lambda: clock[0])
        driver.start_task(store, old_plan.identity, lease, placement="profile", expected_cancel_generation=0, clock=lambda: clock[0])
        if status == "indeterminate":
            clock[0] += 2
            current = driver.acquire_lease(store, placement="profile", room_id="upgrade-room", gateway_id="home", authority_epoch=1,
                process_generation="current", ttl_seconds=60, clock=lambda: clock[0])
            driver.recover_room(store, current, placement="profile", clock=lambda: clock[0])
    history = log(store)
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, store=store, placement="profile")
    service.local_profiles = lambda: PROFILES
    service.runtime.clock = lambda: clock[0]
    monkeypatch.setattr(driver, "admit_task", lambda *a, **kw: pytest.fail("cache rebuild readmitted work"))
    service.prepare_room(service.bindings()[0])
    # Reopening the service repeats only derived-cache preparation, not execution.
    cold = HostedRoomService(server, store=store, placement="profile")
    cold.local_profiles = lambda: PROFILES
    cold.prepare_room(cold.bindings()[0])
    existing = driver.get_task_for_turn(store, old_plan.identity, placement="profile")
    assert existing["status"] == status
    assert existing["identity"] == admitted["identity"]
    assert existing["payload"] == admitted["payload"]
    assert log(store) == history
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_driver_tasks WHERE room_id='upgrade-room'").fetchone()[0] == 1


def test_version_five_checkpoint_replays_unread_interval(tmp_path, store):
    rooms.create_room(store, placement='profile', room_id='upgrade-room', name='Upgrade', authority_gateway_id='home',
                      members=[{'member_id': profile, 'profile': profile} for profile in PROFILES])
    append(store, 'source', 'message.user', {'thread_id': 'thread', 'text': 'Original input'})
    common = {'thread_id': 'thread', 'discussion_event_id': 'source'}
    append(store, 'intervening', 'message.member', {**common, 'member_id': 'reviewer', 'text': 'New evidence'}, 'member')
    append(store, 'late', 'message.member', {**common, 'member_id': 'writer', 'text': 'Finished'}, 'member')
    append(store, 'settled', 'turn.settled', {**common, 'member_id': 'writer', 'message_event_id': 'late',
                                               'task_id': 'slow', 'seen_through_seq': 1}, 'gateway')
    checkpoint = HostedRoomPolicyCheckpoint(store, placement='profile')
    checkpoint.snapshot(room_id='upgrade-room', latest_seq=4)
    # v5 consumed the settlement at 4 and credited the writer through its own message at 3.
    with sqlite3.connect(tmp_path / 'state.db') as conn:
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=5 WHERE room_id='upgrade-room'")
        conn.execute("UPDATE hosted_room_policy_watermarks SET seen_through_seq=3 WHERE member_id='writer'")
    snapshot = checkpoint.snapshot(room_id='upgrade-room', latest_seq=4)
    assert snapshot.watermarks[('thread', 'writer')] == 1
