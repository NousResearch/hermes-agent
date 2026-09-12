"""Same-thread delivery and on-disk policy migration regressions."""

import sqlite3
from pathlib import Path

from gateway import hosted_rooms
from tui_gateway.hosted_room_service import HostedRoomService
from tests.tui_gateway.test_hosted_room_service import (
    _BlockingFirstRPC, _PromptRecordingRPC, _server, _wait_for,
)


def test_same_thread_followup_migrates_and_delivers_committed_peer_reply(
    tmp_path: Path, monkeypatch,
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

    services = [service]
    try:
        service.start()
        service.send(
            room_id="room-1", event_id="user-1",
            payload={"text": "@ops provide the marker", "thread_id": "thread-1"},
        )
        _wait_for(lambda: len(service.rpc.prompts) == 1)
        _wait_for(lambda: any(
            event["kind"] == "room.activity"
            and event["payload"]["discussion_event_id"] == "user-1"
            for event in service._events("room-1")
        ))
        # Build an old-format fixture offline, not by racing a live policy reader.
        assert service.stop(timeout=2)
        canonical = hosted_rooms.read_events(db, room_id="room-1")
        with sqlite3.connect(db) as conn:
            assert conn.execute(
                """SELECT COUNT(*) FROM hosted_room_policy_transcript
                   WHERE room_id='room-1' AND thread_id='thread-1'"""
            ).fetchone()[0] == 2
            conn.execute("DELETE FROM hosted_room_policy_transcript WHERE room_id='room-1'")
            conn.execute("DELETE FROM hosted_room_policy_transcript_state WHERE room_id='room-1'")
        assert hosted_rooms.read_events(db, room_id="room-1") == canonical

        recovered = HostedRoomService(_server(), db_path=db)
        services.append(recovered)
        recovered.rpc = service.rpc
        recovered.runtime.rpc = recovered.rpc
        recovered.local_profiles = service.local_profiles
        backfills = []
        original_backfill = recovered.policy_checkpoint._backfill_transcript

        def observed_backfill(conn, *, room_id, through_seq):
            backfills.append(through_seq)
            return original_backfill(conn, room_id=room_id, through_seq=through_seq)

        monkeypatch.setattr(recovered.policy_checkpoint, "_backfill_transcript", observed_backfill)
        recovered.start()
        recovered.send(
            room_id="room-1", event_id="user-2",
            payload={"text": "@hermes continue", "thread_id": "thread-1"},
        )
        _wait_for(lambda: len(recovered.rpc.prompts) == 2)
        assert recovered.stop(timeout=2)
        assert backfills and backfills[0] > 0
        with sqlite3.connect(db) as conn:
            assert conn.execute(
                "SELECT 1 FROM hosted_room_policy_transcript_state WHERE room_id='room-1'"
            ).fetchone()
        profile, prompt = recovered.rpc.prompts[1]
        assert profile == "default"
        assert "@ops: reply from ops" in prompt
        assert "User (user): @hermes continue" in prompt
    finally:
        for instance in services:
            assert instance.stop(timeout=2)


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
