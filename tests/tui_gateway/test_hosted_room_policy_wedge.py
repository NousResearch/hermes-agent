"""A room whose policy input cannot be validated must be visible, not silently wedged.

Regression: ``DiscussionValidationError`` escaped the room poll on every tick — no task, no
activity, no signal — so the group chat simply stopped answering, and sending more messages
changed nothing. The preparation boundary now counts the failure, records one durable
``member.unavailable`` notice with Retry, and stops re-preparing until the next Send.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from tui_gateway import hosted_room_prepare_stall as prepare_stall
from tui_gateway.hosted_room_service import HostedRoomService

PROFILES = ("writer", "reviewer")
POLICY_ERROR = "turn.settled references no matching user discussion"


def _service(tmp_path, monkeypatch) -> HostedRoomService:
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (home / "profiles" / "reviewer").mkdir(parents=True)
    (home / "profiles" / "reviewer" / "config.yaml").write_text("{}\n")  # identity marker
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=home / "state.db")
    service.create_room(room_id="room", name="Room", members=[
        {"member_id": "writer", "profile": "default", "handle": "writer"},
        {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
    ])
    return service


def _notices(service: HostedRoomService) -> list[dict]:
    return [event for event in service._events("room") if event["kind"] == "member.unavailable"]


def test_policy_validation_failure_is_one_visible_outcome_and_then_stops(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    binding = service.bindings()[0]

    def fail(*_args, **_kwargs):
        raise discussion.DiscussionValidationError(POLICY_ERROR)

    monkeypatch.setattr(discussion, "plan_next_task", fail)

    for attempt in range(prepare_stall.STALL_ATTEMPTS):
        with pytest.raises(discussion.DiscussionValidationError, match=POLICY_ERROR):
            service._prepare_tracked(binding)
        # Only the failure that stalls the room says anything in the room.
        assert len(_notices(service)) == (1 if attempt + 1 == prepare_stall.STALL_ATTEMPTS else 0)
    assert service._prepare_stalls.stalled("room")

    service._prepare_tracked(binding)  # no exception: a stalled room is not polled again
    notices = _notices(service)
    assert len(notices) == 1
    assert notices[0]["payload"]["reason_code"] == "prepare_failed"
    assert POLICY_ERROR in notices[0]["payload"]["error"]
    assert prepare_stall.retry_action(
        "room", hosted_rooms.room_state(service.db_path, room_id="room"))["task_id"] == "prepare:room"


def test_a_new_user_message_after_a_policy_stall_is_stored_and_retried(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    binding = service.bindings()[0]

    def fail(*_args, **_kwargs):
        raise discussion.DiscussionValidationError(POLICY_ERROR)

    monkeypatch.setattr(discussion, "plan_next_task", fail)
    for _ in range(prepare_stall.STALL_ATTEMPTS):
        with pytest.raises(discussion.DiscussionValidationError):
            service._prepare_tracked(binding)
    assert service._prepare_stalls.stalled("room")

    stored = service.send(room_id="room", event_id="continue",
                          payload={"text": "Continue", "thread_id": "thread"})
    assert stored["event_id"] == "continue"
    kinds = [event["kind"] for event in service._events("room")]
    assert kinds.count("message.user") == 1 and kinds.count("member.unavailable") == 1
    # The stall restarts from this Send rather than swallowing it, and says so once more at most.
    assert not service._prepare_stalls.stalled("room")
    assert service._prepare_stalls.last_error("room") == POLICY_ERROR
    assert len(_notices(service)) == 1
