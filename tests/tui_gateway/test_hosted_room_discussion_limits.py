"""Owner-scoped configuration and durable native Discussion budget contracts."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from tui_gateway.hosted_room_service import HostedRoomService


PROFILES = ("research", "build", "review", "ops", "qa", "docs")
ROOM_ID = "budget-room"


def _service(root: Path) -> HostedRoomService:
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=root / "state.db")
    service.local_profiles = lambda: PROFILES
    return service


def _create(service: HostedRoomService, member_count: int) -> None:
    service.create_room(
        room_id=ROOM_ID,
        name="Budget test",
        members=[
            {"member_id": profile, "profile": profile, "handle": profile}
            for profile in PROFILES[:member_count]
        ],
    )
    service.send(
        room_id=ROOM_ID,
        event_id="kickoff",
        payload={"text": "Discuss.", "thread_id": "thread-1"},
    )


def _settle(service: HostedRoomService, index: int, text: str = "@everyone next") -> None:
    binding = service.bindings()[0]
    queued = driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued")
    assert len(queued) == 1, f"expected reply {index + 1} to be admitted"
    task = queued[0]
    events = service._events(ROOM_ID)
    # Exercise the same persisted-task reconstruction used after a worker restart.
    discussion.reconstruct_task_plan(service._room(ROOM_ID), events, task, local_profiles=PROFILES)
    lease = driver.acquire_lease(
        service.db_path,
        room_id=ROOM_ID,
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="budget-test",
        ttl_seconds=30,
        clock=time.time,
    )
    attempt = driver.start_task(
        service.db_path,
        task["identity"],
        lease,
        expected_cancel_generation=0,
        clock=time.time,
    )
    driver.settle_task(
        service.db_path,
        attempt,
        settlement_id=f"reply-{index}",
        status="settled",
        result={"text": text},
        clock=time.time,
    )
    service.prepare_room(binding)


@pytest.mark.parametrize(
    ("member_count", "rounds", "messages", "expected", "reason"),
    [(3, 6, 30, 18, "max_rounds"), (6, 6, 30, 30, "max_messages"),
     (6, 6, 40, 36, "max_rounds"),
     (2, 12, 40, 24, "max_rounds")],
)
def test_owner_config_controls_rounds_shared_messages_and_restart_replay(
    tmp_path, member_count, rounds, messages, expected, reason,
):
    (tmp_path / "config.yaml").write_text(
        f"hosted_rooms:\n  discussion:\n    max_rounds: {rounds}\n    max_messages: {messages}\n"
    )
    service = _service(tmp_path)
    _create(service, member_count)
    for index in range(expected):
        if index == 12:
            # A reconstructed coordinator must retain the configured budget.
            service = _service(tmp_path)
        _settle(service, index)
    events = service._events(ROOM_ID)
    replies = [event for event in events if event["kind"] == "message.member"]
    assert len(replies) == expected
    assert max(event["payload"]["round_index"] for event in replies) == (expected - 1) // member_count
    assert events[-1]["kind"] == "room.activity"
    assert events[-1]["payload"]["reason_code"] == reason
    assert driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued") == []
    # Repeated prepares must not re-admit or duplicate any publication after the bound.
    service.prepare_room(service.bindings()[0])
    assert service._events(ROOM_ID) == events


def test_lowered_budget_keeps_old_higher_round_history_valid(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("hosted_rooms:\n  discussion:\n    max_rounds: 6\n    max_messages: 40\n")
    service = _service(tmp_path)
    _create(service, 6)
    for index in range(33):
        _settle(service, index)
    # Existing round-5 work remains reconstructable; lowering a ceiling is not log corruption.
    config.write_text("hosted_rooms:\n  discussion:\n    max_rounds: 2\n    max_messages: 5\n")
    service = _service(tmp_path)
    _settle(service, 33)
    assert service._events(ROOM_ID)[-1]["payload"]["reason_code"] == "max_messages"
    service.send(
        room_id=ROOM_ID, event_id="next-kickoff",
        payload={"text": "New request.", "thread_id": "thread-2"},
    )
    assert len(driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued")) == 1


def test_terminal_only_turns_do_not_exhaust_the_checkpoint_before_the_reply_budget(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "hosted_rooms:\n  discussion:\n    max_rounds: 6\n    max_messages: 30\n"
    )
    service = _service(tmp_path)
    _create(service, 6)
    for index in range(36):
        _settle(service, index, "(pass)" if index % 6 == 0 else "@everyone next")
    events = service._events(ROOM_ID)
    assert sum(event["kind"] == "message.member" for event in events) == 30
    assert sum(event["kind"] == "turn.settled" for event in events) == 36
    assert events[-1]["payload"]["reason_code"] == "max_messages"


def test_stop_still_fences_work_after_the_previous_ten_reply_ceiling(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "hosted_rooms:\n  discussion:\n    max_rounds: 6\n    max_messages: 30\n"
    )
    service = _service(tmp_path)
    _create(service, 3)
    for index in range(13):
        _settle(service, index)
    assert service.stop_room(ROOM_ID, cancel_id="stop-after-13") == 1
    service.prepare_room(service.bindings()[0])
    assert driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued") == []
    assert sum(event["kind"] == "message.member" for event in service._events(ROOM_ID)) == 13


def test_owner_settings_do_not_leak_from_a_member_scope_and_defaults_are_unchanged(tmp_path):
    from gateway.run import _profile_runtime_scope

    owner, member = tmp_path / "owner", tmp_path / "member"
    owner.mkdir()
    member.mkdir()
    (member / "config.yaml").write_text(
        "hosted_rooms:\n  discussion:\n    max_rounds: 99\n    max_messages: 999\n"
    )
    with _profile_runtime_scope(member, hydrate_secrets=False):
        service = _service(owner)
    assert service.discussion_limits == discussion.DiscussionLimits(max_rounds=3, max_messages=10)
    _create(service, 3)
    for index in range(9):
        _settle(service, index)
    assert service._events(ROOM_ID)[-1]["payload"]["reason_code"] == "max_rounds"


@pytest.mark.parametrize("key", ["max_rounds", "max_messages"])
@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5", "'6'", "null", "[]", "{}"])
def test_invalid_limits_are_rejected_before_any_room_work(tmp_path, key, value):
    (tmp_path / "config.yaml").write_text(f"hosted_rooms:\n  discussion:\n    {key}: {value}\n")
    with pytest.raises(discussion.DiscussionValidationError, match=f"{key} must be a positive integer"):
        _service(tmp_path)
    assert not (tmp_path / "state.db").exists()


@pytest.mark.parametrize("section", ["hosted_rooms: []", "hosted_rooms:\n  discussion: []"])
def test_invalid_config_sections_fail_closed(tmp_path, section):
    (tmp_path / "config.yaml").write_text(section + "\n")
    with pytest.raises(discussion.DiscussionValidationError, match="must be a mapping"):
        _service(tmp_path)
