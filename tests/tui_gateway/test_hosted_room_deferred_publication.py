"""Publishing a turn that outlived its discussion, through the real service and callbacks.

Deferred and uncertain work is not live work, so a discussion can settle and be compacted while
one of its tasks is still unpublished. These drive the public service API and its real publish
callback against a real SQLite room log, a real policy checkpoint and the real Linux profile lock.

The terminal receipts are deliberately fabricated fixtures: this is publication and reconstruction
evidence, never proof of native execution or cessation. Nothing here may dispatch a prompt.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from tui_gateway.hosted_room_service import HostedRoomService


ROOM_ID = "publication-room"
THREAD = "thread-a"
MEMBERS = [
    {"member_id": "default", "profile": "default", "handle": "hermes"},
    {"member_id": "worker", "profile": "worker", "handle": "worker"}]


class ReceiptRPC:
    """A session adapter that can only report history: it never runs or creates anything."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.submits = 0

    def resolve_exact(self, **kwargs):
        return {"session_id": "stored-session"}

    def resume(self, **kwargs):
        return {"session_id": "stored-session"}

    def history(self, **kwargs):
        return list(self.messages)

    def info(self, **kwargs):
        return {"active": False, "task_id": None}

    def submit(self, **kwargs):
        self.submits += 1
        raise AssertionError("late publication must never dispatch a prompt")

    def create(self, **kwargs):
        raise AssertionError("late publication must never create a session")


@pytest.fixture
def room(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    for member in MEMBERS:
        (home / "profiles" / member["profile"]).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    clock = [time.time()]
    rpc = ReceiptRPC()

    def build() -> HostedRoomService:
        service = HostedRoomService(
            ModuleType("offline_server"), db_path=home / "state.db", profiles_root=home)
        service.rpc = service.runtime.rpc = rpc
        service.runtime.clock = lambda: clock[0]
        service.runtime.lease_ttl_seconds = 100
        return service

    service = build()
    service.create_room(room_id=ROOM_ID, name="Publication", members=MEMBERS)
    return {"service": service, "build": build, "rpc": rpc, "clock": clock, "db": home / "state.db"}


def _events(db: Path) -> list[dict[str, Any]]:
    return hosted_rooms.read_events(db, room_id=ROOM_ID, limit=hosted_rooms.MAX_LOG_LIMIT)["events"]


def _own(db: Path, task_id: str, kind: str) -> list[dict[str, Any]]:
    return [
        event for event in _events(db)
        if event["kind"] == kind and event["payload"].get("task_id") == task_id]


def _deferred_task(room, *, foreign: bool, text: str = "@worker inspect this fixture"):
    """Admit one task, park it as deferred, and let its discussion complete and compact."""
    service, db, clock = room["service"], room["db"], room["clock"]
    service.send(room_id=ROOM_ID, event_id="source-user", payload={"text": text, "thread_id": THREAD})
    binding = service.bindings()[0]
    task = driver.list_tasks(db, room_id=ROOM_ID, status="queued")[0]
    identity = task["identity"]
    owner = driver.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch,
        process_generation="prior-owner" if foreign else service.runtime.process_generation,
        ttl_seconds=1, clock=lambda: clock[0])
    attempt = driver.start_task(db, identity, owner, expected_cancel_generation=0, clock=lambda: clock[0])
    driver.fence_task_admission(db, attempt, clock=lambda: clock[0])
    clock[0] = owner.expires_at + 1
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(db, lease, clock=lambda: clock[0])
    driver.defer_indeterminate_task(
        db, identity, lease, expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=0, reason="member_unavailable", clock=lambda: clock[0])
    service.prepare_room(binding)
    service.prepare_room(binding)
    kinds = [event["kind"] for event in _events(db)]
    assert "turn.deferred" in kinds and "room.activity" in kinds, kinds
    assert driver.get_task(db, identity)["status"] == "deferred"
    return binding, identity, attempt


def _receipt(room, identity, attempt, *, status: str = "settled", text: str = "Recovered result"):
    room["rpc"].messages.append({
        "role": "assistant", "task_id": identity.task_id,
        "execution_generation": attempt.execution_generation, "status": status,
        "message_id": f"late-{status}-receipt", "content": text})


@pytest.mark.parametrize("foreign", [False, True], ids=["same-owner", "foreign-owner"])
def test_a_late_receipt_publishes_after_its_discussion_completed(room, foreign: bool):
    service, db = room["service"], room["db"]
    binding, identity, attempt = _deferred_task(room, foreign=foreign)
    _receipt(room, identity, attempt)

    resolved = service.retry_room_task(ROOM_ID, task_id=identity.task_id)

    assert resolved["status"] == "settled"
    assert resolved["execution_generation"] == attempt.execution_generation
    assert resolved["admitted_at"] is not None
    messages = _own(db, identity.task_id, "message.member")
    assert [event["payload"]["text"] for event in messages] == ["Recovered result"]
    assert len(_own(db, identity.task_id, "turn.settled")) == 1
    assert room["rpc"].submits == 0
    assert driver.list_tasks(db, room_id=ROOM_ID, status="queued") == []


def test_repeated_preparation_publishes_the_outcome_exactly_once(room):
    service, db = room["service"], room["db"]
    binding, identity, attempt = _deferred_task(room, foreign=True)
    _receipt(room, identity, attempt)
    service.retry_room_task(ROOM_ID, task_id=identity.task_id)

    before = _events(db)
    for _ in range(3):
        service.prepare_room(binding)

    assert _events(db) == before
    assert len(_own(db, identity.task_id, "message.member")) == 1
    assert len(_own(db, identity.task_id, "turn.settled")) == 1
    assert service.policy_checkpoint.publication_exists(
        room_id=ROOM_ID, task_id=identity.task_id, status="settled", execution_generation=0)
    assert room["rpc"].submits == 0


def test_a_rebuilt_service_publishes_the_same_late_outcome(room):
    db = room["db"]
    _binding, identity, attempt = _deferred_task(room, foreign=True)
    room["clock"][0] += 200.0
    reopened = room["build"]()  # real checkpoint rehydration from the same store
    _receipt(room, identity, attempt)

    resolved = reopened.retry_room_task(ROOM_ID, task_id=identity.task_id)
    reopened.prepare_room(reopened.bindings()[0])

    assert resolved["status"] == "settled"
    assert len(_own(db, identity.task_id, "message.member")) == 1
    assert len(_own(db, identity.task_id, "turn.settled")) == 1
    assert room["rpc"].submits == 0


def test_a_late_failure_is_published_through_the_same_reconstruction(room):
    service, db = room["service"], room["db"]
    _binding, identity, attempt = _deferred_task(room, foreign=True)
    _receipt(room, identity, attempt, status="failed", text="the member reported an error")

    resolved = service.retry_room_task(ROOM_ID, task_id=identity.task_id)

    assert resolved["status"] == "failed"
    assert len(_own(db, identity.task_id, "turn.failed")) == 1
    assert _own(db, identity.task_id, "message.member") == []
    assert room["rpc"].submits == 0


def _settle_without_retry(room, binding, identity, *, text: str = "Recovered result"):
    """Commit the exact receipt at the store, as a worker-side settlement leaves it.

    Explicit Retry is refused while a member is held, so this is how a late outcome reaches the
    room in that state: the durable settlement is already there and the service's own preparation
    publishes it. Fenced on the row's real current generations, which a Stop may have moved.
    """
    lease = room["service"].runtime._ensure_lease(binding)
    task = driver.get_task(room["db"], identity)
    assert task["status"] in {"deferred", "indeterminate"}, task["status"]
    return driver.resolve_indeterminate_task(
        room["db"], identity, lease,
        expected_execution_generation=int(task["execution_generation"]),
        expected_cancel_generation=int(task["cancel_generation"]),
        settlement_id="late-exact-receipt", status="settled", result={"text": text},
        clock=lambda: room["clock"][0], from_status=str(task["status"]))


def test_later_input_in_the_same_thread_supersedes_a_late_result(room):
    """Stale prose must not appear under a newer user message, even across the transcript window."""
    service, db = room["service"], room["db"]
    binding, identity, attempt = _deferred_task(room, foreign=True)
    # Hold the roster so the follow-up messages cannot dispatch new work, then roll the thread
    # transcript well past its retained window (MAX_THREAD_TRANSCRIPT_EVENTS).
    service.send(room_id=ROOM_ID, event_id="stop-all", payload={"text": "@all stop", "thread_id": THREAD})
    for index in range(30):
        service.send(
            room_id=ROOM_ID, event_id=f"later-{index}",
            payload={"text": f"note {index}", "thread_id": THREAD})
    assert _settle_without_retry(room, binding, identity)["status"] == "settled"

    service.prepare_room(binding)

    cancelled = _own(db, identity.task_id, "turn.cancelled")
    assert len(cancelled) == 1
    assert cancelled[0]["payload"]["reason"] == "superseded_by_newer_user_event"
    assert _own(db, identity.task_id, "message.member") == []
    assert room["rpc"].submits == 0


def test_a_late_publication_does_not_release_holds_or_dispatch_new_work(room):
    service, db = room["service"], room["db"]
    binding, identity, attempt = _deferred_task(room, foreign=True)
    # Holds are room-scoped, so this stop reaches the task's member from another thread while
    # leaving the task's own thread free of newer input (which would supersede its result).
    service.send(
        room_id=ROOM_ID, event_id="stop-all", payload={"text": "@all stop", "thread_id": "thread-b"})
    held_before = sorted(hold["handle"] for hold in service.status(ROOM_ID)["holds"])
    assert held_before == ["hermes", "worker"]
    _receipt(room, identity, attempt)

    # Holding the roster also stops that member's outstanding work: the deferred task is asked to
    # stop and, being admitted by a process this one does not run, resolves as uncertain.
    assert driver.get_task(db, identity)["status"] == "indeterminate"
    # Retry stays refused for a held member: a pause is not a recovery gesture.
    with pytest.raises(driver.InvalidTaskTransitionError, match="paused"):
        service.retry_room_task(ROOM_ID, task_id=identity.task_id)
    _settle_without_retry(room, binding, identity)
    service.prepare_room(binding)

    assert len(_own(db, identity.task_id, "message.member")) == 1
    assert sorted(hold["handle"] for hold in service.status(ROOM_ID)["holds"]) == held_before
    assert driver.list_tasks(db, room_id=ROOM_ID, status="queued") == []
    assert room["rpc"].submits == 0


def test_an_outcome_whose_source_event_is_gone_is_still_refused(room):
    """The fix reads the canonical source; it never fabricates one."""
    service, db = room["service"], room["db"]
    _binding, identity, attempt = _deferred_task(room, foreign=True)
    _receipt(room, identity, attempt)
    # Simulate a room whose canonical source event is genuinely unavailable.
    service.policy_checkpoint.events_for_task = lambda **kwargs: []

    with pytest.raises(Exception, match="task source user event is missing"):
        service.retry_room_task(ROOM_ID, task_id=identity.task_id)

    assert _own(db, identity.task_id, "message.member") == []
    assert room["rpc"].submits == 0
