"""Manual holds and persisted stop intent through the real hosted-room coordinator.

Every test drives the actual service against a temp HERMES_HOME and a real SQLite store. No
worker is started and no transport is reachable, so admission, cancellation and status are
exercised while nothing can submit a prompt or create a session.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import ModuleType

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from tui_gateway.hosted_room_service import HostedRoomService


ROOM_ID = "holds-room"
MEMBERS = [
    {"member_id": "member-impl", "profile": "impl", "handle": "impl", "display_name": "Impl Bot"},
    {"member_id": "member-research", "profile": "research", "handle": "research"}]


class _SessionlessRPC:
    """A local session adapter that owns no session: the state after a process restart.

    Every probe answers authoritatively (a hosted turn cannot outlive its canonical session)
    without creating one, so no prompt, model call or child session can happen in these tests.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve_exact(self, *, profile, title, source):
        self.calls.append("resolve_exact")
        return None

    def create(self, **kwargs):
        raise AssertionError("these tests must never create a room session")

    def submit(self, **kwargs):
        raise AssertionError("these tests must never submit a prompt")

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def history(self, *, profile, session_id, source):
        return []

    def info(self, *, profile, session_id, source):
        return {"active": False, "task_id": None}

    def interrupt(self, *, profile, session_id, source, expected_task_id,
                  expected_execution_generation=None):
        return {"interrupted": False}


@pytest.fixture
def service(tmp_path, monkeypatch) -> HostedRoomService:
    home = tmp_path / "runtime"
    for member in MEMBERS:
        (home / "profiles" / member["profile"]).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    service = HostedRoomService(ModuleType("test_server"), db_path=home / "state.db")
    service.rpc = service.runtime.rpc = _SessionlessRPC()
    service.create_room(room_id=ROOM_ID, name="Holds", members=MEMBERS)
    return service


def _binding(service: HostedRoomService):
    return next(b for b in service.bindings() if b.room_id == ROOM_ID)


def _send(service: HostedRoomService, event_id: str, text: str, thread_id: str = "thread-1") -> dict:
    return service.send(
        room_id=ROOM_ID, event_id=event_id, payload={"text": text, "thread_id": thread_id})


def _tasks(service: HostedRoomService, status: str | None = None) -> list[dict]:
    return driver.list_tasks(service.db_path, room_id=ROOM_ID, status=status)


def _held_handles(service: HostedRoomService) -> list[str]:
    return [hold["handle"] for hold in service.status(ROOM_ID)["holds"]]


def _crash_stop(service: HostedRoomService, *, cancel_id: str) -> dict:
    """Commit the durable stop event only, exactly as a crash before cancellation leaves it."""
    room = hosted_rooms.room_state(service.db_path, room_id=ROOM_ID)
    return hosted_rooms.request_room_stop(
        service.db_path, room_id=ROOM_ID, cancel_id=cancel_id,
        expected_gateway_id=str(room["authority_gateway_id"]),
        expected_epoch=int(room["authority_epoch"]))


def _foreign_admitted_attempt(service: HostedRoomService, task: dict, *, ttl_seconds: float = 0.05):
    """Start ``task`` under another process's lease and stamp its admission fence."""
    room = hosted_rooms.room_state(service.db_path, room_id=ROOM_ID)
    lease = driver.acquire_lease(
        service.db_path, room_id=ROOM_ID, gateway_id=str(room["authority_gateway_id"]),
        authority_epoch=int(room["authority_epoch"]), process_generation="paused-owner",
        ttl_seconds=ttl_seconds, clock=time.time)
    attempt = driver.start_task(
        service.db_path, task["identity"], lease, expected_cancel_generation=0, clock=time.time)
    driver.fence_task_admission(service.db_path, attempt, clock=time.time)
    deadline = time.monotonic() + 2.0
    while time.time() < lease.expires_at and time.monotonic() < deadline:
        time.sleep(0.01)
    return attempt


def test_stop_holds_every_member_and_a_fresh_unaddressed_thread_dispatches_nothing(service):
    _send(service, "user-1", "@all start the release checks")
    assert len(_tasks(service)) == 1

    service.stop_room(ROOM_ID, cancel_id="desktop-stop")
    assert sorted(_held_handles(service)) == ["impl", "research"]

    _send(service, "user-2", "any update?", thread_id="thread-2")
    service.prepare_room(_binding(service))

    assert [task["status"] for task in _tasks(service)] == ["cancelled"]
    assert not _tasks(service, "queued")
    assert service.status(ROOM_ID)["working"] is False


def test_a_direct_mention_releases_only_the_addressed_member(service):
    _send(service, "user-1", "@all stop")
    assert sorted(_held_handles(service)) == ["impl", "research"]

    _send(service, "user-2", "@all standup in five")
    assert sorted(_held_handles(service)) == ["impl", "research"], "plain @all is not a release"

    _send(service, "user-3", "@impl what did the logs say?", thread_id="thread-2")

    assert _held_handles(service) == ["research"]
    queued = _tasks(service, "queued")
    assert [task["payload"]["target_member_id"] for task in queued] == ["member-impl"]


def test_status_labels_held_members_from_the_durable_roster(service):
    _send(service, "user-1", "@impl pause")

    assert service.status(ROOM_ID)["holds"] == [{
        "member_id": "member-impl", "handle": "impl", "display_name": "Impl Bot",
        "held_at_seq": 1}]


def test_persisted_stop_intent_cancels_surviving_queued_work_before_dispatch(service):
    """A crash between the durable stop event and cancellation must not leak a dispatch."""
    _send(service, "user-1", "@all start the release checks")
    queued = _tasks(service, "queued")
    assert len(queued) == 1

    _crash_stop(service, cancel_id="stop-that-never-cancelled")
    service.prepare_room(_binding(service))

    assert driver.get_task(service.db_path, queued[0]["identity"])["status"] == "cancelled"
    assert not _tasks(service, "queued")


def test_a_queued_task_for_a_newly_held_member_is_cancelled_before_dispatch(service):
    _send(service, "user-1", "@impl start the release checks")
    queued = _tasks(service, "queued")
    assert [task["payload"]["target_member_id"] for task in queued] == ["member-impl"]

    # A hold that lands after admission (the message supersedes nothing in another thread).
    _send(service, "user-2", "@impl stop", thread_id="thread-2")
    service.prepare_room(_binding(service))

    assert driver.get_task(service.db_path, queued[0]["identity"])["status"] == "cancelled"
    assert not _tasks(service, "queued")


def test_retry_cannot_bypass_a_held_member(service):
    _send(service, "user-1", "@impl inspect the release")
    task = _tasks(service, "queued")[0]
    attempt = _foreign_admitted_attempt(service, task)
    room = hosted_rooms.room_state(service.db_path, room_id=ROOM_ID)
    lease = driver.acquire_lease(
        service.db_path, room_id=ROOM_ID, gateway_id=str(room["authority_gateway_id"]),
        authority_epoch=int(room["authority_epoch"]),
        process_generation=service.runtime.process_generation, ttl_seconds=30, clock=time.time)
    service.runtime._leases[ROOM_ID] = lease
    driver.recover_room(service.db_path, lease, clock=time.time)
    assert driver.get_task(service.db_path, task["identity"])["status"] == "indeterminate"

    _send(service, "user-2", "@impl stop", thread_id="thread-2")

    with pytest.raises(driver.InvalidTaskTransitionError, match="paused"):
        service.retry_room_task(ROOM_ID, task_id=task["identity"].task_id)
    assert driver.get_task(service.db_path, task["identity"])["status"] == "indeterminate"
    assert attempt.execution_generation == 1


def test_holds_survive_a_service_rebuilt_on_the_same_database(service):
    _send(service, "user-1", "@all start the release checks")
    service.stop_room(ROOM_ID, cancel_id="desktop-stop")

    reopened = HostedRoomService(ModuleType("test_server"), db_path=service.db_path)
    reopened.send(
        room_id=ROOM_ID, event_id="user-2", payload={"text": "still there?", "thread_id": "thread-3"})

    assert sorted(hold["handle"] for hold in reopened.status(ROOM_ID)["holds"]) == ["impl", "research"]
    assert not driver.list_tasks(reopened.db_path, room_id=ROOM_ID, status="queued")


def test_stop_never_reports_an_unproven_admitted_attempt_as_finished(service):
    """require_acknowledged Stop must refuse while another process may still dispatch."""
    _send(service, "user-1", "@impl inspect the release")
    task = _tasks(service, "queued")[0]
    _foreign_admitted_attempt(service, task)

    with pytest.raises(RuntimeError, match="still stopping"):
        service.stop_room(ROOM_ID, cancel_id="disband-stop", require_acknowledged=True)

    stopped = driver.get_task(service.db_path, task["identity"])
    assert stopped["status"] == "indeterminate", "an unproven stop is uncertain, never cancelled"
    assert service.status(ROOM_ID)["blocked"] is True


def test_persisted_stop_replay_leaves_an_unproven_admitted_attempt_uncertain(service):
    _send(service, "user-1", "@impl inspect the release")
    task = _tasks(service, "queued")[0]
    _foreign_admitted_attempt(service, task)
    _crash_stop(service, cancel_id="stop-that-never-cancelled")

    service.prepare_room(_binding(service))

    replayed = driver.get_task(service.db_path, task["identity"])
    assert replayed["status"] == "indeterminate"
    assert not _tasks(service, "queued")
