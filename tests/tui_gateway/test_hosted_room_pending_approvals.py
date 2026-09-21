"""Room approval observations must remain bound to a live durable attempt."""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture
def service(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "reviewer").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=home / "state.db")
    service.create_room(room_id="room", name="Room", members=[
        {"member_id": "worker", "profile": "default", "handle": "worker"},
        {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
    ])
    return service


class ApprovalRPC:
    def __init__(self):
        self.task = self.on_terminal = None
        self.generation = 0
        self.completed = threading.Event()
        self.approvals = []

    def resolve_exact(self, **kwargs):
        return {"session_id": "member-session", "title": "Group: room"}

    def resume(self, **kwargs):
        return {"session_id": "member-session"}

    def submit(self, *, task, execution_generation, on_terminal, **kwargs):
        self.task, self.generation, self.on_terminal = task, execution_generation, on_terminal
        return {"accepted": True}

    def history(self, **kwargs):
        if not self.completed.is_set():
            return []
        return [{"role": "assistant", "task_id": self.task.task_id,
                 "execution_generation": self.generation, "status": "settled",
                 "message_id": "reply", "content": "Review complete."}]

    def info(self, **kwargs):
        return {"active": True, "task_id": self.task.task_id,
                "pending_approval": {"request_id": "approval", "choices": ["once", "deny"]}}

    def interrupt(self, **kwargs):
        return {"interrupted": True}

    def approve(self, **kwargs):
        self.approvals.append(kwargs)
        return {"resolved": 1}


def _approve(service, action):
    return service.approve_room_task(
        "room", member_id=action["member_id"], task_id=action["task_id"],
        execution_generation=action["execution_generation"],
        request_id=action["request_id"], choice="once")


@pytest.mark.parametrize("completion", ["callback", "failure", "history", "stop"])
@pytest.mark.parametrize("boundary", ["status", "approve"])
def test_terminal_attempt_drops_approval_without_another_info_poll(service, completion, boundary):
    service.send(room_id="room", event_id="source", payload={"text": "Review", "thread_id": "thread"})
    task = driver.list_tasks(service.db_path, room_id="room", status="queued")[0]
    rpc = ApprovalRPC()
    service.rpc = service.runtime.rpc = rpc
    pending_seen, release_poll = threading.Event(), threading.Event()

    def observe_pending(room_id, member_id, action):
        service._set_pending_action(room_id, member_id, action)
        pending_seen.set()
        assert release_poll.wait(5)

    service.runtime.pending_action = observe_pending
    worker = threading.Thread(target=service.runtime._run_room_once, args=(service.bindings()[0],))
    worker.start()
    try:
        assert pending_seen.wait(5)
        action = service.status("room")["pending_actions"][0]
        assert action["request_id"] == "approval"
        if completion in {"callback", "failure"}:
            rpc.on_terminal({"status": "failed" if completion == "failure" else "settled", "text": "PASS"})
        elif completion == "history":
            rpc.completed.set()
        else:
            service.stop_room("room", cancel_id="stop")
    finally:
        release_poll.set()
        worker.join(5)
    assert not worker.is_alive()
    assert driver.get_task(service.db_path, task["identity"])["status"] in driver.TERMINAL_STATUSES

    # A delayed info response can arrive after completion, including after an
    # earlier status read already removed the first stale observation.
    for _ in range(2):
        service._set_pending_action("room", action["member_id"], action)
        if boundary == "approve":
            with pytest.raises(RuntimeError, match="no longer pending"):
                _approve(service, action)
        assert service.status("room")["pending_actions"] == []
    assert rpc.approvals == []


@pytest.mark.parametrize("mismatch", ["missing", "generation", "member"])
@pytest.mark.parametrize("boundary", ["status", "approve"])
@pytest.mark.parametrize("task_status", ["running", "indeterminate", "deferred"])
def test_pending_approval_is_scoped_to_its_durable_attempt(service, mismatch, boundary, task_status):
    identity = driver.TaskIdentity("room", "task", "thread", "turn")
    driver.admit_task(service.db_path, identity, payload={
        "target_profile": "default", "target_member_id": "worker", "source_event_seq": 1,
        "prompt": "Review"}, clock=service.runtime.clock)
    lease = service.runtime._ensure_lease(service.bindings()[0])
    attempt = driver.start_task(service.db_path, identity, lease, expected_cancel_generation=0,
                                clock=service.runtime.clock)
    # Exercise an actual generation change rather than inventing a stale number.
    driver.requeue_not_admitted_task(service.db_path, attempt, clock=service.runtime.clock)
    attempt = driver.start_task(service.db_path, identity, lease, expected_cancel_generation=0,
                                clock=service.runtime.clock)
    if task_status != "running":
        recovery_now = lease.expires_at + 1
        def recovery_clock():
            return recovery_now
        lease = driver.acquire_lease(
            service.db_path, room_id="room", gateway_id=lease.gateway_id,
            authority_epoch=lease.authority_epoch, process_generation="recovery",
            ttl_seconds=30, clock=recovery_clock)
        driver.recover_room(service.db_path, lease, clock=recovery_clock)
        if task_status == "deferred":
            driver.defer_indeterminate_task(
                service.db_path, identity, lease, expected_execution_generation=attempt.execution_generation,
                expected_cancel_generation=0, reason="observation-unavailable", clock=recovery_clock)
    assert driver.get_task(service.db_path, identity)["status"] == task_status
    rpc = ApprovalRPC()
    service.rpc = rpc
    action = {"kind": "approval", "member_id": "worker", "task_id": identity.task_id,
              "execution_generation": attempt.execution_generation, "request_id": "approval",
              "session_id": "member-session", "approval": {"choices": ["once", "deny"]}}
    stale = dict(action)
    if mismatch == "missing":
        stale["task_id"] = "removed-task"
    elif mismatch == "generation":
        stale["execution_generation"] -= 1
    else:
        stale["member_id"] = "reviewer"
    service._set_pending_action("room", stale["member_id"], stale)
    if boundary == "approve":
        with pytest.raises(RuntimeError, match="no longer pending"):
            _approve(service, stale)
    assert not any(a["kind"] == "approval" for a in service.status("room")["pending_actions"])
    assert rpc.approvals == []

    # Pruning an old observation must not poison a replacement's valid request.
    service._set_pending_action("room", "worker", action)
    assert action in service.status("room")["pending_actions"]
    assert _approve(service, action) == {"resolved": 1}
    assert rpc.approvals == [{"session_id": "member-session", "request_id": "approval", "choice": "once"}]
