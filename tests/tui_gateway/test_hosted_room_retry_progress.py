"""A retried room task starts with no progress of its own: Retry keeps the task id, so the old attempt's
tool and last-frame time must not ride into the new attempt's ``member_activity`` row."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from hermes_agent.gateway import hosted_room_driver as driver
from hermes_agent.tui_gateway.hosted_room_member_activity import record_task_progress, task_progress
from hermes_agent.tui_gateway.hosted_room_service import HostedRoomService


def _deferred_room(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".hermes" / "profiles" / "reviewer").mkdir(parents=True)
    (tmp_path / ".hermes" / "profiles" / "reviewer" / "config.yaml").write_text("{}\n")
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=tmp_path / ".hermes" / "state.db")
    service.create_room(room_id="room", name="Room", members=[
        {"member_id": "worker", "profile": "default", "handle": "worker"},
        {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"}])
    service.send(room_id="room", event_id="source", payload={"text": "Review", "thread_id": "thread"})
    binding = service.bindings()[0]
    task = driver.list_tasks(service.db_path, room_id="room", status="queued")[0]
    now = [time.time()]
    service.runtime.clock = clock = lambda: now[0]
    old = driver.acquire_lease(
        service.db_path, room_id="room", gateway_id=binding.gateway_id, authority_epoch=1,
        process_generation="old", ttl_seconds=1, clock=clock)
    driver.start_task(service.db_path, task["identity"], old, expected_cancel_generation=0, clock=clock)
    now[0] += 2
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(service.db_path, lease, clock=clock)
    driver.defer_indeterminate_task(
        service.db_path, task["identity"], lease, expected_execution_generation=1,
        expected_cancel_generation=0, reason="member_unavailable", clock=clock)
    return service, task["identity"].task_id, lease


def test_retry_drops_the_old_attempts_progress_and_rows_name_their_thread_and_round(tmp_path, monkeypatch):
    service, task_id, _lease = _deferred_room(tmp_path, monkeypatch)
    # The first attempt was last seen in a tool long ago.
    record_task_progress("room", task_id, 1, {"task_id": task_id, "tool": "terminal", "progress_at": 1.0})

    service.retry_room_task("room", task_id=task_id, member_id="worker", execution_generation=1)

    assert task_progress("room", task_id, 1) == {}
    [row] = [a for a in service.status("room")["member_activity"] if a["member_id"] == "worker"]
    assert row["status"] == "queued" and "tool" not in row and "progress_at" not in row
    assert (row["thread_id"], row["round_index"]) == ("thread", 0)


def test_a_late_poll_of_the_replaced_attempt_never_becomes_the_retry_attempts_progress(tmp_path, monkeypatch):
    service, task_id, lease = _deferred_room(tmp_path, monkeypatch)
    [gen1] = driver.list_tasks(service.db_path, room_id="room", status="deferred")
    gen1_info = {"active": True, "task_id": task_id, "execution_generation": 1, "tool": "terminal", "progress_at": 1.0}

    def poll(task, info):  # the driver's info poll, answered for the attempt it was started for
        service.runtime._report_pending_action(task, session_id="worker-session", info=info)

    service.retry_room_task("room", task_id=task_id, member_id="worker", execution_generation=1)
    [queued] = driver.list_tasks(service.db_path, room_id="room", status="queued")
    attempt = driver.start_task(service.db_path, queued["identity"], lease, expected_cancel_generation=0,
                                clock=service.runtime.clock)
    assert attempt.execution_generation == 2
    # The poll of generation 1 that was in flight across Retry answers after generation 2 started.
    poll(gen1, gen1_info)
    row = _worker_row(service)
    assert row["status"] == "running" and "tool" not in row and "progress_at" not in row

    [gen2] = driver.list_tasks(service.db_path, room_id="room", status="running")
    poll(gen2, {**gen1_info, "execution_generation": 2, "tool": "web_search", "progress_at": 5.0})
    poll(gen1, gen1_info)
    assert {k: _worker_row(service)[k] for k in ("tool", "progress_at")} == {"tool": "web_search", "progress_at": 5.0}


def _worker_row(service):
    [row] = [a for a in service.status("room")["member_activity"] if a["member_id"] == "worker"]
    return row
