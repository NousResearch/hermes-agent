"""A retried room task starts with no progress of its own: Retry keeps the task id, so the old attempt's
tool and last-frame time must not ride into the new attempt's ``member_activity`` row."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from gateway import hosted_room_driver as driver
from tui_gateway.hosted_room_member_activity import record_task_progress, task_progress
from tui_gateway.hosted_room_service import HostedRoomService


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
    return service, task["identity"].task_id


def test_retry_drops_the_old_attempts_progress_and_rows_name_their_thread_and_round(tmp_path, monkeypatch):
    service, task_id = _deferred_room(tmp_path, monkeypatch)
    # The first attempt was last seen in a tool long ago.
    record_task_progress("room", task_id, {"task_id": task_id, "tool": "terminal", "progress_at": 1.0})

    service.retry_room_task("room", task_id=task_id, member_id="worker", execution_generation=1)

    assert task_progress("room", task_id) == {}
    [row] = [a for a in service.status("room")["member_activity"] if a["member_id"] == "worker"]
    assert row["status"] == "queued" and "tool" not in row and "progress_at" not in row
    assert (row["thread_id"], row["round_index"]) == ("thread", 0)
