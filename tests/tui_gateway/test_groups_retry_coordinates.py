"""Standalone ``groups.state`` retry rows carry the exact attempt ``groups.retry`` accepts back."""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import closing
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv
from gateway import hosted_room_driver as driver
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
    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: service)
    return service, task["identity"].task_id


def test_retry_row_round_trips_through_groups_retry(tmp_path, monkeypatch):
    service, task_id = _deferred_room(tmp_path, monkeypatch)

    [action] = [a for a in service.status("room")["pending_actions"] if a["kind"] == "retry"]
    assert action == {"kind": "retry", "member_id": "worker", "task_id": task_id,
                      "execution_generation": 1}
    coordinates = {key: value for key, value in action.items() if key != "kind"}

    stale = srv.handle_request({"id": 1, "method": "groups.retry", "params": {
        "room_id": "room", **coordinates, "execution_generation": 2}})
    assert "error" in stale
    retried = srv.handle_request(
        {"id": 2, "method": "groups.retry", "params": {"room_id": "room", **coordinates}})
    assert retried["result"]["task"]["status"] == "queued"


def test_stale_retry_cannot_requeue_a_newer_attempt(tmp_path, monkeypatch):
    service, task_id = _deferred_room(tmp_path, monkeypatch)
    identity = driver.list_tasks(service.db_path, room_id="room", status="deferred")[0]["identity"]
    select = service._list_tasks

    def select_then_advance(room_id, statuses):
        candidates = list(select(room_id, statuses))
        # Another Retry runs generation 2, which also ends deferred, before this request resumes.
        lease = service.runtime._ensure_lease(service.bindings()[0])
        driver.requeue_deferred_task(service.db_path, identity, lease, expected_execution_generation=1,
                                     expected_cancel_generation=0, clock=service.runtime.clock)
        driver.start_task(service.db_path, identity, lease, expected_cancel_generation=0,
                          clock=service.runtime.clock)
        with closing(sqlite3.connect(service.db_path)) as conn, conn:
            conn.execute("UPDATE hosted_room_driver_tasks SET status='deferred' WHERE task_id=?", (task_id,))
        return candidates

    monkeypatch.setattr(service, "_list_tasks", select_then_advance)
    with pytest.raises(driver.StaleTaskError):
        service.retry_room_task("room", task_id=task_id, member_id="worker", execution_generation=1)
    task = driver.get_task(service.db_path, identity)
    assert (task["status"], task["execution_generation"]) == ("deferred", 2)
