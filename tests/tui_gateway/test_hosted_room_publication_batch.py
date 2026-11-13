"""Terminal scans batch receipt reads without changing retry suppression."""

import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from tui_gateway import hosted_room_service
from tui_gateway.hosted_room_service import HostedRoomService


def service(tmp_path):
    db = tmp_path / "state.db"
    for room_id in ("one", "two"):
        hosted_rooms.create_room(
            db, room_id=room_id, name=room_id,
            members=[{"member_id": "writer", "profile": "writer", "handle": "writer"}],
            authority_gateway_id="gateway",
        )
    instance = HostedRoomService(SimpleNamespace(), db_path=db)
    instance.local_profiles = lambda: ("writer",)
    return instance


def task(task_id, status="settled", generation=1):
    return {"identity": SimpleNamespace(task_id=task_id), "status": status,
            "execution_generation": generation, "payload": {"source_event_seq": 1}}


@pytest.mark.parametrize("count", [1, 100])
def test_published_terminal_scan_uses_one_checkpoint_transaction(tmp_path, monkeypatch, count):
    instance = service(tmp_path)
    tasks = [task(f"task-{index}") for index in range(count)]
    with sqlite3.connect(instance.db_path) as conn:
        conn.executemany(
            "INSERT INTO hosted_room_policy_publications VALUES (?, ?, ?, ?, ?)",
            [("one", item["identity"].task_id, "turn.settled", 1, index + 1)
             for index, item in enumerate(tasks)],
        )
    monkeypatch.setattr(instance, "_list_tasks", lambda *args: iter(tasks))
    original = instance.policy_checkpoint._transaction
    calls = []

    @contextmanager
    def counted():
        calls.append(True)
        with original() as conn:
            yield conn

    monkeypatch.setattr(instance.policy_checkpoint, "_transaction", counted)
    assert not instance._publish_terminal_tasks(instance._room("one"))
    assert len(calls) == 1
    assert len(instance._events("one")) == 0


def test_scan_preserves_terminal_retry_and_room_scope(tmp_path, monkeypatch):
    instance = service(tmp_path)
    receipts = [("one", "settled", "turn.settled", 1, 1),
                ("one", "failed", "turn.failed", 1, 2),
                ("one", "cancelled", "turn.cancelled", 1, 3),
                ("one", "retry", "turn.deferred", 1, 4),
                ("two", "foreign", "turn.settled", 1, 1)]
    with sqlite3.connect(instance.db_path) as conn:
        conn.executemany("INSERT INTO hosted_room_policy_publications VALUES (?, ?, ?, ?, ?)", receipts)
    tasks = [task("settled", generation=2), task("failed", "cancelled", 2),
             task("cancelled", "failed", 2), task("retry", "deferred", 1),
             task("retry", "deferred", 2), task("foreign"), task("missing")]
    monkeypatch.setattr(instance, "_list_tasks", lambda *args: iter(tasks))
    reconstructed = []
    monkeypatch.setattr(instance.policy_checkpoint, "events_for_task", lambda **kwargs: [])

    def reconstruct(room, events, item, **kwargs):
        reconstructed.append((item["identity"].task_id, item["execution_generation"]))
        return SimpleNamespace()

    monkeypatch.setattr(hosted_room_service.discussion, "reconstruct_task_plan", reconstruct)
    monkeypatch.setattr(hosted_room_service.discussion, "plan_publication",
                        lambda *args, **kwargs: SimpleNamespace(events=()))
    assert instance._publish_terminal_tasks(instance._room("one"))
    assert reconstructed == [("retry", 2), ("foreign", 1), ("missing", 1)]
