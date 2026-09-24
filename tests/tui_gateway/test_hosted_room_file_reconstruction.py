"""Uncovered reconstruction contracts retained from the historical Files owner.

Source: 691bb08a5cd310fffc5f3d01653dc93f394fc080,
``tests/tui_gateway/test_hosted_room_file_reconstruction.py``. Existing accepted
canonical tests cover partial batches, frozen publication, cursor races, peer
progress, and publication rollback. These two contracts retain identity-tamper
and bounded interrupted-migration coverage without starting a service worker.
"""

import sqlite3
import threading
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from tui_gateway.hosted_room_service import HostedRoomService


def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


def _service(db):
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "ops")
    return service


def _new(tmp_path):
    service = _service(tmp_path / "state.db")
    service.create_room(
        room_id="files",
        name="Files",
        members=[
            {"member_id": "planner", "profile": "default", "handle": "planner"},
            {"member_id": "builder", "profile": "ops", "handle": "builder"},
        ],
    )
    return service


def _send(service, name, attachments=(), thread="work"):
    return service.send(
        room_id="files",
        event_id=name,
        payload={
            "text": "@builder " + name,
            "thread_id": thread,
            **({"attachments": list(attachments)} if attachments else {}),
        },
    )


def _file(service, name):
    uploaded = service.put_attachment(
        room_id="files",
        upload_id=name,
        kind="file",
        name=name + ".txt",
        mime="text/plain",
        data=name.encode(),
    )
    return {
        key: uploaded[key] for key in ("attachment_id", "kind", "name", "size", "mime")
    }


def _tick(service):
    service.prepare_room(service.bindings()[0])


def _queued(service):
    tasks = driver.list_tasks(service.db_path, room_id="files", status="queued")
    assert len(tasks) == 1
    return tasks[0]


def _start(service, task):
    binding = service.bindings()[0]
    lease = driver.acquire_lease(
        service.db_path,
        room_id="files",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="file-test",
        ttl_seconds=300,
        clock=time.time,
    )
    return driver.start_task(
        service.db_path,
        task["identity"],
        lease,
        expected_cancel_generation=task["cancel_generation"],
        clock=time.time,
    )


def _settle(service, task, text="Reviewed"):
    driver.settle_task(
        service.db_path,
        _start(service, task),
        settlement_id="result",
        status="settled",
        result={"text": text},
        clock=time.time,
    )


def _task_events(service, task):
    return service.policy_checkpoint.events_for_task(
        room_id="files",
        source_event_seq=task["payload"]["source_event_seq"],
        input_context=task["payload"].get("input_context"),
        task_id=task["identity"].task_id,
    )


def _reconstruct(service, task):
    return discussion.reconstruct_task_plan(
        hosted_rooms.room_state(service.db_path, room_id="files"),
        _task_events(service, task),
        task,
        local_profiles=service.local_profiles(),
    )


@pytest.mark.parametrize("change", ["watermark", "events", "target", "attachment"])
def test_historical_input_and_task_identity_cannot_be_retargeted(tmp_path, change):
    service = _new(tmp_path)
    _send(service, "initial", [_file(service, "input")])
    task = deepcopy(_queued(service))
    if change == "watermark":
        task["payload"]["input_context"]["watermark"] = 1
    elif change == "events":
        task["payload"]["input_context"]["event_seqs"] = [2]
    elif change == "target":
        task["payload"]["target_member_id"] = "planner"
    else:
        task["payload"]["attachments"][0]["name"] = "different.txt"
    with pytest.raises((ValueError, RuntimeError)):
        _reconstruct(service, task)


def test_interrupted_migration_resumes_bounded_pages_and_input_read_is_indexed(
    tmp_path, monkeypatch
):
    service = _new(tmp_path)
    _send(service, "initial", [_file(service, "input")])
    task = _queued(service)
    room = hosted_rooms.room_state(service.db_path, room_id="files")
    for index in range(1100):
        hosted_rooms.append_event(
            service.db_path,
            room_id="files",
            event_id=f"noise-{index}",
            kind="message.user",
            actor={"kind": "user", "id": "desktop"},
            authority_gateway_id=room["authority_gateway_id"],
            authority_epoch=room["authority_epoch"],
            payload={"text": "unrelated", "thread_id": "other"},
        )
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=1")
    original = hosted_rooms.read_events
    pages = []

    def interrupted(*args, **kwargs):
        pages.append(kwargs["since_seq"])
        if kwargs["since_seq"] >= 500:
            raise OSError("migration interrupted")
        return original(*args, **kwargs)

    with monkeypatch.context() as pause:
        pause.setattr(hosted_rooms, "read_events", interrupted)
        with pytest.raises(OSError, match="migration interrupted"):
            service.policy_checkpoint.sync(room_id="files", latest_seq=1101)
    assert pages == [0, 500]
    with sqlite3.connect(service.db_path) as conn:
        assert (
            conn.execute(
                "SELECT through_seq FROM hosted_room_policy_cursors"
            ).fetchone()[0]
            == 500
        )
    cold = _service(service.db_path)
    snapshot = cold.policy_checkpoint.snapshot(room_id="files", latest_seq=1101)
    assert snapshot.through_seq == 1101
    with monkeypatch.context() as check:
        check.setattr(
            hosted_rooms,
            "read_events",
            lambda *a, **kw: pytest.fail("caught-up replay"),
        )
        cold.policy_checkpoint.snapshot(room_id="files", latest_seq=1101)
    connect = cold.policy_checkpoint._connect
    steps = []

    def bounded_connection():
        conn = connect()
        conn.set_progress_handler(lambda: steps.append(1) or int(len(steps) > 500), 1)
        return conn

    monkeypatch.setattr(cold.policy_checkpoint, "_connect", bounded_connection)
    assert _reconstruct(cold, task).payload == task["payload"]
    assert 0 < len(steps) <= 500
