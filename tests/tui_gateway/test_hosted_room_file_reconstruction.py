"""Files-only service recovery, without Bot outputs or Messaging dependencies."""

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_attachments import AttachmentNotFoundError
from tests.tui_gateway.hosted_room_service_fixtures import _server
from tui_gateway.hosted_room_service import HostedRoomService


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


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("outcome", ["cancelled", "pass", "failed", "deferred"])
def test_nonvisible_file_turn_preserves_later_text_admission(
    tmp_path, monkeypatch, legacy, outcome
):
    if legacy:
        original = discussion.plan_next_task
        monkeypatch.setattr(
            discussion,
            "plan_next_task",
            lambda *a, **kw: original(
                *a,
                **{**kw, "freeze_input_context": False},
            ),
        )
    service = _new(tmp_path)
    attachment = _file(service, "input")
    _send(service, "initial", [attachment])
    prior = _queued(service)
    if outcome == "cancelled":
        assert service.stop_room("files", cancel_id="stop") == 1
    else:
        attempt = _start(service, prior)
        if outcome == "deferred":
            driver.defer_not_admitted_task(
                service.db_path, attempt, reason="offline", clock=time.time
            )
        else:
            driver.settle_task(
                service.db_path,
                attempt,
                settlement_id="prior",
                clock=time.time,
                status="settled" if outcome == "pass" else "failed",
                result={"text": "(pass)"} if outcome == "pass" else {"error": "failed"},
            )
    _tick(service)
    _send(service, "followup")
    task = _queued(service)
    assert not task["payload"].get("attachments")
    assert ("input_context" in task["payload"]) != legacy
    _settle(service, task)
    immutable = driver.get_task(service.db_path, task["identity"])
    cold = _service(service.db_path)
    _tick(cold)
    assert _reconstruct(cold, immutable).payload == task["payload"]
    messages = [
        event for event in cold._events("files") if event["kind"] == "message.member"
    ]
    assert len(messages) == 1 and messages[0]["payload"]["text"] == "Reviewed"
    assert (
        driver.get_task(service.db_path, task["identity"])["result"]
        == immutable["result"]
    )
    before = cold._events("files")
    _tick(cold)
    assert cold._events("files") == before
    assert (
        cold.read_attachment(
            room_id="files",
            attachment_id=attachment["attachment_id"],
            recipient_member_id="builder",
        ).data
        == b"input"
    )


@pytest.mark.parametrize("migrate", [False, True])
def test_partial_visible_reply_keeps_remaining_file_batch(
    tmp_path, monkeypatch, migrate
):
    service = _new(tmp_path)
    attachments = []
    with monkeypatch.context() as pause:
        pause.setattr(service, "prepare_room", lambda binding: None)
        for batch, count in enumerate((8, 8, 1)):
            group = [_file(service, f"file-{batch}-{index}") for index in range(count)]
            attachments.extend(group)
            _send(service, f"batch-{batch}", group)
    _tick(service)
    first = _queued(service)
    assert first["payload"]["attachments"] == attachments[:16]
    _settle(service, first, "First sixteen reviewed")
    room = hosted_rooms.room_state(service.db_path, room_id="files")
    assert service._publish_terminal_tasks(room)
    if migrate:
        with sqlite3.connect(service.db_path) as conn:
            conn.execute(
                "UPDATE hosted_room_policy_transcript_state SET schema_version=1"
            )
            conn.execute("UPDATE hosted_room_policy_watermarks SET seen_through_seq=4")
    cold = _service(service.db_path)
    _tick(cold)
    second = _queued(cold)
    assert second["payload"]["attachments"] == attachments[16:]
    assert second["identity"] != first["identity"]
    assert _reconstruct(cold, second).payload == second["payload"]
    assert (
        cold.policy_checkpoint.snapshot(
            room_id="files",
            latest_seq=hosted_rooms.room_state(service.db_path, room_id="files")[
                "latest_seq"
            ],
        ).watermarks[("work", "builder")]
        == 2
    )


def test_frozen_input_survives_projection_cleanup_and_rejects_missing_canonical_input(
    tmp_path, monkeypatch
):
    service = _new(tmp_path)
    _send(service, "initial", [_file(service, "input")])
    task = _queued(service)
    with monkeypatch.context() as pause:
        pause.setattr(service, "prepare_room", lambda binding: None)
        for index in range(30):
            _send(service, f"later-{index}")
    room = hosted_rooms.room_state(service.db_path, room_id="files")
    service.policy_checkpoint.snapshot(room_id="files", latest_seq=room["latest_seq"])
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("DELETE FROM hosted_room_policy_events WHERE seq=1")
        conn.execute("DELETE FROM hosted_room_policy_transcript WHERE seq=1")
        conn.execute("UPDATE hosted_room_policy_watermarks SET seen_through_seq=999")
    assert _reconstruct(service, task).payload == task["payload"]
    _settle(service, task)
    _tick(service)
    assert not any(
        event["kind"] == "message.member" for event in service._events("files")
    )
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("DELETE FROM hosted_room_events WHERE room_id='files' AND seq=1")
    with pytest.raises(RuntimeError, match="input event is missing"):
        _task_events(service, task)


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


@pytest.mark.parametrize(
    "context",
    [
        {"watermark": True, "event_seqs": [1]},
        {"watermark": 0, "event_seqs": [True]},
        {"watermark": 0, "event_seqs": [2, 1]},
        {"watermark": 0, "event_seqs": [2**63]},
        {"watermark": 0, "event_seqs": list(range(1, 130))},
    ],
)
def test_invalid_input_context_is_rejected_by_real_driver_admission(tmp_path, context):
    service = _new(tmp_path)
    _send(service, "initial")
    task = _queued(service)
    with pytest.raises(driver.DriverValidationError):
        driver.admit_task(
            service.db_path,
            task["identity"],
            payload={**task["payload"], "input_context": context},
            clock=time.time,
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_published_text_wins_newer_request_after_crash(tmp_path, monkeypatch, legacy):
    service = _new(tmp_path)
    with monkeypatch.context() as admission:
        if legacy:
            original = discussion.plan_next_task
            admission.setattr(
                discussion,
                "plan_next_task",
                lambda *a, **kw: original(
                    *a,
                    **{**kw, "freeze_input_context": False},
                ),
            )
        _send(service, "initial", [_file(service, "input")])
    task = _queued(service)
    _settle(service, task)
    append = hosted_rooms.append_event

    def interrupt(*args, **kwargs):
        if kwargs["kind"] == "turn.settled":
            raise OSError("interrupted terminal append")
        return append(*args, **kwargs)

    with monkeypatch.context() as crash:
        crash.setattr(hosted_rooms, "append_event", interrupt)
        with pytest.raises(OSError, match="interrupted"):
            _tick(service)
    with monkeypatch.context() as pause:
        pause.setattr(service, "prepare_room", lambda binding: None)
        _send(service, "newer")
    cold = _service(service.db_path)
    _tick(cold)
    own = [
        event
        for event in cold._events("files")
        if event["payload"].get("task_id") == task["identity"].task_id
    ]
    assert [event["kind"] for event in own] == ["message.member", "turn.settled"]
    assert _reconstruct(cold, task).payload == task["payload"]


@pytest.mark.parametrize("same_thread", [False, True])
def test_new_request_at_terminal_append_retries_without_rerunning_task(
    tmp_path, monkeypatch, same_thread
):
    service = _new(tmp_path)
    _send(service, "initial", [_file(service, "input")])
    task = _queued(service)
    _settle(service, task)
    original = hosted_rooms.append_event

    def newer(*args, **kwargs):
        if kwargs.get("kind") != "message.member":
            return original(*args, **kwargs)
        other = _service(service.db_path)
        with monkeypatch.context() as pause:
            pause.setattr(other, "prepare_room", lambda binding: None)
            _send(other, "newer", thread="work" if same_thread else "other")
        return original(*args, **kwargs)

    with monkeypatch.context() as pause:
        pause.setattr(hosted_rooms, "append_event", newer)
        _tick(service)
    assert not any(
        event["kind"] == "message.member" for event in service._events("files")
    )
    cold = _service(service.db_path)
    _tick(cold)
    own = [
        event
        for event in cold._events("files")
        if event["payload"].get("task_id") == task["identity"].task_id
    ]
    assert [event["kind"] for event in own] == (
        ["turn.cancelled"] if same_thread else ["message.member", "turn.settled"]
    )
    assert driver.get_task(service.db_path, task["identity"])["result"] == {
        "text": "Reviewed"
    }


def test_legacy_admission_winning_prepare_race_keeps_its_original_identity(
    tmp_path, monkeypatch
):
    service = _new(tmp_path)
    with monkeypatch.context() as pause:
        pause.setattr(service, "prepare_room", lambda binding: None)
        _send(service, "initial", [_file(service, "input")])
    room = hosted_rooms.room_state(service.db_path, room_id="files")
    snapshot = service._policy_snapshot(room)
    legacy = discussion.plan_next_task(
        room,
        snapshot.events,
        local_profiles=service.local_profiles(),
        initial_watermarks=snapshot.watermarks,
    ).task
    assert legacy is not None and "input_context" not in legacy.payload
    original = driver.get_task_for_turn

    def other_admission(db, identity):
        driver.admit_task(db, legacy.identity, payload=legacy.payload, clock=time.time)
        return original(db, identity)

    monkeypatch.setattr(driver, "get_task_for_turn", other_admission)
    _tick(service)
    task = _queued(service)
    assert task["identity"] == legacy.identity
    assert task["payload"] == legacy.payload


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


@pytest.mark.parametrize("publish_before_rollback", [False, True])
def test_competing_input_senders_cannot_publish_revoked_files(
    tmp_path, monkeypatch, publish_before_rollback
):
    first = _new(tmp_path)
    second = _service(first.db_path)
    attachment = _file(first, "input")
    payload = {"text": "input", "thread_id": "work", "attachments": [attachment]}
    actual = hosted_rooms.append_event
    first_ready, second_ready, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    failures = []

    def first_send():
        try:
            first.send(room_id="files", event_id="shared", payload=payload)
        except OSError as exc:
            failures.append(str(exc))

    def append(*args, **kwargs):
        if threading.current_thread().name.startswith("first-sender"):
            first_ready.set()
            assert release.wait(10)
            raise OSError("sender lost before append")
        second_ready.set()
        if publish_before_rollback:
            result = actual(*args, **kwargs)
            release.set()
            running.result(timeout=10)
            return result
        release.set()
        running.result(timeout=10)
        return actual(*args, **kwargs)

    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="first-sender") as pool,
        monkeypatch.context() as pause,
    ):
        pause.setattr(hosted_rooms, "append_event", append)
        pause.setattr(second, "prepare_room", lambda binding: None)
        running = pool.submit(first_send)
        try:
            assert first_ready.wait(10)
            if publish_before_rollback:
                second.send(room_id="files", event_id="shared", payload=payload)
            else:
                with pytest.raises(hosted_rooms.EventAttachmentConflictError):
                    second.send(room_id="files", event_id="shared", payload=payload)
                assert not second._events("files")
                with pytest.raises(AttachmentNotFoundError):
                    second.read_attachment(
                        room_id="files",
                        attachment_id=attachment["attachment_id"],
                        recipient_member_id="builder",
                    )
        finally:
            release.set()
            running.result(timeout=10)
    assert first_ready.is_set() and second_ready.is_set() and len(failures) == 1
    with monkeypatch.context() as pause:
        pause.setattr(second, "prepare_room", lambda binding: None)
        second.send(room_id="files", event_id="shared", payload=payload)
    cold = _service(first.db_path)
    assert len(cold._events("files")) == 1
    assert (
        cold.read_attachment(
            room_id="files",
            event_id="shared",
            attachment_id=attachment["attachment_id"],
            recipient_member_id="builder",
        ).data
        == b"input"
    )
