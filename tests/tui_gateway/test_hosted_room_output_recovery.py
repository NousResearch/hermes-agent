"""Bot-output source contracts, composed on the Files reconstruction helpers."""

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_artifacts import (
    RoomArtifactError,
    RoomArtifactOutbox,
    RoomArtifactScope,
    terminal_artifact_manifest,
)
from gateway.hosted_room_attachments import AttachmentNotFoundError
from tests.tui_gateway.test_hosted_room_file_reconstruction import (
    _file,
    _new,
    _queued,
    _send,
    _service,
    _start,
    _tick,
)


FILES = {
    "index.html": b"<!doctype html><title>Verified output</title>",
    "app.js": b"export const ready = true;\n",
    "README.md": b"# Verified three-file handoff\n",
}


def _legacy(monkeypatch, enabled):
    if enabled:
        planner = discussion.plan_next_task
        monkeypatch.setattr(
            discussion,
            "plan_next_task",
            lambda *args, **kwargs: planner(
                *args, **{**kwargs, "freeze_input_context": False}
            ),
        )


def _output(service, task):
    attempt = _start(service, task)
    room = hosted_rooms.room_state(service.db_path, room_id="files")
    scope = RoomArtifactScope.from_mapping({
        "room_id": "files",
        "task_id": task["identity"].task_id,
        "execution_generation": attempt.execution_generation,
        "member_id": "builder",
        "target_profile": "ops",
        "home_install_id": room["authority_gateway_id"],
        "target_install_id": room["authority_gateway_id"],
        "authority_gateway_id": room["authority_gateway_id"],
        "authority_epoch": room["authority_epoch"],
    })
    outbox = RoomArtifactOutbox(service.root / "profiles" / "ops" / "state.db")
    for name, data in FILES.items():
        outbox.put_bytes(scope=scope, source_name=name, data=data)
    result = {
        "text": "Three verified files.",
        "artifacts": terminal_artifact_manifest(outbox.db_path, scope),
    }
    driver.settle_task(
        service.db_path,
        attempt,
        settlement_id="output-result",
        status="settled",
        result=result,
        clock=time.time,
    )
    return outbox, scope, deepcopy(result)


def _own(service, task):
    return [
        event
        for event in service._events("files")
        if event["payload"].get("task_id") == task["identity"].task_id
    ]


def _assert_publication(service, task, result):
    events = _own(service, task)
    assert [event["kind"] for event in events] == ["message.member", "turn.settled"]
    message = events[0]
    attachments = message["payload"]["attachments"]
    assert {entry["name"] for entry in attachments} == set(FILES)
    assert len({entry["attachment_id"] for entry in attachments}) == 3
    for entry in attachments:
        for viewer in (False, True):
            stored = service.read_attachment(
                room_id="files",
                event_id=message["event_id"],
                attachment_id=entry["attachment_id"],
                recipient_member_id="planner",
                viewer=viewer,
            )
            assert stored.data == FILES[entry["name"]]
        with pytest.raises(AttachmentNotFoundError):
            service.read_attachment(
                room_id="files",
                event_id=message["event_id"],
                attachment_id=entry["attachment_id"],
                recipient_member_id="outside",
            )
    durable = driver.get_task(service.db_path, task["identity"])
    assert durable["execution_generation"] == 1
    assert durable["result"] == result
    with sqlite3.connect(service.db_path) as conn:
        assert (
            conn.execute(
                "SELECT state,expires_at FROM hosted_room_attachments WHERE event_id=?",
                (message["event_id"],),
            ).fetchall()
            == [("committed", None)] * 3
        )
    return message


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("outcome", ["cancelled", "pass", "failed", "deferred"])
def test_interrupted_input_then_three_outputs_survives_cold_ack_without_model_rerun(
    tmp_path, monkeypatch, legacy, outcome
):
    _legacy(monkeypatch, legacy)
    service = _new(tmp_path)
    _send(service, "input", [_file(service, "requirements")])
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
                status="settled" if outcome == "pass" else "failed",
                result={"text": "(pass)"} if outcome == "pass" else {"error": "failed"},
                clock=time.time,
            )
    _tick(service)
    _send(service, "resume")
    task = _queued(service)
    assert not task["payload"].get("attachments")
    assert ("input_context" in task["payload"]) != legacy
    outbox, scope, result = _output(service, task)

    def no_work(*args, **kwargs):
        raise AssertionError("terminal recovery reran model work")

    monkeypatch.setattr(driver, "start_task", no_work)
    monkeypatch.setattr(driver, "settle_task", no_work)
    cold = _service(service.db_path)
    _tick(cold)
    _assert_publication(cold, task, result)
    assert not outbox.list(scope)
    assert outbox.retirement_complete(scope)
    outbox.prune_acknowledged_receipts(now=time.time() + 86401)
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("DELETE FROM hosted_room_policy_events")
        conn.execute("DELETE FROM hosted_room_policy_transcript")
    monkeypatch.setattr(RoomArtifactOutbox, "read", no_work)
    reopened = _service(service.db_path)
    before = reopened._events("files")
    _tick(reopened)
    _tick(reopened)
    _assert_publication(reopened, task, result)
    assert reopened._events("files") == before
    assert not reopened._artifact_retry_keys("files")


@pytest.mark.parametrize("legacy", [False, True])
def test_superseded_output_never_imports_or_exposes_private_bytes(
    tmp_path, monkeypatch, legacy
):
    _legacy(monkeypatch, legacy)
    service = _new(tmp_path)
    _send(service, "first")
    task = _queued(service)
    outbox, scope, result = _output(service, task)
    with monkeypatch.context() as pause:
        pause.setattr(service, "prepare_room", lambda binding: None)
        for index in range(1 if legacy else 30):
            _send(service, f"newer-{index}")

    def no_import(**kwargs):
        raise AssertionError("superseded output was imported")

    monkeypatch.setattr(service, "_import_terminal_artifacts", no_import)
    _tick(service)
    assert [event["kind"] for event in _own(service, task)] == ["turn.cancelled"]
    assert not outbox.list(scope)
    assert outbox.retirement_complete(scope)
    assert driver.get_task(service.db_path, task["identity"])["result"] == result
    with sqlite3.connect(service.db_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM hosted_room_attachments").fetchone()[0]
            == 0
        )
    _tick(_service(service.db_path))
    assert not service._artifact_retry_keys("files")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("append_wins", [False, True])
def test_competing_output_publishers_preserve_readability_in_both_orders(
    tmp_path, monkeypatch, legacy, append_wins
):
    _legacy(monkeypatch, legacy)
    first = _new(tmp_path)
    _send(first, "first")
    task = _queued(first)
    outbox, scope, result = _output(first, task)
    second = _service(first.db_path)
    append_first, append_second = first._append_plan, second._append_plan
    staged, release = threading.Event(), threading.Event()

    def stale_append(*args, **kwargs):
        staged.set()
        assert release.wait(10)
        return append_first(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as pool, monkeypatch.context() as pause:
        pause.setattr(first, "_append_plan", stale_append)
        running = pool.submit(_tick, first)
        try:
            assert staged.wait(10)
            with monkeypatch.context() as quiet:
                quiet.setattr(second, "prepare_room", lambda binding: None)
                _send(second, "unrelated", thread="other")
            pending = []

            def fresh_append(*args, **kwargs):
                if append_wins:
                    return append_second(*args, **kwargs)
                pending.extend((args, kwargs))
                raise OSError("fresh publisher paused before append")

            pause.setattr(second, "_append_plan", fresh_append)
            if append_wins:
                _tick(second)
            else:
                with pytest.raises(OSError, match="paused before append"):
                    _tick(second)
            release.set()
            running.result(timeout=10)
            if not append_wins:
                with pytest.raises(hosted_rooms.EventCursorConflictError):
                    append_second(*pending[0], **pending[1])
                assert len(outbox.list(scope)) == 3
                assert not _own(first, task)
        finally:
            release.set()
            running.result(timeout=10)
    cold = _service(first.db_path)
    cold._artifact_clock = lambda: time.time() + 1000
    _tick(cold)
    _assert_publication(cold, task, result)
    assert not outbox.list(scope)
    assert not cold._artifact_retry_keys("files")
    before = cold._events("files")
    _tick(cold)
    assert cold._events("files") == before


@pytest.mark.parametrize("legacy", [False, True])
def test_permanent_failure_rollback_fences_a_competing_staged_manifest(
    tmp_path, monkeypatch, legacy
):
    _legacy(monkeypatch, legacy)
    first = _new(tmp_path)
    _send(first, "first")
    task = _queued(first)
    outbox, scope, result = _output(first, task)
    second = _service(first.db_path)
    append_first, append_second = first._append_plan, second._append_plan
    staged, release = threading.Event(), threading.Event()
    fenced, manifest = [], []

    def waiting(*args, **kwargs):
        manifest.extend(args[1].events[0].payload["attachments"])
        staged.set()
        assert release.wait(10)
        try:
            return append_first(*args, **kwargs)
        except hosted_rooms.EventCursorConflictError:
            fenced.append(True)
            raise

    def bad_bytes(*args, **kwargs):
        raise RoomArtifactError("permanent verification failure")

    def failed_append(*args, **kwargs):
        assert args[1].terminal_kind == "turn.failed"
        assert not outbox.list(scope)
        release.set()
        running.result(timeout=10)
        return append_second(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as pool, monkeypatch.context() as pause:
        pause.setattr(first, "_append_plan", waiting)
        running = pool.submit(_tick, first)
        try:
            assert staged.wait(10)
            pause.setattr(RoomArtifactOutbox, "read", bad_bytes)
            pause.setattr(second, "_append_plan", failed_append)
            _tick(second)
        finally:
            release.set()
            running.result(timeout=10)
    assert fenced == [True]
    assert [event["kind"] for event in _own(first, task)] == ["turn.failed"]
    cold = _service(first.db_path)
    _tick(cold)
    for item in manifest:
        for viewer in (False, True):
            with pytest.raises(AttachmentNotFoundError):
                cold.read_attachment(
                    room_id="files",
                    event_id=f"dmessage:{task['identity'].task_id.removeprefix('dtask:')}",
                    attachment_id=item["attachment_id"],
                    recipient_member_id="planner",
                    viewer=viewer,
                )
    assert driver.get_task(first.db_path, task["identity"])["result"] == result
    assert not cold._artifact_retry_keys("files")


def test_finite_retirement_horizon_keeps_canonical_bytes_but_not_infinite_cleanup(
    tmp_path, monkeypatch
):
    service = _new(tmp_path)
    _send(service, "first")
    task = _queued(service)
    _, _, result = _output(service, task)
    _tick(service)
    message = _assert_publication(service, task, result)
    with sqlite3.connect(service.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_artifact_completions"
        ).fetchone()[0] == 1
    before = service._events("files")
    future = time.time() + 61 * 86400
    monkeypatch.setattr(time, "time", lambda: future)
    cold = _service(service.db_path)
    cold.runtime.clock = lambda: future
    cold._artifact_clock = lambda: future
    _tick(cold)
    _tick(cold)
    assert cold._events("files") == before
    for entry in message["payload"]["attachments"]:
        assert cold.read_attachment(
            room_id="files",
            event_id=message["event_id"],
            attachment_id=entry["attachment_id"],
            recipient_member_id="planner",
        ).data == FILES[entry["name"]]
    assert driver.list_tasks(service.db_path, room_id="files") == []
    with sqlite3.connect(service.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_artifact_retries"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_artifact_completions"
        ).fetchone()[0] == 0
