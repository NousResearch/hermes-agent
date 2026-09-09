"""Runtime tests for the hosted-room session adapter."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from tui_gateway.hosted_room_driver import (
    MAX_TERMINAL_TEXT_BYTES,
    ROOM_SESSION_SOURCE,
    HostedRoomBinding,
    HostedRoomRuntime,
    room_session_title,
)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import (
    PeerHostedRoomTransport,
    PeerMemberRoute,
)


ROOM_ID = "room-1"
PROFILE = "ops"
BINDING = HostedRoomBinding(
    room_id=ROOM_ID,
    gateway_id="gateway-a",
    authority_epoch=1,
)


from tests.tui_gateway.test_hosted_room_driver_runtime import (
    RecordingTurnLocks,
    _identity,
    _runtime,
    _wait_for,
    db,
)


class FakeSessionRPC:
    """Normalized in-memory session adapter with no model or network."""

    def __init__(
        self,
        *,
        auto_complete: bool = True,
        required_lock: RecordingTurnLocks | None = None,
        fail_stage: bool = False,
        fail_stage_at: int | None = None,
        fail_submit_not_admitted: bool = False,
    ) -> None:
        self.auto_complete = auto_complete
        self.required_lock = required_lock
        self.fail_stage = fail_stage
        self.fail_stage_at = fail_stage_at
        self.fail_submit_not_admitted = fail_submit_not_admitted
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.submitted = threading.Event()
        self.on_interrupt = None
        self.on_info = None
        self.history_failures = 0
        self._next_id = 1
        self._stage_count = 0
        self._pending_attachments: dict[str, list[str]] = {}
        self._attachment_snapshots: dict[tuple[str, int], list[str]] = {}
        self._lock = threading.Lock()

    def _assert_lock(self, profile: str) -> None:
        if self.required_lock is not None:
            assert self.required_lock.held_for(profile)

    def add_session(
        self,
        *,
        profile: str = PROFILE,
        title: str = room_session_title(ROOM_ID),
        active: bool = False,
        task_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        with self._lock:
            session_id = f"session-{self._next_id}"
            self._next_id += 1
            session = {"session_id": session_id, "title": title}
            self.sessions[(profile, title)] = session
            self.states[session_id] = {
                "active": active,
                "task_id": task_id,
                "execution_generation": None,
                "history": list(history or []),
                "on_terminal": None,
                "pending_approval": None,
            }
            return session_id

    def complete(
        self,
        task_id: str,
        *,
        content: str = "Finished once.",
        status: str = "settled",
    ) -> None:
        callback = None
        receipt = None
        with self._lock:
            for session_id, session_state in self.states.items():
                if session_state["task_id"] != task_id:
                    continue
                receipt = {
                    "role": "assistant",
                    "task_id": task_id,
                    "execution_generation": session_state["execution_generation"],
                    "status": status,
                    "message_id": f"reply:{task_id}",
                    "content": content,
                }
                session_state["history"].append(receipt)
                session_state["active"] = False
                callback = session_state.get("on_terminal")
                self.calls.append(("complete", {"session_id": session_id}))
                break
        if receipt is None:
            raise AssertionError(f"no active session for {task_id}")
        if callback is not None:
            callback({
                "status": status,
                "settlement_id": receipt["message_id"],
                "message_id": receipt["message_id"],
                "text": content,
            })

    def resolve_exact(self, *, profile: str, title: str, source: str):
        self._assert_lock(profile)
        params = {"profile": profile, "title": title, "source": source}
        self.calls.append(("resolve_exact", params))
        with self._lock:
            session = self.sessions.get((profile, title))
            return dict(session) if session is not None else None

    def create(self, *, profile: str, title: str, source: str):
        self._assert_lock(profile)
        params = {"profile": profile, "title": title, "source": source}
        self.calls.append(("create", params))
        session_id = self.add_session(profile=profile, title=title)
        return {"session_id": session_id, "title": title}

    def resume(self, *, profile: str, session_id: str, source: str):
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
        }
        self.calls.append(("resume", params))
        return {"session_id": session_id}

    def submit(
        self,
        *,
        profile: str,
        session_id: str,
        prompt: str,
        source: str,
        task: state.TaskIdentity,
        execution_generation: int,
        on_terminal,
    ):
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "prompt": prompt,
            "source": source,
            "task": task,
            "execution_generation": execution_generation,
            "on_terminal": on_terminal,
            "staged_attachment_ids": list(
                self._pending_attachments.get(session_id, [])
            ),
        }
        self.calls.append(("submit", params))
        if self.fail_submit_not_admitted:
            error = RuntimeError("prompt submit was refused before admission")
            error.not_admitted = True
            raise error
        with self._lock:
            self.states[session_id]["active"] = True
            self.states[session_id]["task_id"] = task.task_id
            self.states[session_id]["execution_generation"] = execution_generation
            self.states[session_id]["on_terminal"] = on_terminal
        self.submitted.set()
        if self.auto_complete:
            self.complete(task.task_id)
        return {"accepted": True}

    def stage_attachment(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        attachment,
        data: bytes,
        execution_generation: int,
    ):
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
            "attachment": dict(attachment),
            "data": data,
            "execution_generation": execution_generation,
        }
        self.calls.append(("stage_attachment", params))
        self._stage_count += 1
        if self.fail_stage or self._stage_count == self.fail_stage_at:
            raise RuntimeError(
                "attachment staging failed at /Users/private/.hermes/state.db"
            )
        self._pending_attachments.setdefault(session_id, []).append(
            str(attachment.get("attachment_id") or "")
        )
        return {
            "attached": True,
            **(
                {"ref_text": f"@file:attachments/{attachment['name']}"}
                if attachment.get("kind") == "file"
                else {}
            ),
        }

    def begin_attachment_staging(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        execution_generation: int,
    ) -> None:
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
            "execution_generation": execution_generation,
        }
        self.calls.append(("begin_attachment_staging", params))
        self._attachment_snapshots.setdefault(
            (session_id, execution_generation),
            list(self._pending_attachments.get(session_id, [])),
        )

    def commit_attachment_staging(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        execution_generation: int,
    ) -> None:
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
            "execution_generation": execution_generation,
        }
        self.calls.append(("commit_attachment_staging", params))
        self._attachment_snapshots.pop((session_id, execution_generation), None)

    def rollback_attachment_staging(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        execution_generation: int,
    ) -> None:
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
            "execution_generation": execution_generation,
        }
        self.calls.append(("rollback_attachment_staging", params))
        snapshot = self._attachment_snapshots.pop(
            (session_id, execution_generation), None
        )
        if snapshot is not None:
            self._pending_attachments[session_id] = snapshot

    def history(self, *, profile: str, session_id: str, source: str):
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
        }
        self.calls.append(("history", params))
        if self.history_failures > 0:
            self.history_failures -= 1
            raise RuntimeError("transient history read failed")
        with self._lock:
            return [dict(message) for message in self.states[session_id]["history"]]

    def info(self, *, profile: str, session_id: str, source: str):
        self._assert_lock(profile)
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
        }
        self.calls.append(("info", params))
        with self._lock:
            session_state = self.states[session_id]
            result = {
                "active": session_state["active"],
                "task_id": session_state["task_id"],
            }
            if session_state.get("pending_approval"):
                result["status"] = "waiting_for_approval"
                result["pending_approval"] = dict(session_state["pending_approval"])
        if self.on_info is not None:
            self.on_info()
        return result

    def interrupt(
        self,
        *,
        profile: str,
        session_id: str,
        source: str,
        expected_task_id: str,
    ):
        params = {
            "profile": profile,
            "session_id": session_id,
            "source": source,
            "expected_task_id": expected_task_id,
        }
        with self._lock:
            current = self.states[session_id]
            if not current["active"] or current["task_id"] != expected_task_id:
                self.calls.append(("interrupt_skipped", params))
                return {"interrupted": False}
            current["active"] = False
        self.calls.append(("interrupt", params))
        if self.on_interrupt is not None:
            self.on_interrupt()
        return {"interrupted": True}


def _admit(
    db: Path,
    identity: state.TaskIdentity,
    *,
    prompt: str = "Inspect the release candidate.",
    attachments: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "target_profile": PROFILE,
        "prompt": prompt,
        "source_event_seq": 1,
    }
    if attachments:
        payload["attachments"] = attachments
    state.admit_task(
        db,
        identity,
        payload=payload,
        clock=time.time,
    )


def test_attachments_stage_before_submit_and_file_ref_is_runtime_only(db: Path):
    identity = _identity()
    manifests = [
        {
            "attachment_id": "att_11111111111111111111111111111111",
            "kind": "image",
            "name": "diagram.png",
            "size": 12,
            "mime": "image/png",
        },
        {
            "attachment_id": "att_22222222222222222222222222222222",
            "kind": "file",
            "name": "notes.txt",
            "size": 5,
            "mime": "text/plain",
        },
    ]
    _admit(db, identity, attachments=manifests)
    rpc = FakeSessionRPC()
    runtime = _runtime(
        db,
        rpc,
        attachment_loader=lambda _binding, _task: (
            (manifests[0], b"image-bytes"),
            (manifests[1], b"notes"),
        ),
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    methods = [method for method, _params in rpc.calls]
    assert methods.index("stage_attachment") < methods.index("submit")
    assert methods.count("stage_attachment") == 2
    submit = next(params for method, params in rpc.calls if method == "submit")
    assert "@file:attachments/notes.txt" in submit["prompt"]
    durable = state.get_task(db, identity)
    assert "@file:" not in durable["payload"]["prompt"]
    assert "image-bytes" not in repr(durable)


def test_attachment_stage_failure_marks_turn_failed_without_text_only_submit(db: Path):
    identity = _identity()
    manifest = {
        "attachment_id": "att_11111111111111111111111111111111",
        "kind": "image",
        "name": "diagram.png",
        "size": 12,
        "mime": "image/png",
    }
    _admit(db, identity, attachments=[manifest])
    rpc = FakeSessionRPC(fail_stage=True)
    runtime = _runtime(
        db,
        rpc,
        attachment_loader=lambda _binding, _task: ((manifest, b"image-bytes"),),
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "failed")
    assert runtime.stop(timeout=1.0)

    assert any(method == "stage_attachment" for method, _params in rpc.calls)
    assert not any(method == "submit" for method, _params in rpc.calls)
    error = state.get_task(db, identity)["result"]["error"]
    assert error == "Group Chat member turn failed."
    assert "/Users/private" not in error


def test_second_attachment_failure_rolls_back_before_later_text_only_turn(db: Path):
    first = {
        "attachment_id": "att_11111111111111111111111111111111",
        "kind": "image",
        "name": "diagram.png",
        "size": 12,
        "mime": "image/png",
    }
    second = {
        "attachment_id": "att_22222222222222222222222222222222",
        "kind": "pdf",
        "name": "brief.pdf",
        "size": 12,
        "mime": "application/pdf",
    }
    failed_identity = _identity("task-with-files")
    _admit(db, failed_identity, attachments=[first, second])
    rpc = FakeSessionRPC(fail_stage_at=2)
    runtime = _runtime(
        db,
        rpc,
        attachment_loader=lambda _binding, _task: (
            (first, b"image-bytes"),
            (second, b"pdf-bytes"),
        ),
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, failed_identity)["status"] == "failed")

    session_id = next(iter(rpc.states))
    assert rpc._pending_attachments[session_id] == []
    assert any(method == "rollback_attachment_staging" for method, _ in rpc.calls)

    text_identity = _identity("task-text-only")
    _admit(db, text_identity, prompt="Continue without any attachment.")
    runtime.wakeup()
    _wait_for(lambda: state.get_task(db, text_identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    text_submit = next(
        params
        for method, params in rpc.calls
        if method == "submit" and params["task"] == text_identity
    )
    assert text_submit["staged_attachment_ids"] == []


def test_not_admitted_submit_rolls_back_before_later_text_only_turn(db: Path):
    manifest = {
        "attachment_id": "att_11111111111111111111111111111111",
        "kind": "image",
        "name": "diagram.png",
        "size": 12,
        "mime": "image/png",
    }
    refused = _identity("task-refused-after-stage")
    _admit(db, refused, attachments=[manifest])
    rpc = FakeSessionRPC(fail_submit_not_admitted=True)
    runtime = _runtime(
        db,
        rpc,
        attachment_loader=lambda _binding, _task: ((manifest, b"image-bytes"),),
    )

    runtime.start()
    _wait_for(
        lambda: any(
            method == "rollback_attachment_staging" for method, _ in rpc.calls
        )
    )
    assert runtime.stop(timeout=1.0)
    assert state.get_task(db, refused)["status"] == "queued"
    session_id = next(iter(rpc.states))
    assert rpc._pending_attachments[session_id] == []

    state.cancel_task(
        db,
        refused,
        cancel_id="cancel-refused",
        expected_cancel_generation=0,
        clock=time.time,
    )
    text_task = _identity("task-after-refused")
    _admit(db, text_task, prompt="Continue with text only.")
    rpc.fail_submit_not_admitted = False
    resumed = _runtime(db, rpc, attachment_loader=lambda _binding, _task: ())
    resumed.start()
    _wait_for(lambda: state.get_task(db, text_task)["status"] == "settled")
    assert resumed.stop(timeout=1.0)

    text_submit = next(
        params
        for method, params in rpc.calls
        if method == "submit" and params["task"] == text_task
    )
    assert text_submit["staged_attachment_ids"] == []


def test_attachment_task_uses_the_selected_member_transport(db: Path):
    identity = _identity()
    manifest = {
        "attachment_id": "att_11111111111111111111111111111111",
        "kind": "image",
        "name": "diagram.png",
        "size": 12,
        "mime": "image/png",
    }
    _admit(db, identity, attachments=[manifest])
    peer = FakeSessionRPC()
    runtime = HostedRoomRuntime(
        db_path=db,
        rooms=[BINDING],
        transport_resolver=lambda _binding, _task: peer,
        turn_lock=RecordingTurnLocks(),
        attachment_loader=lambda _binding, _task: ((manifest, b"image-bytes"),),
        lease_ttl_seconds=0.4,
        poll_interval_seconds=0.01,
    )

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    assert any(method == "stage_attachment" for method, _params in peer.calls)
    assert any(method == "submit" for method, _params in peer.calls)


def test_task_can_stage_a_bounded_aggregate_from_multiple_messages(db: Path):
    identity = _identity()
    manifests = [
        {
            "attachment_id": f"att_{index:032x}",
            "kind": "file",
            "name": f"notes-{index}.txt",
            "size": 1,
            "mime": "text/plain",
        }
        for index in range(9)
    ]
    _admit(db, identity, attachments=manifests)
    rpc = FakeSessionRPC()

    def load_one_at_a_time(_binding, _task):
        for manifest in manifests:
            yield manifest, b"x"

    runtime = _runtime(db, rpc, attachment_loader=load_one_at_a_time)
    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=1.0)

    assert sum(method == "stage_attachment" for method, _params in rpc.calls) == 9
