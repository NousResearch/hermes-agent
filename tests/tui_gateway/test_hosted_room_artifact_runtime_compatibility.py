"""Historical terminal Output remains held without fabricated authority."""

import hashlib
import json
import threading
import time
from contextlib import nullcontext

import pytest

from gateway import hosted_room_driver as state, hosted_rooms
from gateway.hosted_room_artifacts import RoomArtifactError
from gateway.hosted_room_attachments import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS_PER_MESSAGE,
)
from tui_gateway import hosted_room_driver as driver

_find_terminal_receipt = driver._find_terminal_receipt


def _manifest(*, items=None, version=1):
    items = items or [
        {
            "artifact_id": "rart_0123456789abcdef0123456789abcdef",
            "kind": "file",
            "name": "handoff.md",
            "size": 8,
            "mime": "text/markdown",
            "sha256": "b" * 64,
        }
    ]
    return {
        "version": version,
        "manifest_digest": hashlib.sha256(
            json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "items": items,
    }


def test_legacy_output_receipt_is_held_without_blocking_text_receipt(monkeypatch):
    legacy = state.TaskIdentity("room", "legacy-output", "thread", "turn-legacy")
    text = state.TaskIdentity("room", "text-output", "thread", "turn-text")
    manifest = _manifest()
    history = [
        {
            "role": "assistant",
            "task_id": legacy.task_id,
            "execution_generation": 3,
            "status": "settled",
            "message_id": "peer-run:legacy",
            "content": "Legacy private file.",
            "artifacts": manifest,
            "run_id": "legacy",
        },
        {
            "role": "assistant",
            "task_id": text.task_id,
            "execution_generation": 3,
            "status": "settled",
            "message_id": "peer-run:text",
            "content": "Text remains recoverable.",
        },
    ]

    text_receipt = _find_terminal_receipt(history, text, 3)
    assert text_receipt is not None
    assert text_receipt.result["text"] == "Text remains recoverable."

    # A legacy run_id cannot be upgraded into the current artifact_scope plus
    # peer admission/generation/result commitment. Keep it unresolved for an
    # owner-authorized retirement path instead of crashing or inventing scope.
    with monkeypatch.context() as strict_parser:
        strict_parser.setattr(
            "tui_gateway.hosted_room_driver._bounded_terminal_result",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("legacy receipt reached canonical parsing")
            ),
        )
        assert _find_terminal_receipt(history, legacy, 3) is None


def test_legacy_classification_does_not_absorb_current_or_malformed_receipts():
    identity = state.TaskIdentity("room", "current-output", "thread", "turn-current")
    manifest = _manifest()
    scope = {
        "room_id": "room",
        "task_id": identity.task_id,
        "execution_generation": 3,
        "member_id": "writer",
        "target_profile": "default",
        "home_install_id": "home",
        "target_install_id": "home",
        "authority_gateway_id": "home",
        "authority_epoch": 1,
    }
    current = {
        "role": "assistant",
        "task_id": identity.task_id,
        "execution_generation": 3,
        "status": "settled",
        "message_id": "reply:current",
        "content": "Current canonical file.",
        "artifacts": manifest,
        "artifact_scope": scope,
    }

    assert driver._legacy_artifact_receipt_is_held(current) is False
    receipt = _find_terminal_receipt([current], identity, 3)
    assert receipt is not None
    assert receipt.result["text"] == "Current canonical file."

    malformed_current = {**current, "artifact_scope": None}
    assert driver._legacy_artifact_receipt_is_held(malformed_current) is False

    malformed_legacy = {
        **current,
        "artifact_scope": None,
        "run_id": "run-old-but-malformed",
        "artifacts": {"version": 1, "items": []},
    }
    malformed_legacy.pop("artifact_scope")
    assert driver._legacy_artifact_receipt_is_held(malformed_legacy) is False

    mismatched_legacy = {
        "role": "assistant",
        "task_id": identity.task_id,
        "execution_generation": 3,
        "status": "settled",
        "message_id": "peer-run:other-run",
        "content": "Unrelated historical run identity.",
        "artifacts": manifest,
        "run_id": "expected-run",
    }
    assert driver._legacy_artifact_receipt_is_held(mismatched_legacy) is False
    with pytest.raises(RoomArtifactError, match="scope fields are invalid"):
        _find_terminal_receipt([mismatched_legacy], identity, 3)


def test_malformed_current_output_receipt_still_fails_closed():
    identity = state.TaskIdentity("room", "malformed-current", "thread", "turn-current")
    items = [{
        "artifact_id": "rart_0123456789abcdef0123456789abcdef",
        "kind": "file",
        "name": "handoff.md",
        "size": 8,
        "mime": "text/markdown",
        "sha256": "b" * 64,
    }]
    manifest = {
        "version": 1,
        "manifest_digest": hashlib.sha256(
            json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "items": items,
    }
    current = {
        "role": "assistant",
        "task_id": identity.task_id,
        "execution_generation": 3,
        "status": "settled",
        "message_id": "reply:malformed-current",
        "content": "Missing current scope.",
        "artifacts": manifest,
    }

    with pytest.raises(RoomArtifactError, match="scope fields are invalid"):
        driver._find_terminal_receipt([current], identity, 3)


def _legacy_message(manifest, *, run_id="legacy"):
    return {
        "role": "assistant",
        "task_id": "legacy-output",
        "execution_generation": 3,
        "status": "settled",
        "message_id": f"peer-run:{run_id}",
        "content": "Legacy private file.",
        "artifacts": manifest,
        "run_id": run_id,
    }


@pytest.mark.parametrize(
    ("manifest", "run_id", "error"),
    [
        (
            _manifest(items=[{
                **_manifest()["items"][0], "unexpected": "digest-consistent but invalid",
            }]),
            "legacy",
            "item fields are invalid",
        ),
        (_manifest(version=True), "legacy", "version is unsupported"),
        (
            _manifest(items=[{
                **_manifest()["items"][0],
                "artifact_id": f"rart_{index:032x}",
                "name": f"item-{index}.txt",
                "size": 1,
            } for index in range(MAX_ATTACHMENTS_PER_MESSAGE + 1)]),
            "legacy",
            "count is invalid",
        ),
        (
            _manifest(items=[{
                **_manifest()["items"][0],
                "artifact_id": f"rart_{index:032x}",
                "name": f"large-{index}.bin",
                "size": MAX_ATTACHMENT_BYTES,
                "mime": "application/octet-stream",
            } for index in range(2)]),
            "legacy",
            "byte size is invalid",
        ),
        (_manifest(), "invalid/run", "scope fields are invalid"),
    ],
)
def test_legacy_hold_requires_strict_bounded_manifest_and_run_identity(
    manifest, run_id, error
):
    identity = state.TaskIdentity("room", "legacy-output", "thread", "turn-legacy")
    with pytest.raises(RoomArtifactError, match=error):
        _find_terminal_receipt([_legacy_message(manifest, run_id=run_id)], identity, 3)


class _PollingRPC:
    """Inert session transport used by the real runtime polling path."""

    def __init__(self, histories):
        self.histories = histories
        self.calls = []
        self.legacy_polled = threading.Event()

    def resolve_exact(self, *, profile, title, source):
        self.calls.append(("resolve_exact", title))
        return {"session_id": title}

    def resume(self, *, profile, session_id, source):
        self.calls.append(("resume", session_id))
        return {"session_id": session_id}

    def create(self, **_kwargs):
        raise AssertionError("pre-existing inert sessions must be reused")

    def submit(self, *, profile, session_id, prompt, source, task,
               execution_generation, on_terminal):
        self.calls.append(("submit", session_id))
        return {"accepted": True}

    def history(self, *, profile, session_id, source):
        self.calls.append(("history", session_id))
        if session_id == "Group: legacy-room":
            self.legacy_polled.set()
        return [dict(item) for item in self.histories[session_id]]

    def info(self, *, profile, session_id, source):
        self.calls.append(("info", session_id))
        return {"active": session_id == "Group: legacy-room", "task_id": "legacy-output"}

    def interrupt(self, **_kwargs):
        raise AssertionError("polling a held legacy receipt must not interrupt it")

    def acknowledge(self, **_kwargs):
        raise AssertionError("legacy hold must not ACK")

    def discard(self, **_kwargs):
        raise AssertionError("legacy hold must not discard")


def test_runtime_poll_holds_valid_legacy_output_while_text_task_progresses(tmp_path):
    db = tmp_path / "state.db"
    bindings = [
        driver.HostedRoomBinding("text-room", "gateway", 1),
        driver.HostedRoomBinding("legacy-room", "gateway", 1),
    ]
    identities = [
        state.TaskIdentity("text-room", "text-output", "thread-text", "turn-text"),
        state.TaskIdentity("legacy-room", "legacy-output", "thread-legacy", "turn-legacy"),
    ]
    for binding, identity in zip(bindings, identities):
        hosted_rooms.create_room(
            db,
            room_id=binding.room_id,
            name=binding.room_id,
            members=[{"profile": "ops", "handle": "ops"}],
            authority_gateway_id=binding.gateway_id,
            now=time.time(),
        )
        state.admit_task(
            db,
            identity,
            payload={"target_profile": "ops", "prompt": "Inert task", "source_event_seq": 1},
            clock=time.time,
        )

    histories = {
        "Group: legacy-room": [{
            **_legacy_message(_manifest()),
            "execution_generation": 1,
        }],
        "Group: text-room": [{
            "role": "assistant",
            "task_id": "text-output",
            "execution_generation": 1,
            "status": "settled",
            "message_id": "reply:text-output",
            "content": "Independent text progress.",
        }],
    }
    rpc = _PollingRPC(histories)
    publications = []

    def publish(_binding, task):
        publications.append((task["identity"].task_id, task["status"]))
        if task["identity"].task_id == "legacy-output":
            raise AssertionError("held legacy output must not publish or settle")

    runtime = driver.HostedRoomRuntime(
        db_path=db,
        rooms=bindings,
        rpc=rpc,
        turn_lock=lambda _profile: nullcontext(),
        publish_terminal=publish,
        lease_ttl_seconds=0.4,
        poll_interval_seconds=0.01,
        active_poll_interval_seconds=0.01,
        turn_timeout_seconds=5,
        max_concurrent_rooms=2,
    )
    runtime._process_room(bindings[0])
    assert state.get_task(db, identities[0])["status"] == "settled"
    poller = threading.Thread(target=runtime._process_room, args=(bindings[1],))
    poller.start()
    assert rpc.legacy_polled.wait(timeout=5), runtime.status()["last_error"]
    runtime._stop.set()
    runtime._wake.set()
    poller.join(timeout=5)
    assert not poller.is_alive()
    runtime._release_idle_leases()

    text = state.get_task(db, identities[0])
    legacy = state.get_task(db, identities[1])
    assert legacy["status"] == "running" and legacy["result"] is None
    assert text["status"] == "settled"
    assert text["result"]["text"] == "Independent text progress."
    assert publications == []
    methods = [method for method, _value in rpc.calls]
    assert methods.count("history") >= 2
    assert "acknowledge" not in methods and "discard" not in methods
