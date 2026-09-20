"""Historical terminal Output remains held without fabricated authority."""

import hashlib
import json

import pytest

from gateway import hosted_room_driver as state
from gateway.hosted_room_artifacts import RoomArtifactError
from tui_gateway.hosted_room_driver import _find_terminal_receipt


def test_legacy_output_receipt_is_held_without_blocking_text_receipt():
    legacy = state.TaskIdentity("room", "legacy-output", "thread", "turn-legacy")
    text = state.TaskIdentity("room", "text-output", "thread", "turn-text")
    items = [
        {
            "artifact_id": "rart_0123456789abcdef0123456789abcdef",
            "kind": "file",
            "name": "handoff.md",
            "size": 8,
            "mime": "text/markdown",
            "sha256": "b" * 64,
        }
    ]
    manifest = {
        "version": 1,
        "manifest_digest": hashlib.sha256(
            json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "items": items,
    }
    history = [
        {
            "role": "assistant",
            "task_id": legacy.task_id,
            "execution_generation": 3,
            "status": "settled",
            "message_id": "peer-run:legacy",
            "content": "Legacy private file.",
            "artifacts": manifest,
            "run_id": "run-legacy",
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
    assert _find_terminal_receipt(history, legacy, 3) is None


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
        _find_terminal_receipt([current], identity, 3)
