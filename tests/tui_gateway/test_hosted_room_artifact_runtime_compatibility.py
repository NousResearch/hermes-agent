"""Historical terminal Output remains held without fabricated authority."""

import hashlib
import json

from gateway import hosted_room_driver as state
from tui_gateway import hosted_room_driver as driver

_find_terminal_receipt = driver._find_terminal_receipt


def _manifest():
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
    return {
        "version": 1,
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
