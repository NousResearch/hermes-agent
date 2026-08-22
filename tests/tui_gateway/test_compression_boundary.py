"""Tests for the durable no-replay compression boundary contract."""

from __future__ import annotations

import json

from hermes_cli.compression_boundary import (
    build_compression_checkpoint,
    compression_boundary_resume_payload,
    checkpoint_meta_key,
    load_compression_checkpoint,
    persist_compression_checkpoint,
)


class _MetaDB:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set_meta(self, key: str, value: str) -> None:
        self.values[key] = value

    def get_meta(self, key: str) -> str | None:
        return self.values.get(key)


def test_checkpoint_is_bounded_redacted_and_does_not_embed_transcript():
    prompt = "review the current task\n" + ("x" * 10_000)
    prompt += "\ncredential=" + ("sk-" + ("a" * 24))

    checkpoint = build_compression_checkpoint(
        "session-old",
        prompt=prompt,
        queued_prompt="finish the current step",
        goal="complete the repair",
        cwd="/workspace",
        model="model-a",
        error="context length exceeded",
    )

    assert checkpoint["old_session_id"] == "session-old"
    assert checkpoint["reason"] == "compression_exhausted"
    assert len(checkpoint["prompt_excerpt"]) <= 1_600
    assert len(checkpoint["resume_prompt"]) <= 3_200
    assert "sk-" not in checkpoint["prompt_excerpt"]
    assert "Do not paste or replay the previous transcript." in checkpoint["resume_prompt"]
    assert "finish the current step" in checkpoint["resume_prompt"]


def test_checkpoint_round_trips_through_namespaced_state_meta():
    db = _MetaDB()
    checkpoint = build_compression_checkpoint("session-round-trip", prompt="continue")

    assert persist_compression_checkpoint(db, checkpoint) is True
    assert checkpoint_meta_key("session-round-trip") in db.values
    loaded = load_compression_checkpoint(db, "session-round-trip")

    assert loaded == checkpoint


def test_invalid_or_cross_session_checkpoint_is_not_accepted():
    db = _MetaDB()
    db.set_meta(
        checkpoint_meta_key("session-a"),
        json.dumps(
            {
                "schema_version": 1,
                "reason": "compression_exhausted",
                "old_session_id": "session-b",
            }
        ),
    )

    assert load_compression_checkpoint(db, "session-a") is None
    assert load_compression_checkpoint(db, "session-missing") is None


def test_resume_payload_is_transcript_free_and_preserves_handoff_prompt():
    db = _MetaDB()
    checkpoint = build_compression_checkpoint(
        "session-sealed",
        prompt="continue the implementation",
        queued_prompt="inspect the failing test",
    )
    assert persist_compression_checkpoint(db, checkpoint) is True

    payload = compression_boundary_resume_payload(db, "session-sealed")

    assert payload is not None
    assert payload["compression_boundary"]["old_session_id"] == "session-sealed"
    assert payload["messages"] == []
    assert payload["inflight"] is None
    assert "inspect the failing test" in payload["compression_boundary"]["resume_prompt"]
