"""One-shot next-session handoff routing and isolation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.memory_tool import MemoryStore, apply_memory_pending, memory_tool
from tools.next_session_handoff import (
    looks_like_next_session_handoff,
    next_session_handoff_prompt,
    queue_next_session_handoffs,
    transfer_next_session_handoff,
)


@pytest.fixture(autouse=True)
def _isolated_handoffs(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.next_session_handoff._handoff_dir", lambda: tmp_path / "handoffs")
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path / "memories")


@pytest.fixture()
def store():
    result = MemoryStore(memory_char_limit=500, user_char_limit=300)
    result.load_from_disk()
    return result


@pytest.mark.parametrize("content", [
    "TASK for next session: Start the next session with a concise Turkish recap.",
    "Bir sonraki oturum için görev: araştırmaya devam et.",
])
def test_classifier_accepts_explicit_directives(content):
    assert looks_like_next_session_handoff(content)


@pytest.mark.parametrize("content", [
    "The next session is scheduled for Tuesday.",
    "We discussed browser sessions and conversation state.",
    "Always start tests with scripts/run_tests.sh.",
])
def test_classifier_ignores_factual_or_durable_notes(content):
    assert not looks_like_next_session_handoff(content)


def test_handoffs_are_source_and_target_session_scoped():
    assert queue_next_session_handoffs(
        ["TASK for next session: continue alpha"], source_session_id="source-a")["success"]
    assert queue_next_session_handoffs(
        ["TASK for next session: continue beta"], source_session_id="source-b")["success"]

    assert transfer_next_session_handoff("source-a", "target-a")
    assert "continue alpha" in next_session_handoff_prompt("target-a")
    assert next_session_handoff_prompt("compression-child") is None
    assert next_session_handoff_prompt("target-b") is None

    assert transfer_next_session_handoff("source-b", "target-b")
    assert "continue beta" in next_session_handoff_prompt("target-b")
    assert "continue alpha" not in next_session_handoff_prompt("target-b")


def test_only_first_explicit_new_session_can_claim_source():
    queue_next_session_handoffs(
        ["TASK for next session: continue once"], source_session_id="source")
    assert transfer_next_session_handoff("source", "first-target")
    assert not transfer_next_session_handoff("source", "racing-target")
    assert next_session_handoff_prompt("racing-target") is None


def test_memory_batch_routes_handoff_without_durable_write(store):
    result = json.loads(memory_tool(
        target="memory",
        operations=[{"action": "add", "content": "TASK for next session: continue the audit"}],
        store=store,
        session_id="source",
    ))
    assert result["success"] is True
    assert result["handoff"] is True
    assert store.memory_entries == []
    assert transfer_next_session_handoff("source", "target")
    assert "continue the audit" in next_session_handoff_prompt("target")


def test_mixed_durable_and_handoff_batch_is_refused_atomically(store):
    result = json.loads(memory_tool(
        target="memory",
        operations=[
            {"action": "add", "content": "Python 3.12 project"},
            {"action": "add", "content": "TASK for next session: continue the audit"},
        ],
        store=store,
        session_id="source",
    ))
    assert result["success"] is False
    assert "separate memory call" in result["error"]
    assert store.memory_entries == []
    assert not transfer_next_session_handoff("source", "target")


def test_staged_handoff_replays_with_original_source_session(store):
    result = apply_memory_pending({
        "action": "batch",
        "target": "memory",
        "session_id": "source-at-staging-time",
        "operations": [{"action": "add", "content": "TASK for next session: continue staged work"}],
    }, store)
    assert result["success"] is True
    assert transfer_next_session_handoff("source-at-staging-time", "target")
    assert "continue staged work" in next_session_handoff_prompt("target")


def test_prompt_read_is_idempotent_only_for_assigned_session():
    queue_next_session_handoffs(
        ["TASK for next session: continue prompt work"], source_session_id="source")
    transfer_next_session_handoff("source", "target")
    first = next_session_handoff_prompt("target")
    assert first == next_session_handoff_prompt("target")
    assert next_session_handoff_prompt("other") is None


def test_system_prompt_includes_only_the_exact_assigned_session():
    from agent.system_prompt import _memory_parts

    queue_next_session_handoffs(
        ["TASK for next session: continue prompt work"], source_session_id="source")
    transfer_next_session_handoff("source", "target")
    target = SimpleNamespace(_memory_store=None, _memory_manager=None, session_id="target")
    compression_child = SimpleNamespace(
        _memory_store=None, _memory_manager=None, session_id="compression-child")
    assert "continue prompt work" in "\n".join(_memory_parts(target))
    assert _memory_parts(compression_child) == []
