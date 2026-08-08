"""memory tool must reject non-string content/old_text without crashing."""

from __future__ import annotations

import json

import pytest

from tools.memory_tool import MemoryStore, memory_tool


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return MemoryStore(memory_char_limit=2000, user_char_limit=1000)


def test_add_rejects_non_string_content(store):
    result = store.add("memory", ["not", "a", "string"])
    assert result["success"] is False
    assert "must be a string" in result["error"]
    assert store.memory_entries == []


def test_replace_rejects_non_string_old_text(store):
    store.add("memory", "keep me")
    result = store.replace("memory", 42, "replacement")
    assert result["success"] is False
    assert "old_text must be a string" in result["error"]
    assert store.memory_entries == ["keep me"]


def test_remove_rejects_non_string_old_text(store):
    store.add("memory", "keep me")
    result = store.remove("memory", ["keep"])
    assert result["success"] is False
    assert "old_text must be a string" in result["error"]
    assert store.memory_entries == ["keep me"]


def test_batch_rejects_non_string_content(store):
    result = store.apply_batch(
        "memory",
        [{"action": "add", "content": 123}],
    )
    assert result["success"] is False
    assert "content must be a string" in result["error"]
    assert store.memory_entries == []


def test_memory_tool_add_non_string_returns_tool_error(store):
    """Dispatcher path: model JSON int content must not AttributeError."""
    result = json.loads(
        memory_tool(action="add", content=99, store=store)
    )
    assert result["success"] is False
    assert "must be a string" in result["error"]
