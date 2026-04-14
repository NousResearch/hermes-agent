"""Tests for V3 Phase 3 memory type classification and recall behavior."""

from __future__ import annotations

import json

import pytest

from tools.memory_tool import (
    DEFAULT_MEMORY_TYPE,
    MemoryStore,
    _infer_memory_type,
    memory_tool,
)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    memory_store = MemoryStore(memory_char_limit=1000, user_char_limit=1000)
    memory_store.load_from_disk()
    return memory_store


def test_read_group_by_type(store):
    store.add("memory", "I am a developer", memory_type="user")
    store.add("memory", "Don't add summary footers", memory_type="feedback")
    store.add("memory", "Release milestone is Friday", memory_type="project")

    result = json.loads(
        memory_tool(action="read", target="memory", group_by_type=True, store=store)
    )

    assert result["success"] is True
    assert result["group_by_type"] is True
    assert result["rendered"] == (
        "## [Type] user\n"
        "- I am a developer\n\n"
        "## [Type] feedback\n"
        "- Don't add summary footers\n\n"
        "## [Type] project\n"
        "- Release milestone is Friday\n"
    )


def test_read_flat_default(store):
    store.add("memory", "I am a developer", memory_type="user")
    store.add("memory", "Don't add summary footers", memory_type="feedback")

    result = json.loads(memory_tool(action="read", target="memory", store=store))

    assert result["success"] is True
    assert result["group_by_type"] is False
    assert result["rendered"] == "- I am a developer\n- Don't add summary footers\n"


def test_infer_feedback_type():
    assert _infer_memory_type("Please don't do that again.") == "feedback"


def test_infer_project_type():
    assert _infer_memory_type("The release deadline is this Friday.") == "project"


def test_infer_reference_type():
    assert _infer_memory_type("Dashboard URL: http://example.com/docs") == "reference"


def test_infer_user_type():
    assert _infer_memory_type("I am a developer on the platform team.") == "user"


def test_infer_uncategorized():
    assert _infer_memory_type("Updated several files yesterday.") == DEFAULT_MEMORY_TYPE


def test_auto_classify_on_write(store):
    result = json.loads(
        memory_tool(
            action="add",
            target="memory",
            content="Please don't add a summary footer.",
            store=store,
        )
    )

    assert result["success"] is True
    assert result["typed_entries"][0]["type"] == "feedback"


def test_recall_by_types(store):
    store.add("memory", "Project milestone freezes on Friday", memory_type="project")
    store.add("memory", "Reference docs live at http://example.com", memory_type="reference")
    store.add("memory", "Don't add summary footers", memory_type="feedback")
    store.add("user", "I am a developer", memory_type="user")
    store.add("memory", "Generic note", memory_type=DEFAULT_MEMORY_TYPE)

    recalled = store.recall_by_types(["reference", "feedback", "user", "project"], limit=10)

    assert [item["type"] for item in recalled] == ["user", "feedback", "project", "reference"]
    assert [item["title"] for item in recalled] == [
        "I am a developer",
        "Don't add summary footers",
        "Project milestone freezes on Friday",
        "Reference docs live at http://example.com",
    ]
    assert [item["content"] for item in recalled] == [
        "I am a developer",
        "Don't add summary footers",
        "Project milestone freezes on Friday",
        "Reference docs live at http://example.com",
    ]
