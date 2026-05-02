"""Tests for the todo tool module."""

import json

from tools.todo_tool import TodoStore, todo_tool


class TestWriteAndRead:
    def test_write_replaces_list(self):
        store = TodoStore()
        items = [
            {"id": "1", "content": "First task", "status": "pending"},
            {"id": "2", "content": "Second task", "status": "in_progress"},
        ]
        result = store.write(items)
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["status"] == "in_progress"

    def test_read_returns_copy(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        items = store.read()
        items[0]["content"] = "MUTATED"
        assert store.read()[0]["content"] == "Task"

    def test_write_deduplicates_duplicate_ids(self):
        store = TodoStore()
        result = store.write([
            {"id": "1", "content": "First version", "status": "pending"},
            {"id": "2", "content": "Other task", "status": "pending"},
            {"id": "1", "content": "Latest version", "status": "in_progress"},
        ])
        assert result == [
            {"id": "2", "content": "Other task", "status": "pending"},
            {"id": "1", "content": "Latest version", "status": "in_progress"},
        ]


class TestHasItems:
    def test_empty_store(self):
        store = TodoStore()
        assert store.has_items() is False

    def test_non_empty_store(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "x", "status": "pending"}])
        assert store.has_items() is True


class TestFormatForInjection:
    def test_empty_returns_none(self):
        store = TodoStore()
        assert store.format_for_injection() is None

    def test_non_empty_has_markers(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Do thing", "status": "completed"},
            {"id": "2", "content": "Next", "status": "pending"},
            {"id": "3", "content": "Working", "status": "in_progress"},
        ])
        text = store.format_for_injection()
        # Completed items are filtered out of injection
        assert "[x]" not in text
        assert "Do thing" not in text
        # Active items are included
        assert "[ ]" in text
        assert "[>]" in text
        assert "Next" in text
        assert "Working" in text
        assert "context compression" in text.lower()


class TestMergeMode:
    def test_update_existing_by_id(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Original", "status": "pending"},
        ])
        store.write(
            [{"id": "1", "status": "completed"}],
            merge=True,
        )
        items = store.read()
        assert len(items) == 1
        assert items[0]["status"] == "completed"
        assert items[0]["content"] == "Original"

    def test_merge_appends_new(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "First", "status": "pending"}])
        store.write(
            [{"id": "2", "content": "Second", "status": "pending"}],
            merge=True,
        )
        items = store.read()
        assert len(items) == 2


class TestModelProviderFields:
    def test_write_with_model_and_provider(self):
        store = TodoStore()
        items = [
            {"id": "1", "content": "Task with model", "status": "pending", "model": "gemini-flash", "provider": "openrouter"},
        ]
        result = store.write(items)
        assert len(result) == 1
        assert result[0]["model"] == "gemini-flash"
        assert result[0]["provider"] == "openrouter"

    def test_write_with_model_only(self):
        store = TodoStore()
        items = [
            {"id": "1", "content": "Task with model only", "status": "pending", "model": "gpt-4o"},
        ]
        result = store.write(items)
        assert len(result) == 1
        assert result[0]["model"] == "gpt-4o"
        assert "provider" not in result[0]

    def test_write_with_provider_only(self):
        store = TodoStore()
        items = [
            {"id": "1", "content": "Task with provider only", "status": "pending", "provider": "openai"},
        ]
        result = store.write(items)
        assert len(result) == 1
        assert result[0]["provider"] == "openai"
        assert "model" not in result[0]

    def test_merge_updates_model_provider(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Original", "status": "pending"}])
        store.write(
            [{"id": "1", "model": "claude-3-sonnet", "provider": "anthropic"}],
            merge=True,
        )
        items = store.read()
        assert len(items) == 1
        assert items[0]["model"] == "claude-3-sonnet"
        assert items[0]["provider"] == "anthropic"
        assert items[0]["content"] == "Original"

    def test_model_provider_strip_whitespace(self):
        store = TodoStore()
        items = [
            {"id": "1", "content": "Task", "status": "pending", "model": "  gemini-flash  ", "provider": "  openai  "},
        ]
        result = store.write(items)
        assert result[0]["model"] == "gemini-flash"
        assert result[0]["provider"] == "openai"


class TestTodoToolFunction:
    def test_read_mode(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = json.loads(todo_tool(store=store))
        assert result["summary"]["total"] == 1
        assert result["summary"]["pending"] == 1

    def test_write_mode(self):
        store = TodoStore()
        result = json.loads(todo_tool(
            todos=[{"id": "1", "content": "New", "status": "in_progress"}],
            store=store,
        ))
        assert result["summary"]["in_progress"] == 1

    def test_no_store_returns_error(self):
        result = json.loads(todo_tool())
        assert "error" in result
