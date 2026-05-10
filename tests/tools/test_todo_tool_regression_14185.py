"""
Regression tests for todo_tool defensive normalization (issue #14185).

LLMs occasionally emit the `todos` parameter as a JSON-encoded string
instead of the declared array, or include non-dict items in the list.
These tests verify that `_normalize_todos()` coerces all observed bad
patterns into valid todo dicts without crashing.
"""

import json
import pytest
from tools.todo_tool import TodoStore, todo_tool, _normalize_todos


class TestNormalizeTodos:
    """Unit tests for the _normalize_todos() helper."""

    def test_well_formed_list_of_dicts(self):
        """Normal input passes through unchanged."""
        raw = [{"id": "t1", "content": "x", "status": "pending"}]
        assert _normalize_todos(raw) == raw

    def test_json_string_double_encoded(self):
        """Pattern A: todos field is a JSON-encoded string."""
        raw = '[{"id":"t1","content":"x","status":"pending"}]'
        expected = [{"id": "t1", "content": "x", "status": "pending"}]
        assert _normalize_todos(raw) == expected

    def test_json_string_invalid(self):
        """Invalid JSON string → empty list (graceful degradation)."""
        assert _normalize_todos("not json") == []

    def test_json_string_not_array(self):
        """JSON string that decodes to a dict instead of list → empty list."""
        assert _normalize_todos('{"id":"t1"}') == []

    def test_list_of_json_strings(self):
        """Pattern B: each item is a JSON-encoded dict string."""
        raw = [
            '{"id":"t1","content":"a","status":"pending"}',
            '{"id":"t2","content":"b","status":"in_progress"}',
        ]
        expected = [
            {"id": "t1", "content": "a", "status": "pending"},
            {"id": "t2", "content": "b", "status": "in_progress"},
        ]
        assert _normalize_todos(raw) == expected

    def test_list_of_invalid_strings(self):
        """Non-JSON strings in the list are silently skipped."""
        raw = [
            '{"id":"t1","content":"a","status":"pending"}',
            "not-a-dict",
            42,
            None,
        ]
        expected = [{"id": "t1", "content": "a", "status": "pending"}]
        assert _normalize_todos(raw) == expected

    def test_mixed_list_dicts_and_strings(self):
        """A mix of dicts and JSON strings is handled correctly."""
        raw = [
            {"id": "t1", "content": "a", "status": "pending"},
            '{"id":"t2","content":"b","status":"completed"}',
        ]
        expected = [
            {"id": "t1", "content": "a", "status": "pending"},
            {"id": "t2", "content": "b", "status": "completed"},
        ]
        assert _normalize_todos(raw) == expected

    def test_none_input(self):
        """None → empty list."""
        assert _normalize_todos(None) == []

    def test_dict_input(self):
        """Plain dict (not wrapped in list) → empty list."""
        assert _normalize_todos({"id": "t1"}) == []

    def test_int_input(self):
        """Completely wrong type → empty list."""
        assert _normalize_todos(42) == []


class TestTodoToolRegression14185:
    """Integration tests that exercise the full todo_tool entry point."""

    def test_todo_tool_accepts_json_string(self):
        """The full tool entry point accepts a JSON string for `todos`."""
        store = TodoStore()
        raw = '[{"id":"t1","content":"fix bug","status":"in_progress"}]'
        result = todo_tool(todos=raw, store=store)
        data = json.loads(result)
        assert data["summary"]["total"] == 1
        assert data["todos"][0]["id"] == "t1"

    def test_todo_tool_accepts_list_of_json_strings(self):
        """The full tool entry point accepts a list of JSON strings."""
        store = TodoStore()
        raw = [
            '{"id":"t1","content":"a","status":"pending"}',
            '{"id":"t2","content":"b","status":"completed"}',
        ]
        result = todo_tool(todos=raw, store=store)
        data = json.loads(result)
        assert data["summary"]["total"] == 2
        assert data["summary"]["completed"] == 1

    def test_todo_tool_skips_non_dict_items(self):
        """Non-dict items are silently dropped; the rest is processed."""
        store = TodoStore()
        raw = [
            {"id": "t1", "content": "a", "status": "pending"},
            "not-a-dict",
            42,
            None,
        ]
        result = todo_tool(todos=raw, store=store)
        data = json.loads(result)
        assert data["summary"]["total"] == 1
        assert data["todos"][0]["id"] == "t1"

    def test_todo_tool_merge_with_json_string(self):
        """Merge mode also works when todos arrives as a JSON string."""
        store = TodoStore()
        store.write([{"id": "t1", "content": "original", "status": "pending"}])
        raw = '[{"id":"t1","content":"updated","status":"in_progress"}]'
        result = todo_tool(todos=raw, merge=True, store=store)
        data = json.loads(result)
        assert data["todos"][0]["content"] == "updated"
        assert data["todos"][0]["status"] == "in_progress"

    def test_todo_tool_empty_json_string(self):
        """Empty JSON array string → empty list, read mode."""
        store = TodoStore()
        store.write([{"id": "t1", "content": "x", "status": "pending"}])
        result = todo_tool(todos="[]", store=store)
        data = json.loads(result)
        assert data["summary"]["total"] == 0

    def test_todo_tool_invalid_json_string(self):
        """Invalid JSON string → graceful empty list, no crash."""
        store = TodoStore()
        result = todo_tool(todos="not-json", store=store)
        data = json.loads(result)
        assert data["summary"]["total"] == 0
