"""Tests for defensive type coercion in todo_tool (issue #14185).

Covers three crash patterns:
1. todos is a JSON string instead of a list
2. todos list contains non-dict items (e.g., bare strings)
3. Well-formed input continues to work unchanged
"""

import json

from tools.todo_tool import TodoStore, todo_tool


class TestJsonStringCoercion:
    """Guard 1: todo_tool() recovers when LLM sends todos as a JSON string."""

    def test_json_string_is_parsed_into_list(self):
        store = TodoStore()
        todos_str = json.dumps([
            {"id": "t1", "content": "Do A", "status": "pending"},
            {"id": "t2", "content": "Do B", "status": "in_progress"},
        ])
        result = json.loads(todo_tool(todos=todos_str, store=store))
        assert "error" not in result
        assert result["summary"]["total"] == 2
        assert result["todos"][0]["id"] == "t1"
        assert result["todos"][1]["status"] == "in_progress"

    def test_unparseable_string_returns_error(self):
        store = TodoStore()
        result = json.loads(todo_tool(todos="not valid json [", store=store))
        assert "error" in result

    def test_json_string_that_parses_to_non_list_returns_error(self):
        store = TodoStore()
        # Valid JSON, but a dict instead of a list
        result = json.loads(todo_tool(todos='{"id": "1"}', store=store))
        assert "error" in result

    def test_non_list_non_string_returns_error(self):
        store = TodoStore()
        result = json.loads(todo_tool(todos=42, store=store))
        assert "error" in result


class TestNonDictListItems:
    """Guards 2 & 3: _validate and _dedupe_by_id handle non-dict items."""

    def test_string_item_in_list_does_not_crash(self):
        store = TodoStore()
        result = store.write(["not-a-dict"])
        assert len(result) == 1
        assert result[0]["id"] == "?"
        assert result[0]["content"] == "(invalid item)"
        assert result[0]["status"] == "pending"

    def test_mixed_valid_and_invalid_items(self):
        store = TodoStore()
        result = store.write([
            {"id": "1", "content": "Real task", "status": "pending"},
            "garbage",
            42,
            {"id": "2", "content": "Another task", "status": "completed"},
        ])
        assert len(result) == 4
        # Valid items are preserved
        assert result[0]["id"] == "1"
        assert result[0]["content"] == "Real task"
        assert result[3]["id"] == "2"
        # Invalid items get placeholder values
        assert result[1]["content"] == "(invalid item)"
        assert result[2]["content"] == "(invalid item)"

    def test_none_item_in_list(self):
        store = TodoStore()
        result = store.write([None])
        assert len(result) == 1
        assert result[0]["id"] == "?"

    def test_integer_item_in_list(self):
        store = TodoStore()
        result = store.write([123])
        assert len(result) == 1
        assert result[0]["content"] == "(invalid item)"

    def test_non_dict_items_via_todo_tool(self):
        """End-to-end: non-dict list items produce valid output, not a crash."""
        store = TodoStore()
        result = json.loads(todo_tool(todos=["bad", "also bad"], store=store))
        assert "error" not in result
        assert result["summary"]["total"] == 2
        assert result["summary"]["pending"] == 2


class TestWellFormedInputUnchanged:
    """Regression: normal usage must not be affected by the guards."""

    def test_normal_write_and_read(self):
        store = TodoStore()
        items = [
            {"id": "a", "content": "First", "status": "pending"},
            {"id": "b", "content": "Second", "status": "in_progress"},
        ]
        result = json.loads(todo_tool(todos=items, store=store))
        assert result["summary"]["total"] == 2
        assert result["summary"]["pending"] == 1
        assert result["summary"]["in_progress"] == 1

    def test_merge_mode_still_works(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Original", "status": "pending"}])
        result = json.loads(todo_tool(
            todos=[{"id": "1", "status": "completed"}],
            merge=True,
            store=store,
        ))
        assert result["summary"]["completed"] == 1
        assert result["todos"][0]["content"] == "Original"

    def test_read_mode_still_works(self):
        store = TodoStore()
        store.write([{"id": "x", "content": "Task", "status": "pending"}])
        result = json.loads(todo_tool(store=store))
        assert result["summary"]["total"] == 1

    def test_dedup_still_works(self):
        store = TodoStore()
        result = store.write([
            {"id": "1", "content": "v1", "status": "pending"},
            {"id": "1", "content": "v2", "status": "in_progress"},
        ])
        assert len(result) == 1
        assert result[0]["content"] == "v2"
