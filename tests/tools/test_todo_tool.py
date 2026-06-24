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


class TestTodoStoreBounds:
    """Bounds on persisted todo state (GHSA-5g4g-6jrg-mw3g hardening).

    The todo list is re-injected into context after every compression event,
    so an unbounded item — whether authored by the model or replayed from
    caller-supplied history on the API server's _hydrate_todo_store path —
    would defeat the compression it rides through. These pin the caps.
    Not a security boundary (the API surface is authenticated and the caller
    supplies their own history); this is footgun containment / parity.
    """

    def test_oversized_content_is_truncated(self):
        from tools.todo_tool import MAX_TODO_CONTENT_CHARS
        store = TodoStore()
        store.write([{"id": "1", "content": "A" * 50001, "status": "pending"}])
        item = store.read()[0]
        assert len(item["content"]) <= MAX_TODO_CONTENT_CHARS
        assert item["content"].endswith("… [truncated]")

    def test_injection_block_is_bounded(self):
        from tools.todo_tool import MAX_TODO_CONTENT_CHARS
        store = TodoStore()
        store.write([{"id": "1", "content": "A" * 50001, "status": "pending"}])
        inj = store.format_for_injection()
        # Before the fix this was ~50085 chars; now it tracks the cap.
        assert len(inj) < MAX_TODO_CONTENT_CHARS + 200

    def test_merge_update_content_is_capped(self):
        """The merge path updates content directly, bypassing _validate —
        verify it is capped too."""
        from tools.todo_tool import MAX_TODO_CONTENT_CHARS
        store = TodoStore()
        store.write([{"id": "1", "content": "short", "status": "pending"}])
        store.write([{"id": "1", "content": "B" * 50001}], merge=True)
        assert len(store.read()[0]["content"]) <= MAX_TODO_CONTENT_CHARS

    def test_item_count_is_bounded(self):
        from tools.todo_tool import MAX_TODO_ITEMS
        store = TodoStore()
        store.write([
            {"id": str(i), "content": f"task {i}", "status": "pending"}
            for i in range(5000)
        ])
        assert len(store.read()) == MAX_TODO_ITEMS

    def test_normal_list_is_unchanged(self):
        """No regression: ordinary plans pass through untouched (no marker,
        same content, same order)."""
        store = TodoStore()
        store.write([
            {"id": "1", "content": "write the report", "status": "in_progress"},
            {"id": "2", "content": "review PR", "status": "pending"},
        ])
        items = store.read()
        assert [i["content"] for i in items] == ["write the report", "review PR"]
        assert "[truncated]" not in items[0]["content"]


class TestMergeEdgeCases:
    """Cover merge-mode branches not exercised by the basic tests."""

    def test_merge_skips_item_without_id(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Keep", "status": "pending"}])
        store.write(
            [{"id": "", "content": "No id", "status": "pending"}],
            merge=True,
        )
        items = store.read()
        assert len(items) == 1
        assert items[0]["id"] == "1"

    def test_merge_ignores_invalid_status(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Orig", "status": "pending"}])
        store.write([{"id": "1", "status": "bogus"}], merge=True)
        item = store.read()[0]
        assert item["status"] == "pending"

    def test_merge_preserves_order(self):
        store = TodoStore()
        store.write([
            {"id": "a", "content": "A", "status": "pending"},
            {"id": "b", "content": "B", "status": "pending"},
            {"id": "c", "content": "C", "status": "pending"},
        ])
        store.write([{"id": "b", "status": "completed"}], merge=True)
        ids = [i["id"] for i in store.read()]
        assert ids == ["a", "b", "c"]

class TestValidate:
    """Cover _validate normalisation paths."""

    def test_empty_id_becomes_placeholder(self):
        store = TodoStore()
        store.write([{"id": "", "content": "Task", "status": "pending"}])
        assert store.read()[0]["id"] == "?"

    def test_empty_content_becomes_placeholder(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "", "status": "pending"}])
        assert store.read()[0]["content"] == "(no description)"

    def test_invalid_status_defaults_to_pending(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "bogus"}])
        assert store.read()[0]["status"] == "pending"

    def test_content_is_stripped(self):
        store = TodoStore()
        store.write([{"id": " 1 ", "content": "  spaced  ", "status": " pending "}])
        item = store.read()[0]
        assert item["id"] == "1"
        assert item["content"] == "spaced"
        assert item["status"] == "pending"


class TestDedupeById:
    """Cover _dedupe_by_id edge cases."""

    def test_empty_id_dedupe_uses_placeholder(self):
        store = TodoStore()
        store.write([
            {"id": "", "content": "First", "status": "pending"},
            {"id": "", "content": "Second", "status": "pending"},
        ])
        items = store.read()
        assert len(items) == 1
        assert items[0]["content"] == "Second"

class TestTodoToolFunctionEdgeCases:
    """Cover todo_tool function paths not exercised by basic tests."""

    def test_cancelled_count_in_summary(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "A", "status": "cancelled"},
            {"id": "2", "content": "B", "status": "completed"},
        ])
        result = json.loads(todo_tool(store=store))
        assert result["summary"]["cancelled"] == 1
        assert result["summary"]["completed"] == 1

class TestCheckTodoRequirements:
    def test_always_returns_true(self):
        from tools.todo_tool import check_todo_requirements
        assert check_todo_requirements() is True
