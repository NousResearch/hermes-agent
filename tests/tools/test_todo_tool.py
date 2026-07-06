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
            {"id": "2", "content": "Other task", "status": "pending", "activeForm": ""},
            {"id": "1", "content": "Latest version", "status": "in_progress", "activeForm": ""},
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


class TestActiveForm:
    """Issue #59544 — Claude Code-style ``activeForm`` field for the
    in_progress item, surfaced in both the system prompt and the
    post-compression injection block.
    """

    def test_active_form_default_is_empty_string(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Edit todo_tool.py", "status": "in_progress"},
        ])
        item = store.read()[0]
        assert item["activeForm"] == ""

    def test_active_form_is_persisted_on_write(self):
        store = TodoStore()
        store.write([
            {
                "id": "1",
                "content": "Edit todo_tool.py",
                "status": "in_progress",
                "activeForm": "Editing todo_tool.py",
            },
        ])
        item = store.read()[0]
        assert item["activeForm"] == "Editing todo_tool.py"

    def test_active_form_renders_in_injection_for_in_progress(self):
        store = TodoStore()
        store.write([
            {
                "id": "1",
                "content": "Wire session persistence",
                "status": "in_progress",
                "activeForm": "Wiring session persistence",
            },
            {"id": "2", "content": "Write tests", "status": "pending"},
        ])
        text = store.format_for_injection()
        assert "Wiring session persistence" in text
        # Em-dash separates content from activeForm when both differ
        assert "— Wiring session persistence" in text

    def test_active_form_falls_back_to_content_when_missing(self):
        """No activeForm supplied -> the injection just shows content.

        The format_for_injection helper is the source of truth for the
        fallback contract — used by both the post-compression block and
        the system prompt surface.
        """
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Edit todo_tool.py", "status": "in_progress"},
        ])
        text = store.format_for_injection()
        assert "Edit todo_tool.py" in text
        # No dangling "—" when no activeForm was supplied.
        assert "—" not in text

    def test_active_form_renders_in_active_block(self):
        store = TodoStore()
        store.write([
            {
                "id": "1",
                "content": "Edit todo_tool.py",
                "status": "in_progress",
                "activeForm": "Editing todo_tool.py",
            },
            {"id": "2", "content": "Write tests", "status": "pending"},
            {"id": "3", "content": "Wire persistence", "status": "completed"},
        ])
        text = store.format_for_active_block()
        assert "[Active task list" in text
        assert "3 items" in text
        assert "1 done" in text
        assert "[>]" in text
        assert "[x]" in text
        assert "[ ]" in text
        assert "Editing todo_tool.py" in text

    def test_active_block_returns_none_when_empty(self):
        store = TodoStore()
        assert store.format_for_active_block() is None

    def test_active_block_omitted_when_no_valid_status(self):
        store = TodoStore()
        # Bypass _validate by injecting raw items with an unknown status.
        # Valid statuses are filtered out, so the block should be empty.
        store._items = [{"id": "1", "content": "x", "status": "bogus", "activeForm": ""}]
        assert store.format_for_active_block() is None


class TestPersistenceRoundTrip:
    """Issue #59544 — the todo state survives across agent restarts via
    ``SessionDB.save_todo_state``/``load_todo_state``. We exercise the
    in-memory half here (``TodoStore.to_dict`` / ``TodoStore.from_dict``)
    so the tests stay focused on the tool layer; the SessionDB half is
    exercised by ``test_hermes_state_todo.py`` in the state module.
    """

    def test_round_trip_preserves_items_and_active_form(self):
        original = TodoStore()
        original.write([
            {
                "id": "1",
                "content": "Edit todo_tool.py",
                "status": "in_progress",
                "activeForm": "Editing todo_tool.py",
            },
            {"id": "2", "content": "Wire persistence", "status": "pending"},
            {"id": "3", "content": "Add tests", "status": "completed"},
        ])

        payload = original.to_dict()
        assert payload["version"] == 1
        assert len(payload["items"]) == 3

        restored = TodoStore.from_dict(payload)
        assert restored.read() == original.read()
        # activeForm round-trips
        assert restored.read()[0]["activeForm"] == "Editing todo_tool.py"

    def test_from_dict_handles_missing_payload(self):
        assert TodoStore.from_dict(None).has_items() is False
        assert TodoStore.from_dict({}).has_items() is False
        assert TodoStore.from_dict({"items": []}).has_items() is False

    def test_from_dict_handles_malformed_payload(self):
        """Defensive: a corrupted persisted blob should not crash the
        session — fall back to an empty store (fail-open)."""
        restored = TodoStore.from_dict({"items": "not a list"})
        assert restored.has_items() is False
        restored = TodoStore.from_dict({"items": [None, "garbage", 42]})
        # _validate synthesizes placeholder rows for bad inputs, so we
        # land on the same defensive behavior as the live write path.
        items = restored.read()
        assert len(items) == 3
        assert all(item["id"] == "?" for item in items)

    def test_from_dict_caps_oversized_payload(self):
        from tools.todo_tool import MAX_TODO_ITEMS
        items = [
            {"id": str(i), "content": f"task {i}", "status": "pending"}
            for i in range(5000)
        ]
        restored = TodoStore.from_dict({"version": 1, "items": items})
        assert len(restored.read()) == MAX_TODO_ITEMS

    def test_to_dict_is_json_serializable(self):
        """The payload is meant for a SQLite TEXT column — make sure
        ``json.dumps`` accepts it without a custom encoder."""
        import json
        store = TodoStore()
        store.write([
            {
                "id": "1",
                "content": "Edit todo_tool.py",
                "status": "in_progress",
                "activeForm": "Editing todo_tool.py",
            },
        ])
        encoded = json.dumps(store.to_dict())
        decoded = json.loads(encoded)
        assert decoded["version"] == 1
        assert decoded["items"][0]["activeForm"] == "Editing todo_tool.py"


class TestStatusTransitions:
    """Explicit pending → in_progress → completed transitions via the
    public tool entry point. Issue #59544 requires status to be a first-
    class field the model can flip without rebuilding the whole plan.
    """

    def test_pending_to_in_progress_to_completed(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Write tests", "status": "pending"},
        ])

        # Promote to in_progress with an activeForm.
        store.write(
            [{"id": "1", "status": "in_progress", "activeForm": "Writing tests"}],
            merge=True,
        )
        item = store.read()[0]
        assert item["status"] == "in_progress"
        assert item["activeForm"] == "Writing tests"

        # Complete the task — content + activeForm persist, only status flips.
        store.write(
            [{"id": "1", "status": "completed"}],
            merge=True,
        )
        item = store.read()[0]
        assert item["status"] == "completed"
        assert item["activeForm"] == "Writing tests"

    def test_active_form_can_be_cleared_via_merge(self):
        store = TodoStore()
        store.write([
            {
                "id": "1",
                "content": "Edit todo_tool.py",
                "status": "in_progress",
                "activeForm": "Editing todo_tool.py",
            },
        ])
        store.write(
            [{"id": "1", "activeForm": ""}],
            merge=True,
        )
        assert store.read()[0]["activeForm"] == ""
