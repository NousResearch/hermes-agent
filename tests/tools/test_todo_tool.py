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


class TestTiming:
    """Wall-clock per-item timing (started_at on -> in_progress, ended_at on
    -> completed/cancelled). These prove the gap: an unmeasured item returns
    None while a measured one returns a frozen span."""

    def test_pending_item_has_no_elapsed(self):
        store = TodoStore()
        store.write([{"id": "a", "content": "task", "status": "pending"}])
        assert store.elapsed_for("a") is None

    def test_in_progress_starts_clock(self):
        store = TodoStore()
        store.write([{"id": "a", "content": "task", "status": "pending"}])
        store.write([{"id": "a", "status": "in_progress"}], merge=True)
        elapsed = store.elapsed_for("a")
        assert elapsed is not None and elapsed >= 0

    def test_completed_freezes_elapsed(self, monkeypatch):
        import tools.todo_tool as todo_mod
        clock = {"t": 1000.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        store.write([{"id": "a", "content": "task", "status": "pending"}])
        clock["t"] = 1000.0
        store.write([{"id": "a", "status": "in_progress"}], merge=True)
        clock["t"] = 1005.0
        store.write([{"id": "a", "status": "completed"}], merge=True)
        assert store.elapsed_for("a") == 5.0
        # Time marches on, but a finished item's span stays frozen.
        clock["t"] = 1100.0
        assert store.elapsed_for("a") == 5.0

    def test_completed_without_in_progress_has_no_elapsed(self):
        store = TodoStore()
        store.write([{"id": "a", "content": "task", "status": "pending"}])
        store.write([{"id": "a", "status": "completed"}], merge=True)
        assert store.elapsed_for("a") is None

    def test_cancelled_freezes_elapsed(self, monkeypatch):
        import tools.todo_tool as todo_mod
        clock = {"t": 0.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        store.write([{"id": "a", "content": "task", "status": "in_progress"}])
        clock["t"] = 3.0
        store.write([{"id": "a", "status": "cancelled"}], merge=True)
        assert store.elapsed_for("a") == 3.0

    def test_reopen_resumes_accrual(self, monkeypatch):
        import tools.todo_tool as todo_mod
        clock = {"t": 0.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        store.write([{"id": "a", "content": "task", "status": "in_progress"}])
        clock["t"] = 4.0
        store.write([{"id": "a", "status": "completed"}], merge=True)
        assert store.elapsed_for("a") == 4.0
        # Re-open: ended_at clears, span keeps growing from the original start.
        clock["t"] = 10.0
        store.write([{"id": "a", "status": "in_progress"}], merge=True)
        assert store.elapsed_for("a") == 10.0

    def test_result_json_exposes_elapsed_and_total(self, monkeypatch):
        import tools.todo_tool as todo_mod
        clock = {"t": 100.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        store.write([
            {"id": "a", "content": "A", "status": "in_progress"},
            {"id": "b", "content": "B", "status": "pending"},
        ])
        clock["t"] = 102.0
        result = json.loads(todo_tool(
            todos=[{"id": "a", "status": "completed"}], merge=True, store=store,
        ))
        by_id = {i["id"]: i for i in result["todos"]}
        assert by_id["a"]["elapsed_seconds"] == 2.0
        assert "started_at" in by_id["a"] and "ended_at" in by_id["a"]
        # b never started -> unmeasured.
        assert by_id["b"]["elapsed_seconds"] is None
        assert result["summary"]["total_elapsed_seconds"] == 2.0

    def test_read_only_path_preserves_timing(self, monkeypatch):
        import tools.todo_tool as todo_mod
        clock = {"t": 0.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        store.write([{"id": "a", "content": "A", "status": "in_progress"}])
        clock["t"] = 7.0
        store.write([{"id": "a", "status": "completed"}], merge=True)
        # Pure read (todos=None) must not re-stamp or reset the clock.
        result = json.loads(todo_tool(store=store))
        assert result["todos"][0]["elapsed_seconds"] == 7.0

    def test_replace_mode_resets_timing_for_reused_ids_without_replay_stamps(self, monkeypatch):
        """A fresh replace-mode plan must not inherit week-old clocks by id.

        Gateway history hydration replays raw started_at/ended_at stamps; normal
        model-authored replace calls do not. Reusing generic ids such as
        "setup" or "phase-a" across task topics must therefore start a fresh
        clock instead of preserving stale timing from an older plan.
        """
        import tools.todo_tool as todo_mod
        clock = {"t": 1000.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        store.write([{"id": "setup", "content": "old setup", "status": "in_progress"}])

        clock["t"] = 1000.0 + 183 * 3600
        store.write([{"id": "setup", "content": "new setup", "status": "in_progress"}], merge=False)

        item = store.read_with_timing()[0]
        assert item["content"] == "new setup"
        assert item["started_at"] == clock["t"]
        assert item["elapsed_seconds"] == 0.0

    def test_model_replace_with_copied_timing_fields_resets_reused_id(self, monkeypatch):
        """Live todo_tool writes must not trust model-copied timing fields."""
        import tools.todo_tool as todo_mod
        clock = {"t": 1000.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        store.write([{"id": "setup", "content": "old setup", "status": "in_progress"}])

        clock["t"] = 1000.0 + 183 * 3600
        result = json.loads(todo_tool(
            todos=[{
                "id": "setup",
                "content": "new setup",
                "status": "in_progress",
                "started_at": 1000.0,
                "elapsed_seconds": 183 * 3600,
            }],
            merge=False,
            store=store,
        ))

        item = result["todos"][0]
        assert item["content"] == "new setup"
        assert item["started_at"] == clock["t"]
        assert item["elapsed_seconds"] == 0.0

    def test_model_replace_with_copied_terminal_timing_resets_reused_id(self, monkeypatch):
        """Copied terminal spans on different content are model input, not hydration."""
        import tools.todo_tool as todo_mod
        clock = {"t": 1000.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        store.write([{"id": "setup", "content": "old setup", "status": "in_progress"}])
        clock["t"] = 1005.0
        store.write([{"id": "setup", "status": "completed"}], merge=True)
        assert store.elapsed_for("setup") == 5.0

        clock["t"] = 1000.0 + 183 * 3600
        result = json.loads(todo_tool(
            todos=[{
                "id": "setup",
                "content": "new setup",
                "status": "completed",
                "started_at": 1000.0,
                "ended_at": 1005.0,
                "elapsed_seconds": 5.0,
            }],
            merge=False,
            store=store,
        ))

        item = result["todos"][0]
        assert item["content"] == "new setup"
        assert "started_at" not in item
        assert "ended_at" not in item
        assert item["elapsed_seconds"] is None


class TestTimingHydration:
    """Timing across history replay: terminal spans survive, live work restarts.

    The gateway recreates the AIAgent (and its TodoStore) per message, then
    replays the last todo result. Closed started_at/ended_at spans must be
    adopted, but in-progress clocks must restart on the new turn so idle time
    between turns is not rendered as active work.
    """

    def test_hydration_round_trip_preserves_frozen_span(self, monkeypatch):
        import tools.todo_tool as todo_mod
        clock = {"t": 500.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        store.write([
            {"id": "a", "content": "A", "status": "in_progress"},
            {"id": "b", "content": "B", "status": "pending"},
        ])
        clock["t"] = 506.0
        first = json.loads(todo_tool(
            todos=[{"id": "a", "status": "completed"}], merge=True, store=store,
        ))

        # Fresh store (simulates per-message AIAgent recreation). Replay items
        # carrying started_at/ended_at in REPLACE mode, as _hydrate_todo_store
        # does.
        clock["t"] = 9999.0  # wall clock advanced a lot during the gap
        fresh = TodoStore()
        fresh.write(first["todos"], merge=False)
        # The closed span is preserved exactly, not recomputed against now.
        assert fresh.elapsed_for("a") == 6.0
        assert fresh.elapsed_for("b") is None

    def test_hydration_restarts_live_in_progress_items(self, monkeypatch):
        import tools.todo_tool as todo_mod
        clock = {"t": 500.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        live = json.loads(todo_tool(
            todos=[{"id": "a", "content": "A", "status": "in_progress"}],
            store=store,
        ))
        assert live["todos"][0]["started_at"] == 500.0

        # Fresh store simulates the next gateway turn hydrating from history much
        # later. An in-progress item was not actively worked during that gap, so
        # its clock must restart now instead of showing multi-day elapsed time.
        clock["t"] = 500.0 + 184 * 3600
        fresh = TodoStore()
        fresh.write(live["todos"], merge=False)
        item = fresh.read_with_timing()[0]
        assert item["started_at"] == clock["t"]
        assert item["elapsed_seconds"] == 0.0

    def test_live_clock_wins_over_replayed(self, monkeypatch):
        import tools.todo_tool as todo_mod
        clock = {"t": 0.0}
        monkeypatch.setattr(todo_mod.time, "time", lambda: clock["t"])
        store = TodoStore()
        # Item already has a live started_at at t=0.
        store.write([{"id": "a", "content": "A", "status": "in_progress"}])
        # Replayed payload claims a different (later) started_at — must NOT
        # clobber the live one (setdefault semantics).
        store.write(
            [{"id": "a", "status": "in_progress", "started_at": 50.0}],
            merge=True,
        )
        clock["t"] = 10.0
        assert store.elapsed_for("a") == 10.0  # from the live t=0 start, not 50
