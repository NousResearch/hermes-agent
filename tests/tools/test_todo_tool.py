"""Tests for the todo tool module."""

import json

from tools.todo_tool import TodoStore, todo_tool


class TestPersistence:
    def test_write_persists_to_disk(self, tmp_path):
        path = tmp_path / "todos.json"
        store = TodoStore(persist_path=path)
        store.write([
            {"id": "1", "content": "Task A", "status": "pending"},
            {"id": "2", "content": "Task B", "status": "in_progress"},
        ])
        assert path.exists()
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk == [
            {"id": "1", "content": "Task A", "status": "pending"},
            {"id": "2", "content": "Task B", "status": "in_progress"},
        ]

    def test_init_loads_from_disk(self, tmp_path):
        path = tmp_path / "todos.json"
        path.write_text(json.dumps([
            {"id": "a", "content": "Existing", "status": "completed"},
        ]), encoding="utf-8")
        store = TodoStore(persist_path=path)
        assert store.read() == [
            {"id": "a", "content": "Existing", "status": "completed"},
        ]

    def test_survives_fresh_store_construction(self, tmp_path):
        path = tmp_path / "todos.json"
        first = TodoStore(persist_path=path)
        first.write([{"id": "1", "content": "Persist me", "status": "pending"}])

        # Simulate a gateway restart -- new store, same path.
        second = TodoStore(persist_path=path)
        assert second.read() == [
            {"id": "1", "content": "Persist me", "status": "pending"},
        ]

    def test_session_store_recovers_only_its_own_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        first = TodoStore.for_session("session-one")
        first.write([{"id": "1", "content": "Persist me", "status": "pending"}])

        restarted = TodoStore.for_session("session-one")
        other_session = TodoStore.for_session("session-two")

        assert restarted.read() == [
            {"id": "1", "content": "Persist me", "status": "pending"},
        ]
        assert other_session.read() == []

    def test_merge_writes_persist(self, tmp_path):
        path = tmp_path / "todos.json"
        store = TodoStore(persist_path=path)
        store.write([{"id": "1", "content": "Original", "status": "pending"}])
        store.write([{"id": "1", "status": "completed"}], merge=True)
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk == [
            {"id": "1", "content": "Original", "status": "completed"},
        ]

    def test_corrupt_file_falls_back_to_empty(self, tmp_path):
        path = tmp_path / "todos.json"
        path.write_text("not json {{{", encoding="utf-8")
        store = TodoStore(persist_path=path)
        assert store.read() == []
        # Subsequent writes still work and rewrite the file cleanly.
        store.write([{"id": "1", "content": "fresh", "status": "pending"}])
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk == [{"id": "1", "content": "fresh", "status": "pending"}]

    def test_non_array_file_falls_back_to_empty(self, tmp_path):
        path = tmp_path / "todos.json"
        path.write_text(json.dumps({"todos": []}), encoding="utf-8")
        store = TodoStore(persist_path=path)
        assert store.read() == []

    def test_missing_file_starts_empty(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        store = TodoStore(persist_path=path)
        assert store.read() == []
        assert not path.exists()

    def test_no_persist_path_does_not_create_file(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        store = TodoStore()
        store.write([{"id": "1", "content": "Mem only", "status": "pending"}])
        # No file written -- persistence is opt-in.
        assert list(scratch.iterdir()) == []
        assert store._persist_path is None

    def test_load_skips_invalid_entries(self, tmp_path):
        path = tmp_path / "todos.json"
        path.write_text(json.dumps([
            {"id": "ok", "content": "valid", "status": "pending"},
            "not a dict",
            {"id": "dup", "content": "first", "status": "pending"},
            {"id": "dup", "content": "second", "status": "pending"},
        ]), encoding="utf-8")
        store = TodoStore(persist_path=path)
        items = store.read()
        # Non-dict skipped, duplicate id kept once (first occurrence).
        ids = [item["id"] for item in items]
        assert ids == ["ok", "dup"]
        assert items[1]["content"] == "first"


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
