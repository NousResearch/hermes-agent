"""Tests for the persistent todo store module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.todo_tool import TodoStore
from tools.todo_store import (
    PersistentTodoStore,
    delete_session_todos,
    list_all_sessions,
    load_session_todos,
    save_session_todos,
)


@pytest.fixture(autouse=True)
def tmp_todos_dir(tmp_path):
    """Redirect TODOS_DIR to a temp directory for all tests."""
    with patch("tools.todo_store.TODOS_DIR", tmp_path):
        yield tmp_path


class TestSaveAndLoad:
    def test_round_trip(self):
        todos = [{"id": "1", "content": "Task A", "status": "pending"}]
        doc = save_session_todos("sess-1", todos)
        assert doc["session_id"] == "sess-1"
        assert doc["todos"] == todos
        assert "created_at" in doc
        assert "updated_at" in doc

        loaded = load_session_todos("sess-1")
        assert loaded is not None
        assert loaded["todos"] == todos

    def test_load_missing_returns_none(self):
        assert load_session_todos("nonexistent") is None

    def test_save_preserves_created_at(self):
        original = "2024-01-01T00:00:00+00:00"
        save_session_todos("sess-1", [], created_at=original)
        loaded = load_session_todos("sess-1")
        assert loaded["created_at"] == original


class TestListAllSessions:
    def test_empty_dir(self):
        assert list_all_sessions() == []

    def test_lists_sessions_with_summary(self):
        save_session_todos("sess-1", [
            {"id": "1", "content": "A", "status": "pending"},
            {"id": "2", "content": "B", "status": "completed"},
        ])
        save_session_todos("sess-2", [
            {"id": "1", "content": "C", "status": "in_progress"},
        ])

        sessions = list_all_sessions()
        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert ids == {"sess-1", "sess-2"}

        # Check summary
        s1 = next(s for s in sessions if s["session_id"] == "sess-1")
        assert s1["summary"]["total"] == 2
        assert s1["summary"]["pending"] == 1
        assert s1["summary"]["completed"] == 1


class TestDeleteSessionTodos:
    def test_delete_existing(self):
        save_session_todos("sess-1", [{"id": "1", "content": "X", "status": "pending"}])
        assert delete_session_todos("sess-1") is True
        assert load_session_todos("sess-1") is None

    def test_delete_nonexistent(self):
        assert delete_session_todos("ghost") is False


class TestPersistentTodoStore:
    def test_inherits_todostore(self):
        store = PersistentTodoStore("sess-1")
        assert isinstance(store, TodoStore)

    def test_write_persists_to_disk(self):
        store = PersistentTodoStore("sess-1")
        store.write([{"id": "1", "content": "Task", "status": "pending"}])

        loaded = load_session_todos("sess-1")
        assert loaded is not None
        assert len(loaded["todos"]) == 1
        assert loaded["todos"][0]["content"] == "Task"

    def test_loads_existing_on_init(self):
        save_session_todos("sess-1", [
            {"id": "1", "content": "Existing", "status": "in_progress"},
        ])

        store = PersistentTodoStore("sess-1")
        items = store.read()
        assert len(items) == 1
        assert items[0]["content"] == "Existing"

    def test_merge_persists(self):
        store = PersistentTodoStore("sess-1")
        store.write([{"id": "1", "content": "First", "status": "pending"}])
        store.write(
            [{"id": "1", "status": "completed"}],
            merge=True,
        )

        loaded = load_session_todos("sess-1")
        assert loaded["todos"][0]["status"] == "completed"
        assert loaded["todos"][0]["content"] == "First"

    def test_second_store_picks_up_changes(self):
        store1 = PersistentTodoStore("sess-1")
        store1.write([{"id": "1", "content": "Hello", "status": "pending"}])

        store2 = PersistentTodoStore("sess-1")
        items = store2.read()
        assert len(items) == 1
        assert items[0]["content"] == "Hello"
