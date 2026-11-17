"""Hierarchy progress distinguishes phase acceptance from its individual checks."""

import json

from tools.todo_tool import TodoStore, todo_tool


def item(key, status="pending", parent=None):
    return {"id": key, "content": key, "status": status, **({"parent": parent} if parent else {})}


def test_completed_checks_do_not_complete_their_phase():
    store = TodoStore()
    result = json.loads(todo_tool([
        item("accept", "in_progress"), item("test", "completed", "accept"),
        item("review", "completed", "accept"), item("obsolete", "cancelled", "accept"),
    ], store=store))
    assert result["summary"] == {
        "total": 4, "pending": 0, "in_progress": 1, "completed": 2, "cancelled": 1,
    }
    assert result["progress"] == {
        "phases": {"total": 1, "completed": 0},
        "subtasks": {"total": 2, "completed": 2}, "unresolved": 1,
    }
    assert json.loads(todo_tool(store=store))["progress"] == result["progress"]


def test_reopened_descendant_keeps_completed_ancestors_unfinished():
    store = TodoStore()
    todo_tool([item("release", "completed"), item("review", "completed", "release"),
               item("probe", "completed", "review")], store=store)
    before = store.snapshot()["revision"]
    result = json.loads(todo_tool([{"id": "probe", "status": "pending"}], merge=True, store=store))
    assert result["revision"] > before
    assert result["progress"] == {
        "phases": {"total": 1, "completed": 0},
        "subtasks": {"total": 2, "completed": 0}, "unresolved": 1,
    }
    assert [entry["status"] for entry in result["todos"]] == ["completed", "completed", "pending"]
    restored = TodoStore()
    restored.restore(result["todos"], revision=result["revision"])
    assert restored.snapshot() == store.snapshot()


def test_cancelled_parent_does_not_hide_an_active_check():
    store = TodoStore()
    result = json.loads(todo_tool([item("phase", "cancelled"), item("check", "pending", "phase")], store=store))
    assert result["progress"] == {
        "phases": {"total": 0, "completed": 0},
        "subtasks": {"total": 1, "completed": 0}, "unresolved": 1,
    }


def test_legacy_parent_cycles_and_orphans_do_not_hide_pending_work():
    store = TodoStore()
    store.restore([item("a", "completed", "b"), item("b", "pending", "a"),
                   item("orphan", "completed", "missing")], revision=9)
    assert json.loads(todo_tool(store=store))["progress"] == {
        "phases": {"total": 3, "completed": 1},
        "subtasks": {"total": 0, "completed": 0}, "unresolved": 1,
    }
    assert store.snapshot()["revision"] == 9
