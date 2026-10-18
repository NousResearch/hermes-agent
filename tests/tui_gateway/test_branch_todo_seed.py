"""A Desktop/TUI branch child inherits the parent's todo list.

Branch history is the visible user/assistant projection (tool rows dropped), so the child
agent's hydration finds no todo result and starts empty while the UI still shows the
parent's list (#136606). The branch seeds the child's store from the parent's authoritative
snapshot instead.
"""

from types import SimpleNamespace


def _parent_session(todos, revision):
    from tools.todo_tool import TodoStore

    store = TodoStore()
    store.restore(todos, revision=revision)
    return {"agent": SimpleNamespace(_todo_store=store), "todo_state": None}


def test_branch_seed_carries_parent_todos_with_revision(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server
    from tools.todo_tool import TodoStore

    todos = [
        {"id": "t1", "content": "first task", "status": "completed"},
        {"id": "t2", "content": "second task", "status": "pending"},
        {"id": "t3", "content": "third task", "status": "pending"},
    ]
    parent = _parent_session(todos, 9)
    child_agent = SimpleNamespace(_todo_store=TodoStore())

    server._seed_branch_todo_state(parent, child_agent)

    snapshot = child_agent._todo_store.snapshot()
    assert snapshot["revision"] == 9
    assert [item["content"] for item in snapshot["todos"]] == [
        "first task", "second task", "third task"]


def test_branch_seed_without_parent_todos_leaves_child_empty(monkeypatch):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server
    from tools.todo_tool import TodoStore

    parent = _parent_session([], 0)
    child_agent = SimpleNamespace(_todo_store=TodoStore())

    server._seed_branch_todo_state(parent, child_agent)

    assert child_agent._todo_store.snapshot() == {"todos": [], "revision": 0}


def test_branch_seed_tolerates_a_child_without_a_store(monkeypatch):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    parent = _parent_session([{"id": "t1", "content": "only task", "status": "pending"}], 3)

    server._seed_branch_todo_state(parent, SimpleNamespace())  # must not raise


def test_branch_seed_uses_db_state_for_a_cold_parent(monkeypatch):
    """The branch parent may have no live agent (cold resume); the persisted projection
    seeds the child, and the parent session dict is not mutated (#136606)."""
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server
    from tools.todo_tool import TodoStore

    parent = {"agent": None, "todo_state": None}
    child_agent = SimpleNamespace(_todo_store=TodoStore())

    server._seed_branch_todo_state(parent, child_agent, db_state={
        "todos": [{"id": "t1", "content": "persisted task", "status": "pending"}],
        "revision": 12})

    assert child_agent._todo_store.snapshot() == {
        "todos": [{"id": "t1", "content": "persisted task", "status": "pending"}],
        "revision": 12}
    assert "_branch_todo_state" not in parent


def test_branch_seed_prefers_live_state_over_db_state(monkeypatch):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server
    from tools.todo_tool import TodoStore

    parent = _parent_session([{"id": "t1", "content": "live task", "status": "in_progress"}], 9)
    child_agent = SimpleNamespace(_todo_store=TodoStore())

    server._seed_branch_todo_state(parent, child_agent, db_state={
        "todos": [{"id": "t2", "content": "stale persisted task", "status": "pending"}],
        "revision": 3})

    snapshot = child_agent._todo_store.snapshot()
    assert snapshot["revision"] == 9
    assert snapshot["todos"][0]["content"] == "live task"
