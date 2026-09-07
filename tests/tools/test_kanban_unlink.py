"""Real temporary-board integration tests for guarded dependency removal."""
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tools import kanban_tools as kt


@pytest.fixture
def boards(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)
    (home / "config.yaml").write_text("toolsets: [kanban]\n")
    kb._INITIALIZED_PATHS.clear()
    for board in ("alpha", "beta"):
        kb.init_db(board=board)
    return monkeypatch


def graph(board="alpha", extra=False):
    conn = kb.connect(board=board)
    try:
        root = kb.create_task(conn, title="unfinished goal", assignee="owner")
        child = kb.create_task(conn, title="execution", assignee="worker", parents=[root])
        other = kb.create_task(conn, title="safety prerequisite", assignee="owner")
        if extra:
            kb.link_tasks(conn, other, child)
        return root, child, other
    finally:
        conn.close()


def unlink(root, child, **kw):
    return json.loads(kt._handle_unlink(dict(parent_id=root, child_id=child, **kw)))


def test_unlink_promotes_and_audits_once(boards):
    root, child, _ = graph()
    out = unlink(root, child, board="alpha")
    assert out["ok"] and out["removed"] is True
    assert unlink(root, child, board="alpha")["removed"] is False
    conn = kb.connect(board="alpha")
    try:
        assert kb.parent_ids(conn, child) == []
        assert kb.get_task(conn, child).status == "ready"
        assert kb.get_task(conn, root).status != "done"
        events = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='unlinked'", (child,)).fetchall()
        assert len(events) == 1
        assert json.loads(events[0]["payload"]) == {"parent": root, "child": child}
    finally:
        conn.close()


def test_preserves_other_parents_and_board(boards):
    root, child, other = graph(extra=True)
    beta_root, beta_child, _ = graph("beta")
    boards.setenv("HERMES_KANBAN_BOARD", "beta")
    assert unlink(root, child, board="alpha")["removed"] is True
    for board, expected_child, expected_parent in (("alpha", child, other), ("beta", beta_child, beta_root)):
        conn = kb.connect(board=board)
        try:
            assert kb.parent_ids(conn, expected_child) == [expected_parent]
            assert kb.get_task(conn, expected_child).status == "todo"
        finally:
            conn.close()
    assert unlink(beta_root, beta_child)["removed"] is True


@pytest.mark.parametrize("args", [{}, {"parent_id": "x"}, {"parent_id": 1, "child_id": "y"}, {"parent_id": " ", "child_id": "y"}])
def test_missing_or_invalid_arguments(boards, args):
    assert json.loads(kt._handle_unlink(args)).get("error")


def test_nonexistent_edge_is_idempotent(boards):
    assert unlink("t_missing", "t_absent", board="alpha")["removed"] is False


@pytest.mark.parametrize("context", ["worker", "delegated", "unconfigured"])
def test_denied_before_connect(boards, monkeypatch, context):
    if context == "worker":
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    elif context == "delegated":
        monkeypatch.setattr(kt, "_is_delegated_child_context", lambda: True)
    else:
        monkeypatch.setattr(kt, "_profile_has_kanban_toolset", lambda: False)
    def forbidden(*args, **kwargs):
        pytest.fail("unauthorized caller opened a database")
    monkeypatch.setattr(kt, "_connect", forbidden)
    assert unlink("root", "child", board="beta").get("error")


def test_schema_available_only_to_orchestrator(boards):
    from tools.registry import registry, invalidate_check_fn_cache
    from toolsets import resolve_toolset
    def names():
        invalidate_check_fn_cache()
        return {s["function"]["name"] for s in registry.get_definitions(set(resolve_toolset("kanban")), quiet=True)}
    assert "kanban_unlink" in names()
    boards.setenv("HERMES_KANBAN_TASK", "t_worker")
    assert "kanban_unlink" not in names()
