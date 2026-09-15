"""Real registry/store tests against isolated temporary Hermes homes."""

import json
from pathlib import Path
import pytest


@pytest.fixture
def board(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("toolsets: [kanban]\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in [
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_BOARD",
        "HERMES_DELEGATED_CHILD_CONTEXT",
        "HERMES_KANBAN_RUN_ID",
    ]:
        monkeypatch.delenv(key, raising=False)
    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    yield kb, conn
    conn.close()


def call(name, **args):
    import tools.kanban_tools
    from tools.registry import registry

    return json.loads(registry.dispatch(name, args))


def new(board, **kwargs):
    kb, c = board
    return kb.create_task(c, title="old", body="body", assignee="tester", **kwargs)


def test_edit_preserves_state_and_history(board):
    kb, c = board
    t = new(board)
    before = kb.get_task(c, t)
    kb.add_comment(c, t, author="tester", body="evidence")
    out = call(
        "kanban_edit",
        task_id=t,
        title="new",
        body="updated",
        expected_title="old",
        expected_body="body",
    )
    assert out["ok"], out
    after = kb.get_task(c, t)
    assert (after.title, after.body) == ("new", "updated")
    for f in ["id", "status", "assignee", "workspace_path", "claim_lock"]:
        assert getattr(before, f) == getattr(after, f)
    assert (
        c.execute(
            "select count(*) from task_comments where task_id=?", (t,)
        ).fetchone()[0]
        == 1
    )


@pytest.mark.parametrize(
    "args",
    [
        {"title": "", "expected_title": "old"},
        {"title": "new"},
        {"priority": True, "expected_priority": 0},
        {"body": None, "expected_body": "body"},
        {},
    ],
)
def test_invalid_edit_no_mutation(board, args):
    kb, c = board
    t = new(board)
    assert "error" in call("kanban_edit", task_id=t, **args)
    assert kb.get_task(c, t).title == "old"


def test_stale_edit_refuses(board):
    kb, c = board
    t = new(board)
    assert call("kanban_edit", task_id=t, title="other", expected_title="old")["ok"]
    assert "error" in call("kanban_edit", task_id=t, title="mine", expected_title="old")
    assert kb.get_task(c, t).title == "other"


def test_noop_does_not_add_event(board):
    kb, c = board
    t = new(board)
    before = c.execute("select count(*) from task_events").fetchone()[0]
    assert (
        call("kanban_edit", task_id=t, title="old", expected_title="old")[
            "changed_fields"
        ]
        == []
    )
    assert c.execute("select count(*) from task_events").fetchone()[0] == before


def test_archive_preserves_files_and_history(board, tmp_path):
    kb, c = board
    t = new(board)
    f = tmp_path / "evidence.txt"
    f.write_text("keep")
    c.execute("update tasks set workspace_path=? where id=?", (str(tmp_path), t))
    c.commit()
    status = kb.get_task(c, t).status
    out = call(
        "kanban_archive",
        task_id=t,
        expected_status=status,
        reason="finished; no external execution",
    )
    assert out.get("ok"), out
    assert kb.get_task(c, t).status == "archived"
    assert f.read_text() == "keep"
    assert out["workspace_retained"]
    assert "error" in call("kanban_edit", task_id=t, title="new", expected_title="old")


@pytest.mark.parametrize("state", ["running", "archived"])
def test_archive_refuses_active_or_archived(board, state):
    kb, c = board
    t = new(board)
    c.execute("update tasks set status=? where id=?", (state, t))
    c.commit()
    assert "error" in call(
        "kanban_archive", task_id=t, expected_status=state, reason="test"
    )
    assert kb.get_task(c, t).status == state


def test_archive_refuses_children(board):
    kb, c = board
    t = new(board)
    child = new(board, parents=[t])
    before = kb.get_task(c, child).status
    assert "error" in call(
        "kanban_archive",
        task_id=t,
        expected_status=kb.get_task(c, t).status,
        reason="test",
    )
    assert kb.get_task(c, child).status == before
    assert kb.get_task(c, t).status != "archived"


def test_archive_stale_status(board):
    kb, c = board
    t = new(board)
    assert "error" in call(
        "kanban_archive", task_id=t, expected_status="done", reason="test"
    )
    assert kb.get_task(c, t).status != "archived"


@pytest.mark.parametrize(
    "tool,args",
    [
        ("kanban_edit", {"title": "new", "expected_title": "old"}),
        ("kanban_archive", {"expected_status": "ready", "reason": "test"}),
    ],
)
def test_worker_and_delegate_denied(board, monkeypatch, tool, args):
    kb, c = board
    t = new(board)
    monkeypatch.setenv("HERMES_KANBAN_TASK", t)
    assert "error" in call(tool, task_id=t, **args)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        assert "error" in call(tool, task_id=t, **args)
    assert kb.get_task(c, t).title == "old"
    assert kb.get_task(c, t).status != "archived"


def test_schema_gating(board, monkeypatch):
    import tools.kanban_tools
    from tools.registry import registry
    from tools.kanban_toolset_context import scoped_kanban_toolset_selection

    names = {"kanban_edit", "kanban_archive"}
    with scoped_kanban_toolset_selection(["kanban"]):
        defs = registry.get_definitions(names, quiet=True)
        assert {s["function"]["name"] for s in defs} == names
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    assert not registry.get_definitions(names, quiet=True)


def test_concurrent_edits_do_not_lose_updates(board):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_db_admin import edit_fields

    kb, c = board
    tid = new(board)
    barrier = Barrier(2)

    def edit(title):
        conn = kbc.connect()
        try:
            barrier.wait(timeout=10)
            try:
                edit_fields(conn, tid, {"title": title}, {"title": "old"})
                return title
            except ValueError as exc:
                assert "stale" in str(exc)
                return None
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, ["first", "second"]))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert kb.get_task(c, tid).title == winners[0]


def test_archive_checks_older_unfinished_runs(board):
    kb, c = board
    tid = new(board)
    c.execute(
        "INSERT INTO task_runs(task_id,status,started_at) VALUES (?, 'running', 1)",
        (tid,),
    )
    c.execute(
        "INSERT INTO task_runs(task_id,status,started_at,ended_at) VALUES (?, 'done', 2, 3)",
        (tid,),
    )
    c.commit()
    status = kb.get_task(c, tid).status
    assert "error" in call(
        "kanban_archive", task_id=tid, expected_status=status, reason="test"
    )
    assert kb.get_task(c, tid).status == status


@pytest.mark.parametrize("field,value", [("claim_lock", "held"), ("worker_pid", 12345)])
def test_archive_refuses_claimed_but_not_running(board, field, value):
    kb, c = board
    tid = new(board)
    c.execute(f"UPDATE tasks SET {field}=? WHERE id=?", (value, tid))
    c.commit()
    status = kb.get_task(c, tid).status
    assert "error" in call(
        "kanban_archive", task_id=tid, expected_status=status, reason="test"
    )
    assert kb.get_task(c, tid).status == status


def test_edit_multiple_fields_rolls_back_on_one_stale_value(board):
    kb, c = board
    tid = new(board)
    assert "error" in call(
        "kanban_edit",
        task_id=tid,
        title="new",
        body="new body",
        expected_title="old",
        expected_body="stale",
    )
    task = kb.get_task(c, tid)
    assert (task.title, task.body) == ("old", "body")


def test_priority_edit(board):
    kb, c = board
    tid = new(board)
    old = kb.get_task(c, tid).priority
    assert call("kanban_edit", task_id=tid, priority=old + 1, expected_priority=old)[
        "ok"
    ]
    assert kb.get_task(c, tid).priority == old + 1
