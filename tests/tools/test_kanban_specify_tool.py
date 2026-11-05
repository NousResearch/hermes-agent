"""Public creator-only promotion of triage cards through registry dispatch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def board_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "creator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _task(title: str, *, created_by="creator"):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    with connect() as conn:
        return kb.create_task(conn, title=title, body="original", triage=True, created_by=created_by)


def _read(task_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    with connect() as conn:
        return kb.get_task(conn, task_id), kb.list_events(conn, task_id), kb.list_comments(conn, task_id)


def _dispatch(args):
    import tools.kanban_tools  # register the public handler through its production module discovery path
    from tools.registry import registry

    return json.loads(registry.dispatch("kanban_specify", args))


def test_creator_promotes_own_triage_card_and_preserves_graph_audit(board_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    parent = _task("parent")
    child = _task("child")
    with connect() as conn:
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        before_links = conn.execute("SELECT parent_id, child_id FROM task_links WHERE child_id = ?", (child,)).fetchall()
        before_body = kb.get_task(conn, child).body
    out = _dispatch({"task_id": child})
    assert out.get("ok"), out
    assert out["status"] == "todo"
    task, events, comments = _read(child)
    assert task.status == "todo"  # open parent gate remains in force
    assert task.body == before_body
    assert [(r["parent_id"], r["child_id"]) for r in before_links] == [(parent, child)]
    assert any(event.kind == "specified" for event in events)
    assert comments == []


def test_creator_reaches_ready_when_all_parents_done(board_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    parent = _task("done parent")
    child = _task("ready child")
    with connect() as conn:
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
    out = _dispatch({"task_id": child})
    assert out["ok"] and out["status"] == "ready"


def test_noncreator_and_spoofed_identity_are_rejected_without_change(board_env, monkeypatch):
    other_task = _task("owned by someone else", created_by="other")
    out = _dispatch({"task_id": other_task})
    assert not out.get("ok"), out
    assert _read(other_task)[0].status == "triage"

    own_task = _task("owned by creator")
    monkeypatch.setenv("HERMES_PROFILE", "other")
    out = _dispatch({"task_id": own_task, "profile": "creator", "created_by": "creator", "caller": "creator"})
    assert not out.get("ok"), out
    assert _read(own_task)[0].status == "triage"


@pytest.mark.parametrize("status", ["todo", "ready", "running", "blocked", "done", "archived"])
def test_non_triage_statuses_are_unchanged(board_env, status):
    task_id = _task(status)
    from hermes_cli.kanban_db_connect import connect

    with connect() as conn:
        with __import__("hermes_cli.kanban_db", fromlist=["write_txn"]).write_txn(conn):
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    out = _dispatch({"task_id": task_id})
    assert not out.get("ok"), out
    assert _read(task_id)[0].status == status


def test_kanban_specify_is_registered_for_every_configured_kanban_profile(board_env):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import get_tool_definitions

    save_config({"platform_toolsets": {
        "vidscoder": ["kanban"], "default": ["kanban"],
        "coding": ["kanban"], "gpu-queue": ["kanban"], "plain": ["file"],
    }})
    config = load_config()
    for profile, selection in config["platform_toolsets"].items():
        names = {
            row["function"]["name"]
            for row in get_tool_definitions(
                _get_platform_tools(config, profile, include_default_mcp_servers=False),
                quiet_mode=True, skip_tool_search_assembly=True,
            )
        }
        if "kanban" in selection:
            assert "kanban_specify" in names
        else:
            assert "kanban_specify" not in names
