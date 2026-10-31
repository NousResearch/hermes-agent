"""MCP surface for worktree_id (ORCA-06 kanban<->orca auto-link).

Covers the kanban task API workers/orchestrators actually call:
``kanban_create`` accepts and round-trips ``worktree_id`` (visible in the
create ack and ``kanban_show``), invalid identifiers are rejected as tool
errors with nothing persisted, and omitted links read back as null.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_create_schema_declares_worktree_id():
    from tools.kanban_tools_schemas import KANBAN_CREATE_SCHEMA
    props = KANBAN_CREATE_SCHEMA["parameters"]["properties"]
    assert props["worktree_id"]["type"] == "string"
    assert "worktree_id" in props


def test_create_roundtrips_link(isolated_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({
        "title": "orca-linked card",
        "assignee": "worker",
        "worktree_id": "wt_9f2c1a",
    }))
    assert out["ok"] is True
    tid = out["task_id"]
    # The create ack carries the link back to the caller.
    assert out["worktree_id"] == "wt_9f2c1a"

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.worktree_id == "wt_9f2c1a"

    shown = json.loads(kt._handle_show({"task_id": tid}))
    assert shown["task"]["worktree_id"] == "wt_9f2c1a"


def test_create_omitted_link_is_none(isolated_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({
        "title": "plain card",
        "assignee": "worker",
    }))
    assert out["ok"] is True
    tid = out["task_id"]
    assert out["worktree_id"] is None

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.worktree_id is None
    shown = json.loads(kt._handle_show({"task_id": tid}))
    assert shown["task"]["worktree_id"] is None


@pytest.mark.parametrize("bad", ["has space", "!!", "-leading-dash"])
def test_create_rejects_invalid_link(isolated_home, bad):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    with kbc.connect_closing() as conn:
        before = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]

    out = json.loads(kt._handle_create({
        "title": "bad link card",
        "assignee": "worker",
        "worktree_id": bad,
    }))
    assert "worktree_id" in out["error"]

    with kbc.connect_closing() as conn:
        after = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    assert after == before  # rejected writes persist nothing


def test_show_exposes_link_in_worker_context(isolated_home):
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({
        "title": "ctx card",
        "assignee": "worker",
        "worktree_id": "wt_ctx",
    }))
    shown = json.loads(kt._handle_show({"task_id": out["task_id"]}))
    assert "Orca worktree: wt_ctx" in shown["worker_context"]
