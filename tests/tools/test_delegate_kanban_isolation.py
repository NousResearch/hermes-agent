"""Regression tests for delegate_task isolation from parent Kanban workers."""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import pytest

# The subprocess-boundary tests below spawn ``sys.executable -c`` with a tmp
# cwd. Without an explicit PYTHONPATH the child resolves ``hermes_cli`` /
# ``agent`` through whatever install is on sys.path (in a worktree that is the
# MAIN checkout's editable install, which may not contain the code under
# test). Pin the repo root so the child always imports the tree being tested.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _python_with_repo_path(code: str) -> str:
    """Build a shell command running *code* with the repo under test on PYTHONPATH."""
    return (
        f"PYTHONPATH={shlex.quote(str(_REPO_ROOT))} "
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    )


def _make_running_kanban_task(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    attachments_root = tmp_path / "attachments"
    workspace = tmp_path / "parent-workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "parent-worker")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachments_root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="parent",
            assignee="parent-worker",
            workspace_kind="scratch",
            workspace_path=str(workspace),
        )
        claim = kb.claim_task(conn, tid)
        assert claim is not None
        run_id = claim.id
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return kb, tid, workspace, attachments_root


def test_delegated_child_context_suppresses_env_gated_kanban_tools(monkeypatch, tmp_path):
    """A delegate_task child must not inherit the parent's Kanban tool schema.

    The parent process may be a dispatcher worker with HERMES_KANBAN_TASK set;
    the child is only a subagent, not the run owner.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # noqa: F401 - ensure registered
    from agent.delegation_context import delegated_child_context
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    with delegated_child_context():
        schema = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)

    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "terminal" in names
    assert {n for n in names if n and n.startswith("kanban_")} == set()


def test_build_child_agent_strips_kanban_toolset_even_when_parent_is_worker(monkeypatch):
    """Child construction must fail closed even if the parent exposes kanban."""
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.valid_tool_names = {"terminal"}
            self.session_id = "child-session"

    import run_agent
    from tools import delegate_tool
    import tools.delegate_tool_config as delegate_tool_config

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(delegate_tool_config, "_load_config", lambda: {})

    class Parent:
        enabled_toolsets = ["terminal", "kanban"]
        valid_tool_names = {"terminal", "kanban_complete", "kanban_comment"}
        model = "test-model"
        provider = "test-provider"
        base_url = "http://example.invalid"
        api_mode = "chat_completions"
        platform = "cli"
        session_id = "parent-session"

    child = delegate_tool._build_child_agent(
        task_index=0,
        goal="review only",
        context=None,
        toolsets=None,
        model=None,
        max_iterations=3,
        task_count=1,
        parent_agent=Parent(),
    )

    assert child.valid_tool_names == {"terminal"}
    assert "kanban" not in captured["enabled_toolsets"]
    assert "kanban" in captured["disabled_toolsets"]


def test_delegate_child_execute_code_env_bridges_contextvar_and_scrubs_kanban(
    monkeypatch,
    tmp_path,
):
    """The real execute_code child-env builder must bridge ContextVar lineage.

    Regression coverage for the vulnerable path: delegate_task marks child
    execution with a ContextVar, while execute_code used to scrub plain
    ``os.environ`` and therefore never wrote HERMES_DELEGATED_CHILD_CONTEXT into
    the sandbox env.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path / "parent-workspace"))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    from agent.delegation_context import delegated_child_context
    from tools.code_execution_env import _scrub_child_env

    with delegated_child_context():
        env = _scrub_child_env(
            dict(os.environ),
            is_passthrough=lambda k: k.startswith("HERMES_KANBAN_"),
            is_windows=False,
        )

    assert os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT") is None
    assert env["HERMES_HOME"] == str(home)
    assert env["HERMES_DELEGATED_CHILD_CONTEXT"] == "1"
    assert "HERMES_KANBAN_TASK" not in env
    assert "HERMES_KANBAN_RUN_ID" not in env
    assert "HERMES_KANBAN_CLAIM_LOCK" not in env
    # Board location and workspace routing ride along with the fence marker.
    assert env["HERMES_KANBAN_DB"] == str(home / "kanban.db")
    assert env["HERMES_KANBAN_WORKSPACE"] == str(tmp_path / "parent-workspace")


def test_delegate_child_kanban_cli_cannot_delete_parent_board(
    monkeypatch,
    tmp_path,
):
    kb, _tid, _workspace, _attachments_root = _make_running_kanban_task(
        monkeypatch,
        tmp_path,
    )
    kb.create_board("victim")
    assert kb.board_exists("victim")

    from agent.delegation_context import delegated_child_context
    from tools.environments.local import LocalEnvironment

    code = (
        "from hermes_cli import kanban; "
        "import argparse; "
        "p=argparse.ArgumentParser(); "
        "sub=p.add_subparsers(dest='cmd'); "
        "kanban.build_parser(sub); "
        "args=p.parse_args(['kanban','boards','rm','victim','--delete']); "
        "raise SystemExit(kanban.kanban_command(args))"
    )
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    try:
        with delegated_child_context():
            result = env.execute(
                _python_with_repo_path(code),
                timeout=15,
            )
    finally:
        env.cleanup()

    assert result["returncode"] == 1
    assert "delegate_task child contexts cannot mutate Kanban tasks" in result["output"]
    assert kb.board_exists("victim")
    assert kb.board_dir("victim").is_dir()


def test_delegate_child_attach_url_guard_leaves_no_row_or_file(monkeypatch, tmp_path):
    kb, tid, _workspace, attachments_root = _make_running_kanban_task(monkeypatch, tmp_path)
    from hermes_cli import kanban_db_connect as kbc

    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools

    def forbidden_download(*_args, **_kwargs):
        raise AssertionError("delegated child guard must run before URL download")

    monkeypatch.setattr(kanban_tools, "_download_url_with_cap", forbidden_download)

    with delegated_child_context():
        raw = kanban_tools._handle_attach_url({
            "task_id": tid,
            "url": "https://example.com/leak.txt",
        })

    payload = json.loads(raw)
    assert payload["error"]
    assert "delegate_task child" in payload["error"]

    conn = kbc.connect()
    try:
        assert kb.list_attachments(conn, tid) == []
    finally:
        conn.close()
    task_dir = attachments_root / tid
    assert not task_dir.exists() or list(task_dir.iterdir()) == []


def test_child_attempting_default_complete_does_not_finish_parent_or_delete_workspace(
    monkeypatch,
    tmp_path,
):
    """Deterministic E2E: a delegated child cannot complete its parent task."""
    kb, tid, workspace, _attachments_root = _make_running_kanban_task(monkeypatch, tmp_path)
    from hermes_cli import kanban_db_connect as kbc
    from tools import delegate_tool
    from tools import kanban_tools

    class Parent:
        _current_task_id = tid

        def _touch_activity(self, _desc):
            return None

    class Child:
        tool_progress_callback = None
        _delegate_saved_tool_names = []
        _credential_pool = None
        _subagent_id = "sa-test"
        _delegate_depth = 1
        _parent_subagent_id = None
        model = "test-model"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_estimated_cost_usd = 0.0
        session_reasoning_tokens = 0

        def get_activity_summary(self):
            return {"api_call_count": 0, "max_iterations": 1, "current_tool": None}

        def run_conversation(self, user_message, task_id, **_kwargs):
            attempted = kanban_tools._handle_complete({"summary": "wrong child completion"})
            return {
                "final_response": attempted,
                "completed": True,
                "api_calls": 0,
                "messages": [],
            }

        def close(self):
            return None

    result = delegate_tool._run_single_child(0, "try to complete parent", Child(), Parent())

    conn = kbc.connect()
    try:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
    finally:
        conn.close()

    assert result["status"] == "completed"
    assert "delegate_task child" in result["summary"]
    assert task.status == "running"
    assert run.status == "running"
    assert workspace.is_dir()


def _make_delegation_context_unimportable(monkeypatch) -> None:
    """Simulate a process whose ``agent.delegation_context`` cannot be evaluated.

    ``None`` in ``sys.modules`` is what an importer sees for a module that fails to
    load, but ``from agent import delegation_context`` resolves through the parent
    package's attribute first — so drop that as well, exactly as a broken install
    looks to both import forms.
    """
    import agent

    monkeypatch.setitem(sys.modules, "agent.delegation_context", None)
    monkeypatch.delattr(agent, "delegation_context", raising=False)


def test_unavailable_delegation_context_cannot_mutate_the_board(monkeypatch, tmp_path):
    """An un-evaluable delegation context can never move board state.

    Invariant for both call shapes — the dispatcher-owned worker (which may still
    mutate its own task) and a process with no task at all. Already true on main:
    ``kanban_db_connect.connect()`` imports the same module unwrapped and raises
    before any write, so the write fence is not the only guard today. Pinned here so
    that stays true if that access ever goes lazy or guarded.
    """
    import sqlite3

    kb, tid, _workspace, _attachments_root = _make_running_kanban_task(monkeypatch, tmp_path)
    from tools import kanban_tools

    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    _make_delegation_context_unimportable(monkeypatch)

    owned = kanban_tools._handle_comment({"task_id": tid, "body": "must not land"})
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    unowned = kanban_tools._handle_complete({"task_id": tid, "summary": "must not land"})

    # A refused write never reports success, and never reports nothing.
    for raw in (owned, unowned):
        assert json.loads(raw)["error"], raw

    # Read the board straight from its file: the guarded connect() will not open it.
    conn = sqlite3.connect(kb.kanban_db_path(board=None))
    conn.row_factory = sqlite3.Row
    try:
        assert kb.get_task(conn, tid).status == "running"
        assert kb.latest_run(conn, tid).status == "running"
        assert kb.list_comments(conn, tid) == []
    finally:
        conn.close()


def test_unavailable_delegation_context_fence_refuses_when_the_process_owns_no_task(
    monkeypatch,
    tmp_path,
    caplog,
):
    """The write fence itself must fail closed, not merely fail somewhere later.

    ``_delegation_ctx`` returned its ``default`` (``False`` == "not a delegate
    child") straight out of ``except Exception``, so the fence's verdict for an
    un-evaluable context was "allow" and only the board connection stopped the
    write. A process that owns no ``HERMES_KANBAN_TASK`` must be refused here, with
    an audit line naming the real reason.
    """
    import logging

    kb, tid, _workspace, _attachments_root = _make_running_kanban_task(monkeypatch, tmp_path)
    from tools import kanban_tools

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    _make_delegation_context_unimportable(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="tools.kanban_tools"):
        raw = kanban_tools._handle_comment({"task_id": tid, "body": "must not land"})

    payload = json.loads(raw)
    assert payload["error"]
    assert "refused" in payload["error"]
    assert "delegation context" in payload["error"]
    assert any("un-evaluable" in r.getMessage() for r in caplog.records), caplog.text

