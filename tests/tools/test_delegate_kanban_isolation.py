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

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
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

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})

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
    from tools.code_execution_tool import _scrub_child_env

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
    assert "HERMES_KANBAN_DB" not in env
    assert "HERMES_KANBAN_WORKSPACE" not in env
    assert "HERMES_KANBAN_CLAIM_LOCK" not in env


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

    conn = kb.connect()
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

    conn = kb.connect()
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


# ---------------------------------------------------------------------------
# Server-role entrypoints must not inherit the lineage marker
# ---------------------------------------------------------------------------


def _run_child_python(code: str, env_overrides: dict[str, str], tmp_path):
    """Run *code* in a real subprocess with the repo under test importable."""
    import subprocess

    env = dict(os.environ)
    env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["HERMES_HOME"] = str(tmp_path / ".hermes")
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["serve", "--host", "127.0.0.1"], True),
        (["dashboard"], True),
        (["gui"], True),
        (["desktop"], True),
        (["gateway", "run"], True),
        # Short-lived CLI calls are NOT a server role — a delegated child that
        # shells out to any of these must keep its lineage.
        (["gateway", "restart"], False),
        (["gateway"], False),
        (["kanban", "boards", "rm", "victim", "--delete"], False),
        (["chat"], False),
        ([], False),
        (["--version"], False),
    ],
)
def test_is_server_role_argv(argv, expected):
    from agent.delegation_context import is_server_role_argv

    assert is_server_role_argv(argv) is expected


def test_serve_entrypoint_drops_inherited_marker_and_kanban_works(tmp_path):
    """The gateway/dashboard process must pass the guard, guard unchanged.

    Reproduces the live defect: the desktop app (and the ``hermes serve`` it
    spawns) was launched from a delegated child, inherited
    HERMES_DELEGATED_CHILD_CONTEXT=1, and then failed every Kanban open for the
    life of the process — connect()'s first-open migration pass goes through
    write_txn, so the guard killed the read path too.
    """
    code = (
        "import sys; sys.argv = ['hermes', 'serve', '--host', '127.0.0.1', '--port', '0']\n"
        "import os\n"
        "import hermes_cli.main  # noqa: F401 — module-level startup is the fix\n"
        "assert 'HERMES_DELEGATED_CHILD_CONTEXT' not in os.environ, 'marker not cleared'\n"
        "from hermes_cli import kanban_db as kb\n"
        "kb.init_db()\n"
        "conn = kb.connect()\n"
        "try:\n"
        "    tid = kb.create_task(conn, title='from-server')\n"
        "finally:\n"
        "    conn.close()\n"
        "assert tid\n"
        "print('SERVER_OK')\n"
    )
    proc = _run_child_python(
        code, {"HERMES_DELEGATED_CHILD_CONTEXT": "1"}, tmp_path,
    )
    assert "SERVER_OK" in proc.stdout, (proc.returncode, proc.stdout, proc.stderr)


def test_agent_cli_entrypoint_keeps_inherited_marker(tmp_path):
    """The reset is server-role only: a delegated `hermes kanban` stays rejected."""
    code = (
        "import sys; sys.argv = ['hermes', 'kanban', 'list']\n"
        "import os\n"
        "import hermes_cli.main  # noqa: F401\n"
        "assert os.environ.get('HERMES_DELEGATED_CHILD_CONTEXT') == '1', 'marker cleared'\n"
        "from hermes_cli import kanban_db as kb\n"
        "try:\n"
        "    kb.init_db()\n"
        "except PermissionError as exc:\n"
        "    assert 'delegate_task child contexts' in str(exc)\n"
        "    print('CLI_REJECTED')\n"
        "else:\n"
        "    print('CLI_ALLOWED')\n"
    )
    proc = _run_child_python(
        code, {"HERMES_DELEGATED_CHILD_CONTEXT": "1"}, tmp_path,
    )
    assert "CLI_REJECTED" in proc.stdout, (proc.returncode, proc.stdout, proc.stderr)


def test_real_lineage_child_and_grandchild_still_rejected(tmp_path, monkeypatch):
    """Real delegate_task subprocess-env path: child AND grandchild stay rejected.

    Also pins that the fix does not clear or narrow the marker anywhere on that
    lineage — the grandchild inherits it through a plain ``env=None`` spawn.
    """
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from agent.delegation_context import (
        delegated_child_context,
        delegated_child_subprocess_env,
    )

    grandchild = (
        "import os\n"
        "assert os.environ.get('HERMES_DELEGATED_CHILD_CONTEXT') == '1'\n"
        "from hermes_cli import kanban_db as kb\n"
        "try:\n"
        "    kb.init_db()\n"
        "except PermissionError:\n"
        "    print('GRANDCHILD_REJECTED')\n"
        "else:\n"
        "    print('GRANDCHILD_ALLOWED')\n"
    )
    child = (
        "import os, subprocess, sys\n"
        "assert os.environ.get('HERMES_DELEGATED_CHILD_CONTEXT') == '1'\n"
        "from hermes_cli import kanban_db as kb\n"
        "try:\n"
        "    kb.init_db()\n"
        "except PermissionError:\n"
        "    print('CHILD_REJECTED')\n"
        "else:\n"
        "    print('CHILD_ALLOWED')\n"
        # env=None → the grandchild inherits this process's env, marker included.
        f"g = subprocess.run([sys.executable, '-c', {grandchild!r}],\n"
        "                   capture_output=True, text=True, timeout=120)\n"
        "print(g.stdout.strip())\n"
        "print(g.stderr[-500:], file=sys.stderr)\n"
    )

    # Build the child env exactly the way delegate_task's subprocess call sites
    # do, from inside a real delegated-child ContextVar scope.
    with delegated_child_context():
        child_env = delegated_child_subprocess_env(os.environ)
    assert child_env is not None
    assert child_env["HERMES_DELEGATED_CHILD_CONTEXT"] == "1"

    proc = _run_child_python(child, child_env, tmp_path)
    assert "CHILD_REJECTED" in proc.stdout, (proc.returncode, proc.stdout, proc.stderr)
    assert "GRANDCHILD_REJECTED" in proc.stdout, (proc.returncode, proc.stdout, proc.stderr)
