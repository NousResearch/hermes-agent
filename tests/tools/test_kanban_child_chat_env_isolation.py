"""Child hermes chat must not inherit Kanban lifecycle ownership.

Incident: a Kanban worker ran ``hermes chat -Q … --source tool`` for a
Browser Use benchmark. The child inherited ``HERMES_KANBAN_TASK`` and
called ``kanban_complete``, closing the parent card and triggering
scratch-workspace cleanup while the real worker was still running.

These tests lock the spawn-path contract: terminal / subprocess env
builders drop lifecycle ownership vars for every child, while board
routing pins (``HERMES_KANBAN_BOARD`` / ``HERMES_KANBAN_DB``) and
ordinary tool credentials stay intact. Dispatcher ``_default_spawn``
still injects ownership into the real worker.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[2]

_OWNERSHIP_KEYS = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_CLAIM_LOCK",
)

_SAFE_SAMPLE = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/user",
    "USER": "testuser",
    "HERMES_HOME": "/tmp/hermes-home",
    "MY_APP_VAR": "keep-me",
    "BROWSERBASE_API_KEY": "bb-keep",
    "FIRECRAWL_API_KEY": "fc-keep",
}


def _python_with_repo_path(code: str) -> str:
    return (
        f"PYTHONPATH={shlex.quote(str(_REPO_ROOT))} "
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    )


def _worker_env(tmp_path: Path) -> dict[str, str]:
    env = dict(_SAFE_SAMPLE)
    env.update(
        {
            "HERMES_HOME": str(tmp_path / ".hermes"),
            "HERMES_KANBAN_TASK": "t_parent",
            "HERMES_KANBAN_RUN_ID": "18",
            "HERMES_KANBAN_WORKSPACE": str(tmp_path / "parent-workspace"),
            "HERMES_KANBAN_WORKSPACES_ROOT": str(tmp_path / "workspaces"),
            "HERMES_KANBAN_CLAIM_LOCK": "lock-parent",
            "HERMES_KANBAN_BOARD": "default",
            "HERMES_KANBAN_DB": str(tmp_path / ".hermes" / "kanban.db"),
            "HERMES_SESSION_SOURCE": "kanban",
        }
    )
    return env


def _make_running_kanban_task(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    workspace = tmp_path / "parent-workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "parent-worker")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")

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
    if claim.claim_lock:
        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claim.claim_lock)
    return kb, tid, workspace


def _assert_ownership_stripped(env: dict[str, str]) -> None:
    leaked = [key for key in _OWNERSHIP_KEYS if key in env]
    assert not leaked, f"child inherited lifecycle ownership: {leaked}"


class TestTerminalChildEnvStripsLifecycleOwnership:
    def test_make_run_env_strips_ownership_but_keeps_board_and_tools(self, tmp_path):
        from tools.environments.local import _make_run_env

        parent = _worker_env(tmp_path)
        with patch.dict(os.environ, parent, clear=True):
            child = _make_run_env({})
            assert os.environ["HERMES_KANBAN_TASK"] == "t_parent"

        _assert_ownership_stripped(child)
        assert child["HERMES_KANBAN_BOARD"] == "default"
        assert child["HERMES_KANBAN_DB"] == parent["HERMES_KANBAN_DB"]
        assert child["HERMES_HOME"] == parent["HERMES_HOME"]
        assert child["MY_APP_VAR"] == "keep-me"
        assert child["PATH"]

    def test_sanitize_subprocess_env_strips_ownership(self, tmp_path):
        from tools.environments.local import _sanitize_subprocess_env

        parent = _worker_env(tmp_path)
        child = _sanitize_subprocess_env(parent)
        _assert_ownership_stripped(child)
        assert child["HERMES_KANBAN_BOARD"] == "default"
        assert child["MY_APP_VAR"] == "keep-me"

    def test_hermes_subprocess_env_strips_ownership(self, tmp_path):
        from tools.environments.local import hermes_subprocess_env

        with patch.dict(os.environ, _worker_env(tmp_path), clear=True):
            child = hermes_subprocess_env(inherit_credentials=True)

        _assert_ownership_stripped(child)
        assert child["HERMES_KANBAN_BOARD"] == "default"
        assert child["BROWSERBASE_API_KEY"] == "bb-keep"
        assert child["FIRECRAWL_API_KEY"] == "fc-keep"


def test_lifecycle_keys_are_a_subset_of_full_kanban_env_keys():
    from agent.delegation_context import (
        KANBAN_ENV_KEYS,
        KANBAN_LIFECYCLE_OWNERSHIP_KEYS,
    )

    extra = set(KANBAN_LIFECYCLE_OWNERSHIP_KEYS) - set(KANBAN_ENV_KEYS)
    assert not extra, extra
    assert "HERMES_KANBAN_BOARD" not in KANBAN_LIFECYCLE_OWNERSHIP_KEYS
    assert "HERMES_KANBAN_DB" not in KANBAN_LIFECYCLE_OWNERSHIP_KEYS


def test_default_spawn_still_injects_worker_ownership(monkeypatch, tmp_path):
    """Dispatcher-owned workers must still receive lifecycle vars."""
    from hermes_cli import kanban_db as kb

    captured = {}

    class _Proc:
        pid = 4242

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _Proc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    monkeypatch.setattr(kb, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")

    task = kb.Task(
        id="t_board_worker",
        title="ship it",
        body=None,
        assignee="default",
        status="in_progress",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock="lock-worker",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    kb._default_spawn(task, str(workspace))

    env = captured["env"]
    assert env["HERMES_KANBAN_TASK"] == "t_board_worker"
    assert env["HERMES_KANBAN_RUN_ID"] == "7"
    assert env["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert env["HERMES_KANBAN_CLAIM_LOCK"] == "lock-worker"
    assert env["HERMES_SESSION_SOURCE"] == "kanban"


def test_terminal_child_cannot_complete_parent_and_parent_stays_running(
    monkeypatch, tmp_path
):
    """Exact incident: child session finishes, parent card must stay running."""
    kb, tid, workspace = _make_running_kanban_task(monkeypatch, tmp_path)
    from tools.environments.local import LocalEnvironment

    code = (
        "import json, os, sys; "
        "from tools import kanban_tools as kt; "
        "print('CHILD_TASK=' + os.environ.get('HERMES_KANBAN_TASK', '')); "
        "print(kt._handle_complete({'summary': 'child stole the card'}))"
    )
    env = LocalEnvironment(cwd=str(tmp_path), timeout=20)
    try:
        result = env.execute(_python_with_repo_path(code), timeout=20)
    finally:
        env.cleanup()

    assert "CHILD_TASK=" in result["output"]
    child_task_line = next(
        line for line in result["output"].splitlines() if line.startswith("CHILD_TASK=")
    )
    assert child_task_line == "CHILD_TASK="

    payload = None
    for line in result["output"].splitlines():
        line = line.strip()
        if line.startswith("{") and ("ok" in line or "error" in line):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
    assert payload is not None
    assert payload.get("ok") is not True
    assert "error" in payload

    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"
        assert task.workspace_path == str(workspace)
        assert os.path.isdir(workspace)
    finally:
        conn.close()
