"""Tool-surface cwd contract tests for gateway workspaces.

These cover the platform-neutral part of #29265: once the gateway has resolved
``TERMINAL_CWD``, the user-visible tool surfaces should agree on that workspace.

Unlike the system-prompt readers fixed in the gateway-cwd-resolver cluster
(agent/runtime_cwd.py), these tool sites already read ``TERMINAL_CWD``-first and
were deliberately left out of scope. This file is a *characterization* guard: it
pins the already-correct behavior so the supersession of PR #29365 is airtight
and a future refactor of these sites can't silently regress the contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from tools import code_execution_tool, file_tools, terminal_tool


@contextmanager
def _isolated_terminal_file_state():
    """Clear live terminal/file-tool cwd caches that other tests may populate."""
    with terminal_tool._env_lock:
        old_envs = dict(terminal_tool._active_environments)
        old_last_activity = dict(terminal_tool._last_activity)
        terminal_tool._active_environments.clear()
        terminal_tool._last_activity.clear()
    with file_tools._file_ops_lock:
        old_file_ops = dict(file_tools._file_ops_cache)
        old_last_known_cwd = dict(file_tools._last_known_cwd)
        file_tools._file_ops_cache.clear()
        file_tools._last_known_cwd.clear()
    try:
        yield
    finally:
        with file_tools._file_ops_lock:
            file_tools._file_ops_cache.clear()
            file_tools._file_ops_cache.update(old_file_ops)
            file_tools._last_known_cwd.clear()
            file_tools._last_known_cwd.update(old_last_known_cwd)
        with terminal_tool._env_lock:
            terminal_tool._active_environments.clear()
            terminal_tool._active_environments.update(old_envs)
            terminal_tool._last_activity.clear()
            terminal_tool._last_activity.update(old_last_activity)


@pytest.fixture
def isolated_terminal_file_state():
    with _isolated_terminal_file_state():
        yield


def test_isolation_clears_and_restores_last_known_cwd(monkeypatch, tmp_path):
    stale = str(tmp_path / "stale-workspace")
    monkeypatch.setitem(file_tools._last_known_cwd, "default", stale)

    with _isolated_terminal_file_state():
        assert file_tools._last_known_cwd == {}

    assert file_tools._last_known_cwd["default"] == stale


def test_terminal_env_config_uses_terminal_cwd(monkeypatch, tmp_path):
    """The terminal tool's default cwd should come from TERMINAL_CWD."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    config = terminal_tool._get_env_config()

    assert config["cwd"] == str(workspace)


def test_file_tool_relative_paths_use_terminal_cwd(monkeypatch, tmp_path, isolated_terminal_file_state):
    """Relative file/search/patch paths resolve under TERMINAL_CWD."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    resolved = file_tools._resolve_path_for_task("notes/today.md", task_id="cwd-contract")

    assert resolved == (workspace / "notes" / "today.md").resolve()


def test_execute_code_project_mode_uses_terminal_cwd(monkeypatch, tmp_path):
    """Project-mode execute_code should run scripts from TERMINAL_CWD."""
    workspace = tmp_path / "workspace"
    staging = tmp_path / "staging"
    workspace.mkdir()
    staging.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    resolved = code_execution_tool._resolve_child_cwd("project", str(staging))

    assert Path(resolved) == workspace


def test_execute_code_project_mode_falls_back_when_terminal_cwd_missing(monkeypatch, tmp_path):
    """Invalid TERMINAL_CWD should not break execute_code project mode startup."""
    staging = tmp_path / "staging"
    staging.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "missing"))

    resolved = code_execution_tool._resolve_child_cwd("project", str(staging))

    assert Path(resolved).is_dir()
    assert Path(resolved) != tmp_path / "missing"
