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

from pathlib import Path

from gateway.session_context import clear_session_vars, set_session_vars
from tools import code_execution_tool, file_tools, terminal_tool


def test_terminal_env_config_uses_terminal_cwd(monkeypatch, tmp_path):
    """The terminal tool's default cwd should come from TERMINAL_CWD."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    config = terminal_tool._get_env_config()

    assert config["cwd"] == str(workspace)


def test_file_tool_relative_paths_use_terminal_cwd(monkeypatch, tmp_path):
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


def test_tool_surfaces_use_gateway_session_cwd_without_terminal_cwd(monkeypatch, tmp_path):
    """Webhook sessions pin cwd via contextvars, not process-wide TERMINAL_CWD."""
    workspace = tmp_path / "workspace"
    staging = tmp_path / "staging"
    workspace.mkdir()
    staging.mkdir()

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    tokens = set_session_vars(cwd=str(workspace))
    try:
        terminal_cwd = Path(terminal_tool._get_env_config()["cwd"])
        file_cwd = file_tools._resolve_path_for_task("notes/today.md").parent.parent
        execute_cwd = Path(code_execution_tool._resolve_child_cwd("project", str(staging)))
    finally:
        clear_session_vars(tokens)

    assert terminal_cwd == workspace
    assert file_cwd == workspace
    assert execute_cwd == workspace
