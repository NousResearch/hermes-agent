"""Restricted implementer worker tool-surface tests."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


def test_implementer_runtime_refuses_delegation(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "implementer")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    from tools.delegate_tool import delegate_task

    result = json.loads(delegate_task(goal="forbidden", parent_agent=object()))

    assert "bounded executors" in result["error"]


def test_implementer_runtime_refuses_new_cards(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "implementer")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    from tools.kanban_tools import _handle_create

    result = json.loads(_handle_create({"title": "forbidden", "assignee": "dev"}))

    assert "bounded executors" in result["error"]


def test_implementer_tool_schema_hides_orchestration(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "implementer")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    from model_tools import _select_tool_names

    names = _select_tool_names(["coding", "kanban", "delegation"], None, quiet_mode=True)

    assert {"delegate_task", "kanban_create", "kanban_link"}.isdisjoint(names)


@pytest.mark.macos_only
def test_sandbox_profile_allows_only_workspace_and_board_database(tmp_path):
    workspace = tmp_path / 'work"space'
    board = tmp_path / "board" / "kanban.db"
    profile_home = tmp_path / "profile"
    workspace.mkdir()
    board.parent.mkdir()
    profile_home.mkdir()
    from hermes_cli.kanban_implementer_sandbox import sandboxed_implementer_argv

    argv = sandboxed_implementer_argv(
        ["hermes"], workspace=str(workspace), hermes_home=str(profile_home), board_db=str(board),
    )

    profile = argv[2]
    assert str(workspace) not in profile
    assert f'(subpath "{board.parent}")' not in profile
    assert f'(subpath "{profile_home}")' not in profile
    assert f'(literal "{board}")' in profile
    assert f'(literal "{board}-wal")' in profile
    assert 'work\\"space' in profile
    assert '(subpath "/tmp")' not in profile


@pytest.mark.macos_only
def test_real_sandbox_confines_terminal_and_python_writes(tmp_path):
    workspace = tmp_path / "workspace"
    board = tmp_path / "board" / "kanban.db"
    profile_home = tmp_path / "profile"
    original = tmp_path / "original"
    for path in (workspace, board.parent, profile_home / "cache", original):
        path.mkdir(parents=True)
    from hermes_cli.kanban_implementer_sandbox import sandboxed_implementer_argv

    def run(command):
        argv = sandboxed_implementer_argv(
            command, workspace=str(workspace), hermes_home=str(profile_home), board_db=str(board),
        )
        return subprocess.run(argv, capture_output=True, text=True)

    assert run(["/usr/bin/touch", str(workspace / "ok")]).returncode == 0
    assert run(["/usr/bin/touch", str(board)]).returncode == 0
    assert run(["/usr/bin/touch", str(profile_home / "cache" / "ok")]).returncode == 0
    assert run(["/usr/bin/touch", str(profile_home / "config.yaml")]).returncode != 0
    assert run(["/usr/bin/touch", str(original / "blocked")]).returncode != 0
    python_write = [
        sys.executable, "-c", "from pathlib import Path; Path(r'%s').write_text('bad')" % (original / "python"),
    ]
    assert run(python_write).returncode != 0
    assert not (original / "blocked").exists()
    assert not (original / "python").exists()
