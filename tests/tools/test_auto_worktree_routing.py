from pathlib import Path

from agent.runtime_cwd import set_session_worktree_map
import agent.runtime_cwd as rt
from tools.file_tools import _resolve_path_for_task
from tools.terminal_tool import _resolve_command_cwd


class _Env:
    def __init__(self, cwd: str):
        self.cwd = cwd


def test_terminal_workdir_inside_configured_repo_routes_to_session_worktree(tmp_path):
    repo = tmp_path / "repo"
    worktree = repo / ".worktrees" / "session"
    (repo / "src").mkdir(parents=True)
    worktree.mkdir(parents=True)

    token = set_session_worktree_map({str(repo): str(worktree)})
    try:
        assert _resolve_command_cwd(
            workdir=str(repo / "src"),
            env=_Env(str(tmp_path)),
            default_cwd=str(tmp_path),
        ) == str(worktree / "src")
    finally:
        rt._SESSION_WORKTREE_MAP.reset(token)


def test_terminal_live_cwd_inside_configured_repo_routes_to_session_worktree(tmp_path):
    repo = tmp_path / "repo"
    worktree = repo / ".worktrees" / "session"
    (repo / "src").mkdir(parents=True)
    worktree.mkdir(parents=True)

    token = set_session_worktree_map({str(repo): str(worktree)})
    try:
        assert _resolve_command_cwd(
            workdir=None,
            env=_Env(str(repo / "src")),
            default_cwd=str(tmp_path),
        ) == str(worktree / "src")
    finally:
        rt._SESSION_WORKTREE_MAP.reset(token)


def test_file_tool_absolute_path_inside_configured_repo_routes_to_session_worktree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    worktree = repo / ".worktrees" / "session"
    (repo / "src").mkdir(parents=True)
    worktree.mkdir(parents=True)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    token = set_session_worktree_map({str(repo): str(worktree)})
    try:
        assert _resolve_path_for_task(str(repo / "src" / "app.py")) == worktree / "src" / "app.py"
    finally:
        rt._SESSION_WORKTREE_MAP.reset(token)
