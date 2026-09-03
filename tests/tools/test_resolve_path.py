"""Tests for _resolve_path() — TERMINAL_CWD-aware path resolution in file_tools."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace


class TestResolvePath:
    """Verify _resolve_path respects TERMINAL_CWD for worktree isolation."""

    def test_relative_path_uses_terminal_cwd(self, monkeypatch, tmp_path):
        """Relative paths resolve against TERMINAL_CWD, not process CWD."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        from tools.file_tools import _resolve_path

        result = _resolve_path("foo/bar.py")
        assert result == (tmp_path / "foo" / "bar.py")


    def test_relative_path_prefers_recorded_session_cwd(self, monkeypatch, tmp_path):
        """The session's recorded cwd must win after the terminal changes directory."""
        start_dir = tmp_path / "start"
        live_dir = tmp_path / "worktree"
        start_dir.mkdir()
        live_dir.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(start_dir))

        from tools import file_tools, terminal_tool

        task_id = "live-cwd"
        # The session's completed `cd` recorded the new directory.
        terminal_tool.record_session_cwd(task_id, str(live_dir))

        try:
            result = file_tools._resolve_path("nested/file.txt", task_id=task_id)
        finally:
            terminal_tool.clear_session_cwd(task_id)

        assert result == live_dir / "nested" / "file.txt"

    def test_kanban_workspace_beats_stale_recorded_session_cwd(self, monkeypatch, tmp_path):
        """Relative file reads stay in the worker's assigned task worktree."""
        stable = tmp_path / "stable-base"
        workspace = tmp_path / "task-worktree"
        stable.mkdir()
        workspace.mkdir()

        from tools import file_tools, terminal_tool

        task_id = "kanban-worker-session"
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_workspace")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
        monkeypatch.setenv("TERMINAL_CWD", str(workspace))
        terminal_tool.record_session_cwd(task_id, str(stable))
        try:
            result = file_tools._resolve_path("README.md", task_id=task_id)
        finally:
            terminal_tool.clear_session_cwd(task_id)

        assert result == workspace / "README.md"

    def test_delegated_task_inherits_current_session_cwd(self, monkeypatch, tmp_path):
        """A child task resolves files in its parent desktop session workspace."""
        workspace = tmp_path / "conversation-worktree"
        process_cwd = tmp_path / "gateway-launch"
        workspace.mkdir()
        process_cwd.mkdir()
        monkeypatch.chdir(process_cwd)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        from gateway.session_context import clear_session_vars, set_session_vars
        from tools import file_tools, terminal_tool

        session_key = "desktop-session"
        child_task_id = "delegated-child"
        terminal_tool.record_session_cwd(session_key, str(workspace))
        tokens = set_session_vars(session_key=session_key, cwd=str(workspace))
        try:
            result = file_tools._resolve_path("dir/README.md", task_id=child_task_id)
        finally:
            clear_session_vars(tokens)
            terminal_tool.clear_session_cwd(session_key)

        assert result == workspace / "dir" / "README.md"

    def test_session_cwd_inheritance_failure_leaves_debug_breadcrumb(
        self, monkeypatch, tmp_path, caplog
    ):
        """A failed inherited-session lookup is diagnosable without breaking fallback."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

        from tools import file_tools, terminal_tool

        def fail_session_lookup():
            raise RuntimeError("session registry unavailable")

        monkeypatch.setattr(terminal_tool, "_current_session_key", fail_session_lookup)
        with caplog.at_level(logging.DEBUG, logger="tools.file_tools"):
            result = file_tools._resolve_path("README.md", task_id="delegated-child")

        assert result == tmp_path / "README.md"
        assert "session cwd inheritance unavailable" in caplog.text
