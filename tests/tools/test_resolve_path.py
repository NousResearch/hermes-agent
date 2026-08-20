"""Tests for _resolve_path() — TERMINAL_CWD-aware path resolution in file_tools."""

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

    def test_ssh_tilde_not_expanded_on_host(self, monkeypatch):
        """#71201: on the ssh backend a leading ~ must NOT be expanded against
        the gateway HOME — it must pass through for remote-side expansion."""
        from tools import file_tools

        monkeypatch.setattr(
            file_tools, "_terminal_env_type_for_task",
            lambda task_id="default": "ssh",
        )

        assert str(file_tools._resolve_path("~/notes.txt")) == "~/notes.txt"
        assert str(file_tools._resolve_path("~")) == "~"
        assert str(file_tools._resolve_path("~user/file.txt")) == "~user/file.txt"

    def test_local_tilde_still_expanded(self, monkeypatch):
        """On the local backend ~ is still expanded (regression guard)."""
        from tools import file_tools

        monkeypatch.setattr(
            file_tools, "_terminal_env_type_for_task",
            lambda task_id="default": "local",
        )

        result = file_tools._resolve_path("~/notes.txt")
        assert "~" not in str(result)
        assert result.is_absolute()
