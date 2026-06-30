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

    def test_absolute_path_ignores_terminal_cwd(self, monkeypatch, tmp_path):
        """Absolute paths are unaffected by TERMINAL_CWD."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        from tools.file_tools import _resolve_path

        absolute = (tmp_path / "already-absolute.txt").resolve()
        result = _resolve_path(str(absolute))
        assert result == absolute

    def test_falls_back_to_cwd_without_terminal_cwd(self, monkeypatch):
        """Without TERMINAL_CWD, falls back to os.getcwd()."""
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        from tools.file_tools import _resolve_path

        result = _resolve_path("some_file.txt")
        assert result == Path(os.getcwd()) / "some_file.txt"

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        """~ is expanded before TERMINAL_CWD join (already absolute)."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        from tools.file_tools import _resolve_path

        from hermes_constants import get_real_home
        result = _resolve_path("~/notes.txt")
        # After real-home expansion, ~/notes.txt becomes absolute → TERMINAL_CWD ignored
        assert result == Path(get_real_home()) / "notes.txt"

    def test_tilde_uses_real_home_when_profile_home_is_process_home(self, monkeypatch, tmp_path):
        """~ should mean the OS user home, not <profile>/home.

        Regression: a profile-scoped Desktop/gateway turn resolved
        ~/.life-os/... as <HERMES_HOME>/home/.life-os/... and reported file not
        found even though the user's real file lived under /home/<user>.
        """
        real_home = tmp_path / "real-home"
        profile_home = tmp_path / ".hermes" / "profiles" / "life-advisor"
        profile_user_home = profile_home / "home"
        real_home.mkdir()
        profile_user_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("HOME", str(profile_user_home))
        monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "workspace"))

        from tools.file_tools import _resolve_path

        result = _resolve_path("~/.life-os/projects/Main/memory/graph.json")

        assert result == real_home / ".life-os" / "projects" / "Main" / "memory" / "graph.json"

    def test_result_is_resolved(self, monkeypatch, tmp_path):
        """Output path has no '..' components."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        from tools.file_tools import _resolve_path

        result = _resolve_path("a/../b/file.txt")
        assert ".." not in str(result)
        assert result == (tmp_path / "b" / "file.txt")

    def test_relative_path_prefers_live_file_ops_cwd(self, monkeypatch, tmp_path):
        """Live env.cwd must win after the terminal session changes directory."""
        start_dir = tmp_path / "start"
        live_dir = tmp_path / "worktree"
        start_dir.mkdir()
        live_dir.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(start_dir))

        from tools import file_tools

        task_id = "live-cwd"
        fake_ops = SimpleNamespace(
            env=SimpleNamespace(cwd=str(live_dir)),
            cwd=str(start_dir),
        )

        with file_tools._file_ops_lock:
            previous = file_tools._file_ops_cache.get(task_id)
            file_tools._file_ops_cache[task_id] = fake_ops

        try:
            result = file_tools._resolve_path("nested/file.txt", task_id=task_id)
        finally:
            with file_tools._file_ops_lock:
                if previous is None:
                    file_tools._file_ops_cache.pop(task_id, None)
                else:
                    file_tools._file_ops_cache[task_id] = previous

        assert result == live_dir / "nested" / "file.txt"
