"""Regression tests for profile-aware tilde expansion in file tools.

The bug (#48552): in-process file tools (write_file, read_file, patch,
search_files) resolved ``~`` via ``os.path.expanduser()``, which reads the
gateway process's ``HOME``.  In profile mode (Docker, systemd, s6) the gateway
``HOME`` differs from the profile ``HOME`` that interactive sessions use, so
``~`` expanded to the wrong directory and file operations failed with
"no such file or directory".

The fix adds ``_expand_tilde()`` which delegates to
``hermes_constants.get_subprocess_home()`` — the same policy the terminal tool
uses for subprocess environments.

See: https://github.com/NousResearch/hermes-agent/issues/48552
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.file_tools as ft


# ---------------------------------------------------------------------------
# _expand_tilde() unit tests
# ---------------------------------------------------------------------------

class TestExpandTilde:
    """Verify the _expand_tilde() helper resolves ~ to the profile home."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Path separator mismatch: \\ vs / on Windows")
    def test_tilde_expands_to_profile_home(self):
        """When get_subprocess_home returns a value, ~/path uses it."""
        with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/profiles/coder/home"):
            result = ft._expand_tilde("~/scratch/file.txt")
        assert result == "/opt/data/profiles/coder/home/scratch/file.txt"

    def test_bare_tilde_expands_to_profile_home(self):
        """Bare ~ expands to the profile home."""
        with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/profiles/coder/home"):
            result = ft._expand_tilde("~")
        assert result == "/opt/data/profiles/coder/home"

    def test_falls_back_when_no_profile_home(self):
        """When get_subprocess_home returns None, use os.path.expanduser."""
        with patch("hermes_constants.get_subprocess_home", return_value=None):
            result = ft._expand_tilde("~/Documents")
        assert result == os.path.expanduser("~/Documents")

    def test_other_user_tilde_not_overridden(self):
        """~user/path must NOT use the profile home — it's a different user."""
        with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/profiles/coder/home"):
            result = ft._expand_tilde("~root/file.txt")
        # Should use os.path.expanduser, not the profile home
        assert "/opt/data/profiles/coder/home" not in result

    def test_no_tilde_unchanged(self):
        """Paths without ~ are returned unchanged (modulo expanduser)."""
        with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/profiles/coder/home"):
            result = ft._expand_tilde("/etc/passwd")
        assert result == "/etc/passwd"

    def test_empty_path_unchanged(self):
        """Empty string returns empty."""
        with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/profiles/coder/home"):
            assert ft._expand_tilde("") == ""


# ---------------------------------------------------------------------------
# Integration: _resolve_path_for_task uses profile home
# ---------------------------------------------------------------------------

class TestResolvePathUsesProfileHome:
    """Verify _resolve_path_for_task resolves ~ to the profile home."""

    def test_relative_tilde_resolves_to_profile_home(self, tmp_path, monkeypatch):
        """A ~/path argument resolves under the profile home, not process HOME."""
        profile_home = tmp_path / "profile_home"
        profile_home.mkdir()
        process_home = tmp_path / "process_home"
        process_home.mkdir()

        monkeypatch.setenv("HOME", str(process_home))
        monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)

        with patch("hermes_constants.get_subprocess_home", return_value=str(profile_home)):
            resolved = ft._resolve_path_for_task("~/test_file.txt", task_id="test")

        assert str(resolved).startswith(str(profile_home))
        assert "process_home" not in str(resolved)

    def test_absolute_tilde_in_workspace_root(self, tmp_path, monkeypatch):
        """A workspace root specified with ~ resolves to profile home."""
        profile_home = tmp_path / "profile_home"
        profile_home.mkdir()
        process_home = tmp_path / "process_home"
        process_home.mkdir()

        monkeypatch.setenv("HOME", str(process_home))
        monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)

        with patch("hermes_constants.get_subprocess_home", return_value=str(profile_home)):
            # _resolve_base_dir uses the workspace root from config; if it contains ~,
            # it should resolve to profile home
            resolved = ft._resolve_path_for_task("~/data/config.json", task_id="test")

        assert str(profile_home) in str(resolved)
        assert str(process_home) not in str(resolved)
