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
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.file_tools_paths as ft
from tools import terminal_tool


# ---------------------------------------------------------------------------
# _expand_tilde() unit tests
# ---------------------------------------------------------------------------

class TestExpandTilde:
    """Verify the _expand_tilde() helper resolves ~ to the profile home."""

    @pytest.mark.platforms("linux")
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("~", "/opt/data/profiles/coder/home"),
            ("~/config.json", "/opt/data/profiles/coder/home/config.json"),
            (r"~\config.json", "/opt/data/profiles/coder/home/config.json"),
            (r"~\workspace\file.txt", "/opt/data/profiles/coder/home/workspace/file.txt"),
        ],
    )
    def test_tilde_expands_to_profile_home(self, path, expected):
        """When get_subprocess_home returns a value, ~ and Windows-style variants use it."""
        with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/profiles/coder/home"):
            result = ft._expand_tilde(path)
        assert result == expected

    def test_empty_path_unchanged(self):
        """Empty string returns empty."""
        with patch("hermes_constants.get_subprocess_home", return_value="/opt/data/profiles/coder/home"):
            assert ft._expand_tilde("") == ""

    def test_tilde_falls_back_to_expanduser_when_profile_home_unavailable(self):
        """A missing profile home must fall back to the host default expansion."""
        expected = os.path.expanduser(r"~\config.json")
        with patch("hermes_constants.get_subprocess_home", side_effect=RuntimeError("no home")):
            result = ft._expand_tilde(r"~\config.json")
        assert result == expected


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
        monkeypatch.setattr(terminal_tool, "_session_cwd", {})

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
        monkeypatch.setattr(terminal_tool, "_session_cwd", {})

        with patch("hermes_constants.get_subprocess_home", return_value=str(profile_home)):
            # _resolve_base_dir uses the workspace root from config; if it contains ~,
            # it should resolve to profile home
            resolved = ft._resolve_path_for_task("~/data/config.json", task_id="test")

        assert str(profile_home) in str(resolved)
        assert str(process_home) not in str(resolved)
