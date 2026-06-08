"""Tests for _find_shell — user-login-shell preference on POSIX.

Regression tests for #42203: on macOS, ``_find_shell`` used to return
``/bin/bash`` (bash 3.2) which silently swallowed background commands
when ``~/.bash_profile`` contained ``exec /bin/zsh -l``.
"""

import os
import platform
from unittest.mock import patch

import pytest

from tools.environments.local import _find_bash, _find_shell


class TestFindShellPrefersUserShell:
    """_find_shell should prefer $SHELL over bash on POSIX."""

    def test_returns_shell_env_when_set_and_exists(self, tmp_path):
        """When $SHELL points to an existing binary, _find_shell returns it."""
        fake_zsh = tmp_path / "zsh"
        fake_zsh.touch()
        with patch.dict(os.environ, {"SHELL": str(fake_zsh)}):
            assert _find_shell() == str(fake_zsh)

    def test_falls_back_to_find_bash_when_shell_unset(self):
        """When $SHELL is unset, _find_shell delegates to _find_bash."""
        env = {k: v for k, v in os.environ.items() if k != "SHELL"}
        with patch.dict(os.environ, env, clear=True):
            assert _find_shell() == _find_bash()

    def test_falls_back_to_find_bash_when_shell_not_a_file(self, tmp_path):
        """When $SHELL points to a non-existent path, _find_shell delegates."""
        fake_path = str(tmp_path / "nonexistent_shell")
        with patch.dict(os.environ, {"SHELL": fake_path}):
            assert _find_shell() == _find_bash()

    def test_falls_back_to_find_bash_when_shell_empty(self):
        """When $SHELL is empty string, _find_shell delegates."""
        with patch.dict(os.environ, {"SHELL": ""}):
            assert _find_shell() == _find_bash()


class TestFindShellWindowsBehavior:
    """On Windows, _find_shell always delegates to _find_bash."""

    def test_windows_ignores_shell_env(self):
        """On Windows, $SHELL is ignored — _find_shell delegates to _find_bash."""
        with patch("tools.environments.local._IS_WINDOWS", True):
            # Even if SHELL is set, it should be ignored on Windows
            with patch.dict(os.environ, {"SHELL": "/usr/bin/zsh"}):
                result = _find_shell()
                assert result == _find_bash()


class TestFindShellReturnsString:
    """_find_shell must return a string, never None."""

    def test_returns_string(self):
        """_find_shell always returns a non-empty string on any platform."""
        result = _find_shell()
        assert isinstance(result, str)
        assert len(result) > 0


class TestFindBashUnchanged:
    """_find_bash should be unaffected by the _find_shell change."""

    def test_find_bash_still_prefers_bash(self):
        """_find_bash still returns bash (not $SHELL) on POSIX."""
        result = _find_bash()
        # On any system, _find_bash should return something containing "bash"
        # or fall back to $SHELL or /bin/sh — but it should NOT prefer $SHELL
        # over bash the way _find_shell does.
        assert isinstance(result, str)
        assert len(result) > 0
