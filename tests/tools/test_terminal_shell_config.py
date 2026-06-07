"""Tests for terminal.shell config option.

Verifies that the shell used for terminal() command execution can be
configured via ``terminal.shell`` in config.yaml, with auto-detection
from ``$SHELL`` and fallback to ``/bin/bash``.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from tools.environments.local import (
    _find_bash,
    _read_terminal_shell_config,
    _resolve_terminal_shell,
    _resolve_shell_init_files,
)


class TestReadTerminalShellConfig:
    """Unit tests for _read_terminal_shell_config()."""

    def test_returns_configured_shell(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"terminal": {"shell": "/bin/zsh"}},
        ):
            assert _read_terminal_shell_config() == "/bin/zsh"

    def test_returns_empty_when_not_set(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"terminal": {}},
        ):
            assert _read_terminal_shell_config() == ""

    def test_returns_empty_on_exception(self):
        with patch(
            "hermes_cli.config.load_config",
            side_effect=RuntimeError("broken"),
        ):
            assert _read_terminal_shell_config() == ""

    def test_returns_empty_when_none(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value=None,
        ):
            assert _read_terminal_shell_config() == ""


class TestResolveTerminalShell:
    """Unit tests for _resolve_terminal_shell()."""

    def test_config_takes_priority(self, tmp_path):
        fake_shell = tmp_path / "myshell"
        fake_shell.write_text("#!/bin/sh\n")
        fake_shell.chmod(0o755)

        with patch(
            "tools.environments.local._read_terminal_shell_config",
            return_value=str(fake_shell),
        ):
            result = _resolve_terminal_shell()
        assert result == str(fake_shell)

    def test_config_by_which_lookup(self, tmp_path, monkeypatch):
        """Config value found via shutil.which (e.g. 'fish' on PATH)."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fish = bin_dir / "fish"
        fish.write_text("#!/bin/sh\n")
        fish.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))

        with patch(
            "tools.environments.local._read_terminal_shell_config",
            return_value="fish",
        ):
            result = _resolve_terminal_shell()
        assert result == str(fish)

    def test_config_not_found_warns_and_falls_back(self, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/zsh")
        with patch(
            "tools.environments.local._read_terminal_shell_config",
            return_value="/nonexistent/shell",
        ):
            result = _resolve_terminal_shell()
        # Falls back to $SHELL
        assert result == "/bin/zsh"

    def test_auto_detect_from_shell_env(self, monkeypatch):
        monkeypatch.delenv("SHELL", raising=False)
        monkeypatch.setenv("SHELL", "/bin/zsh")

        with patch(
            "tools.environments.local._read_terminal_shell_config",
            return_value="",
        ):
            result = _resolve_terminal_shell()
        assert result == "/bin/zsh"

    def test_fallback_to_bash(self, monkeypatch):
        monkeypatch.delenv("SHELL", raising=False)

        with patch(
            "tools.environments.local._read_terminal_shell_config",
            return_value="",
        ), patch("shutil.which", return_value=None), patch(
            "os.path.isfile", return_value=False
        ):
            result = _resolve_terminal_shell()
        assert result == "/bin/sh"


class TestFindBashBackwardCompat:
    """_find_bash() should delegate to _resolve_terminal_shell()."""

    def test_delegates_to_resolve(self):
        with patch(
            "tools.environments.local._resolve_terminal_shell",
            return_value="/custom/shell",
        ):
            assert _find_bash() == "/custom/shell"


class TestResolveShellInitFilesZsh:
    """Auto-detection of zsh init files when terminal.shell is zsh."""

    def test_auto_sources_zshrc_when_zsh_configured(self, tmp_path, monkeypatch):
        zshrc = tmp_path / ".zshrc"
        zshrc.write_text("export MARKER=zsh\n")
        monkeypatch.setenv("HOME", str(tmp_path))

        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([], True),
        ), patch(
            "tools.environments.local._read_terminal_shell_config",
            return_value="/bin/zsh",
        ):
            resolved = _resolve_shell_init_files()

        assert str(zshrc) in resolved

    def test_auto_sources_zshenv_before_zshrc(self, tmp_path, monkeypatch):
        zshenv = tmp_path / ".zshenv"
        zshenv.write_text("export FROM_ZSHENV=1\n")
        zshrc = tmp_path / ".zshrc"
        zshrc.write_text("export FROM_ZSHRC=1\n")
        monkeypatch.setenv("HOME", str(tmp_path))

        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([], True),
        ), patch(
            "tools.environments.local._read_terminal_shell_config",
            return_value="/bin/zsh",
        ):
            resolved = _resolve_shell_init_files()

        assert resolved.index(str(zshenv)) < resolved.index(str(zshrc))

    def test_fish_skips_auto_init(self, tmp_path, monkeypatch):
        """fish handles its own config — no bash-style source needed."""
        monkeypatch.setenv("HOME", str(tmp_path))

        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([], True),
        ), patch(
            "tools.environments.local._read_terminal_shell_config",
            return_value="/opt/homebrew/bin/fish",
        ):
            resolved = _resolve_shell_init_files()

        assert resolved == []

    def test_explicit_list_overrides_zsh_auto(self, tmp_path, monkeypatch):
        """Explicit shell_init_files take priority over auto-detection."""
        custom = tmp_path / "custom_init"
        custom.write_text("# custom\n")
        monkeypatch.setenv("HOME", str(tmp_path))

        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([str(custom)], True),
        ), patch(
            "tools.environments.local._read_terminal_shell_config",
            return_value="/bin/zsh",
        ):
            resolved = _resolve_shell_init_files()

        assert resolved == [str(custom)]
