"""Tests for terminal shell selection (auto/bash/zsh).

Hermes terminal backend historically hardcodes ``_find_bash()`` for spawning
the command shell. On systems where the user's login shell is zsh (common on
macOS Catalina+ and many Linux setups), this forces bash to source zsh-syntax
files (``~/.zshrc``), which contain ``eval "$(mise activate zsh)"`` /
``pyenv init - zsh`` — these emit zsh-only syntax that crashes the bash
arithmetic parser, killing the shell with exit 1 and empty output.

The fix: add ``terminal.shell`` config (auto/bash/zsh). When ``auto``, the
terminal respects ``$SHELL`` and uses the user's actual login shell (if it's
a POSIX-sh-family shell). This lets zsh users inherit their full environment
without bash-to-zsh translation fragility.

Regression coverage for the "empty output, exit 1" failure mode reported when
shell_init_files included ~/.zshrc but the terminal spawned bash.
"""

import os
from unittest.mock import patch

import pytest


class TestResolveTerminalShell:
    """Pure-function unit tests for shell resolution."""

    def test_auto_returns_user_sHELL_when_sh_family(self, monkeypatch):
        """auto mode: $SHELL=/usr/bin/zsh → returns zsh path."""
        from tools.environments.local import _resolve_terminal_shell

        monkeypatch.setenv("SHELL", "/usr/bin/zsh")
        result = _resolve_terminal_shell("auto")
        assert result == "/usr/bin/zsh"

    def test_auto_returns_bash_when_sHELL_is_non_sh_family(self, monkeypatch):
        """auto mode: $SHELL=/usr/bin/fish → falls back to bash."""
        from tools.environments.local import _resolve_terminal_shell

        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        result = _resolve_terminal_shell("auto")
        # Must be a bash binary, NOT fish
        assert result.endswith("bash")
        assert "fish" not in result

    def test_auto_returns_bash_when_sHELL_unset(self, monkeypatch):
        """auto mode: $SHELL unset → falls back to bash."""
        from tools.environments.local import _resolve_terminal_shell

        monkeypatch.delenv("SHELL", raising=False)
        result = _resolve_terminal_shell("auto")
        assert result.endswith("bash")

    def test_explicit_bash_returns_bash(self, monkeypatch):
        """explicit 'bash' → returns bash binary."""
        from tools.environments.local import _resolve_terminal_shell

        monkeypatch.setenv("SHELL", "/usr/bin/zsh")
        result = _resolve_terminal_shell("bash")
        assert result.endswith("bash")
        assert "zsh" not in result

    def test_explicit_zsh_returns_zsh(self, monkeypatch):
        """explicit 'zsh' → returns zsh binary (if it exists)."""
        from tools.environments.local import _resolve_terminal_shell

        result = _resolve_terminal_shell("zsh")
        assert result.endswith("zsh")

    def test_explicit_zsh_falls_back_when_no_zsh(self, monkeypatch):
        """explicit 'zsh' but no zsh binary → falls back to bash."""
        from tools.environments.local import _resolve_terminal_shell

        with patch("os.path.isfile", return_value=False):
            result = _resolve_terminal_shell("zsh")
        assert result.endswith("bash")

    def test_invalid_value_falls_back_to_auto(self, monkeypatch):
        """invalid value → treats as auto."""
        from tools.environments.local import _resolve_terminal_shell

        monkeypatch.setenv("SHELL", "/bin/bash")
        result = _resolve_terminal_shell("powershell")
        assert result.endswith("bash")


class TestShellKindDetection:
    """Detect whether a resolved shell path is zsh-family or bash-family."""

    def test_zsh_path_detected_as_zsh(self):
        from tools.environments.local import _shell_kind

        assert _shell_kind("/usr/bin/zsh") == "zsh"

    def test_bash_path_detected_as_bash(self):
        from tools.environments.local import _shell_kind

        assert _shell_kind("/bin/bash") == "bash"

    def test_dash_path_detected_as_bash_family(self):
        """dash/sh → use bash semantics (snapshot bootstrap is bash-compatible)."""
        from tools.environments.local import _shell_kind

        assert _shell_kind("/bin/dash") == "bash"
        assert _shell_kind("/bin/sh") == "bash"

    def test_fish_path_detected_as_bash_family(self):
        """Non-sh-family → falls back to bash bootstrap (best effort)."""
        from tools.environments.local import _shell_kind

        assert _shell_kind("/usr/bin/fish") == "bash"
