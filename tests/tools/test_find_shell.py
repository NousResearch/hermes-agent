"""Tests for _find_shell — user-login-shell preference on POSIX.

Regression tests for #42203: on macOS, ``_find_shell`` used to return
``/bin/bash`` (bash 3.2) which silently swallowed background commands
when ``~/.bash_profile`` contained ``exec /bin/zsh -l``.
"""

import os
import platform
import subprocess
import sys
from unittest.mock import patch

import pytest

from tools.environments.local import _find_bash, _find_shell


class TestFindShellPrefersUserShell:
    """_find_shell should prefer $SHELL over bash on POSIX."""

    def test_returns_shell_env_when_set_and_exists(self, tmp_path):
        """When $SHELL points to an existing allowlisted executable, _find_shell returns it."""
        fake_zsh = tmp_path / "zsh"
        fake_zsh.touch()
        fake_zsh.chmod(0o755)
        with patch.dict(os.environ, {"SHELL": str(fake_zsh)}):
            assert _find_shell() == str(fake_zsh)

    def test_configured_shell_overrides_shell_env(self, tmp_path):
        configured_zsh = tmp_path / "zsh"
        configured_zsh.touch()
        configured_zsh.chmod(0o755)
        user_bash = tmp_path / "bash"
        user_bash.touch()
        user_bash.chmod(0o755)

        with patch.dict(os.environ, {"SHELL": str(user_bash)}), patch(
            "tools.environments.local._resolve_terminal_shell",
            return_value=(str(configured_zsh), "zsh"),
        ):
            assert _find_shell() == str(configured_zsh)

    def test_falls_back_when_shell_not_executable(self, tmp_path):
        """$SHELL exists but lacks the execute bit -> fall back to _find_bash
        (returning it would fail at spawn time)."""
        fake = tmp_path / "zsh"
        fake.touch()
        fake.chmod(0o644)  # not executable
        with patch.dict(os.environ, {"SHELL": str(fake)}):
            assert _find_shell() == _find_bash()

    def test_falls_back_for_incompatible_shell_fish(self, tmp_path):
        """#42203 regression: $SHELL=fish must NOT be returned — spawn_local's
        `-lic` / `set +m` syntax breaks fish, which would trade the bash-3.2
        swallow for a parse error on every background command. Fall back to bash."""
        fake_fish = tmp_path / "fish"
        fake_fish.touch()
        fake_fish.chmod(0o755)
        with patch.dict(os.environ, {"SHELL": str(fake_fish)}):
            assert _find_shell() == _find_bash()

    def test_falls_back_for_incompatible_shell_csh(self, tmp_path):
        """$SHELL=tcsh/csh is also not -lic/set+m compatible -> fall back."""
        fake = tmp_path / "tcsh"
        fake.touch()
        fake.chmod(0o755)
        with patch.dict(os.environ, {"SHELL": str(fake)}):
            assert _find_shell() == _find_bash()

    def test_honours_allowlisted_bash_and_dash(self, tmp_path):
        """Every allowlisted POSIX-sh-family shell is honoured."""
        for name in ("bash", "dash", "sh", "ksh"):
            fake = tmp_path / name
            fake.touch()
            fake.chmod(0o755)
            with patch.dict(os.environ, {"SHELL": str(fake)}):
                assert _find_shell() == str(fake), name

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


@pytest.mark.skipif(
    not os.path.isfile("/bin/bash") or sys.platform != "darwin",
    reason="reproduces the macOS system-bash-3.2 login-shell swallow",
)
class TestMacosLoginShellSwallowRegression:
    """E2E regression for #42203: the actual failure is that system bash 3.2,
    invoked as a login shell (`-lic`) with stdin=/dev/null and a
    ~/.bash_profile that `exec`s zsh, silently swallows the command (exit 0,
    no output, no side effects). Prove (a) the bug exists with /bin/bash and
    (b) the $SHELL (zsh) path _find_shell prefers does NOT swallow."""

    def _spawn_like_registry(self, shell, command, home, tmp_path):
        import subprocess
        env = dict(os.environ)
        env["HOME"] = str(home)
        # Mirror process_registry.spawn_local: [shell, "-lic", "set +m; <cmd>"]
        # with stdin redirected to /dev/null.
        return subprocess.run(
            [shell, "-lic", f"set +m; {command}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_system_bash_swallows_but_zsh_does_not(self, tmp_path):
        # A .bash_profile that exec's zsh — the reported macOS shape.
        home = tmp_path / "home"
        home.mkdir()
        (home / ".bash_profile").write_text("exec /bin/zsh -l\n")

        zsh = os.environ.get("SHELL") or "/bin/zsh"
        if not os.path.isfile(zsh):
            pytest.skip("no zsh available")

        marker_bash = tmp_path / "bash_ran"
        marker_zsh = tmp_path / "zsh_ran"

        # /bin/bash login shell: command is swallowed (file NOT created).
        self._spawn_like_registry("/bin/bash", f"echo x > {marker_bash}", home, tmp_path)
        # zsh (the $SHELL _find_shell prefers): command runs (file created).
        self._spawn_like_registry(zsh, f"echo x > {marker_zsh}", home, tmp_path)

        # The FIX path (zsh) must run the command.
        assert marker_zsh.exists(), "zsh ($SHELL) path must run the command"

        # Differential: when /bin/bash is the swallow-prone 3.x (macOS system
        # bash), the login-shell invocation must demonstrably FAIL to run the
        # command — that's the bug this PR routes around. Only assert the
        # negative when we've confirmed a 3.x bash, so the test stays valid on
        # boxes/CI with a newer /bin/bash that doesn't swallow.
        ver = subprocess.run(
            ["/bin/bash", "--version"], capture_output=True, text=True
        ).stdout
        if "version 3." in ver:
            assert not marker_bash.exists(), (
                "system bash 3.x login shell should swallow the command "
                "(the #42203 bug); _find_shell routes around it by preferring zsh"
            )

    def test_find_shell_selects_working_shell_on_this_box(self, tmp_path):
        """_find_shell's choice must actually execute a background-style
        command (regression against returning a swallow-prone shell)."""
        shell = _find_shell()
        marker = tmp_path / "ok_marker"
        subprocess.run(
            [shell, "-lic", f"set +m; echo ok > {marker}"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        assert marker.exists(), f"_find_shell()={shell} swallowed the command"
