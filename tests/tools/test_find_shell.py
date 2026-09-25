"""Tests for _find_shell — user-login-shell preference on POSIX.

Regression tests for #42203: on macOS, ``_find_shell`` used to return
``/bin/bash`` (bash 3.2) which silently swallowed background commands
when ``~/.bash_profile`` contained ``exec /bin/zsh -l``.
"""

import os
import shutil
import subprocess
import time
from unittest.mock import patch

import pytest

from tools.environments.local import _find_bash, _find_shell


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    except ImportError:
        try:
            os.kill(pid, 0)  # windows-footgun: ok — psutil fallback only on POSIX hosts without it
        except OSError:
            return False
        return True


class TestFindShellPrefersUserShell:
    """_find_shell should prefer $SHELL over bash on POSIX."""

    @pytest.mark.platforms("linux")
    def test_returns_shell_env_when_set_and_exists(self, tmp_path):
        """When $SHELL points to an existing allowlisted executable, _find_shell returns it."""
        fake_zsh = tmp_path / "zsh"
        fake_zsh.touch()
        fake_zsh.chmod(0o755)
        with patch.dict(os.environ, {"SHELL": str(fake_zsh)}):
            assert _find_shell() == str(fake_zsh)

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


    @pytest.mark.platforms("linux")
    def test_honours_allowlisted_bash_and_dash(self, tmp_path):
        """Every allowlisted POSIX-sh-family shell is honoured."""
        for name in ("bash", "dash", "sh", "ksh"):
            fake = tmp_path / name
            fake.touch()
            fake.chmod(0o755)
            with patch.dict(os.environ, {"SHELL": str(fake)}):
                assert _find_shell() == str(fake), name


    def test_falls_back_to_find_bash_when_shell_empty(self):
        """When $SHELL is empty string, _find_shell delegates."""
        with patch.dict(os.environ, {"SHELL": ""}):
            assert _find_shell() == _find_bash()


class TestFindShellWindowsBehavior:
    """On Windows, _find_shell always delegates to _find_bash."""

    @pytest.mark.platforms("windows")
    def test_windows_ignores_shell_env(self):
        """On Windows, $SHELL is ignored — _find_shell delegates to _find_bash.

        Windows-only: faking ``_IS_WINDOWS`` selected the branch but left
        ``_find_bash`` resolving a POSIX bash, so the equality proved nothing
        about Git-Bash resolution on the real host.
        """
        # Even if SHELL is set, it should be ignored on Windows
        with patch.dict(os.environ, {"SHELL": "/usr/bin/zsh"}):
            result = _find_shell()
            assert result == _find_bash()


class TestFindBashCollapsedToPmShell:
    """_find_bash is now a thin wrapper over pm.shell(); the Windows
    candidate ladder (HERMES_GIT_BASH_PATH → %LOCALAPPDATA%\\hermes\\git →
    Program Files) and the ASLR diagnostic were deleted — the store is the
    authority on bundled bash."""

    @pytest.mark.platforms("windows")
    def test_delegates_to_pm_shell(self, monkeypatch):
        """_find_bash returns whatever pm.shell() resolves (store bash or
        provisioned PATH)."""
        monkeypatch.setattr(
            "pm.shell.bash", lambda: r"C:\store\tools\git-x\usr\bin\bash.exe"
        )
        assert _find_bash() == r"C:\store\tools\git-x\usr\bin\bash.exe"

    def test_raises_when_pm_shell_finds_nothing(self, monkeypatch):
        """A store with no bash (and no PATH bash) surfaces a clear error
        pointing at `hermes pm install` instead of hunting locations."""
        monkeypatch.setattr("pm.shell.bash", lambda: None)
        with pytest.raises(RuntimeError) as exc_info:
            _find_bash()
        assert "No shell found" in str(exc_info.value)
        assert "hermes pm install" in str(exc_info.value)



@pytest.mark.platforms("macos")
@pytest.mark.skipif(
    not os.path.isfile("/bin/bash"),
    reason="reproduces the macOS system-bash-3.2 login-shell swallow",
)
class TestMacosLoginShellSwallowRegression:
    """E2E regression for #42203: the actual failure is that system bash 3.2,
    invoked as a login shell (`-lic`) with stdin=/dev/null and a
    ~/.bash_profile that `exec`s zsh, silently swallows the command (exit 0,
    no output, no side effects). Prove (a) the bug exists with /bin/bash and
    (b) the zsh path _find_shell prefers does NOT swallow."""

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
