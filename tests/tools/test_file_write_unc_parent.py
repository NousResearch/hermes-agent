"""UNC (``\\\\server\\share``) parents in the atomic write path.

On Windows the shell-form of a UNC path keeps a ``//`` root (``_bash_safe_path`` has no
drive letter to translate). MSYS bash's own builtins (``[ -d ]``, ``open``, ``mktemp``,
``mv``) reach such a share, but GNU ``mkdir`` reads the leading ``//`` as its own
read-only POSIX root and dies with "Read-only file system" — and because the mkdir was
folded into a ``set -e`` script, that aborted the entire write even when the parent
directory already existed, so ``write_file`` could never reach ``\\\\wsl.localhost\\...``.

These tests pin both halves of the fix — mkdir only when the parent is missing, and
Python creating a missing UNC parent — using a fake backend, so they run anywhere.
"""

import os

import pytest

from tools.environments import local as local_env
from tools.file_operations import ExecuteResult, ShellFileOperations

UNC_TARGET = r"\\wsl.localhost\Ubuntu\home\tt\proj\f.py"
DRIVE_TARGET = r"C:\Users\tt\proj\f.py"
POSIX_TARGET = "/home/u/proj/f.py"

GUARD = '[ -d "$d" ] || mkdir -p "$d"; '


class _FakeEnv:
    """Local backend that records the script instead of running it."""

    is_local = True
    cwd = "."

    def __init__(self, is_local: bool = True):
        self.is_local = is_local
        self.commands = []

    def execute(self, command, cwd=None, **kwargs):  # noqa: D401 - mirrors env API
        self.commands.append(command)
        return {"output": "", "returncode": 0}


@pytest.fixture
def windows(monkeypatch):
    """Pretend we are on Windows, so the path translation under test is active."""
    monkeypatch.setattr(local_env, "_IS_WINDOWS", True)


@pytest.fixture
def makedirs_spy(monkeypatch):
    """Record os.makedirs calls instead of touching the filesystem."""
    calls = []

    def fake_makedirs(path, *args, **kwargs):
        calls.append((path, kwargs.get("exist_ok")))
        return None

    monkeypatch.setattr(os, "makedirs", fake_makedirs)
    return calls


def _ops(env):
    return ShellFileOperations(env, cwd=".")


class TestIsUncBashPath:
    def test_backslash_and_forward_slash_unc_forms_are_detected(self, windows):
        assert local_env._is_unc_bash_path(UNC_TARGET) is True
        assert local_env._is_unc_bash_path("//wsl.localhost/Ubuntu/home/tt/f.py") is True

    def test_drive_and_posix_paths_are_not_unc(self, windows):
        assert local_env._is_unc_bash_path(DRIVE_TARGET) is False
        assert local_env._is_unc_bash_path("/home/u/proj/f.py") is False
        assert local_env._is_unc_bash_path("") is False

    def test_never_unc_off_windows(self):
        # Off Windows _bash_safe_path is a no-op, so a // path is a plain POSIX path.
        assert local_env._is_unc_bash_path("//wsl.localhost/Ubuntu/home/tt/f.py") is False


class TestAtomicWriteParentHandling:
    def test_unc_parent_is_created_from_python(self, windows, makedirs_spy):
        env = _FakeEnv()
        result = _ops(env)._atomic_write(UNC_TARGET, "hello\n")

        assert result.exit_code == 0
        # The shell cannot mkdir a UNC parent, so Python must have made it.
        assert makedirs_spy == [(r"\\wsl.localhost\Ubuntu\home\tt\proj", True)]
        script = env.commands[0]
        assert GUARD in script, "mkdir must stay behind an existence guard"
        assert "d='//wsl.localhost/Ubuntu/home/tt/proj'" in script

    def test_missing_unc_parent_is_an_error_when_python_cannot_make_it(
        self, windows, monkeypatch
    ):
        def boom(path, *args, **kwargs):
            raise OSError(1, "Operation not permitted")

        monkeypatch.setattr(os, "makedirs", boom)
        env = _FakeEnv()
        result = _ops(env)._atomic_write(UNC_TARGET, "hello\n")

        assert result.exit_code != 0
        assert "Operation not permitted" in result.stdout
        assert env.commands == [], "a refused parent must not run the write script"

    def test_drive_path_keeps_the_shell_in_charge_of_mkdir(self, windows, makedirs_spy):
        env = _FakeEnv()
        result = _ops(env)._atomic_write(DRIVE_TARGET, "hello\n")

        assert result.exit_code == 0
        assert makedirs_spy == [], "normal drive paths must not gain a Python makedirs"
        assert GUARD in env.commands[0]

    def test_posix_path_is_unchanged(self, makedirs_spy):
        env = _FakeEnv()
        result = _ops(env)._atomic_write(POSIX_TARGET, "hello\n")

        assert result.exit_code == 0
        assert makedirs_spy == []
        script = env.commands[0]
        assert GUARD in script
        assert "d='/home/u/proj'" in script

    def test_remote_backend_never_touches_the_local_filesystem(self, windows, makedirs_spy):
        env = _FakeEnv(is_local=False)
        result = _ops(env)._atomic_write(UNC_TARGET, "hello\n")

        assert result.exit_code == 0
        assert makedirs_spy == [], "a remote backend's paths are not this process's to make"
        assert GUARD in env.commands[0]
