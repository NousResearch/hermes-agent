"""The very next command after the terminal cwd is deleted must still RUN.

``LocalEnvironment._recover_cwd()`` exists so a command that ``rm -rf``'d its own
working directory does not wedge every later call (#17558). It repairs
``self.cwd`` so ``Popen(cwd=...)`` gets a real directory, but
``BaseEnvironment.execute()`` builds the wrapped bash script — which embeds
``builtin cd -- <cwd>`` — from ``self.cwd`` BEFORE ``_recover_cwd()`` runs inside
``_run_bash()``. The recovery therefore lands one step too late: the script still
``cd``s into the directory that no longer exists and exits 126, so the command the
user asked for never runs even though the log says the cwd was repaired.

The existing coverage in ``test_local_env_cwd_recovery.py`` patches ``Popen``
out, so it only ever asserted the ``cwd=`` argument — never that the command ran.
These tests drive a real bash so the script's own ``cd`` is the thing under test.
"""

import os
import shutil

import pytest

from tools.environments.local import LocalEnvironment


def _env(cwd: str) -> LocalEnvironment:
    """A LocalEnvironment without the init shell snapshot (not under test here)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LocalEnvironment, "init_session", lambda self, *a, **k: None)
        return LocalEnvironment(cwd=cwd, timeout=30)


@pytest.fixture
def deleted_cwd(tmp_path):
    """Yield a LocalEnvironment whose working directory was removed after init."""
    doomed = tmp_path / "doomed"
    doomed.mkdir()
    env = _env(str(doomed))
    try:
        # The previous tool call deleted the session's working directory.
        shutil.rmtree(doomed)
        assert env.cwd == str(doomed) and not os.path.isdir(env.cwd)
        yield env
    finally:
        try:
            env.cleanup()
        except Exception:
            pass


def test_command_runs_after_cwd_deletion(deleted_cwd):
    """The next command executes instead of dying on the wrapper's ``cd``."""
    result = deleted_cwd.execute("echo alive", timeout=30)

    # Recovery ran, so self.cwd is a directory that exists.
    assert os.path.isdir(deleted_cwd.cwd)
    # The command actually ran rather than being refused by the wrapper's cd.
    assert result["returncode"] == 0, result["output"]
    assert "alive" in result["output"]


def test_repeated_commands_after_recovery_keep_working(deleted_cwd):
    """Recovery is not one-shot: later commands run too, not just the first."""
    first = deleted_cwd.execute("echo one", timeout=30)
    second = deleted_cwd.execute("echo two", timeout=30)

    assert first["returncode"] == 0, first["output"]
    assert second["returncode"] == 0, second["output"]
    assert "one" in first["output"] and "two" in second["output"]


def test_no_cwd_recovery_when_cwd_is_intact(tmp_path):
    """The common path is untouched: a live cwd runs the command in place."""
    env = _env(str(tmp_path))
    try:
        result = env.execute("echo alive", timeout=30)

        assert result["returncode"] == 0, result["output"]
        assert "alive" in result["output"]
        assert env.cwd == str(tmp_path)
    finally:
        try:
            env.cleanup()
        except Exception:
            pass


def test_file_operations_command_runs_after_cwd_deletion(deleted_cwd):
    """The file tool's own exec path recovers too.

    ``ShellFileOperations._exec`` used to read ``env.cwd`` and pass it back as an
    explicit ``cwd=``, which outranks the recovery inside ``execute`` and left the
    file tool's commands dying on the wrapper's ``cd`` with exit 126.
    """
    from tools.file_operations import ShellFileOperations

    fops = ShellFileOperations(deleted_cwd)

    result = fops._exec("echo alive")

    assert result.cwd_error == ""
    assert result.exit_code == 0, result.stdout
    assert "alive" in result.stdout


def test_file_operations_honours_an_explicit_cwd(deleted_cwd, tmp_path):
    """An explicit caller cwd still wins — the recovery must not hijack it."""
    from tools.file_operations import ShellFileOperations

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    fops = ShellFileOperations(deleted_cwd)

    result = fops._exec("pwd", cwd=str(elsewhere))

    assert result.exit_code == 0, result.stdout
    # ``pwd`` prints one path (Git Bash may render it MSYS-style), and the wrapper can
    # append terminal control sequences after it, so compare the path component.
    printed = result.stdout.strip().splitlines()[0]
    assert printed.replace("\\", "/").endswith("/elsewhere"), result.stdout
