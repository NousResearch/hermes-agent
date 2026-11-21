"""The execute_code kernel's PATH carries the runtimes Hermes manages for itself.

``tools/environments/local.py`` gives the terminal subshell a completed PATH
(``_append_missing_sane_path_entries``): the pm store's dirs, ``$HERMES_HOME/bin``
and ``~/.local/bin``. The kernel child went without it, so on an install whose
only ``uv`` is the managed one, ``subprocess.run(["uv", ...])`` inside a cell
raised ``FileNotFoundError: 'uv'`` while the very same command worked from the
terminal tool. ``_build_child_env`` now applies the same completion, appended so
the user's own PATH entries keep precedence.
"""

import os
import sys

import pytest

from tools.code_execution_env import _build_child_env

pytestmark = pytest.mark.platforms("posix")  # PATH completion is POSIX-only (Windows is a no-op)


def _child_env():
    return _build_child_env(
        rpc_endpoint="sock",
        rpc_token="tok",
        tmpdir="/tmp/hermes-test",
        child_python=sys.executable,
    )


def test_managed_hermes_home_bin_reaches_the_kernel_path(monkeypatch, tmp_path):
    """``$HERMES_HOME/bin`` holds the managed uv; a cell must be able to resolve it."""
    home = tmp_path / "home"
    managed_bin = home / "bin"
    managed_bin.mkdir(parents=True)
    # The regression's shape: a PATH with neither the managed bin nor the sane defaults.
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    entries = _child_env()["PATH"].split(os.pathsep)

    assert str(managed_bin) in entries
    # Appended, not prepended: the inherited order survives and the dir lands last.
    assert entries[:2] == ["/usr/bin", "/bin"]
    assert entries.index(str(managed_bin)) > 1


def test_kernel_path_keeps_user_precedence_and_stays_duplicate_free(monkeypatch, tmp_path):
    """A managed dir already on PATH keeps its position instead of being appended again."""
    home = tmp_path / "home"
    managed_bin = home / "bin"
    managed_bin.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", f"{managed_bin}:/usr/bin:/bin")

    entries = _child_env()["PATH"].split(os.pathsep)

    assert entries[0] == str(managed_bin), "user precedence must not be overridden"
    assert entries.count(str(managed_bin)) == 1
