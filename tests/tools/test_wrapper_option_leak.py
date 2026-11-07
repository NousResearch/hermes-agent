"""A command that flips shell options (``set -e``) must not leak into the wrapper epilogue (#144060).

``_wrap_command_script`` runs the user command through ``eval`` in the wrapper's own shell.
``_atomic_write`` starts its script with ``set -e``, so after a successful write the wrapper's
epilogue (env re-dump, CWD marker, ``exit $__hermes_ec``) runs with errexit still on. Under the
``bash -l`` fallback (``init_session`` failed) a login shell then sources ``~/.bash_logout`` on
exit, and with errexit active a failing logout script (stock Ubuntu ``clear_console``) replaces
the command's status 0 with 1 — write_file/patch then report ``Failed to write file: `` for a
write that succeeded. Saving and restoring the shell options around the eval keeps the epilogue
running with the wrapper's own options.
"""

import os
import subprocess
import sys

import pytest

from tools.environments.base_session_env import _wrap_command_script


def _wrapped(command: str) -> str:
    return _wrap_command_script(
        command,
        quoted_cwd="'.'",
        quoted_snap="'/tmp/hermes-snap'",
        snap_tmp_template="'/tmp/hermes-snap.XXXXXX'",
        passthrough_names=(),
        snapshot_ready=False,
        cwd_marker="__HERMES_CWD__",
    )


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("command", ["true", "set -e; true"])
def test_login_shell_logout_failure_does_not_mask_exit_zero(tmp_path, command):
    """``bash -l`` with a failing ~/.bash_logout must not turn the command's 0 into 1."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_logout").write_text("false\n")  # clear_console-class failure
    env = {**os.environ, "HOME": str(home)}
    proc = subprocess.run(
        ["bash", "-l", "-c", _wrapped(command)],
        capture_output=True, text=True, env=env, cwd=str(home), timeout=60, check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
