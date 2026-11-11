"""An internal probe must not rewind the shell session state a user command just wrote.

Every ``env.execute`` ends by re-dumping the shell's exports over the shared session
snapshot and by setting ``env.cwd`` from its cwd marker. The env poller and the remote
code kernel poll the sandbox through that same ``execute``. A probe that sourced the
snapshot before a foreground command finished and ends after it publishes its stale
copy, so the user's ``export`` (or ``source venv/bin/activate``) vanishes from the next
command, and its cwd marker replaces the directory the user ``cd``-ed into.
"""

from __future__ import annotations

import os
import shlex
import shutil
import threading

import pytest

from tests.waiting import wait_until
from tools.code_kernel_remote import _sh
from tools.environments.local import LocalEnvironment


def _open_writer(fifo: str):
    """Writer fd once the probe holds the read end open, else None."""
    try:
        return os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return None


@pytest.mark.platforms("linux", "macos")
@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_a_probe_spanning_a_foreground_command_keeps_its_exports_and_cwd(tmp_path):
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    result_file = tmp_path / "cell_res_000001.json"
    os.mkfifo(result_file)
    env = LocalEnvironment(cwd=str(work), timeout=30)
    try:
        # The remote kernel's result poll, held open until the test releases it.
        probe = threading.Thread(
            target=_sh, args=(env, f"cat {shlex.quote(str(result_file))} 2>/dev/null"), kwargs={"timeout": 30})
        probe.start()
        writer = wait_until(lambda: _open_writer(str(result_file)), what="probe reading the result file")

        user = env.execute("cd sub && export HERMES_PROBE_EXPORT=kept", timeout=30)
        assert user["returncode"] == 0, user
        os.close(writer)
        probe.join(timeout=30)
        assert not probe.is_alive()

        assert env.cwd == str((work / "sub").resolve())
        assert env.execute("echo ${HERMES_PROBE_EXPORT:-lost}", timeout=30)["output"].strip() == "kept"
    finally:
        env.cleanup()
