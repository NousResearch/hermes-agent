"""rg can exit between the native runner's poll() and its group kill.

macOS answers getpgid() on an exited, unreaped child with ESRCH, so the kill helper must use
the pgid recorded at spawn and must not raise when none was recorded.
"""

import os
import subprocess

import pytest

from tools.environments.local import LocalEnvironment, _kill_process_group_posix, _record_process_group
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.platforms("posix")


def _wait_until_exited_unreaped(pid):
    os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)


def test_kill_helper_tolerates_an_exited_unreaped_child_without_a_recorded_pgid():
    proc = subprocess.Popen(["true"], start_new_session=True)
    try:
        _wait_until_exited_unreaped(proc.pid)
        _kill_process_group_posix(proc)
    finally:
        proc.wait()


def test_spawn_records_the_group_and_its_start_fingerprint():
    proc = subprocess.Popen(["sleep", "5"], start_new_session=True)
    try:
        _record_process_group(proc)
        assert proc._hermes_pgid == proc.pid  # a new session leader leads its own group
        assert proc._hermes_pgid_start is not None
    finally:
        proc.kill()
        proc.wait()


def test_native_rg_runner_survives_rg_exiting_after_poll(tmp_path, monkeypatch):
    real_popen = subprocess.Popen

    class ExitsAfterPoll(real_popen):
        """poll() reports "running" once, after the child has already exited (unreaped)."""
        lied = False

        def poll(self):
            if not self.lied:
                self.lied = True
                _wait_until_exited_unreaped(self.pid)
                return None
            return super().poll()

    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    monkeypatch.setattr(subprocess, "Popen", ExitsAfterPoll)
    result = ops._run_rg_native(["sh", "-c", "'echo needle'"], 5, timeout=10)
    assert result.stdout == "needle\n"
    assert result.exit_code == 0
