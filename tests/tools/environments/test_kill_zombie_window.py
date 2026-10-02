"""Zombie-window race in ``_kill_process_group_posix`` (search_files Errno 3).

A child that hits its output bound (rg at the fetch limit) can exit between the
caller's ``proc.poll()`` and the kill helper's ``os.getpgid`` — poll() still says
alive, getpgid already raises ProcessLookupError (Linux #116855, Darwin too).
The helper used to ``raise`` when no ``_hermes_pgid`` fallback was set, so the
raw ``OSError [Errno 3] No such process`` escaped as an opaque search_files
error on small/fast trees. It must return cleanly instead: the group is dying
on its own and the caller owns its drained output.
"""

import os
import subprocess
import sys
import time

import pytest

from tools.environments.local import _kill_process_group_posix

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only helper (caller gates on _IS_WINDOWS)")


def _free_pid() -> int:
    """A PID that is not alive right now (not a zombie either)."""
    pid = os.getpid() + 977
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        pid += 1


class _DeadButPolling:
    """Looks alive to poll(); the kernel disagrees — the exact race window."""

    def __init__(self, pid: int, pgid: int | None = None):
        self.pid = pid
        self.poll = lambda: None  # never-exited lie, like a fresh Popen
        if pgid is not None:
            self._hermes_pgid = pgid


def test_dead_pid_without_pgid_fallback_returns_cleanly():
    """The regression: no raise (which surfaced as '[Errno 3] No such process')."""
    with pytest.raises(ProcessLookupError):
        os.getpgid(_free_pid())  # sanity: the pid really is gone
    assert _kill_process_group_posix(_DeadButPolling(_free_pid())) is None


def test_dead_pid_with_pgid_fallback_still_kills_live_group():
    """The _hermes_pgid shim fallback keeps working through the patched path."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    try:
        time.sleep(0.05)  # let the child settle into its own group
        live_pgid = os.getpgid(proc.pid)
        assert _kill_process_group_posix(_DeadButPolling(_free_pid(), pgid=live_pgid)) is None
        deadline = time.monotonic() + 3.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc.poll() is not None, "group via _hermes_pgid fallback must be killed"
    finally:
        proc.kill()
        proc.wait()


def test_waitless_poll_only_shim_reaches_killpg_without_attributeerror():
    """Review catch (#131316): shims without ``.wait`` (poll-only fakes) died with
    AttributeError on the post-KILL ``proc.wait(timeout=0.2)`` convenience call —
    the helper now guards it. The killpg path itself must still work."""
    proc = subprocess.Popen(
        ["sleep", "5"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(0.1)

    class WaitLess:
        """poll-only + kill: exactly the shim shape the reviewer's fake had."""

        def __init__(self, p):
            self.pid = p.pid
            self._p = p

        def poll(self):
            return self._p.poll()

        def kill(self):
            self._p.kill()

    shim = WaitLess(proc)
    result = _kill_process_group_posix(shim)
    deadline = time.monotonic() + 3.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert result is None
        assert proc.poll() is not None, "killpg must terminate a live group"
    finally:
        proc.kill()
        proc.wait()