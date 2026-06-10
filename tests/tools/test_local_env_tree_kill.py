"""Process-tree kill coverage for ``LocalEnvironment._kill_process``.

Bug: on Windows ``_kill_process`` used a bare ``proc.terminate()``, which
kills only the direct child (the shell wrapper). Grandchildren survived as
orphans whenever a terminal command hit its timeout — observed in production
as ``hermes.exe``/``python.exe`` pairs burning a CPU core each for hours
after a 5s tool timeout. ``_kill_process_tree`` closes that gap; these tests
run on every platform because the psutil sweep is platform-neutral.
"""

import subprocess
import sys
import time

import psutil
import pytest

from tools.environments import local as local_mod
from tools.environments.local import _kill_process_tree

# Parent script: spawn a long-sleeping grandchild, report its pid, then hang
# around long enough for the test to do its work.
_PARENT_SCRIPT = (
    "import subprocess, sys, time\n"
    'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])\n'
    "print(child.pid, flush=True)\n"
    "time.sleep(120)\n"
)


def _spawn_tree(tmp_path):
    """Start parent + grandchild; return (parent_proc, grandchild_pid)."""
    script = tmp_path / "parent.py"
    script.write_text(_PARENT_SCRIPT)
    parent = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    line = parent.stdout.readline().strip()
    if not line:
        parent.kill()
        pytest.fail("test setup: parent never reported the grandchild pid")
    return parent, int(line)


def _force_cleanup(*pids):
    for pid in pids:
        try:
            psutil.Process(pid).kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _wait_for_dead(pid, timeout=5.0):
    """True once the pid is confirmed gone (allows for reaping lag)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(0.05)
    return not psutil.pid_exists(pid)


def test_kill_process_tree_kills_grandchildren(tmp_path):
    parent, grandchild_pid = _spawn_tree(tmp_path)
    try:
        assert psutil.pid_exists(parent.pid), "sanity: parent should be alive"
        assert psutil.pid_exists(grandchild_pid), "sanity: grandchild should be alive"

        assert _kill_process_tree(parent.pid) is True

        assert _wait_for_dead(parent.pid), "parent survived the tree kill"
        assert _wait_for_dead(grandchild_pid), "grandchild survived the tree kill"
    finally:
        _force_cleanup(parent.pid, grandchild_pid)
        parent.stdout.close()
        try:
            parent.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError):
            pass


def test_kill_process_tree_on_missing_pid_returns_true():
    # Harvest a pid that is known to be free: spawn a trivial process, wait
    # for it to exit, and confirm the pid hasn't been reused. PID reuse in
    # this window is rare on every platform; retry a few times if it happens.
    for _ in range(10):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        if not psutil.pid_exists(proc.pid):
            assert _kill_process_tree(proc.pid) is True
            return
    pytest.fail("test setup: could not obtain an unused pid in 10 attempts")


def test_windows_branch_uses_tree_kill(monkeypatch, tmp_path):
    """Route _kill_process through the Windows branch on every platform.

    The branch only depends on module globals and the proc handle, so
    forcing ``_IS_WINDOWS`` exercises the real Windows code path (helper
    plus wrapper reap) without a Windows host.
    """
    parent, grandchild_pid = _spawn_tree(tmp_path)
    try:
        assert psutil.pid_exists(grandchild_pid), "sanity: grandchild should be alive"

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        env = object.__new__(local_mod.LocalEnvironment)
        env._kill_process(parent)

        assert _wait_for_dead(parent.pid), "parent survived _kill_process"
        assert _wait_for_dead(grandchild_pid), (
            "grandchild survived _kill_process — Windows branch is not "
            "killing the process tree"
        )
    finally:
        _force_cleanup(parent.pid, grandchild_pid)
        parent.stdout.close()
        try:
            parent.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError):
            pass
