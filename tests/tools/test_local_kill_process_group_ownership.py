"""Process-group cleanup must never signal a group the child does not lead.

Regression for #97296 (teardown half). ``_kill_process_group_posix`` looked up the
child's PGID and ``killpg``'d it. A child that did not get its own session (a
``sitecustomize`` that drops ``start_new_session`` to keep macOS on posix_spawn, or any
spawner that skips ``setsid``) shares the CALLER's group, so every terminal timeout and
every bounded ``search_files`` stop SIGTERM'd the caller's whole group — on Desktop the
Electron main, its renderer and the Python backend at once. When the group cannot be
looked up at all (macOS ``getpgid()`` fails with ESRCH for an exited, unreaped child —
the bounded ``search_files`` stop racing rg's own exit) or no longer be signalled
(macOS ``killpg()`` of a zombie-only group fails with EPERM), the kill must not raise.

Real processes, no mocks: the helper runs inside a throwaway session, so a regression can
only take that harness down, never the test runner.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_HARNESS = textwrap.dedent("""
    import json, os, subprocess, sys, time
    import psutil
    try:
        os.setsid()
    except PermissionError:
        pass  # already a session leader (the runner honoured start_new_session)
    if os.getpgrp() != os.getpid():
        sys.exit(3)  # never run the kill outside a throwaway group
    from tools.environments.local import _kill_process_group_posix

    def gone(pid, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.05)
        return False

    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    bystander = subprocess.Popen(sleep, **quiet)  # an unrelated member of the caller's group
    try:
        # The child shares the caller's group (no new session) and forks a grandchild.
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess, sys; g = subprocess.Popen(sys.argv[1:], stdout=subprocess.DEVNULL); "
             "print(g.pid, flush=True); g.wait()", *sleep],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        grandchild = int(child.stdout.readline())
        _kill_process_group_posix(child)
        exited = subprocess.Popen([sys.executable, "-c", "pass"], **quiet)
        deadline = time.monotonic() + 30  # wait until it has exited but is not reaped yet
        while (psutil.Process(exited.pid).status() != psutil.STATUS_ZOMBIE
               and time.monotonic() < deadline):
            time.sleep(0.01)
        _kill_process_group_posix(exited)
        # An exited group leader whose PGID was cached at spawn (as _run_bash does): the
        # zombie-only group rejects killpg with EPERM on macOS.
        leader = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True, **quiet)
        leader._hermes_pgid = leader.pid
        while (psutil.Process(leader.pid).status() != psutil.STATUS_ZOMBIE
               and time.monotonic() < deadline):
            time.sleep(0.01)
        _kill_process_group_posix(leader)
        report = {"child_reaped": child.poll() is not None, "grandchild_gone": gone(grandchild),
                  "exited_child_reaped": exited.returncode is not None,
                  "exited_leader_reaped": leader.wait(timeout=10) is not None,
                  "bystander_alive": bystander.poll() is None}
    finally:
        bystander.kill()
        bystander.wait()
    print(json.dumps(report))
""")


@pytest.mark.live_system_guard_bypass
def test_kill_never_signals_a_group_the_child_does_not_lead():
    harness = subprocess.run(
        [sys.executable, "-c", _HARNESS], cwd=_REPO_ROOT, capture_output=True, text=True,
        timeout=120, start_new_session=True)
    assert harness.returncode == 0, (
        f"harness exited {harness.returncode}: the cleanup signalled its caller's own "
        f"process group, or raised\n{harness.stderr[-2000:]}")
    assert json.loads(harness.stdout.splitlines()[-1]) == {
        "child_reaped": True, "grandchild_gone": True, "exited_child_reaped": True,
        "exited_leader_reaped": True, "bystander_alive": True}
