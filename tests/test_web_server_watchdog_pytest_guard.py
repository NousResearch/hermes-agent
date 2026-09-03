"""The parent-death watchdog must never arm inside a test process.

``_start_parent_death_watchdog`` spawns a thread whose terminal action is
``os._exit(0)`` -- an unconditional, unwindable process kill.  Inside pytest
that kills the *runner*: the session stops mid-file, ``pytest_sessionfinish``
never fires, no summary line and no ``--junit-xml`` are written, and the
shell still observes exit status 0.  A truncated run is indistinguishable
from a green one unless you notice the missing summary.

The trigger is ordinary inheritance: a test process launched from the Hermes
desktop app inherits ``HERMES_PARENT_PID`` (plus marker/nonce) pointing at
the desktop.  Any test that reaches ``start_server()`` then arms a watchdog
aimed at a parent that has nothing to do with the test run.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

from hermes_cli import web_server

REPO_ROOT = Path(__file__).resolve().parent.parent


def _watchdog_threads():
    return [t for t in threading.enumerate() if t.name == "serve-parent-watchdog"]


class TestWatchdogNeverArmsUnderPytest:
    def test_no_thread_started_with_full_parent_identity(self):
        """Marker+nonce present and valid -- still must not arm under pytest."""
        before = len(_watchdog_threads())
        env = {
            "HERMES_PARENT_PID": str(os.getpid()),
            "HERMES_PARENT_START_MARKER": "123456",
            "HERMES_PARENT_NONCE": "abc123",
            "PYTEST_CURRENT_TEST": "test_x (call)",
        }
        with patch.dict(os.environ, env, clear=False):
            web_server._start_parent_death_watchdog()
        assert len(_watchdog_threads()) == before

    def test_no_thread_started_with_legacy_pid_only(self):
        """The PID-only legacy path is guarded too."""
        before = len(_watchdog_threads())
        env = {
            "HERMES_PARENT_PID": str(os.getpid()),
            "PYTEST_CURRENT_TEST": "test_x (call)",
        }
        with patch.dict(os.environ, env, clear=False):
            for stale in ("HERMES_PARENT_START_MARKER", "HERMES_PARENT_NONCE"):
                os.environ.pop(stale, None)
            web_server._start_parent_death_watchdog()
        assert len(_watchdog_threads()) == before

    def test_os_exit_is_never_reached(self):
        """The guard returns before any thread that could _exit is created."""
        env = {
            "HERMES_PARENT_PID": str(os.getpid()),
            "PYTEST_CURRENT_TEST": "test_x (call)",
        }
        with patch.dict(os.environ, env, clear=False), \
             patch.object(os, "_exit") as fake_exit, \
             patch.object(threading, "Thread") as fake_thread:
            web_server._start_parent_death_watchdog()
            assert not fake_thread.called, "watchdog thread must not be constructed"
            assert not fake_exit.called


class TestGuardDoesNotLeakIntoChildProcesses:
    """The guard must not disarm production processes spawned BY a test.

    ``PYTEST_CURRENT_TEST`` is an environment variable, so it is inherited by
    every subprocess a test starts -- including a production-shaped
    ``hermes serve`` whose parent-death watchdog is the very thing under test
    (``tests/hermes_cli/test_serve_port_in_use.py`` spawns exactly this, and
    scrubbing the ``HERMES_PARENT_*`` family does not remove the pytest var).
    An env-only guard would therefore weaken the lifecycle contract in the
    child while claiming to only affect tests.  Pairing it with an
    in-interpreter ``sys.modules`` check keeps the guard scoped to the runner.
    """

    def test_pytest_env_var_is_inherited_by_children(self):
        """Baseline: the env half alone cannot distinguish runner from child."""
        proc = subprocess.run(
            [sys.executable, "-c",
             "import os; print('PYTEST_CURRENT_TEST' in os.environ)"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.stdout.strip() == "True", (
            "expected the pytest env var to leak into the child; if this ever "
            "fails the guard's second condition may no longer be needed"
        )

    def test_child_process_still_arms_the_watchdog(self):
        """A serve-shaped child spawned from pytest must NOT be disarmed.

        Bites the env-only guard: that version returns early here (the var is
        inherited) and the watchdog never arms, so the assertion fails.
        """
        code = (
            "import os, sys, threading\n"
            "from hermes_cli import web_server\n"
            "os.environ['HERMES_PARENT_PID'] = str(os.getppid())\n"
            "os.environ.pop('HERMES_PARENT_START_MARKER', None)\n"
            "os.environ.pop('HERMES_PARENT_NONCE', None)\n"
            "assert 'PYTEST_CURRENT_TEST' in os.environ, 'env var did not inherit'\n"
            "assert 'pytest' not in sys.modules, 'child must not have pytest loaded'\n"
            "web_server._start_parent_death_watchdog()\n"
            "armed = [t for t in threading.enumerate()\n"
            "         if t.name == 'serve-parent-watchdog']\n"
            # Distinct tokens on purpose: 'ARMED' is a substring of 'DISARMED',
            # so a naive membership check would pass in BOTH directions and the
            # test would never bite.
            "print('WATCHDOG_RESULT=' + ('armed' if armed else 'not-armed'))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, f"child failed: {proc.stdout}\n{proc.stderr}"
        assert "WATCHDOG_RESULT=armed" in proc.stdout, (
            "the parent-death watchdog must still arm in a production-shaped "
            "child spawned from a test -- the guard must scope to the pytest "
            f"runner, not to anything that merely inherits its environment. "
            f"stdout={proc.stdout!r} stderr={proc.stderr[-2000:]!r}"
        )
