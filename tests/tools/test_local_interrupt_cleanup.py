"""Regression tests for _wait_for_process subprocess cleanup on exception exit.

When the poll loop exits via KeyboardInterrupt or SystemExit (SIGTERM via
cli.py signal handler, SIGINT on the main thread in non-interactive -q mode,
or explicit sys.exit from some caller), the child subprocess must be killed
before the exception propagates — otherwise the local backend's use of
os.setsid leaves an orphan with PPID=1.

The live repro that motivated this: hermes chat -q ... 'sleep 300', SIGTERM
to the python process, sleep 300 survived with PPID=1 for the full 300 s
because _wait_for_process never got to call _kill_process before python
died.  See commit message for full context.
"""
import os
import signal
import subprocess
import threading
import time

import pytest

from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "logs").mkdir(exist_ok=True)


def _pgid_still_alive(pgid: int) -> bool:
    """Return True if any non-zombie process in the group is still alive."""
    try:
        ps = subprocess.run(
            ["ps", "-axo", "pgid=,stat=,pid=,command="],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in ps.stdout.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) < 2:
                continue
            try:
                line_pgid = int(parts[0])
            except ValueError:
                continue
            if line_pgid != pgid:
                continue
            stat = parts[1]
            if not stat.startswith("Z"):
                return True
        return False
    except Exception:
        try:
            os.killpg(pgid, 0)  # signal 0 = existence check
            return True
        except ProcessLookupError:
            return False


def test_wait_for_process_kills_subprocess_on_keyboardinterrupt():
    """When KeyboardInterrupt arrives mid-poll, the subprocess group must be
    killed before the exception is re-raised."""
    env = LocalEnvironment(cwd="/tmp")
    try:
        result_holder = {}
        proc_holder = {}
        started = threading.Event()
        raise_at = [None]  # set by the main thread to tell worker when

        # Record the actual process handle instead of depending on ps/psutil
        # command-line formatting, which differs across macOS/Linux shells.
        original_run_bash = env._run_bash

        def recording_run_bash(cmd_string, *args, **kwargs):
            proc = original_run_bash(cmd_string, *args, **kwargs)
            if "sleep 30" in cmd_string:
                proc_holder["proc"] = proc
                started.set()
            return proc

        env._run_bash = recording_run_bash

        # Drive execute() on a separate thread so we can SIGNAL-interrupt it
        # via a thread-targeted exception without killing our test process.
        def worker():
            # Spawn a subprocess that will definitely be alive long enough
            # to observe the cleanup, via env.execute(...) — the normal path
            # that goes through _wait_for_process.
            try:
                result_holder["result"] = env.execute("sleep 30", timeout=60)
            except BaseException as e:  # noqa: BLE001 — we want to observe it
                result_holder["exception"] = type(e).__name__

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        assert started.wait(timeout=5.0), (
            "test setup: command subprocess was not spawned after 5 s"
        )
        target_proc = proc_holder.get("proc")
        assert target_proc is not None
        pgid = os.getpgid(target_proc.pid)
        assert _pgid_still_alive(pgid), "sanity: subprocess should be alive"

        # Now inject a KeyboardInterrupt into the worker thread the same
        # way CPython's signal machinery would.  We use ctypes.PyThreadState_SetAsyncExc
        # which is how signal delivery to non-main threads is simulated.
        import ctypes
        import sys as _sys
        # py-thread-state exception targets need the ident, not the Thread
        tid = t.ident
        assert tid is not None
        # Fire KeyboardInterrupt into the worker thread
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), ctypes.py_object(KeyboardInterrupt),
        )
        assert ret == 1, f"SetAsyncExc returned {ret}, expected 1"

        # Give the worker a moment to: hit the exception at the next poll,
        # run the except-block cleanup (_kill_process), and exit.
        t.join(timeout=5.0)
        assert not t.is_alive(), "worker didn't exit within 5 s of the interrupt"

        # The critical assertion: the subprocess GROUP must be dead.  Not
        # just the bash wrapper — the 'sleep 30' child too.
        # Give the SIGTERM+1s wait+SIGKILL escalation a moment to complete.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not _pgid_still_alive(pgid):
                break
            time.sleep(0.1)
        assert not _pgid_still_alive(pgid), (
            f"subprocess group {pgid} is STILL ALIVE after worker received "
            f"KeyboardInterrupt — orphan bug regressed.  This is the "
            f"sleep-300-survives-SIGTERM scenario from Physikal's Apr 2026 "
            f"report.  See tools/environments/base.py _wait_for_process "
            f"except-block."
        )
        # And the worker should have observed the KeyboardInterrupt (i.e.
        # it re-raised cleanly, not silently swallowed).
        assert result_holder.get("exception") == "KeyboardInterrupt", (
            f"worker result: {result_holder!r} — expected KeyboardInterrupt "
            f"propagation after cleanup"
        )
    finally:
        try:
            env.cleanup()
        except Exception:
            pass
