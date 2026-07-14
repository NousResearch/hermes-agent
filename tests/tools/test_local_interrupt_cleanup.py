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
import threading
import time
from types import SimpleNamespace

import pytest

from tools.environments import local as local_mod
from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "logs").mkdir(exist_ok=True)


def _pgid_still_alive(pgid: int) -> bool:
    """Return True if any process in the given process group is still alive."""
    try:
        os.killpg(pgid, 0)  # signal 0 = existence check
        return True
    except ProcessLookupError:
        return False


def _process_group_snapshot(pgid: int) -> str:
    """Return a process-table snapshot for diagnostics."""
    return subprocess.run(
        ["ps", "-o", "pid,ppid,pgid,stat,cmd", "-g", str(pgid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _wait_for_pgid_exit(pgid: int, timeout: float = 30.0) -> bool:
    """Wait for a process group to disappear under loaded xdist hosts.

    The cleanup chain is: SIGTERM → 3s TimeoutStopSec → SIGKILL → reap.
    Under heavy xdist load (40 parallel workers, 6-shard CI), the full
    sequence can exceed 10s. Default timeout is generous to avoid CI
    flakes; in practice the wait returns in <1s on quiet hosts.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pgid_still_alive(pgid):
            return True
        time.sleep(0.1)
    return not _pgid_still_alive(pgid)


def test_kill_process_uses_cached_pgid_if_wrapper_already_exited(monkeypatch):
    """If the shell wrapper exits before cleanup, still kill its process group.

    Without the cached pgid fallback, ``os.getpgid(proc.pid)`` raises for the
    dead wrapper and cleanup falls back to ``proc.kill()``, which cannot reach
    orphaned grandchildren still running in the original process group.
    """
    env = object.__new__(LocalEnvironment)
    proc = SimpleNamespace(
        pid=12345,
        _hermes_pgid=67890,
        poll=lambda: 0,
        kill=lambda: None,
    )
    killpg_calls = []

    def fake_getpgid(_pid):
        raise ProcessLookupError

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", fake_getpgid)
    monkeypatch.setattr(os, "killpg", fake_killpg)

    env._kill_process(proc)

    assert killpg_calls == [(67890, signal.SIGTERM), (67890, 0)]


def test_kill_process_uses_windows_tree_kill(monkeypatch):
    """Windows must kill the whole Bash process tree, not just the wrapper."""
    env = object.__new__(LocalEnvironment)
    terminate_calls = []
    waits = []
    killed = []

    def fake_terminate(pid, *, force=False):
        terminate_calls.append((pid, force))

    proc = SimpleNamespace(
        pid=12345,
        kill=lambda: killed.append(True),
        wait=lambda timeout=None: waits.append(timeout),
    )

    monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr("gateway.status.terminate_pid", fake_terminate)

    env._kill_process(proc)

    assert terminate_calls == [(12345, True)]
    assert waits == [2.0]
    assert killed == []


def test_wait_for_process_kills_subprocess_on_keyboardinterrupt():
    """When KeyboardInterrupt arrives mid-poll, the subprocess group must be
    killed before the exception is re-raised."""
    env = LocalEnvironment(cwd="/tmp")
    try:
        result_holder = {}
        proc_holder = {}
        started = threading.Event()
        raise_at = [None]  # set by the main thread to tell worker when

        original_run_bash = env._run_bash

        def tracking_run_bash(*args, **kwargs):
            proc = original_run_bash(*args, **kwargs)
            cmd_string = args[0] if args else kwargs.get("cmd_string", "")
            if "sleep 30" in str(cmd_string):
                proc_holder["proc"] = proc
                started.set()
            return proc

        env._run_bash = tracking_run_bash

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
        # Wait until the subprocess actually exists.  Capture the Popen object
        # from LocalEnvironment._run_bash instead of scraping process text:
        # wrappers vary by shell/platform and can make `sleep 30` cmdline scans
        # flaky even when the process group is alive.
        assert started.wait(timeout=5.0), (
            "test setup: LocalEnvironment did not spawn the 'sleep 30' wrapper within 5 s"
        )
        proc = proc_holder.get("proc")
        assert proc is not None and proc.poll() is None, "sanity: subprocess wrapper should be alive"
        pgid = os.getpgid(proc.pid)
        assert _pgid_still_alive(pgid), "sanity: subprocess group should be alive"

        # Now inject a KeyboardInterrupt into the worker thread the same
        # way CPython's signal machinery would.  We use ctypes.PyThreadState_SetAsyncExc
        # which is how signal delivery to non-main threads is simulated.
        import ctypes
        # py-thread-state exception targets need the ident, not the Thread
        tid = t.ident
        assert tid is not None
        # Fire KeyboardInterrupt into the worker thread
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), ctypes.py_object(KeyboardInterrupt),
        )
        assert ret == 1, f"SetAsyncExc returned {ret}, expected 1"

        # Give the worker a moment to: hit the exception at the next poll,
        # run the except-block cleanup (_kill_process), and exit.  Under
        # xdist load the SIGTERM → 3s wait → SIGKILL chain can take longer
        # than 5s before the worker's join() returns; bumped to 15s.
        t.join(timeout=15.0)
        assert not t.is_alive(), "worker didn't exit within 15 s of the interrupt"

        # The critical assertion: the subprocess GROUP must be dead.  Not
        # just the bash wrapper — the 'sleep 30' child too. Under xdist load,
        # process-group disappearance can lag briefly after the worker exits,
        # especially if the process is already dying or waiting to be reaped.
        assert _wait_for_pgid_exit(pgid), (
            f"subprocess group {pgid} is STILL ALIVE after worker received "
            f"KeyboardInterrupt — orphan bug regressed.  This is the "
            f"sleep-300-survives-SIGTERM scenario from Physikal's Apr 2026 "
            f"report.  See tools/environments/base.py _wait_for_process "
            f"except-block.\n{_process_group_snapshot(pgid)}"
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
