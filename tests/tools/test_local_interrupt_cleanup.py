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


def _wait_for_pgid_exit(pgid: int, timeout: float = 8.0) -> bool:
    """Wait (bounded) for a process group to disappear after cleanup.

    Once _wait_for_process's except-block runs _kill_process, the reap chain
    (SIGTERM → ≤1s → SIGKILL → ≤2s → wait) empties the group in ≤~3.2s
    regardless of host load — those waits are wall-clock deadlines, not
    work-bound (see LocalEnvironment._kill_process). On quiet hosts this
    returns in <1s.

    The timeout is deliberately SMALL. This helper is one of three sequential
    waits in test_wait_for_process_kills_subprocess_on_keyboardinterrupt; their
    sum must stay well under the 30s global pytest-timeout. That cap uses the
    ``thread`` method (pyproject.toml addopts), which on expiry hard-``os._exit``s
    the per-file interpreter — crashing the ENTIRE file ("1 file where no tests
    ran") instead of failing one assertion. A generous timeout here is exactly
    what used to trip that cap under xdist load and spuriously fail unrelated
    PRs. If the group genuinely hasn't died within this budget, the caller
    fails one assertion cleanly, with a process-table snapshot.
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
        # Wait until the subprocess actually exists.  LocalEnvironment.execute
        # does init_session() (one spawn) before the real command, so we need
        # to wait until a sleep 30 is visible.  Use pgrep-style lookup via
        # /proc to find the bash process running our sleep.
        deadline = time.monotonic() + 5.0
        target_pid = None
        while time.monotonic() < deadline:
            # Walk our children and grand-children to find one running 'sleep 30'
            try:
                import psutil  # optional — fall back if absent
                for p in psutil.Process(os.getpid()).children(recursive=True):
                    try:
                        if "sleep 30" in " ".join(p.cmdline()):
                            target_pid = p.pid
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except ImportError:
                # Fall back to ps
                ps = subprocess.run(
                    ["ps", "-eo", "pid,ppid,pgid,cmd"], capture_output=True, text=True,
                )
                for line in ps.stdout.splitlines():
                    if "sleep 30" in line and "grep" not in line:
                        parts = line.split()
                        if parts and parts[0].isdigit():
                            target_pid = int(parts[0])
                            break
            if target_pid:
                break
            time.sleep(0.1)

        assert target_pid is not None, (
            "test setup: couldn't find 'sleep 30' subprocess after 5 s"
        )
        pgid = os.getpgid(target_pid)
        assert _pgid_still_alive(pgid), "sanity: subprocess should be alive"

        # Settle briefly so the worker is reliably parked in _wait_for_process's
        # interruptible poll-loop sleep before we inject. The poll loop is
        # entered within microseconds of the spawn, but the 'sleep 30' becomes
        # visible to our scan a hair before the worker reaches the try-guarded
        # sleep; injecting in that sliver would deliver the async exception
        # outside the poll loop, where _kill_process never runs — a test-harness
        # race, not a product regression. A short settle closes it.
        time.sleep(0.3)
        assert _pgid_still_alive(pgid), "subprocess exited during settle"

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

        # Give the worker time to: hit the exception at the next poll, run the
        # except-block cleanup (_kill_process → reap ≤~3.2s, then a ≤2s drain
        # join), and re-raise. Those waits are wall-clock-bounded, so ~5.5s is
        # the real worst case; 8s absorbs scheduling jitter under xdist load.
        #
        # This ceiling is deliberately part of a budget: setup (≤5s) + settle
        # (0.3s) + join (≤8s) + _wait_for_pgid_exit (≤8s) ≈ 21s stays clear of
        # the 30s global pytest-timeout, whose `thread` method hard-kills the
        # whole test file on expiry (see _wait_for_pgid_exit's docstring and
        # pyproject addopts). The old 15s+30s pair could reach ~50s under load
        # and trip that cap — crashing the shard instead of failing one test.
        # If the worker really hasn't exited in 8s, the assert below fails this
        # one test cleanly.
        t.join(timeout=8.0)
        assert not t.is_alive(), "worker didn't exit within 8 s of the interrupt"

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
