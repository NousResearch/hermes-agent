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
import contextlib
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


def _wait_for_pgid_exit(pgid: int, timeout: float = 60.0) -> bool:
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
        deadline = time.monotonic() + 20.0  # generous: init_session + spawn dilate under CI load
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
        t.join(timeout=30.0)
        assert not t.is_alive(), "worker didn't exit within 30 s of the interrupt"

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


def _descendant_running(marker: str):
    import psutil
    for p in psutil.Process(os.getpid()).children(recursive=True):
        try:
            if marker in " ".join(p.cmdline()):
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def test_exit_cleanup_kills_foreground_command_still_running(monkeypatch):
    """The host-exit funnel (CLI, one-shot, messaging gateway, serve atexit) must take an
    in-flight foreground command's process group with it: it runs in its own session, so
    the host exiting mid-command would otherwise orphan it to init."""
    from tools import terminal_tool_lifecycle

    monkeypatch.setattr(terminal_tool_lifecycle, "_scratch_paths", lambda: [])
    env = LocalEnvironment(cwd="/tmp")
    result: dict = {}
    t = threading.Thread(target=lambda: result.update(env.execute("sleep 3517", timeout=600)), daemon=True)
    try:
        t.start()
        deadline = time.monotonic() + 20.0
        while (proc := _descendant_running("sleep 3517")) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc is not None, "test setup: foreground sleep never started"
        pgid = os.getpgid(proc.pid)

        terminal_tool_lifecycle.cleanup_all_environments()

        assert _wait_for_pgid_exit(pgid, timeout=15.0), (
            f"foreground command survived exit cleanup:\n{_process_group_snapshot(pgid)}")
        t.join(timeout=15.0)
        assert not t.is_alive() and result.get("returncode") not in (None, 0), result
    finally:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        env.cleanup()


# Child for the hard-exit race tests: it really os._exit()s right after the kill, like the watchdogs.
_HARD_EXIT_RACE_CHILD = r"""
import os, sys, threading
from tools.environments import base
from tools.environments.local import LocalEnvironment
scenario, cmd = sys.argv[1], sys.argv[2]
env = LocalEnvironment(cwd=os.getcwd())
if scenario == "spawn_before_publish":
    spawned, release = threading.Event(), threading.Event()
    real_run_bash = LocalEnvironment._run_bash
    def gated(self, command, **kw):
        proc = real_run_bash(self, command, **kw)
        if cmd in command:
            spawned.set()
            release.wait(10)  # barrier: the child exists, it is not published yet
        return proc
    LocalEnvironment._run_bash = gated
    threading.Thread(target=env.execute, args=(cmd,), kwargs={"timeout": 600}, daemon=True).start()
    assert spawned.wait(20)
    threading.Timer(0.2, release.set).start()  # registration lands while the killer is in flight
    base.kill_live_foreground_processes(now=True)
else:  # launch_after_fence
    base.kill_live_foreground_processes(now=True)
    t = threading.Thread(target=env.execute, args=(cmd,), kwargs={"timeout": 600}, daemon=True)
    t.start()
    t.join(3)
os._exit(0)
"""


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
@pytest.mark.live_system_guard_bypass  # a red run must reap survivors reparented to init
@pytest.mark.parametrize("scenario", ["spawn_before_publish", "launch_after_fence"])
def test_hard_exit_leaves_no_foreground_survivor_around_the_spawn(scenario, tmp_path):
    """A hard exit must own every foreground child: one spawned but not yet registered when the kill
    runs, and one launched after the kill took its snapshot (the exit fence refuses it)."""
    import sys

    import psutil

    cmd = f"sleep {35000 + os.getpid() % 1000}.{len(scenario)}"
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = subprocess.run([sys.executable, "-c", _HARD_EXIT_RACE_CHILD, scenario, cmd], cwd=str(tmp_path),
                       env={**os.environ, "PYTHONPATH": repo}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    time.sleep(0.3)
    survivors = [p for p in psutil.process_iter(["cmdline"]) if cmd in " ".join(p.info["cmdline"] or [])]
    for p in survivors:
        with contextlib.suppress(psutil.Error):
            p.kill()
    assert not survivors, f"{scenario}: {[' '.join(p.info['cmdline']) for p in survivors]} outlived the hard exit"


def test_hard_exit_kill_never_blocks_on_a_slow_remote_cancel(monkeypatch):
    """Modal/Daytona/Vercel cancel through a blocking SDK call; the hard-exit kill runs just before
    os._exit, so it must give those one short deadline instead of waiting them out."""
    from tools.environments import base
    from tools.environments.base_output import _ThreadedProcessHandle

    class _SdkEnv:  # the kill every SDK backend inherits: proc.kill() -> cancel_fn
        _kill_process = base.BaseEnvironment._kill_process
        _force_kill_process = base.BaseEnvironment._force_kill_process

    cancelled = threading.Event()
    handle = _ThreadedProcessHandle(lambda: ("", 0), cancel_fn=lambda: (cancelled.set(), time.sleep(8)))
    monkeypatch.setitem(base._live_foreground, id(handle), (_SdkEnv(), handle))
    t0 = time.monotonic()
    base.kill_live_foreground_processes(now=True)
    elapsed = time.monotonic() - t0
    assert cancelled.is_set() and elapsed < 1.0, f"blocked {elapsed:.2f}s"
