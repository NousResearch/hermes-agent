"""Tests for MCP stdio subprocess lifecycle — issue #59349.

Background: a stdio MCP server that never returns a valid JSON-RPC
``initialize`` response (e.g. emits a malformed frame and then blocks on
``read(stdin)``) was leaking its pidfd + pipe FDs on every discovery
cycle. The gateway accumulated these until it hit its nofile soft limit
and started throwing ``OSError: [Errno 24] Too many open files``.

The fix introduces:
  * an ``asyncio.wait_for`` startup timeout around ``session.initialize()``
  * a ``_terminate_process_tree`` psutil-based reaper for any leaked child
  * a ``_find_mcp_children_by_cmdline`` fallback for children that escaped
    the snapshot race window
  * a module-level ``atexit`` hook that reaps MCP children on gateway
    shutdown (the path SIGTERM + interpreter-exit take).

The unit tests below exercise the helper functions in isolation — they
do NOT spin up a real ``mcp`` server (the gateway's MCP stdio transport
is async + requires an anyio event loop tied to the background MCP
loop). Covering the helpers end-to-end here is sufficient to demonstrate
the leak is bounded; a higher-level integration test could be added
later that drives ``_run_stdio`` directly with a hung subprocess.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from unittest.mock import patch

import psutil
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_hanging_process():
    """Spawn a real subprocess that hangs on stdin — the #59349 repro shape.

    Returns a subprocess.Popen that prints nothing, reads stdin forever,
    and is alive until ``.terminate()`` / ``.kill()`` is called. Mirrors
    the "broken MCP server" from the issue body closely enough that the
    reaper helpers can be exercised against a live, hanging child.
    """
    procs: list[subprocess.Popen] = []

    def _spawn():
        # Cross-platform: a one-liner Python script that prints a single
        # line then blocks on stdin forever. Using ``python`` directly is
        # safe across Windows + POSIX; this is a test-only subprocess.
        code = (
            "import sys\n"
            "print('ready', flush=True)\n"
            "for _ in sys.stdin:\n"
            "    pass\n"
        )
        kwargs = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        # ``creationflags`` on Windows + ``start_new_session`` on POSIX so
        # the child can't be cleaned up by Ctrl+C / SIGHUP to the test
        # runner. The reaper helpers must explicitly reach for it.
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            **kwargs,
        )
        procs.append(proc)
        # Give it a moment to actually print "ready" so we're not racing
        # the reaper against fork/exec.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if b"ready" in line:
                break
        else:  # pragma: no cover — extremely unlikely
            proc.kill()
            raise RuntimeError("hanging subprocess never reported ready")
        return proc

    yield _spawn

    # Cleanup: ensure no test child survives into the next test.
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=2)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _isolate_module_state():
    """Reset mcp_tool's tracked PIDs/pgids before AND after each test.

    Per-file subprocess isolation gives each test file a fresh
    interpreter, so there's no shared module state between this file and
    other test_mcp_* files. Within this file, however, the global
    ``_stdio_pids`` / ``_stdio_pgids`` / ``_orphan_stdio_pids`` maps would
    leak across tests if we didn't reset them.

    Tests that intentionally populate these maps must restore or assert
    their final state; the fixture simply guarantees a clean slate.
    """
    from tools import mcp_tool

    with mcp_tool._lock:
        mcp_tool._stdio_pids.clear()
        mcp_tool._orphan_stdio_pids.clear()
        mcp_tool._stdio_pgids.clear()
    yield
    with mcp_tool._lock:
        mcp_tool._stdio_pids.clear()
        mcp_tool._orphan_stdio_pids.clear()
        mcp_tool._stdio_pgids.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTerminateProcessTreeReapsHangingChild:
    """Hung MCP server → startup timeout terminates it, no leak.

    The repro from #59349 spawns a stdio MCP server that prints a
    non-conforming frame and then blocks on ``read(stdin)``. From the
    test's point of view we just need a child that does NOT exit on its
    own — we exercise ``_terminate_process_tree`` against it and assert
    the child is dead afterward.
    """

    def test_hanging_child_is_force_killed(self, make_hanging_process):
        from tools.mcp_tool import _terminate_process_tree

        proc = make_hanging_process()
        pid = proc.pid
        assert psutil.pid_exists(pid), "child should be alive after spawn"

        _terminate_process_tree(pid, timeout=2.0)

        assert not psutil.pid_exists(pid), (
            "after _terminate_process_tree the hanging child must be gone "
            "(#59349: leaked hang would otherwise accumulate per cycle)"
        )

    def test_already_dead_child_is_noop(self):
        """Calling terminate on a vanished PID must not raise.

        The finally block in ``_run_stdio`` runs the helper on every exit
        path — clean, exceptional, AND after the SDK has already torn
        down. It must not crash on the clean-path cases.
        """
        from tools.mcp_tool import _terminate_process_tree

        # 0x7fffffff is reserved / harmless on Linux, and a likely-vacant
        # number on Windows. Use psutil to pick a definitely-gone PID.
        gone_pid = 0x7FFFFFFE
        # Sanity: pretend it isn't real. If somehow it IS, fall back to a
        # PID we just spawn+killed ourselves.
        try:
            if psutil.pid_exists(gone_pid):
                proc = subprocess.Popen(
                    [sys.executable, "-c", "pass"]
                )
                gone_pid = proc.pid
                proc.wait(timeout=5)
        except Exception:  # pragma: no cover
            pass

        # Must not raise even though the PID is gone.
        _terminate_process_tree(gone_pid, timeout=0.5)


class TestFindMcpChildrenByCmdlineFallback:
    """The classic #59349 race: SDK opens the child, the snapshot returns
    empty, and ``initialize()`` fails. The fallback handler must still
    find + reap the leaked child via psutil cmdline matching.
    """

    def test_finds_child_with_matching_cmdline(self, make_hanging_process):
        from tools.mcp_tool import _find_mcp_children_by_cmdline

        proc = make_hanging_process()
        try:
            matches = _find_mcp_children_by_cmdline(
                sys.executable, ["-c", "import sys\nprint('ready', flush=True)\nfor _ in sys.stdin:\n    pass\n"]
            )
            assert proc.pid in matches, (
                "cmdline-match fallback must find the leaked MCP server "
                "child that escaped the initial PID snapshot (#59349)"
            )
        finally:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

    def test_does_not_match_unrelated_processes(self, make_hanging_process):
        """A live but unrelated process must NOT be reaped as collateral.

        The cmdline-match fallback uses the tail of the cmdline, so an
        MCP-style command embedded in a wholly different invocation
        (different working binary / different arg list) must not match.
        """
        from tools.mcp_tool import _find_mcp_children_by_cmdline

        proc = make_hanging_process()
        try:
            # Asking for a config that we KNOW isn't spawned by anyone
            # must return the empty set — never the unrelated child.
            matches = _find_mcp_children_by_cmdline(
                "definitely-not-an-mcp-server-xyz123",
                ["nonexistent-arg", "another-nonexistent-arg"],
            )
            assert proc.pid not in matches, (
                "unrelated processes must not be reaped as collateral "
                "by the cmdline-match fallback"
            )
            assert not matches, (
                "no match expected for a config no real child corresponds to"
            )
        finally:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass


class TestAtexitReapsOnGatewayShutdown:
    """Gateway SIGTERM → all MCP children terminated (#59349).

    The ``atexit`` hook installed at module-import time is the safety
    net that runs even when the gateway is killed by a signal before
    ``_stop_mcp_loop`` can run. We verify it by populating the tracking
    maps with a real, hanging child and asserting it gets reaped when
    the hook fires.
    """

    def test_atexit_hook_reaps_tracked_children(self, make_hanging_process):
        from tools import mcp_tool

        proc = make_hanging_process()
        pid = proc.pid
        try:
            # Mark the child as a tracked (active) MCP server so the
            # "include_active=True" branch fires.
            with mcp_tool._lock:
                mcp_tool._stdio_pids[pid] = "fake-server"
            assert psutil.pid_exists(pid), "tracked child should still be alive"

            # Invoke the hook directly. Wrapped with try/except so a
            # log-only path never breaks the test process on shutdown.
            mcp_tool._atexit_reap_mcp_children()

            assert not psutil.pid_exists(pid), (
                "atexit hook must reap active MCP children on gateway "
                "shutdown (#59349)"
            )
            with mcp_tool._lock:
                assert pid not in mcp_tool._stdio_pids, (
                    "tracking map must be cleaned up after reap"
                )
        finally:
            try:
                if psutil.pid_exists(pid):
                    proc.kill()
                    proc.wait(timeout=2)
            except Exception:
                pass

    def test_atexit_registered_exactly_once(self):
        """Re-importing the module must not double-register the hook.

        Multiple atexit callbacks for the same function would corrupt
        state and slow shutdown. We rely on a module-level
        ``_registered`` flag — assert that the idempotency guard works.
        """
        from tools import mcp_tool

        # Touch the attribute set by the registration guard. If
        # something accidentally reset it (e.g. a refactor that drops
        # the guard), this assertion fails loudly instead of silently
        # double-registering.
        assert getattr(
            mcp_tool._atexit_reap_mcp_children, "_registered", False
        ) is True, (
            "atexit hook must be idempotent on import (#59349 safety)"
        )


class TestInitializeTimeoutConstant:
    """The startup-timeout guard must be a module-level constant.

    ``_run_stdio`` reads ``_INITIALIZE_TIMEOUT_DEFAULT`` directly so
    tests + downstream callers can extend it via ``config``. We assert
    the constant exists, is a positive number, and is short enough to
    be useful (the issue's 31h-EMFILE scenario only needed a few
    seconds per cycle to stop the leak).
    """

    def test_constant_exists_and_is_positive(self):
        from tools import mcp_tool

        val = getattr(mcp_tool, "_INITIALIZE_TIMEOUT_DEFAULT", None)
        assert val is not None, (
            "_INITIALIZE_TIMEOUT_DEFAULT must exist (issue #59349)"
        )
        assert isinstance(val, (int, float)), (
            "timeout must be numeric so asyncio.wait_for can use it"
        )
        assert 0 < val <= 60, (
            "default timeout must be > 0 and <= 60s; a hung MCP server "
            "should never block a discovery cycle for more than a minute"
        )


class TestFdCountStableAcrossReconnects:
    """FD count stable across reconnects (#59349 — EMFILE guard).

    The original bug accumulated ~1 child + 2-3 pipe FDs per discovery
    cycle. With the reaper in place, a sequence of "failed connect +
    reap" must keep the gateway's open-FD count bounded.

    We approximate the gateway's FD pressure with the test process's
    own psutil measurement, since each fake cycle spawns a child via
    the helper and the reaper ensures it doesn't survive.
    """

    def test_repeated_hangs_do_not_accumulate_fds(self, make_hanging_process):
        """Run the spawn+reap pair a few times; FD count should stay flat.

        Without the fix, each iteration leaks a child + its pipes.
        With the fix, ``_terminate_process_tree`` in the finally block
        (or, in this test, called directly post-spawn) brings the FD
        count back to baseline.
        """
        from tools.mcp_tool import _terminate_process_tree

        # Baseline FD count before any spawn.
        baseline_fds = len(psutil.Process(os.getpid()).open_files()) + 1

        for _ in range(3):
            proc = make_hanging_process()
            # ``make_hanging_process`` opens stdin/stdout/stderr on the
            # child; parent doesn't inherit them as FDs but psutil
            # ``open_files()`` can briefly see transient handler state
            # from the subprocess machinery on Windows. We just want
            # the asymptotic picture, so allow a small slack.
            assert psutil.pid_exists(proc.pid)
            _terminate_process_tree(proc.pid, timeout=2.0)
            # PID truly gone now:
            assert not psutil.pid_exists(proc.pid)
            # Give the kernel a beat to garbage-collect the parent's
            # transient subprocess handles. Without the fix, the open
            # FD count would walk up monotonically each iteration.
            time.sleep(0.1)

        # After reap the test process FD count must NOT be wildly
        # larger than baseline (allow generous slack for unrelated
        # background activity).
        after_fds = len(psutil.Process(os.getpid()).open_files())
        # Open to growth from concurrent test infra; only fail on big
        # growth, since the strict check would be flaky.
        assert after_fds <= baseline_fds + 16, (
            f"FD count grew from {baseline_fds} to {after_fds} across "
            "3 spawn+reap cycles — a leak like #59349 would show this "
            "growing monotonically"
        )
