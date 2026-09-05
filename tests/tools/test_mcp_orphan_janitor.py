"""#81880 regression: stdio MCP orphans must be reaped mid-process.

`_release_spawned_children` correctly marks live children as orphans
(`_orphan_stdio_pids`) when a session closes — but in a long-lived
desktop/gateway process nothing swept that set until process exit, so every
session's leaked stdio children accumulated for the process lifetime. Field
reports on the issue: ~30 python + ~28 node processes resident after days of
desktop-app use, OOM on 16GB Macs, sustained thermal load on Windows laptops.

The existing reapers all target different producers (process-exit sweep, CLI
session-rotation #82141, parent-death watchdog for ungraceful exit). The
background orphan janitor added here closes the remaining gap: a
lazily-started daemon thread that periodically calls
`_kill_orphaned_mcp_children()` — the existing race-safe sweep that only
touches the orphan set, never live sessions.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

import tools.mcp_tool_lifecycle as lifecycle
from tools.mcp_tool_common import _core


@pytest.fixture(autouse=True)
def _clean_orphan_state():
    """Isolate the module-global orphan state + janitor around each test."""
    with _core._lock:
        saved_orphans = set(lifecycle._orphan_stdio_pids)
        saved_servers = dict(lifecycle._orphan_stdio_pid_servers)
        lifecycle._orphan_stdio_pids.clear()
        lifecycle._orphan_stdio_pid_servers.clear()
    saved_thread = lifecycle._orphan_janitor_thread
    saved_wakeup = lifecycle._orphan_janitor_wakeup
    saved_interval = lifecycle._orphan_janitor_interval
    saved_idle_stops = lifecycle._orphan_janitor_idle_stops
    lifecycle._orphan_janitor_thread = None
    lifecycle._orphan_janitor_wakeup = None
    try:
        yield
    finally:
        # Stop any janitor the test started: clear the orphan set (a daemon
        # that sweeps only orphans then self-retires) and detach it.
        with _core._lock:
            lifecycle._orphan_stdio_pids.clear()
            lifecycle._orphan_stdio_pid_servers.clear()
        lifecycle._orphan_janitor_thread = None
        lifecycle._orphan_janitor_wakeup = None
        lifecycle._orphan_janitor_interval = saved_interval
        lifecycle._orphan_janitor_idle_stops = saved_idle_stops
        with _core._lock:
            lifecycle._orphan_stdio_pids.update(saved_orphans)
            lifecycle._orphan_stdio_pid_servers.update(saved_servers)
        if saved_thread is not None and saved_thread.is_alive():
            lifecycle._orphan_janitor_thread = saved_thread
            lifecycle._orphan_janitor_wakeup = saved_wakeup


class TestOrphanJanitor:
    def test_recording_orphan_starts_janitor(self):
        """A recorded orphan must start the background reaper."""
        with patch.object(lifecycle, "_kill_orphaned_mcp_children") as sweep:
            lifecycle._orphan_stdio_pids.add(424242)
            lifecycle._orphan_stdio_pid_servers[424242] = "test-server"
            lifecycle._maybe_start_orphan_janitor()

            thread = lifecycle._orphan_janitor_thread
            assert thread is not None
            assert thread.is_alive()
            assert thread.daemon
            assert thread.name == "mcp-orphan-janitor"

            # The janitor wakes and sweeps the orphan set (fast interval for
            # the test only).
            lifecycle._orphan_janitor_interval = 0.05
            if lifecycle._orphan_janitor_wakeup is not None:
                lifecycle._orphan_janitor_wakeup.set()
            deadline = time.monotonic() + 10
            while not sweep.called and time.monotonic() < deadline:
                time.sleep(0.05)
            assert sweep.called

    def test_second_start_is_idempotent(self):
        """Starting twice must not spawn a second thread."""
        lifecycle._maybe_start_orphan_janitor()
        first = lifecycle._orphan_janitor_thread
        assert first is not None
        lifecycle._maybe_start_orphan_janitor()
        assert lifecycle._orphan_janitor_thread is first

    def test_janitor_exits_when_idle(self):
        """With no orphans the thread retires so idle processes pay nothing."""
        lifecycle._orphan_janitor_interval = 0.05
        lifecycle._orphan_janitor_idle_stops = 1
        lifecycle._maybe_start_orphan_janitor()
        thread = lifecycle._orphan_janitor_thread
        assert thread is not None
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert lifecycle._orphan_janitor_thread is None

    def test_janitor_survives_sweep_errors(self):
        """An unexpected sweep error must not kill the janitor."""
        lifecycle._orphan_stdio_pids.add(424243)
        lifecycle._orphan_janitor_interval = 0.05
        with patch.object(
            lifecycle, "_kill_orphaned_mcp_children", side_effect=RuntimeError("boom")
        ):
            lifecycle._maybe_start_orphan_janitor()
            thread = lifecycle._orphan_janitor_thread
            assert thread is not None
            time.sleep(0.3)
            assert thread.is_alive()

    def test_threads_share_one_janitor(self):
        """Concurrent starters still produce exactly one thread."""
        from unittest.mock import patch

        # A pending (mock-swept) orphan keeps the janitor from self-retiring
        # on nudges: without it, the losers' wake-up nudges can be consumed
        # as empty iterations and trip the idle-stop before the assert.
        lifecycle._orphan_stdio_pids.add(424244)
        barrier = threading.Barrier(4)
        lifecycle._orphan_janitor_interval = 30.0

        def _start():
            barrier.wait(timeout=10)
            lifecycle._maybe_start_orphan_janitor()

        with patch.object(lifecycle, "_kill_orphaned_mcp_children"):
            threads = [threading.Thread(target=_start) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            assert lifecycle._orphan_janitor_thread is not None
            assert lifecycle._orphan_janitor_thread.is_alive()
