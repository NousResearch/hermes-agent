"""Regression for #134611: the stdio spawn's child-PID delta must not claim a child that another
gateway thread (a cron worker or its ``systemd-run`` wrapper) spawns inside the same window.

Both arms use real processes and the real ``_snapshot_child_pids`` / ``_filter_mcp_children``;
the MCP arm spawns through the MCP SDK's own ``stdio_client`` from the event-loop thread, the
same way ``_run_stdio`` does.
"""
import asyncio
import subprocess
import sys
import threading

import pytest

from tools import mcp_tool_lifecycle as lifecycle

_SLEEPER = [sys.executable, "-c", "import sys; sys.stdin.read()"]


class _ForeignSpawner:
    """A long-lived non-MCP thread that spawns an isolated child on demand, like the cron thread."""

    def __init__(self):
        self.spawned = threading.Event()
        self.release = threading.Event()
        self.proc = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.proc = subprocess.Popen(_SLEEPER, stdin=subprocess.PIPE, start_new_session=True)
        self.spawned.set()
        self.release.wait()

    def __enter__(self):
        self.thread.start()
        assert self.spawned.wait(10)
        return self

    def __exit__(self, *exc):
        self.proc.kill()
        self.proc.wait(10)
        self.release.set()
        self.thread.join(10)


@pytest.mark.platforms("linux")
def test_foreign_thread_child_spawned_in_window_is_not_attributed():
    async def scenario():
        before = lifecycle._snapshot_child_pids(calling_thread_only=True)
        with _ForeignSpawner() as foreign:
            own = await asyncio.create_subprocess_exec(*_SLEEPER, stdin=subprocess.PIPE,
                                                       start_new_session=True)
            try:
                after = lifecycle._snapshot_child_pids(calling_thread_only=True)
                delta = lifecycle._filter_mcp_children(after - before)
                # The process-wide reading sees both children: that is the window the fix closes.
                assert {own.pid, foreign.proc.pid} <= lifecycle._snapshot_child_pids()
            finally:
                own.kill()
                await own.wait()
        return delta, own.pid, foreign.proc.pid

    delta, own_pid, foreign_pid = asyncio.run(scenario())
    assert own_pid in delta
    assert foreign_pid not in delta


@pytest.mark.platforms("linux")
def test_sdk_stdio_client_child_is_attributed_from_the_loop_thread():
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        before = lifecycle._snapshot_child_pids(calling_thread_only=True)
        params = StdioServerParameters(command=_SLEEPER[0], args=_SLEEPER[1:])
        with _ForeignSpawner() as foreign:
            async with stdio_client(params):
                delta = lifecycle._filter_mcp_children(
                    lifecycle._snapshot_child_pids(calling_thread_only=True) - before)
            return delta, foreign.proc.pid

    delta, foreign_pid = asyncio.run(scenario())
    assert len(delta) == 1
    assert foreign_pid not in delta
