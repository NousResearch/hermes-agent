"""Tests for the bounded parked self-probe loop (#77765).

After a keepalive failure on an HTTP+OAuth MCP server, the long-lived
gateway process can wedge its OAuth/auth flow: every parked self-probe
burns the full ``connect_timeout`` per attempt (default 60s × 3 attempts
≈ 3 minutes) and then parks again — forever, even though the server,
token, and network are healthy (a fresh process connects in ~0.3s).

The fix bounds that loop:

- revival probes run on a short connect budget
  (``_PARKED_PROBE_CONNECT_TIMEOUT``) so a wedged auth flow gives up in
  seconds instead of minutes;
- after ``_MAX_PARKED_PROBE_FAILURES`` consecutive failed probes the timed
  self-probe is abandoned — the server waits only for an explicit
  reconnect request or shutdown.
"""

import asyncio

import pytest


def _patch_fast_sleep(monkeypatch):
    """Make ``asyncio.sleep`` a no-op (except for real-time polling).

    Returns the real ``asyncio.sleep`` so the test can still wait on wall
    clock while the run task's backoff/park sleeps are instant.
    """
    real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await real_sleep(0)

    monkeypatch.setattr("tools.mcp_tool.asyncio.sleep", _fast_sleep)
    return real_sleep


@pytest.mark.no_isolate
def test_parked_probe_runs_on_short_connect_budget(monkeypatch, tmp_path):
    """Revival probes must use the short connect budget, not the full one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)
    real_sleep = _patch_fast_sleep(monkeypatch)

    seen = []

    class _Task(MCPServerTask):
        def _deregister_tools(self):
            self._registered_tool_names = []

        async def _run_http(self, config):
            seen.append(self._effective_connect_timeout(config))
            raise asyncio.TimeoutError()

    async def _scenario():
        task = _Task("srv")
        task._registered_tool_names = ["srv__tool"]
        run_task = asyncio.ensure_future(
            task.run(
                {
                    "url": "https://example.test/mcp",
                    "connect_timeout": 60.0,
                    "auth": "oauth",
                }
            )
        )
        # Exhaust the initial ladder (3 full-budget attempts) and park.
        for _ in range(5000):
            await real_sleep(0.005)
            if task._was_parked:
                break
        assert task._was_parked, "server never parked"

        # First connect used the full budget; the probe must be capped.
        for _ in range(5000):
            await real_sleep(0.005)
            if seen and seen[-1] <= mcp_tool._PARKED_PROBE_CONNECT_TIMEOUT:
                break
        assert seen, "no connect attempts recorded"
        assert seen[0] == 60.0, f"first connect should use the full budget: {seen}"
        assert any(
            t <= mcp_tool._PARKED_PROBE_CONNECT_TIMEOUT for t in seen
        ), f"revival probe did not use the short budget: {seen}"

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())


@pytest.mark.no_isolate
def test_parked_self_probe_gives_up_after_max_failures(monkeypatch, tmp_path):
    """After the failure threshold the timed self-probe stops; an explicit
    reconnect request still revives the server."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)
    monkeypatch.setattr(mcp_tool, "_MAX_PARKED_PROBE_FAILURES", 3)
    real_sleep = _patch_fast_sleep(monkeypatch)

    attempts = []

    class _Task(MCPServerTask):
        def _deregister_tools(self):
            self._registered_tool_names = []

        async def _run_http(self, config):
            attempts.append(1)
            raise asyncio.TimeoutError()

    async def _scenario():
        task = _Task("srv")
        task._registered_tool_names = ["srv__tool"]
        run_task = asyncio.ensure_future(
            task.run({"url": "https://example.test/mcp", "auth": "oauth"})
        )
        # Reach the give-up threshold (3 failed probe cycles).
        for _ in range(10000):
            await real_sleep(0.005)
            if task._parked_probe_failures >= 3:
                break
        assert task._parked_probe_failures >= 3, "give-up threshold not reached"
        count_at_giveup = len(attempts)

        # Give-up: no new probe attempts fire while parked.
        for _ in range(500):
            await real_sleep(0.005)
            assert len(attempts) == count_at_giveup, (
                f"self-probe kept firing after give-up: {len(attempts)} "
                f"attempts (was {count_at_giveup})"
            )

        # An explicit reconnect request still wakes the parked server.
        task._reconnect_event.set()
        for _ in range(5000):
            await real_sleep(0.005)
            if len(attempts) > count_at_giveup:
                break
        assert len(attempts) > count_at_giveup, (
            "explicit reconnect did not revive the parked server"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())


def test_mark_session_proven_resets_parked_probe_failures(monkeypatch, tmp_path):
    """A session that proves healthy restarts the probe-failure counter."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("srv")
    task._session_proven = False
    task._was_parked = True
    task._reconnect_retries = 4
    task._parked_probe_failures = 3
    task._mark_session_proven()
    assert task._session_proven is True
    assert task._was_parked is False
    assert task._reconnect_retries == 0
    assert task._parked_probe_failures == 0


def test_effective_connect_timeout_caps_only_probe_attempts(monkeypatch, tmp_path):
    """The short budget applies only while a revival probe is armed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("srv")
    task._probe_connect_timeout = mcp_tool._PARKED_PROBE_CONNECT_TIMEOUT
    assert (
        task._effective_connect_timeout({"connect_timeout": 60.0})
        == mcp_tool._PARKED_PROBE_CONNECT_TIMEOUT
    )
    # The cap never *raises* a configured timeout.
    assert task._effective_connect_timeout({"connect_timeout": 5.0}) == 5.0
    task._probe_connect_timeout = None
    assert task._effective_connect_timeout({"connect_timeout": 60.0}) == 60.0
