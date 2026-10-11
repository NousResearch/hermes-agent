"""Tests for the MCPServerTask reconnect signal.

When the OAuth layer cannot recover in-place (e.g., external refresh of a
single-use refresh_token made the SDK's in-memory refresh fail), the tool
handler signals MCPServerTask to tear down the current MCP session and
reconnect with fresh credentials. This file exercises the signal plumbing
in isolation from the full stdio/http transport machinery.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest




@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_shutdown_wins_when_both_set():
    """If both events are set simultaneously, shutdown takes precedence."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


def test_keepalive_jitter_is_stable_and_bounded():
    from tools.mcp_tool_server_run import _mcp_keepalive_jitter_seconds

    value = _mcp_keepalive_jitter_seconds("github")
    assert value == _mcp_keepalive_jitter_seconds("github")
    assert 0 <= value <= 15
    assert value != _mcp_keepalive_jitter_seconds("filesystem")


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["shutdown", "reconnect", "probe"])
@pytest.mark.parametrize("interval,jitter", [(10, 3), (5, 15)])
async def test_keepalive_phase_preserves_cadence_and_lifecycle(monkeypatch, event, interval, jitter):
    from tools.mcp_tool import MCPServerTask
    from tools import mcp_tool_server_run as lifecycle

    task = MCPServerTask("github")
    task._config["keepalive_interval"] = interval
    task.session = object()
    timeouts = []
    real_wait = asyncio.wait

    async def phase_wait(waiters, *, timeout, return_when):
        timeouts.append(timeout)
        if len(timeouts) == 2 and event != "probe":
            getattr(task, f"_{event}_event").set()
            return await real_wait(waiters, timeout=2, return_when=return_when)
        if len(timeouts) > 2:
            return await real_wait(waiters, timeout=2, return_when=return_when)
        return set(), set(waiters)

    async def probe():
        task._shutdown_event.set()

    monkeypatch.setattr(lifecycle, "_mcp_keepalive_jitter_seconds", lambda _: jitter)
    monkeypatch.setattr(lifecycle.asyncio, "wait", phase_wait)
    task._keepalive_probe = AsyncMock(side_effect=probe)
    reason = await asyncio.wait_for(task._wait_for_lifecycle_event(), timeout=2)
    assert timeouts[:2] == [interval - min(interval, jitter), min(interval, jitter)]
    assert reason == ("reconnect" if event == "reconnect" else "shutdown")
    assert task._keepalive_probe.await_count == (1 if event == "probe" else 0)
