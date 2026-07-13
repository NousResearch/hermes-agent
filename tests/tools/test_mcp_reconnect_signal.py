"""Tests for the MCPServerTask reconnect signal.

When the OAuth layer cannot recover in-place (e.g., external refresh of a
single-use refresh_token made the SDK's in-memory refresh fail), the tool
handler signals MCPServerTask to tear down the current MCP session and
reconnect with fresh credentials. This file exercises the signal plumbing
in isolation from the full stdio/http transport machinery.
"""
import asyncio

import pytest


@pytest.mark.asyncio
async def test_reconnect_event_attribute_exists():
    """MCPServerTask has a _reconnect_event alongside _shutdown_event."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")
    assert hasattr(task, "_reconnect_event")
    assert isinstance(task._reconnect_event, asyncio.Event)
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_returns_reconnect():
    """When _reconnect_event fires, helper returns 'reconnect' and clears it."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "reconnect"
    # Should have cleared so the next cycle starts fresh
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_returns_shutdown():
    """When _shutdown_event fires, helper returns 'shutdown'."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_shutdown_wins_when_both_set():
    """If both events are set simultaneously, shutdown takes precedence."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


@pytest.mark.asyncio
async def test_transport_connect_attempt_times_out_before_session_is_published():
    """A wedged OAuth reconnect must not block the server task forever.

    A production OAuth failure left ``_run_http`` stuck inside the SDK transport
    before it published ``server.session``. Without an outer setup watchdog,
    the retry loop never reached attempt 2 and reconnect signals could not be
    serviced until the gateway restarted.
    """
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("oauth")
    blocker = asyncio.Event()

    async def _wedged_transport():
        await blocker.wait()

    with pytest.raises(TimeoutError, match="timed out before publishing a session"):
        await task._run_transport_with_connect_timeout(
            _wedged_transport,
            connect_timeout=0.01,
        )


@pytest.mark.asyncio
async def test_transport_connect_watchdog_preserves_lifecycle_result():
    """The setup watchdog must stay transparent after session publication."""
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("test")

    async def _published_transport():
        task.session = object()
        return "recycle"

    reason = await task._run_transport_with_connect_timeout(
        _published_transport,
        connect_timeout=0.1,
    )

    assert reason == "recycle"
