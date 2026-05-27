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
async def test_keepalive_serializes_list_tools_with_rpc_lock(monkeypatch):
    """Keepalive list_tools must not overlap an in-flight MCP RPC."""
    import tools.mcp_tool as mcp_tool
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("test")
    observed_external_lock_states = []
    external_lock_held = True

    class FakeSession:
        async def list_tools(self):
            observed_external_lock_states.append(external_lock_held)
            task._shutdown_event.set()

    original_sleep = asyncio.sleep

    async def fake_wait(tasks, timeout=None, return_when=None):
        await original_sleep(0)
        done = {item for item in tasks if item.done()}
        if done:
            return done, set(tasks) - done
        return set(), set(tasks)

    task.session = FakeSession()
    monkeypatch.setattr(mcp_tool.asyncio, "wait", fake_wait)

    await task._rpc_lock.acquire()
    waiter = asyncio.create_task(task._wait_for_lifecycle_event())
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert observed_external_lock_states == []
    finally:
        external_lock_held = False
        task._rpc_lock.release()

    reason = await asyncio.wait_for(waiter, timeout=1.0)

    assert reason == "shutdown"
    assert observed_external_lock_states == [False]
