"""Tests for the MCPServerTask reconnect signal.

When the OAuth layer cannot recover in-place (e.g., external refresh of a
single-use refresh_token made the SDK's in-memory refresh fail), the tool
handler signals MCPServerTask to tear down the current MCP session and
reconnect with fresh credentials. This file exercises the signal plumbing
in isolation from the full stdio/http transport machinery.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


def test_keepalive_jitter_is_stable_and_bounded(monkeypatch):
    """Per-server keepalive jitter is deterministic and within the cap."""
    import tools.mcp_tool as mcp_tool

    monkeypatch.setattr(mcp_tool, "_MCP_KEEPALIVE_MAX_JITTER_SECONDS", 15.0)

    first = mcp_tool._mcp_keepalive_jitter_seconds("github")
    second = mcp_tool._mcp_keepalive_jitter_seconds("github")

    assert first == second
    assert 0.0 <= first <= 15.0


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_jitters_before_keepalive(monkeypatch):
    """Idle keepalive probes wait the per-server jitter before list_tools."""
    import tools.mcp_tool as mcp_tool
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("github")

    async def fake_list_tools():
        task._shutdown_event.set()

    task.initialize_result = SimpleNamespace(
        capabilities=SimpleNamespace(tools=SimpleNamespace())
    )
    task.session = type(
        "Session",
        (),
        {
            "send_ping": AsyncMock(side_effect=Exception("Unknown method: ping")),
            "list_tools": AsyncMock(side_effect=fake_list_tools),
        },
    )()

    task._config["keepalive_interval"] = 0.01
    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.001)
    monkeypatch.setattr(mcp_tool, "_mcp_keepalive_jitter_seconds", lambda _name: 0.05)

    waiter = asyncio.create_task(task._wait_for_lifecycle_event())
    await asyncio.sleep(0.025)
    task.session.list_tools.assert_not_awaited()

    reason = await asyncio.wait_for(waiter, timeout=0.2)

    assert reason == "shutdown"
    task.session.send_ping.assert_awaited_once()
    task.session.list_tools.assert_awaited_once()
