"""Tests for the MCPServerTask reconnect signal.

When the OAuth layer cannot recover in-place (e.g., external refresh of a
single-use refresh_token made the SDK's in-memory refresh fail), the tool
handler signals MCPServerTask to tear down the current MCP session and
reconnect with fresh credentials. This file exercises the signal plumbing
in isolation from the full stdio/http transport machinery.
"""
import asyncio
from types import SimpleNamespace

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
async def test_post_ready_disconnect_retries_past_reconnect_cap(monkeypatch):
    """After initial readiness, reconnect failures should not park the task."""
    import tools.mcp_tool as mcp_tool
    from tools.mcp_tool import MCPServerTask

    class FlakyPostReadyTask(MCPServerTask):
        __slots__ = ("attempts", "shutdown_after")

        def __init__(self, shutdown_after: int):
            super().__init__("test")
            self.attempts = 0
            self.shutdown_after = shutdown_after

        async def _run_stdio(self, config):
            self.attempts += 1
            if self.attempts >= self.shutdown_after:
                self._shutdown_event.set()
            raise ConnectionError("transient disconnect")

    sleep_delays = []

    async def fake_sleep(delay):
        sleep_delays.append(delay)

    async def fail_if_parked(self):
        raise AssertionError("post-ready reconnects should keep retrying")

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        MCPServerTask,
        "_wait_for_reconnect_or_shutdown",
        fail_if_parked,
        raising=False,
    )

    shutdown_after = mcp_tool._MAX_RECONNECT_RETRIES + 2
    task = FlakyPostReadyTask(shutdown_after=shutdown_after)
    task._ready.set()  # Simulate a server that had connected successfully.

    await task.run({"command": "fake-mcp-server"})

    assert task.attempts == shutdown_after
    assert len(sleep_delays) == mcp_tool._MAX_RECONNECT_RETRIES + 1
    assert max(sleep_delays) <= mcp_tool._MAX_BACKOFF_SECONDS


@pytest.mark.asyncio
async def test_initial_connection_failures_remain_bounded(monkeypatch):
    """Bad initial configs/offline servers should still fail fast."""
    import tools.mcp_tool as mcp_tool
    from tools.mcp_tool import MCPServerTask

    class NeverReadyTask(MCPServerTask):
        __slots__ = ("attempts",)

        def __init__(self):
            super().__init__("test")
            self.attempts = 0

        async def _run_stdio(self, config):
            self.attempts += 1
            raise ConnectionError("startup failure")

    async def fake_sleep(delay):
        return None

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", fake_sleep)

    task = NeverReadyTask()
    await task.run({"command": "fake-mcp-server"})

    assert task.attempts == mcp_tool._MAX_INITIAL_CONNECT_RETRIES + 1
    assert isinstance(task._error, ConnectionError)
    assert task._ready.is_set()


@pytest.mark.asyncio
async def test_post_threshold_reconnect_reregisters_deregistered_tools(monkeypatch):
    """Recovered servers must publish tools again after stale deregistration."""
    from tools.registry import registry
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    server_name = "reconnect_regression_server"
    tool_name = "mcp_reconnect_regression_server_echo"

    class FakeSession:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="echo",
                        description="Echo input",
                        inputSchema={"type": "object", "properties": {}},
                    )
                ]
            )

    task = MCPServerTask(server_name)
    task._config = {}
    task.session = FakeSession()
    task._ready.set()  # Reconnect path: initial start already completed.

    try:
        task._deregister_tools()
        assert registry.get_toolset_for_tool(tool_name) is None

        await task._discover_tools()

        assert task._registered_tool_names == [tool_name]
        assert registry.get_toolset_for_tool(tool_name) == f"mcp-{server_name}"
        assert mcp_tool._mcp_tool_server_names[tool_name] == server_name
    finally:
        task._deregister_tools()
