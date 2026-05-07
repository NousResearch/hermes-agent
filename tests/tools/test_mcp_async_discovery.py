"""Tests for discover_mcp_tools_async() in tools.mcp_tool."""

import asyncio
from unittest.mock import patch

import pytest  # ty: ignore[unresolved-import]


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    """Ensure clean MCP module state before/after each test."""
    import tools.mcp_tool as mcp

    old_loop = mcp._mcp_loop
    old_thread = mcp._mcp_thread
    old_servers = dict(mcp._servers)
    yield
    # Restore server state.
    mcp._servers.clear()
    mcp._servers.update(old_servers)
    # If the test created a new background thread, stop it cleanly.
    new_loop = mcp._mcp_loop
    new_thread = mcp._mcp_thread
    if new_loop is not old_loop and new_loop is not None:
        new_loop.call_soon_threadsafe(new_loop.stop)
    if new_thread is not old_thread and new_thread is not None:
        new_thread.join(timeout=2)
    mcp._mcp_loop = old_loop
    mcp._mcp_thread = old_thread
    mcp._mcp_loop_ready.clear()


class TestDiscoverMcpToolsAsync:
    """Tests for the fire-and-forget discover_mcp_tools_async function."""

    def test_returns_immediately(self):
        """Should return None immediately, not block."""
        config = {
            "test-server": {
                "command": "echo ok",
                "connect_timeout": 1,
            }
        }
        with patch("tools.mcp_tool._load_mcp_config", return_value=config):
            from tools.mcp_tool import discover_mcp_tools_async

            result = discover_mcp_tools_async()
        assert result is None

    def test_noop_when_mcp_not_available(self):
        with patch("tools.mcp_tool._MCP_AVAILABLE", False):
            from tools.mcp_tool import discover_mcp_tools_async

            result = discover_mcp_tools_async()
        assert result is None

    def test_noop_when_no_servers_configured(self):
        with patch("tools.mcp_tool._load_mcp_config", return_value={}):
            from tools.mcp_tool import discover_mcp_tools_async

            result = discover_mcp_tools_async()
        assert result is None

    def test_noop_when_all_servers_already_connected(self):
        """When all configured servers are already connected, skip."""
        import tools.mcp_tool as mcp

        config = {"test-server": {"command": "echo ok"}}
        with patch.object(mcp, "_load_mcp_config", return_value=config):
            # Pre-populate _servers so the server appears already connected
            fake_server = mcp.MCPServerTask("test-server")
            fake_server._registered_tool_names = ["mcp_test-server_fake"]
            mcp._servers["test-server"] = fake_server

            result = mcp.discover_mcp_tools_async()
        # Should skip because server is already in _servers
        assert result is None

    def test_skips_disabled_servers(self):
        config = {
            "enabled-svr": {"command": "echo ok"},
            "disabled-svr": {"command": "echo nope", "enabled": False},
        }
        with patch("tools.mcp_tool._load_mcp_config", return_value=config):
            from tools.mcp_tool import discover_mcp_tools_async

            result = discover_mcp_tools_async()
        assert result is None

    def test_schedules_discovery_on_mcp_loop(self):
        """When new servers exist, schedules via run_coroutine_threadsafe."""
        import tools.mcp_tool as mcp

        config = {"new-server": {"command": "echo ok", "connect_timeout": 1}}

        with patch.object(mcp, "_load_mcp_config", return_value=config):
            # Capture the run_coroutine_threadsafe call
            with patch.object(
                mcp.asyncio, "run_coroutine_threadsafe"
            ) as mock_schedule:
                mcp.discover_mcp_tools_async()

        # Should have been called exactly once with a coroutine and the loop
        assert mock_schedule.call_count == 1
        call_args = mock_schedule.call_args
        coro = call_args[0][0]
        loop = call_args[0][1]
        assert asyncio.iscoroutine(coro)
        assert isinstance(loop, asyncio.AbstractEventLoop)
        # Don't leave the scheduled coroutine dangling
        coro.close()

    def test_no_loop_no_crash(self):
        """When MCP loop isn't running, should log error and return."""
        import tools.mcp_tool as mcp

        config = {"new-server": {"command": "echo ok", "connect_timeout": 1}}

        with patch.object(mcp, "_load_mcp_config", return_value=config):
            # Kill the MCP loop so run_coroutine_threadsafe would fail
            with patch.object(mcp, "_mcp_loop", None):
                result = mcp.discover_mcp_tools_async()
        assert result is None
