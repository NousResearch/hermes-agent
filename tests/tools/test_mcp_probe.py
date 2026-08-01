"""Tests for probe_mcp_server_tools() in tools.mcp_tool."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    """Ensure clean MCP module state before/after each test."""
    import tools.mcp_tool as mcp
    old_loop = mcp._mcp_loop
    old_thread = mcp._mcp_thread
    old_servers = dict(mcp._servers)
    yield
    mcp._servers.clear()
    mcp._servers.update(old_servers)
    mcp._mcp_loop = old_loop
    mcp._mcp_thread = old_thread


class TestProbeMcpServerTools:
    """Tests for the lightweight probe_mcp_server_tools function."""

    def test_returns_empty_when_mcp_not_available(self):
        with patch("tools.mcp_tool._MCP_AVAILABLE", False):
            from tools.mcp_tool import probe_mcp_server_tools
            result = probe_mcp_server_tools()
        assert result == {}


    def test_failed_server_omitted_from_results(self):
        """Servers that fail to connect are silently skipped."""
        config = {
            "github": {"command": "npx", "connect_timeout": 5},
            "broken": {"command": "nonexistent", "connect_timeout": 5},
        }
        mock_tool = SimpleNamespace(name="create_issue", description="Create")
        mock_server = MagicMock()
        mock_server._tools = [mock_tool]
        mock_server.shutdown = AsyncMock()

        async def fake_connect(name, cfg):
            if name == "broken":
                raise ConnectionError("Server not found")
            return mock_server

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run, \
             patch("tools.mcp_tool._stop_mcp_loop"):

            def run_coro(coro_or_factory, timeout=120):
                coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            mock_run.side_effect = run_coro

            from tools.mcp_tool import probe_mcp_server_tools
            result = probe_mcp_server_tools()

        assert "github" in result
        assert "broken" not in result


    def test_skips_disabled_servers(self):
        """Disabled servers are not probed."""
        config = {
            "github": {"command": "npx", "connect_timeout": 5},
            "disabled_one": {"command": "npx", "enabled": False},
        }
        mock_tool = SimpleNamespace(name="create_issue", description="Create")
        mock_server = MagicMock()
        mock_server._tools = [mock_tool]
        mock_server.shutdown = AsyncMock()

        connect_calls = []

        async def fake_connect(name, cfg):
            connect_calls.append(name)
            return mock_server

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run, \
             patch("tools.mcp_tool._stop_mcp_loop"):

            def run_coro(coro_or_factory, timeout=120):
                coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            mock_run.side_effect = run_coro

            from tools.mcp_tool import probe_mcp_server_tools
            result = probe_mcp_server_tools()

        assert "github" in result
        assert "disabled_one" not in result
        assert "disabled_one" not in connect_calls
