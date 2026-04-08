"""Tests for MCP dynamic tool discovery (notifications/tools/list_changed)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _register_server_tools
from tools.registry import ToolRegistry


def _make_mcp_tool(name: str, desc: str = ""):
    return SimpleNamespace(name=name, description=desc, inputSchema=None)


class TestRegisterServerTools:
    """Tests for the extracted _register_server_tools helper."""

    @pytest.fixture
    def mock_registry(self):
        return ToolRegistry()

    @pytest.fixture
    def mock_toolsets(self):
        return {
            "hermes-cli": {"tools": ["terminal"], "description": "CLI", "includes": []},
            "hermes-telegram": {"tools": ["terminal"], "description": "TG", "includes": []},
            "custom-toolset": {"tools": [], "description": "Other", "includes": []},
        }

    def test_injects_hermes_toolsets(self, mock_registry, mock_toolsets):
        """Tools are injected into hermes-* toolsets but not custom ones."""
        server = MCPServerTask("my_srv")
        server._tools = [_make_mcp_tool("my_tool", "desc")]
        server.session = MagicMock()

        with patch("tools.registry.registry", mock_registry), \
            patch("toolsets.create_custom_toolset"), \
            patch.dict("toolsets.TOOLSETS", mock_toolsets, clear=True):

            registered = _register_server_tools("my_srv", server, {})

        assert "mcp_my_srv_my_tool" in registered
        assert "mcp_my_srv_my_tool" in mock_registry.get_all_tool_names()

        # Injected into hermes-* toolsets
        assert "mcp_my_srv_my_tool" in mock_toolsets["hermes-cli"]["tools"]
        assert "mcp_my_srv_my_tool" in mock_toolsets["hermes-telegram"]["tools"]
        # NOT into non-hermes toolsets
        assert "mcp_my_srv_my_tool" not in mock_toolsets["custom-toolset"]["tools"]


class TestRefreshTools:
    """Tests for MCPServerTask._refresh_tools nuke-and-repave cycle."""

    @pytest.fixture
    def mock_registry(self):
        return ToolRegistry()

    @pytest.fixture
    def mock_toolsets(self):
        return {
            "hermes-cli": {"tools": ["terminal"], "description": "CLI", "includes": []},
            "hermes-telegram": {"tools": ["terminal"], "description": "TG", "includes": []},
        }

    @pytest.mark.asyncio
    async def test_nuke_and_repave(self, mock_registry, mock_toolsets):
        """Old tools are removed and new tools registered on refresh."""
        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {}

        # Seed initial state: one old tool registered
        mock_registry.register(
            name="mcp_live_srv_old_tool", toolset="mcp-live_srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )
        server._registered_tool_names = ["mcp_live_srv_old_tool"]
        mock_toolsets["hermes-cli"]["tools"].append("mcp_live_srv_old_tool")

        # New tool list from server
        new_tool = _make_mcp_tool("new_tool", "new behavior")
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(tools=[new_tool])
            )
        )

        with patch("tools.registry.registry", mock_registry), \
            patch("toolsets.create_custom_toolset"), \
            patch.dict("toolsets.TOOLSETS", mock_toolsets, clear=True):

            await server._refresh_tools()

        # Old tool completely gone
        assert "mcp_live_srv_old_tool" not in mock_registry.get_all_tool_names()
        assert "mcp_live_srv_old_tool" not in mock_toolsets["hermes-cli"]["tools"]

        # New tool registered
        assert "mcp_live_srv_new_tool" in mock_registry.get_all_tool_names()
        assert "mcp_live_srv_new_tool" in mock_toolsets["hermes-cli"]["tools"]
        assert server._registered_tool_names == ["mcp_live_srv_new_tool"]


class TestMessageHandler:
    """Tests for MCPServerTask._make_message_handler dispatch."""

    @pytest.mark.asyncio
    async def test_dispatches_tool_list_changed(self):
        from tools.mcp_tool import _MCP_NOTIFICATION_TYPES
        if not _MCP_NOTIFICATION_TYPES:
            pytest.skip("MCP SDK ToolListChangedNotification not available")

        from mcp.types import ServerNotification, ToolListChangedNotification

        server = MCPServerTask("notif_srv")
        with patch.object(MCPServerTask, "_refresh_tools", new_callable=AsyncMock) as mock_refresh:
            handler = server._make_message_handler()
            notification = ServerNotification(
                root=ToolListChangedNotification(method="notifications/tools/list_changed")
            )
            await handler(notification)
            mock_refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ignores_exceptions_and_other_messages(self):
        server = MCPServerTask("notif_srv")
        with patch.object(MCPServerTask, "_refresh_tools", new_callable=AsyncMock) as mock_refresh:
            handler = server._make_message_handler()
            # Exceptions should not trigger refresh
            await handler(RuntimeError("connection dead"))
            # Unknown message types should not trigger refresh
            await handler({"jsonrpc": "2.0", "result": "ok"})
            mock_refresh.assert_not_awaited()


class TestDeregister:
    """Tests for ToolRegistry.deregister."""

    def test_removes_tool(self):
        reg = ToolRegistry()
        reg.register(name="foo", toolset="ts1", schema={}, handler=lambda x: x)
        assert "foo" in reg.get_all_tool_names()
        reg.deregister("foo")
        assert "foo" not in reg.get_all_tool_names()

    def test_cleans_up_toolset_check(self):
        reg = ToolRegistry()
        check = lambda: True  # noqa: E731
        reg.register(name="foo", toolset="ts1", schema={}, handler=lambda x: x, check_fn=check)
        assert reg.is_toolset_available("ts1")
        reg.deregister("foo")
        # Toolset check should be gone since no tools remain
        assert "ts1" not in reg._toolset_checks

    def test_preserves_toolset_check_if_other_tools_remain(self):
        reg = ToolRegistry()
        check = lambda: True  # noqa: E731
        reg.register(name="foo", toolset="ts1", schema={}, handler=lambda x: x, check_fn=check)
        reg.register(name="bar", toolset="ts1", schema={}, handler=lambda x: x)
        reg.deregister("foo")
        # bar still in ts1, so check should remain
        assert "ts1" in reg._toolset_checks

    def test_noop_for_unknown_tool(self):
        reg = ToolRegistry()
        reg.deregister("nonexistent")  # Should not raise
