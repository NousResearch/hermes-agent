import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _register_server_tools
from tools.registry import ToolRegistry


def _make_mcp_tool(name: str, desc: str = ""):
    return SimpleNamespace(name=name, description=desc, inputSchema=None)

class TestDynamicToolDiscovery:
    
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

    def test_register_server_tools_injects_hermes_toolsets(self, mock_registry, mock_toolsets):
        """Validates the extracted _register_server_tools handles hermes-* injection."""
        server = MCPServerTask("my_srv")
        server._tools = [_make_mcp_tool("my_tool", "desc")]
        server.session = MagicMock()
        
        with patch("tools.registry.registry", mock_registry), \
            patch("toolsets.create_custom_toolset") as mock_create, \
            patch.dict("toolsets.TOOLSETS", mock_toolsets, clear=True):
            
            registered = _register_server_tools("my_srv", server, {})
            
        assert "mcp_my_srv_my_tool" in registered
        assert "mcp_my_srv_my_tool" in mock_registry.get_all_tool_names()
        
        # Verify TOOLSETS injection
        assert "mcp_my_srv_my_tool" in mock_toolsets["hermes-cli"]["tools"]
        assert "mcp_my_srv_my_tool" in mock_toolsets["hermes-telegram"]["tools"]
        assert "mcp_my_srv_my_tool" not in mock_toolsets["custom-toolset"]["tools"]

    @pytest.mark.asyncio
    async def test_refresh_tools_nuke_and_repave(self, mock_registry, mock_toolsets):
        """Verifies _refresh_tools removes old tools and registers new ones."""
        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {}
        
        # New State reported by MCP
        new_tool = _make_mcp_tool("new_tool", "new behavior")
        
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(tools=[new_tool])
            )
        )
        
        # Initial State
        server._registered_tool_names = ["mcp_live_srv_old_tool"]
        mock_registry.register(
            name="mcp_live_srv_old_tool", toolset="mcp-live_srv", schema={}, 
            handler=lambda x: x, check_fn=lambda: True, is_async=False, description="", emoji=""
        )
        mock_toolsets["hermes-cli"]["tools"].append("mcp_live_srv_old_tool")
        
        # New State reported by MCP
        
        server.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[new_tool])
        )
        
        with patch("tools.registry.registry", mock_registry), \
            patch("toolsets.create_custom_toolset"), \
            patch.dict("toolsets.TOOLSETS", mock_toolsets, clear=True):
                 
            await server._refresh_tools()
            
        # Assert Old Tool is COMPLETELY gone
        assert "mcp_live_srv_old_tool" not in mock_registry.get_all_tool_names()
        assert "mcp_live_srv_old_tool" not in mock_toolsets["hermes-cli"]["tools"]
        
        # Assert New Tool is registered
        assert "mcp_live_srv_new_tool" in mock_registry.get_all_tool_names()
        assert "mcp_live_srv_new_tool" in mock_toolsets["hermes-cli"]["tools"]
        assert server._registered_tool_names == ["mcp_live_srv_new_tool"]

    @pytest.mark.asyncio
    async def test_message_handler_dispatches_tool_list_changed(self):
        from tools.mcp_tool import _MCP_NOTIFICATION_TYPES
        if not _MCP_NOTIFICATION_TYPES:
            pytest.skip("MCP SDK ToolListChangedNotification is not installed")

        from mcp.types import ServerNotification, ToolListChangedNotification

        server = MCPServerTask("notif_srv")

        with patch.object(MCPServerTask, '_refresh_tools', new_callable=AsyncMock) as mock_refresh:
            handler = server._make_message_handler()

            notification = ServerNotification(
                root=ToolListChangedNotification(method="notifications/tools/list_changed")
            )

            await handler(notification)
            mock_refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_message_handler_ignores_exceptions_and_other_messages(self):
        server = MCPServerTask("notif_srv")

        with patch.object(MCPServerTask, '_refresh_tools', new_callable=AsyncMock) as mock_refresh:
            handler = server._make_message_handler()
            
            await handler(RuntimeError("connection dead"))
            await handler({"jsonrpc": "2.0", "result": "ok"})
            
            mock_refresh.assert_not_awaited()