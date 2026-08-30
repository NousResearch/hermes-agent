"""Tests for MCP dynamic tool discovery (notifications/tools/list_changed)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask
from tools import mcp_tool_discovery
from tools.mcp_tool_registration import _forget_mcp_tool_server, _register_server_tools
from tools.registry import ToolRegistry


def _make_mcp_tool(name: str, desc: str = "", annotations=None):
    return SimpleNamespace(
        name=name,
        description=desc,
        inputSchema=None,
        annotations=annotations,
    )


class TestRegisterServerTools:
    """Tests for the extracted _register_server_tools helper."""

    @pytest.fixture
    def mock_registry(self):
        return ToolRegistry()

    def test_exposes_live_server_aliases(self, mock_registry):
        """Registered MCP tools are reachable via live raw-server aliases."""
        server = MCPServerTask("my_srv")
        server._tools = [_make_mcp_tool("my_tool", "desc")]
        server.session = MagicMock()
        from toolsets import resolve_toolset, validate_toolset

        with patch("tools.registry.registry", mock_registry):
            registered = _register_server_tools("my_srv", server, {})
            assert "mcp__my_srv__my_tool" in registered
            assert "mcp__my_srv__my_tool" in mock_registry.get_all_tool_names()
            assert validate_toolset("my_srv") is True
            assert "mcp__my_srv__my_tool" in resolve_toolset("my_srv")


class TestRefreshTools:
    """Tests for MCPServerTask._refresh_tools nuke-and-repave cycle."""

    @pytest.fixture
    def mock_registry(self):
        return ToolRegistry()

    @pytest.mark.asyncio
    async def test_nuke_and_repave(self, mock_registry):
        """Old tools are removed and new tools registered on refresh."""
        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {}
        from toolsets import resolve_toolset

        # Seed initial state: one old tool registered
        mock_registry.register(
            name="mcp__live_srv__old_tool", toolset="mcp-live_srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )
        server._registered_tool_names = ["mcp__live_srv__old_tool"]

        # New tool list from server
        new_tool = _make_mcp_tool("new_tool", "new behavior")
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(tools=[new_tool])
            )
        )

        with patch("tools.registry.registry", mock_registry):
            await server._refresh_tools()
            assert "mcp__live_srv__old_tool" not in mock_registry.get_all_tool_names()
            assert "mcp__live_srv__old_tool" not in resolve_toolset("live_srv")
            assert "mcp__live_srv__new_tool" in mock_registry.get_all_tool_names()
            assert "mcp__live_srv__new_tool" in resolve_toolset("live_srv")
            assert server._registered_tool_names == ["mcp__live_srv__new_tool"]

    @pytest.mark.asyncio
    async def test_read_only_to_write_refresh_fails_closed(self, mock_registry):
        """A live annotation downgrade removes execute_code exposure first."""
        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {"tools": {"resources": False, "prompts": False}}
        server.session = MagicMock()
        server._tools = [
            _make_mcp_tool(
                "query", annotations=SimpleNamespace(readOnlyHint=True)
            )
        ]
        registry_name = "mcp__live_srv__query"

        try:
            with patch("tools.registry.registry", mock_registry):
                server._registered_tool_names = _register_server_tools(
                    "live_srv", server, server._config
                )
                assert mcp_tool_discovery.get_read_only_mcp_tools().get(registry_name) == "live_srv"

                server.session = SimpleNamespace(
                    list_tools=AsyncMock(
                        return_value=SimpleNamespace(
                            tools=[
                                _make_mcp_tool(
                                    "query",
                                    annotations=SimpleNamespace(
                                        readOnlyHint=False
                                    ),
                                )
                            ]
                        )
                    )
                )
                await server._refresh_tools()

                assert registry_name in mock_registry.get_all_tool_names()
                assert registry_name not in mcp_tool_discovery.get_read_only_mcp_tools()
        finally:
            _forget_mcp_tool_server(registry_name)


class TestMessageHandler:
    """Tests for MCPServerTask._make_message_handler dispatch."""

    @pytest.mark.asyncio
    async def test_dispatches_tool_list_changed(self):
        from tools.mcp_tool import _MCP_NOTIFICATION_TYPES
        if not _MCP_NOTIFICATION_TYPES:
            pytest.skip("MCP SDK ToolListChangedNotification not available")

        from mcp.types import ServerNotification, ToolListChangedNotification

        server = MCPServerTask("notif_srv")
        # Product now schedules the refresh as a background task (see
        # _schedule_tools_refresh in mcp_tool.py ~L918) rather than awaiting
        # it directly, to avoid wedging the stdio JSON-RPC stream. Patch at
        # the scheduler seam so we can still assert dispatch happened without
        # reaching into asyncio.create_task internals.
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule:
            handler = server._make_message_handler()
            notification = ToolListChangedNotification(
                method="notifications/tools/list_changed"
            )
            if hasattr(ServerNotification, "model_validate"):
                # mcp < 2.0 wrapped notifications in a RootModel; 2.0 made
                # ServerNotification a plain union of the concrete types, which
                # has no constructor to wrap with.
                notification = ServerNotification(root=notification)
            await handler(notification)
            mock_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignores_exceptions_and_other_messages(self):
        server = MCPServerTask("notif_srv")
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule:
            handler = server._make_message_handler()
            # Exceptions should not trigger refresh
            await handler(RuntimeError("connection dead"))
            # Unknown message types should not trigger refresh
            await handler({"jsonrpc": "2.0", "result": "ok"})
            mock_schedule.assert_not_called()


class TestDeregister:
    """Tests for ToolRegistry.deregister."""

    def test_removes_tool(self):
        reg = ToolRegistry()
        reg.register(name="foo", toolset="ts1", schema={}, handler=lambda x: x)
        assert "foo" in reg.get_all_tool_names()
        reg.deregister("foo")
        assert "foo" not in reg.get_all_tool_names()


    def test_noop_for_unknown_tool(self):
        reg = ToolRegistry()
        reg.deregister("nonexistent")  # Should not raise
