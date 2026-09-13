"""Tests for MCP dynamic tool discovery (notifications/tools/list_changed)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_registration import _register_server_tools
from tools.registry import ToolRegistry


def _make_mcp_tool(name: str, desc: str = ""):
    return SimpleNamespace(name=name, description=desc, inputSchema=None)


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

    def test_colliding_static_toolset_name_merges_both_tool_sets(self, mock_registry):
        """An MCP server named after a built-in toolset must not be shadowed.

        Regression: an MCP server registered as `homeassistant` (colliding
        with the static `homeassistant` toolset) had its tools silently
        dropped because get_toolset() returned the static definition without
        consulting the alias registered by _register_server_tools().
        """
        from toolsets import TOOLSETS, get_toolset, resolve_toolset

        assert "homeassistant" in TOOLSETS  # collision premise
        static_tools = set(TOOLSETS["homeassistant"]["tools"])

        server = MCPServerTask("homeassistant")
        server._tools = [_make_mcp_tool("get_entities", "List HA entities")]
        server.session = MagicMock()

        with patch("tools.registry.registry", mock_registry):
            registered = _register_server_tools("homeassistant", server, {})
            assert "mcp__homeassistant__get_entities" in registered

            ts = get_toolset("homeassistant")
            # Static built-ins are still present...
            assert static_tools <= set(ts["tools"])
            # ...and the MCP server's tools are no longer shadowed.
            assert "mcp__homeassistant__get_entities" in ts["tools"]
            assert "mcp__homeassistant__get_entities" in resolve_toolset("homeassistant")


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
    async def test_restart_with_none_session_skips_without_crash(self, mock_registry):
        """#109824: a list_changed refresh that lands while the transport is
        being rebuilt (``session is None`` between teardown and the next
        handshake) must complete cleanly instead of raising
        ``AttributeError: 'NoneType' object has no attribute 'list_tools'``.

        The reconnect's own _discover_tools re-lists tools, so the refresh is
        a no-op and the previous registration is left intact.
        """
        from toolsets import resolve_toolset

        server = MCPServerTask("restart_srv")
        server._config = {}
        old_tool = _make_mcp_tool("old_tool", "")
        server._tools = [old_tool]
        server.session = SimpleNamespace(list_tools=AsyncMock())
        with patch("tools.registry.registry", mock_registry):
            # Seed initial state the way the initial connect would: a live
            # registration with the old tool published.
            server._registered_tool_names = _register_server_tools(
                "restart_srv", server, {}
            )
            # Teardown state that every run() cycle leaves behind:
            server.session = None

            # Must NOT raise (the #109824 crash) and must not touch the registry.
            await server._refresh_tools()
            assert "mcp__restart_srv__old_tool" in mock_registry.get_all_tool_names()
            assert "mcp__restart_srv__old_tool" in resolve_toolset("restart_srv")
            assert server._registered_tool_names == ["mcp__restart_srv__old_tool"]
            # The pre-restart tool list was left untouched by the skipped refresh.
            assert len(server._tools) == 1

    @pytest.mark.asyncio
    async def test_background_refresh_survives_restart_reconnect_cycle(self, mock_registry):
        """#109824 end-to-end shape: a refresh task scheduled on the dying
        transport is still alive when run() sets ``session = None`` and then
        re-establishes the session. _schedule_tools_refresh must not crash the
        task, and once a live session exists the refresh completes normally.
        """
        server = MCPServerTask("reconnect_srv")
        server._config = {}
        old_tool = _make_mcp_tool("old_tool", "")
        new_tool = _make_mcp_tool("new_tool", "")
        server._tools = [old_tool]
        server._registered_tool_names = ["mcp__reconnect_srv__old_tool"]
        with patch("tools.registry.registry", mock_registry):
            mock_registry.register(
                name="mcp__reconnect_srv__old_tool", toolset="mcp-reconnect_srv",
                schema={}, handler=lambda x: x, check_fn=lambda: True,
                is_async=False, description="", emoji="",
            )
            # Simulate run()'s teardown: session dropped, tools deregistered
            # by park/shutdown. A list_changed refresh from the old transport
            # is still pending in _pending_refresh_tasks.
            server.session = None
            # Schedule the in-flight refresh (as a list_changed would have) and
            # let it observe the None session — it must be a clean no-op.
            task = server._schedule_tools_refresh()
            assert task in server._pending_refresh_tasks  # strongly referenced
            await asyncio.gather(task, return_exceptions=True)
            # No crash recorded: the task completed without raising, and the
            # done-callback removed it from the pending set.
            assert not task.cancelled()
            assert task.exception() is None
            assert task not in server._pending_refresh_tasks
            # Now the transport reconnects and re-publishes tools; a subsequent
            # refresh on the LIVE session must succeed and republish.
            server.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[new_tool])))
            await server._refresh_tools()
            assert server._registered_tool_names == ["mcp__reconnect_srv__new_tool"]

    @pytest.mark.asyncio
    async def test_refresh_after_session_restored_succeeds(self, mock_registry):
        """Once the restart re-establishes the session, _refresh_tools works
        normally (the skip is only while the session is genuinely absent)."""
        from toolsets import resolve_toolset

        server = MCPServerTask("restored_srv")
        server._config = {}
        server.session = None
        server._registered_tool_names = []
        server._tools = []

        live_tool = _make_mcp_tool("live_tool", "")
        with patch("tools.registry.registry", mock_registry):
            # Restore the session, then refresh.
            server.session = SimpleNamespace(
                list_tools=AsyncMock(return_value=SimpleNamespace(tools=[live_tool]))
            )
            await server._refresh_tools()
            assert "mcp__restored_srv__live_tool" in mock_registry.get_all_tool_names()
            assert "mcp__restored_srv__live_tool" in resolve_toolset("restored_srv")
            assert server._registered_tool_names == ["mcp__restored_srv__live_tool"]


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
