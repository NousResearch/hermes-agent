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
    async def test_skips_refresh_before_session_ready(self, mock_registry):
        """Startup notifications can arrive before ClientSession is assigned."""
        server = MCPServerTask("early_srv")
        server._config = {}
        server.session = None
        server._registered_tool_names = ["mcp_early_srv_old_tool"]

        with patch("tools.registry.registry", mock_registry):
            await server._refresh_tools()

        assert server._registered_tool_names == ["mcp_early_srv_old_tool"]

    @pytest.mark.asyncio
    async def test_refresh_tools_ignores_closed_session(self, mock_registry):
        """A queued dynamic refresh can race with shutdown/reconnect."""
        server = MCPServerTask("srv")
        server.session = None
        server._registered_tool_names = ["mcp__srv__old"]

        with patch("tools.registry.registry", mock_registry):
            await server._refresh_tools()

        assert server._registered_tool_names == ["mcp__srv__old"]

    @pytest.mark.asyncio
    async def test_refresh_resolves_session_after_waiting_for_rpc_lock(self):
        """A reconnect while queued must query the replacement session."""
        server = MCPServerTask("srv")
        old_session = MagicMock()
        old_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
        new_session = MagicMock()
        new_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
        server.session = old_session

        await server._rpc_lock.acquire()
        task = asyncio.create_task(server._refresh_tools())
        await asyncio.sleep(0)
        server._adopt_session(new_session)
        server._rpc_lock.release()
        await task

        old_session.list_tools.assert_not_awaited()
        new_session.list_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_discards_response_from_superseded_session(self, mock_registry):
        """An old in-flight response must not mutate the current registry."""
        started = asyncio.Event()
        finish = asyncio.Event()
        server = MCPServerTask("srv")
        old_session = MagicMock()

        async def list_tools():
            started.set()
            await finish.wait()
            return SimpleNamespace(tools=[_make_mcp_tool("stale_tool")])

        old_session.list_tools = AsyncMock(side_effect=list_tools)
        server.session = old_session
        server._registered_tool_names = ["mcp__srv__old"]
        mock_registry.register(
            name="mcp__srv__old", toolset="mcp-srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )

        with patch("tools.registry.registry", mock_registry):
            task = asyncio.create_task(server._refresh_tools())
            await started.wait()
            server._adopt_session(MagicMock())
            finish.set()
            await task

        assert server._registered_tool_names == ["mcp__srv__old"]
        assert "mcp__srv__stale_tool" not in mock_registry.get_all_tool_names()

    @pytest.mark.asyncio
    async def test_refresh_uses_locked_capability_snapshot(self):
        """Capability must come from the locked generation, not the old one."""
        server = MCPServerTask("srv")
        old_session = MagicMock()
        old_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
        new_session = MagicMock()
        new_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
        server.session = old_session
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace(), prompts=None)
        )

        await server._rpc_lock.acquire()
        task = asyncio.create_task(server._refresh_tools())
        await asyncio.sleep(0)
        server._adopt_session(new_session)
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=None, prompts=SimpleNamespace())
        )
        server._rpc_lock.release()
        await task

        old_session.list_tools.assert_not_awaited()
        new_session.list_tools.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config",
        [{}, {"url": "https://mcp.example.test/mcp"}],
        ids=["stdio", "streamable-http"],
    )
    async def test_refresh_skips_closed_transport_during_teardown(
        self, mock_registry, caplog, config
    ):
        """ClientSession can stay referenced while its transport is dying."""
        from anyio import ClosedResourceError

        started = asyncio.Event()
        finish = asyncio.Event()
        server = MCPServerTask("srv")
        server._config = config
        session = MagicMock()

        async def list_tools():
            started.set()
            await finish.wait()
            raise ClosedResourceError()

        session.list_tools = list_tools
        server.session = session
        server._registered_tool_names = ["mcp__srv__old"]
        mock_registry.register(
            name="mcp__srv__old", toolset="mcp-srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )

        with patch("tools.registry.registry", mock_registry), caplog.at_level("DEBUG"):
            task = server._schedule_tools_refresh()
            await started.wait()
            # Same interleaving as ClientSession.__aexit__: pointer remains,
            # transport is invalidated, generation is bumped first.
            server._invalidate_session_generation()
            finish.set()
            await task

        assert server._registered_tool_names == ["mcp__srv__old"]
        assert "mcp__srv__old" in mock_registry.get_all_tool_names()
        assert "dynamic tool refresh failed" not in caplog.text
        assert task not in server._pending_refresh_tasks

    @pytest.mark.asyncio
    async def test_refresh_propagates_genuine_protocol_errors(self, caplog):
        """Do not swallow tools/list failures that are not teardown."""
        server = MCPServerTask("srv")
        session = MagicMock()
        session.list_tools = AsyncMock(side_effect=RuntimeError("tools/list exploded"))
        server.session = session

        with pytest.raises(RuntimeError, match="tools/list exploded"):
            await server._refresh_tools()

        with caplog.at_level("ERROR"):
            await server._refresh_tools_task()
        assert "dynamic tool refresh failed" in caplog.text

    @pytest.mark.asyncio
    async def test_refresh_logs_unexpected_closed_transport(self, caplog):
        """Closed transport is not skipped unless that generation is dying."""
        from anyio import ClosedResourceError

        server = MCPServerTask("srv")
        session = MagicMock()
        session.list_tools = AsyncMock(side_effect=ClosedResourceError())
        server.session = session

        with pytest.raises(ClosedResourceError):
            await server._refresh_tools()

        with caplog.at_level("ERROR"):
            await server._refresh_tools_task()
        assert "dynamic tool refresh failed" in caplog.text

    @pytest.mark.asyncio
    async def test_lifecycle_return_invalidates_generation_before_context_exit(self):
        """Shutdown/reconnect must bump generation while the pointer is still set."""
        server = MCPServerTask("srv")
        live = object()
        server.session = live
        generation = server._session_generation
        server._shutdown_event.set()

        reason = await server._wait_for_lifecycle_event()

        assert reason == "shutdown"
        assert server.session is live
        assert server._session_generation != generation
        assert server._refresh_generation_stale(generation, live)

    @pytest.mark.asyncio
    async def test_stdio_recycle_invalidates_generation(self):
        server = MCPServerTask("srv")
        server.session = object()
        generation = server._session_generation

        server._mark_stdio_recycled("idle_timeout_seconds")

        assert server.session is None
        assert server._session_generation != generation


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
