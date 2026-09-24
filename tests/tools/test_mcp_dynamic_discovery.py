"""Tests for MCP dynamic tool discovery (notifications/tools/list_changed)."""

import asyncio
import logging
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
        """#109824: a list_changed refresh that lands while the transport is being rebuilt
        (``session is None`` between teardown and the next handshake) must complete cleanly
        instead of raising ``AttributeError: 'NoneType' object has no attribute 'list_tools'``.
        The reconnect's own discovery re-lists tools, so the refresh is a no-op that leaves the
        previous registration — and the owned-names bookkeeping — intact."""
        from toolsets import resolve_toolset

        server = MCPServerTask("restart_srv")
        server._config = {}
        server._tools = [_make_mcp_tool("old_tool", "")]
        server.session = SimpleNamespace(list_tools=AsyncMock())
        with patch("tools.registry.registry", mock_registry):
            server._registered_tool_names = _register_server_tools("restart_srv", server, {})
            server.session = None  # what every run() teardown leaves behind

            await server._refresh_tools()

            assert "mcp__restart_srv__old_tool" in mock_registry.get_all_tool_names()
            assert "mcp__restart_srv__old_tool" in resolve_toolset("restart_srv")
            assert server._registered_tool_names == ["mcp__restart_srv__old_tool"]
            assert len(server._tools) == 1

    @pytest.mark.asyncio
    async def test_background_refresh_survives_restart_reconnect_cycle(self, mock_registry, caplog):
        """#109824 end-to-end shape: a refresh task scheduled on the dying transport runs after
        run() has set ``session = None``. The background task must finish without logging the
        'dynamic tool refresh failed' crash, and once the transport reconnects a refresh on the
        live session must republish normally."""
        server = MCPServerTask("reconnect_srv")
        server._config = {}
        server._tools = [_make_mcp_tool("old_tool", "")]
        server._registered_tool_names = ["mcp__reconnect_srv__old_tool"]
        new_tool = _make_mcp_tool("new_tool", "")
        with patch("tools.registry.registry", mock_registry):
            mock_registry.register(
                name="mcp__reconnect_srv__old_tool", toolset="mcp-reconnect_srv",
                schema={}, handler=lambda x: x, check_fn=lambda: True,
                is_async=False, description="", emoji="",
            )
            server.session = None
            with caplog.at_level(logging.ERROR, logger="tools.mcp_tool"):
                task = server._schedule_tools_refresh()
                assert task in server._pending_refresh_tasks
                await asyncio.gather(task, return_exceptions=True)
            assert not task.cancelled()
            assert task.exception() is None
            assert task not in server._pending_refresh_tasks
            assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
            assert server._registered_tool_names == ["mcp__reconnect_srv__old_tool"]

            server.session = SimpleNamespace(
                list_tools=AsyncMock(return_value=SimpleNamespace(tools=[new_tool]))
            )
            await server._refresh_tools()
            assert server._registered_tool_names == ["mcp__reconnect_srv__new_tool"]
            assert "mcp__reconnect_srv__old_tool" not in mock_registry.get_all_tool_names()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", [{}, {"url": "https://example.test/mcp"}], ids=["stdio", "http"])
    async def test_refresh_during_transport_exit_does_not_log_or_publish(self, mock_registry, caplog, config):
        """A referenced ClientSession may already be closing when tools/list fails."""
        from anyio import ClosedResourceError

        server = MCPServerTask("retiring_srv")
        server._config = config
        started, finish = asyncio.Event(), asyncio.Event()

        async def list_tools():
            started.set()
            await finish.wait()
            raise ClosedResourceError()

        session = SimpleNamespace(list_tools=list_tools)
        server._registered_tool_names = ["mcp__retiring_srv__old"]
        mock_registry.register(
            name="mcp__retiring_srv__old", toolset="mcp-retiring_srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )
        with patch("tools.registry.registry", mock_registry), caplog.at_level(logging.ERROR), \
             patch.object(MCPServerTask, "_negotiate_session", new_callable=AsyncMock), \
             patch.object(MCPServerTask, "_discover_tools", new_callable=AsyncMock):
            serving = asyncio.create_task(server._serve_session(session, 1))
            await asyncio.wait_for(server._ready.wait(), 2)
            task = server._schedule_tools_refresh()
            await asyncio.wait_for(started.wait(), 2)
            server._shutdown_event.set()
            # The run loop returns before ClientSession.__aexit__ clears the pointer.
            assert await asyncio.wait_for(serving, 2) == "shutdown"
            # Session withdrawal must happen before transport __aexit__; otherwise
            # a refresh arriving *after* this return reuses the dying pointer.
            assert server.session is None
            finish.set()
            await task
            assert "dynamic tool refresh failed" not in caplog.text
            caplog.clear()
            late = server._schedule_tools_refresh()
            await late
        assert "dynamic tool refresh failed" not in caplog.text
        assert server._registered_tool_names == ["mcp__retiring_srv__old"]
        assert "mcp__retiring_srv__old" in mock_registry.get_all_tool_names()
        assert task not in server._pending_refresh_tasks

        # A closed transport on the current generation is still an error, not a blanket skip.
        live = MCPServerTask("live_srv")
        live.session = SimpleNamespace(list_tools=AsyncMock(side_effect=ClosedResourceError()))
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            await live._schedule_tools_refresh()
        assert "dynamic tool refresh failed" in caplog.text

        # Cancellation cleanup can suspend before _serve_session's finally runs.
        waiting = MCPServerTask("waiting_srv")
        waiting._config = config
        entered, resume = asyncio.Event(), asyncio.Event()

        async def held_cleanup(*waiters):
            entered.set()
            await resume.wait()
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)

        with patch.object(MCPServerTask, "_negotiate_session", new_callable=AsyncMock), \
             patch.object(MCPServerTask, "_discover_tools", new_callable=AsyncMock), \
             patch.object(MCPServerTask, "_cancel_waiters", side_effect=held_cleanup):
            serving = asyncio.create_task(waiting._serve_session(session, 1))
            await asyncio.wait_for(waiting._ready.wait(), 2)
            waiting._shutdown_event.set()
            try:
                await asyncio.wait_for(entered.wait(), 2)
                assert waiting.session is None
            finally:
                resume.set()
                await asyncio.wait_for(serving, 2)

        # Failed discovery must withdraw its newly published pointer before
        # unwinding the ClientSession context, not only bump a generation.
        discovering = MCPServerTask("discovery_srv")
        with patch.object(MCPServerTask, "_negotiate_session", new_callable=AsyncMock), \
             patch.object(MCPServerTask, "_discover_tools", new_callable=AsyncMock,
                          side_effect=RuntimeError("discovery failed")):
            with pytest.raises(RuntimeError, match="discovery failed"):
                await discovering._serve_session(session, 1)
        assert discovering.session is None

    @pytest.mark.asyncio
    async def test_superseded_refresh_is_discarded_before_later_refresh(self, mock_registry):
        """An old response must not publish; a subsequent live refresh can publish."""
        server = MCPServerTask("swap_srv")
        started, finish = asyncio.Event(), asyncio.Event()

        async def old_list():
            started.set()
            await finish.wait()
            return SimpleNamespace(tools=[_make_mcp_tool("stale")])

        old_session = SimpleNamespace(list_tools=old_list)
        server.session = old_session
        with patch("tools.registry.registry", mock_registry):
            task = asyncio.create_task(server._refresh_tools())
            await asyncio.wait_for(started.wait(), 2)
            # Simulate pointer replacement; reconnect's authoritative registry
            # publication is a separate concern (#111227).
            replacement = SimpleNamespace(list_tools=AsyncMock(
                return_value=SimpleNamespace(tools=[_make_mcp_tool("replacement")])
            ))
            server.session = replacement
            finish.set()
            await task
            assert "mcp__swap_srv__stale" not in mock_registry.get_all_tool_names()
            await server._refresh_tools()
            assert "mcp__swap_srv__replacement" in mock_registry.get_all_tool_names()
            assert server._registered_tool_names == ["mcp__swap_srv__replacement"]

        # A queued refresh must evaluate capabilities of the session it actually calls.
        cap_server = MCPServerTask("cap_srv")
        cap_server.session = SimpleNamespace(list_tools=AsyncMock())
        cap_server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        await cap_server._rpc_lock.acquire()
        queued = asyncio.create_task(cap_server._refresh_tools())
        await asyncio.sleep(0)
        prompt_only = SimpleNamespace(list_tools=AsyncMock())
        cap_server.session = prompt_only
        cap_server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=None, prompts=SimpleNamespace())
        )
        cap_server._rpc_lock.release()
        await queued
        prompt_only.list_tools.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refresh_after_session_restored_succeeds(self, mock_registry):
        """The skip applies only while the session is genuinely absent: once the restart
        re-establishes it, a refresh from an empty registration publishes the live tools."""
        from toolsets import resolve_toolset

        server = MCPServerTask("restored_srv")
        server._config = {}
        server.session = None
        server._registered_tool_names = []
        server._tools = []
        with patch("tools.registry.registry", mock_registry):
            server.session = SimpleNamespace(
                list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_make_mcp_tool("live_tool", "")]))
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


