"""Tests for MCP dynamic tool discovery (notifications/tools/list_changed)."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import (
    MCPServerTask,
    _discover_and_register_server,
    _register_server_tools,
)
from tools.registry import ToolRegistry


def _make_mcp_tool(name: str, desc: str = "", input_schema=None):
    return SimpleNamespace(name=name, description=desc, inputSchema=input_schema)


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

    def test_registration_failure_rolls_back_newly_published_tools(
        self, mock_registry
    ):
        server = MCPServerTask("my_srv")
        server._tools = [
            _make_mcp_tool("first", "first behavior"),
            _make_mcp_tool("second", "second behavior"),
        ]
        original_register = mock_registry.register
        call_count = 0

        def _register(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("injected registration failure")
            return original_register(**kwargs)

        with (
            patch("tools.registry.registry", mock_registry),
            patch.object(mock_registry, "register", side_effect=_register),
            pytest.raises(RuntimeError, match="injected registration failure"),
        ):
            _register_server_tools("my_srv", server, {})

        assert mock_registry.get_all_tool_names() == []
        assert server._last_registered_tool_contracts == {}

    def test_registration_failure_restores_previous_snapshot(self, mock_registry):
        server = MCPServerTask("my_srv")
        server._config = {}
        server._tools = [
            _make_mcp_tool("first", "original first"),
            _make_mcp_tool("second", "original second"),
        ]

        with patch("tools.registry.registry", mock_registry):
            server._synchronize_registered_tools()

        original_register = mock_registry.register
        call_count = 0

        def _register(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("injected registration failure")
            return original_register(**kwargs)

        server._tools = [
            _make_mcp_tool("first", "changed first"),
            _make_mcp_tool("second", "changed second"),
        ]
        with (
            patch("tools.registry.registry", mock_registry),
            patch.object(mock_registry, "register", side_effect=_register),
            pytest.raises(RuntimeError, match="injected registration failure"),
        ):
            server._synchronize_registered_tools()

        assert server._registered_tool_names == [
            "mcp__my_srv__first",
            "mcp__my_srv__second",
        ]
        assert mock_registry.get_entry("mcp__my_srv__first").schema[
            "description"
        ] == "original first"
        assert mock_registry.get_entry("mcp__my_srv__second").schema[
            "description"
        ] == "original second"
        assert server._last_registered_tool_contracts[
            "mcp__my_srv__first"
        ]["description"] == "original first"


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
    async def test_warns_when_existing_tool_contract_changes(self, mock_registry, caplog):
        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {}
        server._tools = [_make_mcp_tool("stable_tool", "original behavior")]
        server._registered_tool_names = ["mcp__live_srv__stable_tool"]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[_make_mcp_tool("stable_tool", "changed behavior")]
                )
            )
        )

        with (
            patch("tools.registry.registry", mock_registry),
            caplog.at_level(logging.WARNING, logger="tools.mcp_tool"),
        ):
            await server._refresh_tools()

        assert any(
            "modified: mcp__live_srv__stable_tool" in record.getMessage()
            for record in caplog.records
        )
        assert any(
            "MCP rug pull" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_warns_when_existing_tool_input_schema_changes(self, mock_registry, caplog):
        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {}
        server._tools = [
            _make_mcp_tool(
                "stable_tool",
                "stable behavior",
                {"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]
        server._registered_tool_names = ["mcp__live_srv__stable_tool"]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        _make_mcp_tool(
                            "stable_tool",
                            "stable behavior",
                            {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "recursive": {"type": "boolean"},
                                },
                            },
                        )
                    ]
                )
            )
        )

        with (
            patch("tools.registry.registry", mock_registry),
            caplog.at_level(logging.WARNING, logger="tools.mcp_tool"),
        ):
            await server._refresh_tools()

        assert any(
            "modified: mcp__live_srv__stable_tool" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_does_not_warn_when_required_order_changes(self, mock_registry, caplog):
        properties = {
            "path": {"type": "string"},
            "recursive": {"type": "boolean"},
        }
        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {}
        server._tools = [
            _make_mcp_tool(
                "stable_tool",
                "stable behavior",
                {
                    "type": "object",
                    "properties": properties,
                    "required": ["path", "recursive"],
                },
            )
        ]
        server._registered_tool_names = ["mcp__live_srv__stable_tool"]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        _make_mcp_tool(
                            "stable_tool",
                            "stable behavior",
                            {
                                "type": "object",
                                "properties": properties,
                                "required": ["recursive", "path"],
                            },
                        )
                    ]
                )
            )
        )

        with (
            patch("tools.registry.registry", mock_registry),
            caplog.at_level(logging.WARNING, logger="tools.mcp_tool"),
        ):
            await server._refresh_tools()

        assert not any(
            "modified: mcp__live_srv__stable_tool" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_warns_when_instance_data_required_order_changes(self, mock_registry, caplog):
        def _schema(required_order):
            return {
                "type": "object",
                "properties": {
                    "policy": {
                        "const": {
                            "type": "object",
                            "properties": {
                                "safe": {"type": "string"},
                                "dangerous": {"type": "string"},
                            },
                            "required": required_order,
                        }
                    }
                },
            }

        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {}
        server._tools = [
            _make_mcp_tool(
                "stable_tool",
                "stable behavior",
                _schema(["safe", "dangerous"]),
            )
        ]
        server._registered_tool_names = ["mcp__live_srv__stable_tool"]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        _make_mcp_tool(
                            "stable_tool",
                            "stable behavior",
                            _schema(["dangerous", "safe"]),
                        )
                    ]
                )
            )
        )

        with (
            patch("tools.registry.registry", mock_registry),
            caplog.at_level(logging.WARNING, logger="tools.mcp_tool"),
        ):
            await server._refresh_tools()

        assert any(
            "modified: mcp__live_srv__stable_tool" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.parametrize(
        ("keyword", "old_value", "new_value"),
        [
            ("const", True, 1),
            ("default", False, 0),
            ("enum", [True], [1]),
            ("examples", [False], [0]),
        ],
    )
    @pytest.mark.asyncio
    async def test_warns_when_json_instance_value_types_change(
        self, mock_registry, caplog, keyword, old_value, new_value
    ):
        def _schema(value):
            return {
                "type": "object",
                "properties": {
                    "policy": {keyword: value},
                },
            }

        server = MCPServerTask("live_srv")
        server._config = {}
        server._tools = [
            _make_mcp_tool(
                "stable_tool", "stable behavior", _schema(old_value)
            )
        ]
        server._registered_tool_names = ["mcp__live_srv__stable_tool"]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        _make_mcp_tool(
                            "stable_tool", "stable behavior", _schema(new_value)
                        )
                    ]
                )
            )
        )

        with patch("tools.registry.registry", mock_registry):
            with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
                await server._refresh_tools()

        assert any(
            "modified: mcp__live_srv__stable_tool" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_warns_when_tool_contract_changes_across_reconnect(
        self, mock_registry, caplog
    ):
        server = MCPServerTask("live_srv")
        server._ready.set()
        server._config = {}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._tools = [_make_mcp_tool("stable_tool", "original behavior")]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[_make_mcp_tool("stable_tool", "changed behavior")]
                )
            )
        )

        with (
            patch("tools.registry.registry", mock_registry),
            caplog.at_level(logging.WARNING, logger="tools.mcp_tool"),
        ):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            await server._discover_tools()

        assert any(
            "modified after reconnect: mcp__live_srv__stable_tool" in record.getMessage()
            for record in caplog.records
        )
        assert any(
            "MCP rug pull" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_warns_when_capability_change_replaces_remote_with_utility(
        self, mock_registry, caplog
    ):
        tool_name = "mcp__live_srv__list_resources"
        server = MCPServerTask("live_srv")
        server._ready.set()
        server._config = {}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._tools = [
            _make_mcp_tool("list_resources", "remote list behavior")
        ]
        server.session = SimpleNamespace()

        with patch("tools.registry.registry", mock_registry):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            remote_handler = mock_registry.get_entry(tool_name).handler
            server.initialize_result = SimpleNamespace(
                capabilities=SimpleNamespace(resources=SimpleNamespace())
            )
            with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
                await server._discover_tools()

        assert server._registered_tool_names == [
            "mcp__live_srv__list_resources",
            "mcp__live_srv__read_resource",
        ]
        assert mock_registry.get_entry(tool_name).handler is not remote_handler
        assert sum(
            f"modified after reconnect: {tool_name}" in record.getMessage()
            for record in caplog.records
        ) == 1

    @pytest.mark.asyncio
    async def test_no_tools_capability_reconnect_uses_transition_lock(
        self, mock_registry
    ):
        server = MCPServerTask("live_srv")
        server._ready.set()
        server._config = {}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(resources=SimpleNamespace())
        )
        server.session = SimpleNamespace()

        with patch("tools.registry.registry", mock_registry):
            async with server._refresh_lock:
                discovery_task = asyncio.create_task(server._discover_tools())
                await asyncio.sleep(0)
                assert not discovery_task.done()
            await discovery_task

    @pytest.mark.asyncio
    async def test_modified_reconnect_deregisters_removed_tool(
        self, mock_registry
    ):
        stable_name = "mcp__live_srv__stable_tool"
        removed_name = "mcp__live_srv__removed_tool"
        server = MCPServerTask("live_srv")
        server._ready.set()
        server._config = {}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._tools = [
            _make_mcp_tool("stable_tool", "original behavior"),
            _make_mcp_tool("removed_tool", "removed behavior"),
        ]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[_make_mcp_tool("stable_tool", "changed behavior")]
                )
            )
        )

        with patch("tools.registry.registry", mock_registry):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            await server._discover_tools()

        assert server._registered_tool_names == [stable_name]
        assert stable_name in mock_registry.get_all_tool_names()
        assert removed_name not in mock_registry.get_all_tool_names()
        assert removed_name in server._last_registered_tool_contracts

    @pytest.mark.asyncio
    async def test_reconnect_synchronizes_additions_and_removals_without_mutation(
        self, mock_registry
    ):
        stable_name = "mcp__live_srv__stable_tool"
        removed_name = "mcp__live_srv__removed_tool"
        added_name = "mcp__live_srv__added_tool"
        server = MCPServerTask("live_srv")
        server._ready.set()
        server._config = {}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._tools = [
            _make_mcp_tool("stable_tool", "stable behavior"),
            _make_mcp_tool("removed_tool", "removed behavior"),
        ]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        _make_mcp_tool("stable_tool", "stable behavior"),
                        _make_mcp_tool("added_tool", "added behavior"),
                    ]
                )
            )
        )

        with patch("tools.registry.registry", mock_registry):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            await server._discover_tools()

        assert set(server._registered_tool_names) == {stable_name, added_name}
        assert stable_name in mock_registry.get_all_tool_names()
        assert added_name in mock_registry.get_all_tool_names()
        assert removed_name not in mock_registry.get_all_tool_names()
        assert removed_name in server._last_registered_tool_contracts

    @pytest.mark.parametrize(
        ("second_description", "expected_warning_count"),
        [
            ("changed behavior", 1),
            ("original behavior", 2),
        ],
    )
    @pytest.mark.parametrize("parked", [False, True])
    @pytest.mark.asyncio
    async def test_serializes_reconnect_discovery_with_dynamic_refresh(
        self,
        mock_registry,
        caplog,
        second_description,
        expected_warning_count,
        parked,
    ):
        server = MCPServerTask("live_srv")
        server._ready.set()
        server._config = {}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._tools = [_make_mcp_tool("stable_tool", "original behavior")]

        first_list_started = asyncio.Event()
        release_first_list = asyncio.Event()
        list_call_count = 0

        async def _list_tools():
            nonlocal list_call_count
            list_call_count += 1
            if list_call_count == 1:
                first_list_started.set()
                await release_first_list.wait()
                description = "changed behavior"
            else:
                description = second_description
            return SimpleNamespace(
                tools=[_make_mcp_tool("stable_tool", description)]
            )

        server.session = SimpleNamespace(list_tools=_list_tools)

        with (
            patch("tools.registry.registry", mock_registry),
            caplog.at_level(logging.WARNING, logger="tools.mcp_tool"),
        ):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            if parked:
                server._deregister_tools()

            discovery_task = asyncio.create_task(server._discover_tools())
            await first_list_started.wait()
            refresh_task = asyncio.create_task(server._refresh_tools())
            await asyncio.sleep(0)
            release_first_list.set()
            await asyncio.gather(discovery_task, refresh_task)

        rug_pull_warnings = [
            record
            for record in caplog.records
            if "MCP rug pull" in record.getMessage()
        ]
        assert len(rug_pull_warnings) == expected_warning_count
        assert mock_registry._tools[
            "mcp__live_srv__stable_tool"
        ].schema["description"] == second_description
        assert server._last_registered_tool_contracts[
            "mcp__live_srv__stable_tool"
        ]["description"] == second_description

    @pytest.mark.asyncio
    async def test_serializes_initial_registration_with_dynamic_refresh(
        self, mock_registry
    ):
        tool_name = "mcp__startup_srv__stable_tool"
        server = MCPServerTask("startup_srv")
        server._config = {}
        server._tools = [_make_mcp_tool("stable_tool", "original behavior")]
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )

        refresh_list_started = asyncio.Event()
        release_refresh_list = asyncio.Event()
        connect_returned = asyncio.Event()

        async def _list_tools():
            refresh_list_started.set()
            await release_refresh_list.wait()
            return SimpleNamespace(
                tools=[_make_mcp_tool("stable_tool", "changed behavior")]
            )

        async def _connect_server(_name, _config):
            connect_returned.set()
            return server

        server.session = SimpleNamespace(list_tools=_list_tools)

        with (
            patch("tools.registry.registry", mock_registry),
            patch("tools.mcp_tool._connect_server", _connect_server),
            patch.dict("tools.mcp_tool._servers", {}, clear=True),
        ):
            refresh_task = asyncio.create_task(server._refresh_tools())
            await refresh_list_started.wait()
            registration_task = asyncio.create_task(
                _discover_and_register_server("startup_srv", {})
            )
            await connect_returned.wait()
            for _ in range(3):
                await asyncio.sleep(0)

            registration_completed_while_refresh_pending = registration_task.done()
            tool_exposed_while_refresh_pending = (
                tool_name in mock_registry.get_all_tool_names()
            )

            release_refresh_list.set()
            await asyncio.gather(refresh_task, registration_task)

        assert not registration_completed_while_refresh_pending
        assert not tool_exposed_while_refresh_pending
        assert mock_registry._tools[tool_name].schema["description"] == (
            "changed behavior"
        )

    @pytest.mark.asyncio
    async def test_startup_refresh_timeout_does_not_leave_cached_server(
        self, mock_registry
    ):
        server = MCPServerTask("startup_srv")
        server._config = {"connect_timeout": 1}
        server._tools = [_make_mcp_tool("stable_tool", "original behavior")]
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        refresh_list_started = asyncio.Event()

        async def _list_tools():
            refresh_list_started.set()
            await asyncio.Event().wait()

        async def _connect_server(_name, _config):
            return server

        server.session = SimpleNamespace(list_tools=_list_tools)

        with (
            patch("tools.registry.registry", mock_registry),
            patch("tools.mcp_tool._connect_server", _connect_server),
            patch.dict("tools.mcp_tool._servers", {}, clear=True) as servers,
        ):
            refresh_task = asyncio.create_task(server._refresh_tools())
            await refresh_list_started.wait()
            with pytest.raises(asyncio.TimeoutError):
                await _discover_and_register_server(
                    "startup_srv", {"connect_timeout": 0.01}
                )
            refresh_task.cancel()
            await asyncio.gather(refresh_task, return_exceptions=True)

        assert "startup_srv" not in servers
        assert server._registered_tool_names == []
        assert mock_registry.get_all_tool_names() == []

    @pytest.mark.asyncio
    async def test_startup_timeout_does_not_wait_for_cancellation_resistant_task(
        self, mock_registry
    ):
        server = MCPServerTask("startup_srv")
        server._config = {"connect_timeout": 1}
        server._tools = [_make_mcp_tool("stable_tool", "original behavior")]
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        refresh_list_started = asyncio.Event()
        release_server_task = asyncio.Event()

        async def _list_tools():
            refresh_list_started.set()
            await asyncio.Event().wait()

        async def _resist_cancellation():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release_server_task.wait()

        server.session = SimpleNamespace(list_tools=_list_tools)
        server._task = asyncio.create_task(_resist_cancellation())

        async def _connect_server(_name, _config):
            return server

        with (
            patch("tools.registry.registry", mock_registry),
            patch("tools.mcp_tool._connect_server", _connect_server),
            patch.dict("tools.mcp_tool._servers", {}, clear=True) as servers,
        ):
            refresh_task = asyncio.create_task(server._refresh_tools())
            await refresh_list_started.wait()
            discovery_task = asyncio.create_task(
                _discover_and_register_server(
                    "startup_srv", {"connect_timeout": 0.01}
                )
            )
            done, _ = await asyncio.wait({discovery_task}, timeout=0.1)
            try:
                assert discovery_task in done
                with pytest.raises(asyncio.TimeoutError):
                    await discovery_task
                assert "startup_srv" not in servers
                assert server._registered_tool_names == []
                assert mock_registry.get_all_tool_names() == []
            finally:
                server._task.cancel()
                release_server_task.set()
                refresh_task.cancel()
                await asyncio.gather(
                    discovery_task,
                    refresh_task,
                    server._task,
                    return_exceptions=True,
                )

    @pytest.mark.asyncio
    async def test_refresh_list_timeout_releases_transition_locks(self):
        server = MCPServerTask("live_srv")
        server._config = {"connect_timeout": 0.01}

        async def _list_tools():
            await asyncio.Event().wait()

        server.session = SimpleNamespace(list_tools=_list_tools)

        with pytest.raises(asyncio.TimeoutError):
            await server._refresh_tools()

        assert not server._refresh_lock.locked()
        assert not server._rpc_lock.locked()

    @pytest.mark.asyncio
    async def test_does_not_warn_for_never_exposed_tool_across_reconnect(self, caplog):
        server = MCPServerTask("live_srv")
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._tools = [_make_mcp_tool("hidden_tool", "original behavior")]
        server._registered_tool_names = []
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[_make_mcp_tool("hidden_tool", "changed behavior")]
                )
            )
        )

        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            await server._discover_tools()

        assert not any(
            "modified after reconnect" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_parked_reconnect_ignores_changed_filtered_tool(
        self, mock_registry, caplog
    ):
        visible_name = "mcp__live_srv__visible_tool"
        hidden_name = "mcp__live_srv__hidden_tool"
        server = MCPServerTask("live_srv")
        server._ready.set()
        server._config = {"tools": {"include": ["visible_tool"]}}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._tools = [
            _make_mcp_tool("visible_tool", "stable behavior"),
            _make_mcp_tool("hidden_tool", "original hidden behavior"),
        ]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        _make_mcp_tool("visible_tool", "stable behavior"),
                        _make_mcp_tool("hidden_tool", "changed hidden behavior"),
                    ]
                )
            )
        )

        with patch("tools.registry.registry", mock_registry):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            server._deregister_tools()
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
                await server._discover_tools()

        assert server._registered_tool_names == [visible_name]
        assert hidden_name not in server._last_registered_tool_contracts
        assert not any(
            hidden_name in record.getMessage()
            or "MCP rug pull" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_warns_when_collided_tool_is_reexposed(self, mock_registry, caplog):
        tool_name = "mcp__live_srv__stable_tool"
        server = MCPServerTask("live_srv")
        server._ready.set()
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._config = {}
        server._tools = [_make_mcp_tool("stable_tool", "original behavior")]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[_make_mcp_tool("stable_tool", "changed behavior")]
                )
            )
        )

        with patch("tools.registry.registry", mock_registry):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            server._deregister_tools()
            mock_registry.register(
                name=tool_name,
                toolset="builtin",
                schema={},
                handler=lambda _args: None,
            )

            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
                await server._discover_tools()
            assert server._registered_tool_names == []
            assert not any(
                "MCP rug pull" in record.getMessage()
                for record in caplog.records
            )

            caplog.clear()
            mock_registry.deregister(tool_name)
            with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
                await server._discover_tools()

        assert server._registered_tool_names == [tool_name]
        assert sum(
            f"modified after reconnect: {tool_name}" in record.getMessage()
            for record in caplog.records
        ) == 1

    @pytest.mark.asyncio
    async def test_dynamic_removal_preserves_new_registry_owner(self, mock_registry):
        tool_name = "mcp__live_srv__shared_tool"
        server = MCPServerTask("live_srv")
        server._ready.set()
        server._config = {}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._tools = [_make_mcp_tool("shared_tool", "remote behavior")]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[]))
        )

        with patch("tools.registry.registry", mock_registry):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            mock_registry.deregister(tool_name)
            mock_registry.register(
                name=tool_name,
                toolset="builtin-owner",
                schema={
                    "name": tool_name,
                    "description": "replacement behavior",
                    "parameters": {"type": "object", "properties": {}},
                },
                handler=lambda **_: None,
            )

            await server._refresh_tools()

        assert mock_registry.get_toolset_for_tool(tool_name) == "builtin-owner"
        assert server._registered_tool_names == []
        assert tool_name in server._last_registered_tool_contracts

    def test_deregistration_preserves_new_registry_owner(self, mock_registry):
        tool_name = "mcp__live_srv__shared_tool"
        server = MCPServerTask("live_srv")
        server._registered_tool_names = [tool_name]
        mock_registry.register(
            name=tool_name,
            toolset="builtin-owner",
            schema={
                "name": tool_name,
                "description": "replacement behavior",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda **_: None,
        )

        with (
            patch("tools.registry.registry", mock_registry),
            patch.dict(
                "tools.mcp_tool._mcp_tool_server_names",
                {tool_name: "live_srv"},
                clear=True,
            ) as provenance,
        ):
            server._deregister_tools()
            assert tool_name not in provenance

        assert mock_registry.get_toolset_for_tool(tool_name) == "builtin-owner"
        assert server._registered_tool_names == []

    @pytest.mark.parametrize("cleanup_path", ["refresh", "deregister"])
    def test_cleanup_preserves_replacement_mcp_provenance(
        self, mock_registry, cleanup_path
    ):
        """Same-name instance replacement keeps provenance for the live owner.

        Distinct raw server names that sanitize to one tool prefix still map to
        different toolsets (``mcp-{raw}``). Current registry policy rejects
        cross-MCP ownership transfer, so this covers same-name instance swap
        under one raw server name (the realistic reconnect/replace path).
        """
        from tools.mcp_tool import (
            has_registered_mcp_tools,
            is_mcp_tool_parallel_safe,
        )
        import tools.mcp_tool as mcp_tool

        tool_name = "mcp__shared_owner__shared_tool"
        old_server = MCPServerTask("shared_owner")
        old_server._config = {}
        old_server._tools = [_make_mcp_tool("shared_tool", "old behavior")]
        new_server = MCPServerTask("shared_owner")
        new_server._config = {"supports_parallel_tool_calls": True}
        new_server._tools = [_make_mcp_tool("shared_tool", "new behavior")]

        with (
            patch("tools.registry.registry", mock_registry),
            patch.dict("tools.mcp_tool._mcp_tool_server_names", {}, clear=True),
            patch("tools.mcp_tool._parallel_safe_servers", {"shared_owner"}),
            patch.object(mcp_tool, "_servers", {"shared_owner": old_server}),
        ):
            old_server._synchronize_registered_tools()
            mcp_tool._servers["shared_owner"] = new_server
            new_server._synchronize_registered_tools()
            assert mock_registry.get_toolset_for_tool(tool_name) == (
                "mcp-shared_owner"
            )
            assert mock_registry.get_entry(tool_name).schema["description"] == (
                "new behavior"
            )
            if cleanup_path == "refresh":
                old_server._tools = []
                old_server._synchronize_registered_tools()
            else:
                old_server._deregister_tools()

            assert mock_registry.get_toolset_for_tool(tool_name) == (
                "mcp-shared_owner"
            )
            assert mock_registry.get_entry(tool_name).schema["description"] == (
                "new behavior"
            )
            assert has_registered_mcp_tools()
            assert is_mcp_tool_parallel_safe(tool_name)

    def test_registration_skips_server_name_already_connecting(self):
        import tools.mcp_tool as mcp_tool

        config = {"same": {"command": "ignored"}}
        with (
            patch.object(mcp_tool, "_MCP_AVAILABLE", True),
            patch.object(
                mcp_tool,
                "_filter_suspicious_mcp_servers",
                return_value=config,
            ),
            patch.object(mcp_tool, "_connect_cooldown_active", return_value=False),
            patch.object(mcp_tool, "_servers", {}),
            patch.object(mcp_tool, "_server_connecting", {"same"}),
            patch.object(mcp_tool, "_server_connect_errors", {}),
            patch.object(mcp_tool, "_parallel_safe_servers", set()),
            patch.object(mcp_tool, "_ensure_mcp_loop") as ensure_loop,
            patch.object(mcp_tool, "_run_on_mcp_loop") as run_on_loop,
            patch.object(
                mcp_tool,
                "_existing_tool_names",
                return_value=["existing"],
            ),
        ):
            registered = mcp_tool.register_mcp_servers(config)

        assert registered == ["existing"]
        ensure_loop.assert_not_called()
        run_on_loop.assert_not_called()

    def test_outer_discovery_timeout_releases_connecting_name(self):
        import tools.mcp_tool as mcp_tool

        config = {"same": {"command": "ignored"}}
        connecting = set()
        with (
            patch.object(mcp_tool, "_MCP_AVAILABLE", True),
            patch.object(
                mcp_tool,
                "_filter_suspicious_mcp_servers",
                return_value=config,
            ),
            patch.object(mcp_tool, "_servers", {}),
            patch.object(mcp_tool, "_server_connecting", connecting),
            patch.object(mcp_tool, "_server_connect_errors", {}),
            patch.object(mcp_tool, "_parallel_safe_servers", set()),
            patch.object(mcp_tool, "_connect_cooldown_active", return_value=False),
            patch.object(mcp_tool, "_ensure_mcp_loop"),
            patch.object(
                mcp_tool,
                "_run_on_mcp_loop",
                side_effect=TimeoutError("outer discovery timed out"),
            ),
        ):
            with pytest.raises(TimeoutError, match="outer discovery timed out"):
                mcp_tool.register_mcp_servers(config)

        assert connecting == set()

    def test_parallel_safety_uses_raw_owner_across_sanitized_collision(self):
        import tools.mcp_tool as mcp_tool

        tool_name = "mcp__a_b__shared_tool"
        config = {
            "a-b": {
                "command": "ignored",
                "supports_parallel_tool_calls": True,
            },
            "a_b": {
                "command": "ignored",
                "supports_parallel_tool_calls": False,
            },
        }
        parallel_safe_servers = set()
        provenance = {}
        with (
            patch.object(mcp_tool, "_MCP_AVAILABLE", True),
            patch.object(
                mcp_tool,
                "_filter_suspicious_mcp_servers",
                return_value=config,
            ),
            patch.object(mcp_tool, "_servers", {}),
            patch.object(mcp_tool, "_server_connecting", set()),
            patch.object(mcp_tool, "_server_connect_errors", {}),
            patch.object(mcp_tool, "_parallel_safe_servers", parallel_safe_servers),
            patch.object(mcp_tool, "_mcp_tool_server_names", provenance),
            patch.object(mcp_tool, "_connect_cooldown_active", return_value=False),
            patch.object(mcp_tool, "_ensure_mcp_loop"),
            patch.object(mcp_tool, "_run_on_mcp_loop"),
            patch.object(mcp_tool, "_existing_tool_names", return_value=[]),
        ):
            mcp_tool.register_mcp_servers(config)

            mcp_tool._track_mcp_tool_server(tool_name, "a_b")
            assert not mcp_tool.is_mcp_tool_parallel_safe(tool_name)

            mcp_tool._track_mcp_tool_server(tool_name, "a-b")
            assert mcp_tool.is_mcp_tool_parallel_safe(tool_name)

        assert parallel_safe_servers == {"a-b"}
        assert provenance[tool_name] == "a-b"

    def test_stale_same_name_deregistration_preserves_current_instance(
        self, mock_registry
    ):
        import tools.mcp_tool as mcp_tool

        tool_name = "mcp__same__shared_tool"
        stale_server = MCPServerTask("same")
        stale_server._config = {}
        stale_server._tools = [_make_mcp_tool("shared_tool", "stale behavior")]
        current_server = MCPServerTask("same")
        current_server._config = {}
        current_server._tools = [_make_mcp_tool("shared_tool", "current behavior")]

        with (
            patch("tools.registry.registry", mock_registry),
            patch.dict("tools.mcp_tool._mcp_tool_server_names", {}, clear=True),
            patch.object(mcp_tool, "_servers", {"same": stale_server}),
        ):
            stale_server._synchronize_registered_tools()
            mcp_tool._servers["same"] = current_server
            current_server._synchronize_registered_tools()
            stale_server._deregister_tools()

            assert mock_registry.get_entry(tool_name).schema["description"] == (
                "current behavior"
            )
            assert mcp_tool._mcp_tool_server_names[tool_name] == "same"

    @pytest.mark.asyncio
    async def test_cross_mcp_collision_warns_for_visible_contract_change(
        self, mock_registry, caplog
    ):
        """Sanitized-name collisions keep the first owner and warn on drift.

        Current registry policy rejects cross-MCP toolset shadowing, so a
        second server must not steal the model-visible entry. A differing
        contract is still a rug-pull signal and must be logged.
        """
        tool_name = "mcp__a_b__shared_tool"
        first_server = MCPServerTask("a-b")
        first_server._config = {}
        first_server._tools = [_make_mcp_tool("shared_tool", "A-safe")]
        second_server = MCPServerTask("a_b")
        second_server._config = {}
        second_server._tools = [_make_mcp_tool("shared_tool", "B-dangerous")]

        with (
            patch("tools.registry.registry", mock_registry),
            patch.dict("tools.mcp_tool._mcp_tool_server_names", {}, clear=True),
        ):
            first_server._synchronize_registered_tools()
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
                second_server._synchronize_registered_tools()

        assert mock_registry.get_entry(tool_name).schema["description"] == "A-safe"
        assert mock_registry.get_toolset_for_tool(tool_name) == "mcp-a-b"
        assert sum(
            "MCP rug pull" in record.getMessage()
            and tool_name in record.getMessage()
            for record in caplog.records
        ) == 1

    def test_snapshot_publication_is_atomic_for_registry_readers(
        self, mock_registry
    ):
        import threading

        server = MCPServerTask("atomic")
        server._config = {}
        server._tools = [
            _make_mcp_tool("first", "first behavior"),
            _make_mcp_tool("second", "second behavior"),
        ]
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(
                tools=SimpleNamespace(), resources=None, prompts=None
            )
        )
        first_published = threading.Event()
        release_publication = threading.Event()
        reader_finished = threading.Event()
        observed_names = []
        register_calls = 0
        original_register = mock_registry.register

        def blocking_register(**spec):
            nonlocal register_calls
            original_register(**spec)
            register_calls += 1
            if register_calls == 1:
                first_published.set()
                release_publication.wait(timeout=2)

        def publish_snapshot():
            server._synchronize_registered_tools()

        def read_snapshot():
            observed_names.extend(
                mock_registry.get_tool_names_for_toolset("mcp-atomic")
            )
            reader_finished.set()

        with (
            patch("tools.registry.registry", mock_registry),
            patch.object(mock_registry, "register", side_effect=blocking_register),
            patch.dict("tools.mcp_tool._mcp_tool_server_names", {}, clear=True),
        ):
            publisher = threading.Thread(target=publish_snapshot)
            reader = threading.Thread(target=read_snapshot)
            publisher.start()
            assert first_published.wait(timeout=1)
            reader.start()
            try:
                assert not reader_finished.wait(timeout=0.05)
            finally:
                release_publication.set()
                publisher.join(timeout=2)
                reader.join(timeout=2)

        assert not publisher.is_alive()
        assert not reader.is_alive()
        assert observed_names == ["mcp__atomic__first", "mcp__atomic__second"]

    def test_bounds_retained_inactive_contracts_by_count(self, mock_registry):
        server = MCPServerTask("live_srv")
        server._config = {}

        with (
            patch("tools.registry.registry", mock_registry),
            patch("tools.mcp_tool._MAX_RETAINED_INACTIVE_MCP_CONTRACTS", 2),
        ):
            for tool_name in ("one", "two", "three", "four"):
                server._tools = [_make_mcp_tool(tool_name, f"{tool_name} behavior")]
                server._synchronize_registered_tools()

        assert set(server._last_registered_tool_contracts) == {
            "mcp__live_srv__two",
            "mcp__live_srv__three",
            "mcp__live_srv__four",
        }

    def test_evicts_oversized_inactive_contract(self, mock_registry):
        tool_name = "mcp__live_srv__large_tool"
        server = MCPServerTask("live_srv")
        server._config = {}
        server._tools = [_make_mcp_tool("large_tool", "x" * 1024)]

        with (
            patch("tools.registry.registry", mock_registry),
            patch("tools.mcp_tool._MAX_RETAINED_INACTIVE_MCP_CONTRACT_BYTES", 128),
        ):
            server._synchronize_registered_tools()
            assert tool_name in server._last_registered_tool_contracts
            server._deregister_tools()

        assert tool_name not in server._last_registered_tool_contracts

    @pytest.mark.asyncio
    @pytest.mark.parametrize("refresh_method", ["_refresh_tools", "_discover_tools"])
    async def test_utility_name_collision_tracks_only_visible_contract(
        self, refresh_method, mock_registry, caplog
    ):
        tool_name = "mcp__live_srv__list_resources"
        server = MCPServerTask("live_srv")
        server._ready.set()
        server._config = {}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(
                tools=SimpleNamespace(),
                resources=SimpleNamespace(),
                prompts=None,
            )
        )
        server._tools = [_make_mcp_tool("list_resources", "remote behavior A")]
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[
                        _make_mcp_tool("list_resources", "remote behavior B")
                    ]
                )
            )
        )

        with patch("tools.registry.registry", mock_registry):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            visible_description = mock_registry._tools[tool_name].schema[
                "description"
            ]
            assert server._last_registered_tool_contracts[tool_name][
                "description"
            ] == visible_description

            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
                await getattr(server, refresh_method)()

        assert mock_registry._tools[tool_name].schema["description"] == (
            visible_description
        )
        assert server._last_registered_tool_contracts[tool_name][
            "description"
        ] == visible_description
        assert not any(
            "MCP rug pull" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_warns_when_removed_tool_is_readded_with_changed_contract(
        self, mock_registry, caplog
    ):
        tool_name = "mcp__live_srv__stable_tool"
        server = MCPServerTask("live_srv")
        server._config = {}
        server._tools = [_make_mcp_tool("stable_tool", "original behavior")]
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                side_effect=[
                    SimpleNamespace(tools=[]),
                    SimpleNamespace(
                        tools=[
                            _make_mcp_tool("stable_tool", "changed behavior")
                        ]
                    ),
                ]
            )
        )

        with patch("tools.registry.registry", mock_registry):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            await server._refresh_tools()
            assert server._registered_tool_names == []

            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
                await server._refresh_tools()

        assert server._registered_tool_names == [tool_name]
        assert any(
            f"modified: {tool_name}" in record.getMessage()
            for record in caplog.records
        )


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
            notification = ServerNotification(
                root=ToolListChangedNotification(method="notifications/tools/list_changed")
            )
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

    @pytest.mark.asyncio
    async def test_coalesces_rapid_refresh_notifications(self):
        server = MCPServerTask("notif_srv")
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()
        refresh_calls = 0

        async def blocked_refresh(_server):
            nonlocal refresh_calls
            refresh_calls += 1
            if refresh_calls == 1:
                refresh_started.set()
                await release_refresh.wait()

        with patch.object(MCPServerTask, "_refresh_tools", new=blocked_refresh):
            first_task = server._schedule_tools_refresh()
            await asyncio.wait_for(refresh_started.wait(), timeout=1)
            scheduled_tasks = [
                server._schedule_tools_refresh() for _ in range(99)
            ]
            try:
                assert all(task is first_task for task in scheduled_tasks)
                assert len(server._pending_refresh_tasks) == 1
            finally:
                release_refresh.set()
                await asyncio.gather(
                    *server._pending_refresh_tasks,
                    return_exceptions=True,
                )

        assert refresh_calls == 2
        assert not server._pending_refresh_tasks


class TestDeregister:
    """Tests for ToolRegistry.deregister."""

    def test_preserves_exposed_contracts_for_reconnect_checks(self):
        server = MCPServerTask("parked_srv")
        server._config = {}
        server.initialize_result = SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace())
        )
        server._tools = [
            _make_mcp_tool("visible_tool", "original behavior")
        ]

        with patch("tools.registry.registry", ToolRegistry()):
            server._registered_tool_names = _register_server_tools(
                server.name, server, server._config
            )
            server._deregister_tools()

        assert server._registered_tool_names == []
        assert server._last_registered_tool_contracts[
            "mcp__parked_srv__visible_tool"
        ]["description"] == "original behavior"

    def test_removes_tool(self):
        reg = ToolRegistry()
        reg.register(name="foo", toolset="ts1", schema={}, handler=lambda x: x)
        assert "foo" in reg.get_all_tool_names()
        reg.deregister("foo")
        assert "foo" not in reg.get_all_tool_names()


    def test_noop_for_unknown_tool(self):
        reg = ToolRegistry()
        reg.deregister("nonexistent")  # Should not raise
