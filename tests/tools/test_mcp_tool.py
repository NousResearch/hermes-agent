"""Tests for the MCP (Model Context Protocol) client support.

All tests use mocks -- no real MCP servers or subprocesses are started.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mcp_tool(name="read_file", description="Read a file", input_schema=None):
    """Create a fake MCP Tool object matching the SDK interface."""
    tool = SimpleNamespace()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
        },
        "required": ["path"],
    }
    return tool


def _make_call_result(text="file contents here", is_error=False):
    """Create a fake MCP CallToolResult."""
    block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[block], isError=is_error)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadMCPConfig:
    def test_no_config_returns_empty(self):
        """No mcp_servers key in config -> empty dict."""
        with patch("tools.mcp_tool.load_config", create=True) as mock_lc:
            # Patch the actual import inside the function
            with patch("hermes_cli.config.load_config", return_value={"model": "test"}):
                from tools.mcp_tool import _load_mcp_config
                result = _load_mcp_config()
                assert result == {}

    def test_valid_config_parsed(self):
        """Valid mcp_servers config is returned as-is."""
        servers = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "env": {},
            }
        }
        with patch("hermes_cli.config.load_config", return_value={"mcp_servers": servers}):
            from tools.mcp_tool import _load_mcp_config
            result = _load_mcp_config()
            assert "filesystem" in result
            assert result["filesystem"]["command"] == "npx"

    def test_mcp_servers_not_dict_returns_empty(self):
        """mcp_servers set to non-dict value -> empty dict."""
        with patch("hermes_cli.config.load_config", return_value={"mcp_servers": "invalid"}):
            from tools.mcp_tool import _load_mcp_config
            result = _load_mcp_config()
            assert result == {}


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

class TestSchemaConversion:
    def test_converts_mcp_tool_to_hermes_schema(self):
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(name="read_file", description="Read a file")
        schema = _convert_mcp_schema("filesystem", mcp_tool)

        assert schema["name"] == "mcp_filesystem_read_file"
        assert schema["description"] == "Read a file"
        assert "properties" in schema["parameters"]

    def test_empty_input_schema_gets_default(self):
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(name="ping", description="Ping", input_schema=None)
        mcp_tool.inputSchema = None
        schema = _convert_mcp_schema("test", mcp_tool)

        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"]["properties"] == {}

    def test_tool_name_prefix_format(self):
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(name="list_dir")
        schema = _convert_mcp_schema("my_server", mcp_tool)

        assert schema["name"] == "mcp_my_server_list_dir"

    def test_hyphens_sanitized_to_underscores(self):
        """Hyphens in tool/server names are replaced with underscores for LLM compat."""
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(name="get-sum")
        schema = _convert_mcp_schema("my-server", mcp_tool)

        assert schema["name"] == "mcp_my_server_get_sum"
        assert "-" not in schema["name"]


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

class TestCheckFunction:
    def test_disconnected_returns_false(self):
        from tools.mcp_tool import _make_check_fn, _connections

        # Ensure no connection exists
        _connections.pop("test_server", None)
        check = _make_check_fn("test_server")
        assert check() is False

    def test_connected_returns_true(self):
        from tools.mcp_tool import _make_check_fn, _connections, MCPConnection

        conn = MCPConnection(
            server_name="test_server",
            session=MagicMock(),
            stack=MagicMock(),
        )
        _connections["test_server"] = conn
        try:
            check = _make_check_fn("test_server")
            assert check() is True
        finally:
            _connections.pop("test_server", None)

    def test_session_none_returns_false(self):
        from tools.mcp_tool import _make_check_fn, _connections, MCPConnection

        conn = MCPConnection(
            server_name="test_server",
            session=None,
            stack=MagicMock(),
        )
        _connections["test_server"] = conn
        try:
            check = _make_check_fn("test_server")
            assert check() is False
        finally:
            _connections.pop("test_server", None)


# ---------------------------------------------------------------------------
# Tool handler (async)
# ---------------------------------------------------------------------------

class TestToolHandler:
    """Tool handlers are sync functions that schedule work on the MCP loop."""

    def _patch_mcp_loop(self, coro_side_effect=None):
        """Return a patch for _run_on_mcp_loop that runs the coroutine directly."""
        def fake_run(coro, timeout=30):
            return asyncio.get_event_loop().run_until_complete(coro)
        if coro_side_effect:
            return patch("tools.mcp_tool._run_on_mcp_loop", side_effect=coro_side_effect)
        return patch("tools.mcp_tool._run_on_mcp_loop", side_effect=fake_run)

    def test_successful_call(self):
        from tools.mcp_tool import _make_tool_handler, _connections, MCPConnection

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("hello world", is_error=False)
        )
        conn = MCPConnection("test_srv", session=mock_session, stack=MagicMock())
        _connections["test_srv"] = conn

        try:
            handler = _make_tool_handler("test_srv", "greet")
            with self._patch_mcp_loop():
                result = json.loads(handler({"name": "world"}))
            assert result["result"] == "hello world"
            mock_session.call_tool.assert_called_once_with("greet", arguments={"name": "world"})
        finally:
            _connections.pop("test_srv", None)

    def test_mcp_error_result(self):
        from tools.mcp_tool import _make_tool_handler, _connections, MCPConnection

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("something went wrong", is_error=True)
        )
        conn = MCPConnection("test_srv", session=mock_session, stack=MagicMock())
        _connections["test_srv"] = conn

        try:
            handler = _make_tool_handler("test_srv", "fail_tool")
            with self._patch_mcp_loop():
                result = json.loads(handler({}))
            assert "error" in result
            assert "something went wrong" in result["error"]
        finally:
            _connections.pop("test_srv", None)

    def test_disconnected_server(self):
        from tools.mcp_tool import _make_tool_handler, _connections

        _connections.pop("ghost", None)
        handler = _make_tool_handler("ghost", "any_tool")
        # Disconnected check happens before _run_on_mcp_loop, no patch needed
        result = json.loads(handler({}))
        assert "error" in result
        assert "not connected" in result["error"]

    def test_exception_during_call(self):
        from tools.mcp_tool import _make_tool_handler, _connections, MCPConnection

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))
        conn = MCPConnection("test_srv", session=mock_session, stack=MagicMock())
        _connections["test_srv"] = conn

        try:
            handler = _make_tool_handler("test_srv", "broken_tool")
            with self._patch_mcp_loop():
                result = json.loads(handler({}))
            assert "error" in result
            assert "connection lost" in result["error"]
        finally:
            _connections.pop("test_srv", None)


# ---------------------------------------------------------------------------
# Tool registration (discovery + register)
# ---------------------------------------------------------------------------

class TestDiscoverAndRegister:
    def test_tools_registered_in_registry(self):
        """_discover_and_register_server registers tools with correct names."""
        from tools.registry import ToolRegistry, registry as real_registry
        from tools.mcp_tool import _discover_and_register_server, _connections, MCPConnection

        mock_registry = ToolRegistry()
        mock_tools = [
            _make_mcp_tool("read_file", "Read a file"),
            _make_mcp_tool("write_file", "Write a file"),
        ]

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        async def fake_connect(name, config):
            return MCPConnection(name, session=mock_session, stack=MagicMock())

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("fs", {"command": "npx", "args": []})
            )

        assert "mcp_fs_read_file" in registered
        assert "mcp_fs_write_file" in registered
        assert "mcp_fs_read_file" in mock_registry.get_all_tool_names()
        assert "mcp_fs_write_file" in mock_registry.get_all_tool_names()

        _connections.pop("fs", None)

    def test_toolset_created(self):
        """A custom toolset is created for the MCP server."""
        from tools.mcp_tool import _discover_and_register_server, _connections, MCPConnection

        mock_tools = [_make_mcp_tool("ping", "Ping")]

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        async def fake_connect(name, config):
            return MCPConnection(name, session=mock_session, stack=MagicMock())

        mock_create = MagicMock()
        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("toolsets.create_custom_toolset", mock_create):
            asyncio.run(
                _discover_and_register_server("myserver", {"command": "test"})
            )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs[1]["name"] == "mcp-myserver" or call_kwargs[0][0] == "mcp-myserver"

        _connections.pop("myserver", None)

    def test_schema_format_correct(self):
        """Registered schemas have the correct format."""
        from tools.registry import ToolRegistry, registry as real_registry
        from tools.mcp_tool import _discover_and_register_server, _connections, MCPConnection

        mock_registry = ToolRegistry()
        mock_tools = [_make_mcp_tool("do_thing", "Do something")]

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        async def fake_connect(name, config):
            return MCPConnection(name, session=mock_session, stack=MagicMock())

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            asyncio.run(
                _discover_and_register_server("srv", {"command": "test"})
            )

        entry = mock_registry._tools.get("mcp_srv_do_thing")
        assert entry is not None
        assert entry.schema["name"] == "mcp_srv_do_thing"
        assert "parameters" in entry.schema
        assert entry.is_async is False
        assert entry.toolset == "mcp-srv"

        _connections.pop("srv", None)


# ---------------------------------------------------------------------------
# _connect_server (SDK interaction)
# ---------------------------------------------------------------------------

class TestConnectServer:
    def test_calls_sdk_with_correct_params(self):
        """_connect_server creates StdioServerParameters and calls stdio_client."""
        from tools.mcp_tool import _connect_server, MCPConnection

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        mock_read = MagicMock()
        mock_write = MagicMock()

        with patch("tools.mcp_tool.StdioServerParameters") as mock_params, \
             patch("tools.mcp_tool.stdio_client") as mock_stdio, \
             patch("tools.mcp_tool.ClientSession") as mock_cs, \
             patch("tools.mcp_tool.AsyncExitStack") as mock_stack_cls:

            mock_stack = MagicMock()
            mock_stack.enter_async_context = AsyncMock(
                side_effect=[(mock_read, mock_write), mock_session]
            )
            mock_stack_cls.return_value = mock_stack

            conn = asyncio.run(_connect_server("test_srv", {
                "command": "npx",
                "args": ["-y", "some-server"],
                "env": {"MY_KEY": "secret"},
            }))

        # StdioServerParameters called with correct values
        mock_params.assert_called_once_with(
            command="npx",
            args=["-y", "some-server"],
            env={"MY_KEY": "secret"},
        )
        # ClientSession created with the streams
        mock_cs.assert_called_once_with(mock_read, mock_write)
        # initialize() was called
        mock_session.initialize.assert_called_once()
        # Returned connection is valid
        assert conn.server_name == "test_srv"
        assert conn.session is mock_session

    def test_no_command_raises(self):
        """Missing 'command' in config raises ValueError."""
        from tools.mcp_tool import _connect_server

        with pytest.raises(ValueError, match="no 'command'"):
            asyncio.run(_connect_server("bad", {"args": []}))

    def test_empty_env_passed_as_none(self):
        """Empty env dict is passed as None to StdioServerParameters."""
        from tools.mcp_tool import _connect_server

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        with patch("tools.mcp_tool.StdioServerParameters") as mock_params, \
             patch("tools.mcp_tool.stdio_client"), \
             patch("tools.mcp_tool.ClientSession", return_value=mock_session), \
             patch("tools.mcp_tool.AsyncExitStack") as mock_stack_cls:

            mock_stack = MagicMock()
            mock_stack.enter_async_context = AsyncMock(
                side_effect=[
                    (MagicMock(), MagicMock()),
                    mock_session,
                ]
            )
            mock_stack_cls.return_value = mock_stack

            asyncio.run(_connect_server("srv", {
                "command": "node",
                "env": {},
            }))

        # Empty dict -> None
        assert mock_params.call_args[1]["env"] is None or \
               mock_params.call_args.kwargs.get("env") is None


# ---------------------------------------------------------------------------
# discover_mcp_tools toolset injection
# ---------------------------------------------------------------------------

class TestToolsetInjection:
    def test_mcp_tools_added_to_platform_toolsets(self):
        """Discovered MCP tools are injected into hermes-cli and platform toolsets."""
        from tools.mcp_tool import _connections, MCPConnection

        mock_tools = [_make_mcp_tool("list_files", "List files")]
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        async def fake_connect(name, config):
            return MCPConnection(name, session=mock_session, stack=MagicMock())

        fake_toolsets = {
            "hermes-cli": {"tools": ["terminal", "web_search"], "description": "CLI", "includes": []},
            "hermes-telegram": {"tools": ["terminal"], "description": "Telegram", "includes": []},
        }
        fake_config = {
            "fs": {"command": "npx", "args": []},
        }

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.mcp_tool.TOOLSETS", fake_toolsets, create=True), \
             patch("toolsets.TOOLSETS", fake_toolsets):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()

        assert "mcp_fs_list_files" in result
        assert "mcp_fs_list_files" in fake_toolsets["hermes-cli"]["tools"]
        assert "mcp_fs_list_files" in fake_toolsets["hermes-telegram"]["tools"]
        # Original tools preserved
        assert "terminal" in fake_toolsets["hermes-cli"]["tools"]

        _connections.pop("fs", None)

    def test_server_connection_failure_skipped(self):
        """If one server fails to connect, others still proceed."""
        from tools.mcp_tool import _connections, MCPConnection

        mock_tools = [_make_mcp_tool("ping", "Ping")]
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        call_count = 0

        async def flaky_connect(name, config):
            nonlocal call_count
            call_count += 1
            if name == "broken":
                raise ConnectionError("cannot reach server")
            return MCPConnection(name, session=mock_session, stack=MagicMock())

        fake_config = {
            "broken": {"command": "bad"},
            "good": {"command": "npx", "args": []},
        }
        fake_toolsets = {
            "hermes-cli": {"tools": [], "description": "CLI", "includes": []},
        }

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=flaky_connect), \
             patch("toolsets.TOOLSETS", fake_toolsets):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()

        # Only good server's tool registered
        assert "mcp_good_ping" in result
        assert "mcp_broken_ping" not in result
        assert call_count == 2  # Both were attempted

        _connections.pop("good", None)


# ---------------------------------------------------------------------------
# Graceful fallback
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    def test_mcp_unavailable_returns_empty(self):
        """When _MCP_AVAILABLE is False, discover_mcp_tools is a no-op."""
        with patch("tools.mcp_tool._MCP_AVAILABLE", False):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()
            assert result == []

    def test_no_servers_returns_empty(self):
        """No MCP servers configured -> empty list."""
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value={}):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()
            assert result == []


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:
    def test_no_connections_safe(self):
        """shutdown_mcp_servers with no connections does nothing."""
        from tools.mcp_tool import shutdown_mcp_servers, _connections

        _connections.clear()
        shutdown_mcp_servers()  # Should not raise

    def test_shutdown_clears_connections(self):
        """shutdown_mcp_servers closes stacks and clears the dict."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import shutdown_mcp_servers, _connections, MCPConnection

        _connections.clear()
        mock_stack = MagicMock()
        mock_stack.aclose = AsyncMock()
        conn = MCPConnection("test", session=MagicMock(), stack=mock_stack)
        _connections["test"] = conn

        # Start a real background loop so shutdown can schedule on it
        mcp_mod._ensure_mcp_loop()
        try:
            shutdown_mcp_servers()
        finally:
            # _stop_mcp_loop is called by shutdown, but ensure cleanup
            mcp_mod._mcp_loop = None
            mcp_mod._mcp_thread = None

        assert len(_connections) == 0
        mock_stack.aclose.assert_called_once()

    def test_shutdown_handles_errors(self):
        """shutdown_mcp_servers handles errors during close gracefully."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import shutdown_mcp_servers, _connections, MCPConnection

        _connections.clear()
        mock_stack = MagicMock()
        mock_stack.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
        conn = MCPConnection("broken", session=MagicMock(), stack=mock_stack)
        _connections["broken"] = conn

        mcp_mod._ensure_mcp_loop()
        try:
            shutdown_mcp_servers()  # Should not raise
        finally:
            mcp_mod._mcp_loop = None
            mcp_mod._mcp_thread = None

        assert len(_connections) == 0
