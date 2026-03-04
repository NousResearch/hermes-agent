"""Tests for the MCP (Model Context Protocol) client support.

All tests use mocks -- no real MCP servers or subprocesses are started.
"""

import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock MCP types if mcp package is not installed
# ---------------------------------------------------------------------------

try:
    from mcp.types import CreateMessageResult, TextContent, ErrorData
except ImportError:
    # Create lightweight stand-ins so that _make_sampling_callback can be tested
    # without the real mcp package installed.
    class TextContent:
        def __init__(self, type="text", text=""):
            self.type = type
            self.text = text

    class CreateMessageResult:
        def __init__(self, role="", content=None, model="", stopReason=""):
            self.role = role
            self.content = content
            self.model = model
            self.stopReason = stopReason

    class ErrorData:
        def __init__(self, code=0, message=""):
            self.code = code
            self.message = message

    # Inject into the mcp_tool module namespace so the callback can use them
    import tools.mcp_tool as _mcp_mod
    _mcp_mod.CreateMessageResult = CreateMessageResult
    _mcp_mod.TextContent = TextContent
    _mcp_mod.ErrorData = ErrorData
    _mcp_mod._MCP_SAMPLING_TYPES = True


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


def _make_mock_server(name, session=None, tools=None):
    """Create an MCPServerTask with mock attributes for testing."""
    from tools.mcp_tool import MCPServerTask
    server = MCPServerTask(name)
    server.session = session
    server._tools = tools or []
    return server


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadMCPConfig:
    def test_no_config_returns_empty(self):
        """No mcp_servers key in config -> empty dict."""
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
        from tools.mcp_tool import _make_check_fn, _servers

        _servers.pop("test_server", None)
        check = _make_check_fn("test_server")
        assert check() is False

    def test_connected_returns_true(self):
        from tools.mcp_tool import _make_check_fn, _servers

        server = _make_mock_server("test_server", session=MagicMock())
        _servers["test_server"] = server
        try:
            check = _make_check_fn("test_server")
            assert check() is True
        finally:
            _servers.pop("test_server", None)

    def test_session_none_returns_false(self):
        from tools.mcp_tool import _make_check_fn, _servers

        server = _make_mock_server("test_server", session=None)
        _servers["test_server"] = server
        try:
            check = _make_check_fn("test_server")
            assert check() is False
        finally:
            _servers.pop("test_server", None)


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

class TestToolHandler:
    """Tool handlers are sync functions that schedule work on the MCP loop."""

    def _patch_mcp_loop(self, coro_side_effect=None):
        """Return a patch for _run_on_mcp_loop that runs the coroutine directly."""
        def fake_run(coro, timeout=30):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        if coro_side_effect:
            return patch("tools.mcp_tool._run_on_mcp_loop", side_effect=coro_side_effect)
        return patch("tools.mcp_tool._run_on_mcp_loop", side_effect=fake_run)

    def test_successful_call(self):
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("hello world", is_error=False)
        )
        server = _make_mock_server("test_srv", session=mock_session)
        _servers["test_srv"] = server

        try:
            handler = _make_tool_handler("test_srv", "greet", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({"name": "world"}))
            assert result["result"] == "hello world"
            mock_session.call_tool.assert_called_once_with("greet", arguments={"name": "world"})
        finally:
            _servers.pop("test_srv", None)

    def test_mcp_error_result(self):
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("something went wrong", is_error=True)
        )
        server = _make_mock_server("test_srv", session=mock_session)
        _servers["test_srv"] = server

        try:
            handler = _make_tool_handler("test_srv", "fail_tool", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({}))
            assert "error" in result
            assert "something went wrong" in result["error"]
        finally:
            _servers.pop("test_srv", None)

    def test_disconnected_server(self):
        from tools.mcp_tool import _make_tool_handler, _servers

        _servers.pop("ghost", None)
        handler = _make_tool_handler("ghost", "any_tool", 120)
        result = json.loads(handler({}))
        assert "error" in result
        assert "not connected" in result["error"]

    def test_exception_during_call(self):
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))
        server = _make_mock_server("test_srv", session=mock_session)
        _servers["test_srv"] = server

        try:
            handler = _make_tool_handler("test_srv", "broken_tool", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({}))
            assert "error" in result
            assert "connection lost" in result["error"]
        finally:
            _servers.pop("test_srv", None)


# ---------------------------------------------------------------------------
# Tool registration (discovery + register)
# ---------------------------------------------------------------------------

class TestDiscoverAndRegister:
    def test_tools_registered_in_registry(self):
        """_discover_and_register_server registers tools with correct names."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()
        mock_tools = [
            _make_mcp_tool("read_file", "Read a file"),
            _make_mcp_tool("write_file", "Write a file"),
        ]
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("fs", {"command": "npx", "args": []})
            )

        assert "mcp_fs_read_file" in registered
        assert "mcp_fs_write_file" in registered
        assert "mcp_fs_read_file" in mock_registry.get_all_tool_names()
        assert "mcp_fs_write_file" in mock_registry.get_all_tool_names()

        _servers.pop("fs", None)

    def test_toolset_created(self):
        """A custom toolset is created for the MCP server."""
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_tools = [_make_mcp_tool("ping", "Ping")]
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        mock_create = MagicMock()
        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("toolsets.create_custom_toolset", mock_create):
            asyncio.run(
                _discover_and_register_server("myserver", {"command": "test"})
            )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs[1]["name"] == "mcp-myserver" or call_kwargs[0][0] == "mcp-myserver"

        _servers.pop("myserver", None)

    def test_schema_format_correct(self):
        """Registered schemas have the correct format."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()
        mock_tools = [_make_mcp_tool("do_thing", "Do something")]
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

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

        _servers.pop("srv", None)


# ---------------------------------------------------------------------------
# MCPServerTask (run / start / shutdown)
# ---------------------------------------------------------------------------

class TestMCPServerTask:
    """Test the MCPServerTask lifecycle with mocked MCP SDK."""

    def _mock_stdio_and_session(self, session):
        """Return patches for stdio_client and ClientSession as async CMs."""
        mock_read, mock_write = MagicMock(), MagicMock()

        mock_stdio_cm = MagicMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        mock_cs_cm = MagicMock()
        mock_cs_cm.__aenter__ = AsyncMock(return_value=session)
        mock_cs_cm.__aexit__ = AsyncMock(return_value=False)

        return (
            patch("tools.mcp_tool.stdio_client", return_value=mock_stdio_cm),
            patch("tools.mcp_tool.ClientSession", return_value=mock_cs_cm),
            mock_read, mock_write,
        )

    def test_start_connects_and_discovers_tools(self):
        """start() creates a Task that connects, discovers tools, and waits."""
        from tools.mcp_tool import MCPServerTask

        mock_tools = [_make_mcp_tool("echo")]
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        p_stdio, p_cs, _, _ = self._mock_stdio_and_session(mock_session)

        async def _test():
            with patch("tools.mcp_tool.StdioServerParameters"), p_stdio, p_cs:
                server = MCPServerTask("test_srv")
                await server.start({"command": "npx", "args": ["-y", "test"]})

                assert server.session is mock_session
                assert len(server._tools) == 1
                assert server._tools[0].name == "echo"
                mock_session.initialize.assert_called_once()

                await server.shutdown()
                assert server.session is None

        asyncio.run(_test())

    def test_no_command_raises(self):
        """Missing 'command' in config raises ValueError."""
        from tools.mcp_tool import MCPServerTask

        async def _test():
            server = MCPServerTask("bad")
            with pytest.raises(ValueError, match="no 'command'"):
                await server.start({"args": []})

        asyncio.run(_test())

    def test_empty_env_gets_safe_defaults(self):
        """Empty env dict gets safe default env vars (PATH, HOME, etc.)."""
        from tools.mcp_tool import MCPServerTask

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[])
        )

        p_stdio, p_cs, _, _ = self._mock_stdio_and_session(mock_session)

        async def _test():
            with patch("tools.mcp_tool.StdioServerParameters") as mock_params, \
                 p_stdio, p_cs, \
                 patch.dict("os.environ", {"PATH": "/usr/bin", "HOME": "/home/test"}, clear=False):
                server = MCPServerTask("srv")
                await server.start({"command": "node", "env": {}})

                # Empty dict -> safe env vars (not None)
                call_kwargs = mock_params.call_args
                env_arg = call_kwargs.kwargs.get("env")
                assert env_arg is not None
                assert isinstance(env_arg, dict)
                assert "PATH" in env_arg
                assert "HOME" in env_arg

                await server.shutdown()

        asyncio.run(_test())

    def test_shutdown_signals_task_exit(self):
        """shutdown() signals the event and waits for task completion."""
        from tools.mcp_tool import MCPServerTask

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[])
        )

        p_stdio, p_cs, _, _ = self._mock_stdio_and_session(mock_session)

        async def _test():
            with patch("tools.mcp_tool.StdioServerParameters"), p_stdio, p_cs:
                server = MCPServerTask("srv")
                await server.start({"command": "npx"})

                assert server.session is not None
                assert not server._task.done()

                await server.shutdown()

                assert server.session is None
                assert server._task.done()

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# discover_mcp_tools toolset injection
# ---------------------------------------------------------------------------

class TestToolsetInjection:
    def test_mcp_tools_added_to_all_hermes_toolsets(self):
        """Discovered MCP tools are dynamically injected into all hermes-* toolsets."""
        from tools.mcp_tool import MCPServerTask

        mock_tools = [_make_mcp_tool("list_files", "List files")]
        mock_session = MagicMock()

        fresh_servers = {}

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        fake_toolsets = {
            "hermes-cli": {"tools": ["terminal"], "description": "CLI", "includes": []},
            "hermes-telegram": {"tools": ["terminal"], "description": "TG", "includes": []},
            "hermes-gateway": {"tools": [], "description": "GW", "includes": []},
            "non-hermes": {"tools": [], "description": "other", "includes": []},
        }
        fake_config = {"fs": {"command": "npx", "args": []}}

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._servers", fresh_servers), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("toolsets.TOOLSETS", fake_toolsets):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()

        assert "mcp_fs_list_files" in result
        # All hermes-* toolsets get injection
        assert "mcp_fs_list_files" in fake_toolsets["hermes-cli"]["tools"]
        assert "mcp_fs_list_files" in fake_toolsets["hermes-telegram"]["tools"]
        assert "mcp_fs_list_files" in fake_toolsets["hermes-gateway"]["tools"]
        # Non-hermes toolset should NOT get injection
        assert "mcp_fs_list_files" not in fake_toolsets["non-hermes"]["tools"]
        # Original tools preserved
        assert "terminal" in fake_toolsets["hermes-cli"]["tools"]

    def test_server_connection_failure_skipped(self):
        """If one server fails to connect, others still proceed."""
        from tools.mcp_tool import MCPServerTask

        mock_tools = [_make_mcp_tool("ping", "Ping")]
        mock_session = MagicMock()

        fresh_servers = {}
        call_count = 0

        async def flaky_connect(name, config):
            nonlocal call_count
            call_count += 1
            if name == "broken":
                raise ConnectionError("cannot reach server")
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        fake_config = {
            "broken": {"command": "bad"},
            "good": {"command": "npx", "args": []},
        }
        fake_toolsets = {
            "hermes-cli": {"tools": [], "description": "CLI", "includes": []},
        }

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._servers", fresh_servers), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=flaky_connect), \
             patch("toolsets.TOOLSETS", fake_toolsets):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()

        assert "mcp_good_ping" in result
        assert "mcp_broken_ping" not in result
        assert call_count == 2

    def test_partial_failure_retry_on_second_call(self):
        """Failed servers are retried on subsequent discover_mcp_tools() calls."""
        from tools.mcp_tool import MCPServerTask

        mock_tools = [_make_mcp_tool("ping", "Ping")]
        mock_session = MagicMock()

        # Use a real dict so idempotency logic works correctly
        fresh_servers = {}
        call_count = 0
        broken_fixed = False

        async def flaky_connect(name, config):
            nonlocal call_count
            call_count += 1
            if name == "broken" and not broken_fixed:
                raise ConnectionError("cannot reach server")
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        fake_config = {
            "broken": {"command": "bad"},
            "good": {"command": "npx", "args": []},
        }
        fake_toolsets = {
            "hermes-cli": {"tools": [], "description": "CLI", "includes": []},
        }

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._servers", fresh_servers), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=flaky_connect), \
             patch("toolsets.TOOLSETS", fake_toolsets):
            from tools.mcp_tool import discover_mcp_tools

            # First call: good connects, broken fails
            result1 = discover_mcp_tools()
            assert "mcp_good_ping" in result1
            assert "mcp_broken_ping" not in result1
            first_attempts = call_count

            # "Fix" the broken server
            broken_fixed = True
            call_count = 0

            # Second call: should retry broken, skip good
            result2 = discover_mcp_tools()
            assert "mcp_good_ping" in result2
            assert "mcp_broken_ping" in result2
            assert call_count == 1  # Only broken retried


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
             patch("tools.mcp_tool._servers", {}), \
             patch("tools.mcp_tool._load_mcp_config", return_value={}):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()
            assert result == []


# ---------------------------------------------------------------------------
# Shutdown (public API)
# ---------------------------------------------------------------------------

class TestShutdown:
    def test_no_servers_safe(self):
        """shutdown_mcp_servers with no servers does nothing."""
        from tools.mcp_tool import shutdown_mcp_servers, _servers

        _servers.clear()
        shutdown_mcp_servers()  # Should not raise

    def test_shutdown_clears_servers(self):
        """shutdown_mcp_servers calls shutdown() on each server and clears dict."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import shutdown_mcp_servers, _servers

        _servers.clear()
        mock_server = MagicMock()
        mock_server.name = "test"
        mock_server.shutdown = AsyncMock()
        _servers["test"] = mock_server

        mcp_mod._ensure_mcp_loop()
        try:
            shutdown_mcp_servers()
        finally:
            mcp_mod._mcp_loop = None
            mcp_mod._mcp_thread = None

        assert len(_servers) == 0
        mock_server.shutdown.assert_called_once()

    def test_shutdown_handles_errors(self):
        """shutdown_mcp_servers handles errors during close gracefully."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import shutdown_mcp_servers, _servers

        _servers.clear()
        mock_server = MagicMock()
        mock_server.name = "broken"
        mock_server.shutdown = AsyncMock(side_effect=RuntimeError("close failed"))
        _servers["broken"] = mock_server

        mcp_mod._ensure_mcp_loop()
        try:
            shutdown_mcp_servers()  # Should not raise
        finally:
            mcp_mod._mcp_loop = None
            mcp_mod._mcp_thread = None

        assert len(_servers) == 0

    def test_shutdown_is_parallel(self):
        """Multiple servers are shut down in parallel via asyncio.gather."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import shutdown_mcp_servers, _servers
        import time

        _servers.clear()

        # 3 servers each taking 1s to shut down
        for i in range(3):
            mock_server = MagicMock()
            mock_server.name = f"srv_{i}"
            async def slow_shutdown():
                await asyncio.sleep(1)
            mock_server.shutdown = slow_shutdown
            _servers[f"srv_{i}"] = mock_server

        mcp_mod._ensure_mcp_loop()
        try:
            start = time.monotonic()
            shutdown_mcp_servers()
            elapsed = time.monotonic() - start
        finally:
            mcp_mod._mcp_loop = None
            mcp_mod._mcp_thread = None

        assert len(_servers) == 0
        # Parallel: ~1s, not ~3s. Allow some margin.
        assert elapsed < 2.5, f"Shutdown took {elapsed:.1f}s, expected ~1s (parallel)"


# ---------------------------------------------------------------------------
# _build_safe_env
# ---------------------------------------------------------------------------

class TestBuildSafeEnv:
    """Tests for _build_safe_env() environment filtering."""

    def test_only_safe_vars_passed(self):
        """Only safe baseline vars and XDG_* from os.environ are included."""
        from tools.mcp_tool import _build_safe_env

        fake_env = {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "USER": "test",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "C",
            "TERM": "xterm",
            "SHELL": "/bin/bash",
            "TMPDIR": "/tmp",
            "XDG_DATA_HOME": "/home/test/.local/share",
            "SECRET_KEY": "should_not_appear",
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
        }
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)

        # Safe vars present
        assert result["PATH"] == "/usr/bin"
        assert result["HOME"] == "/home/test"
        assert result["USER"] == "test"
        assert result["LANG"] == "en_US.UTF-8"
        assert result["XDG_DATA_HOME"] == "/home/test/.local/share"
        # Unsafe vars excluded
        assert "SECRET_KEY" not in result
        assert "AWS_ACCESS_KEY_ID" not in result

    def test_user_env_merged(self):
        """User-specified env vars are merged into the safe env."""
        from tools.mcp_tool import _build_safe_env

        with patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
            result = _build_safe_env({"MY_CUSTOM_VAR": "hello"})

        assert result["PATH"] == "/usr/bin"
        assert result["MY_CUSTOM_VAR"] == "hello"

    def test_user_env_overrides_safe(self):
        """User env can override safe defaults."""
        from tools.mcp_tool import _build_safe_env

        with patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
            result = _build_safe_env({"PATH": "/custom/bin"})

        assert result["PATH"] == "/custom/bin"

    def test_none_user_env(self):
        """None user_env still returns safe vars from os.environ."""
        from tools.mcp_tool import _build_safe_env

        with patch.dict("os.environ", {"PATH": "/usr/bin", "HOME": "/root"}, clear=True):
            result = _build_safe_env(None)

        assert isinstance(result, dict)
        assert result["PATH"] == "/usr/bin"
        assert result["HOME"] == "/root"

    def test_secret_vars_excluded(self):
        """Sensitive env vars from os.environ are NOT passed through."""
        from tools.mcp_tool import _build_safe_env

        fake_env = {
            "PATH": "/usr/bin",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "OPENAI_API_KEY": "sk-proj-abc123",
            "DATABASE_URL": "postgres://user:pass@localhost/db",
            "API_SECRET": "supersecret",
        }
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)

        assert "PATH" in result
        assert "AWS_SECRET_ACCESS_KEY" not in result
        assert "GITHUB_TOKEN" not in result
        assert "OPENAI_API_KEY" not in result
        assert "DATABASE_URL" not in result
        assert "API_SECRET" not in result


# ---------------------------------------------------------------------------
# _sanitize_error
# ---------------------------------------------------------------------------

class TestSanitizeError:
    """Tests for _sanitize_error() credential stripping."""

    def test_strips_github_pat(self):
        from tools.mcp_tool import _sanitize_error
        result = _sanitize_error("Error with ghp_abc123def456")
        assert result == "Error with [REDACTED]"

    def test_strips_openai_key(self):
        from tools.mcp_tool import _sanitize_error
        result = _sanitize_error("key sk-projABC123xyz")
        assert result == "key [REDACTED]"

    def test_strips_bearer_token(self):
        from tools.mcp_tool import _sanitize_error
        result = _sanitize_error("Authorization: Bearer eyJabc123def")
        assert result == "Authorization: [REDACTED]"

    def test_strips_token_param(self):
        from tools.mcp_tool import _sanitize_error
        result = _sanitize_error("url?token=secret123")
        assert result == "url?[REDACTED]"

    def test_no_credentials_unchanged(self):
        from tools.mcp_tool import _sanitize_error
        result = _sanitize_error("normal error message")
        assert result == "normal error message"

    def test_multiple_credentials(self):
        from tools.mcp_tool import _sanitize_error
        result = _sanitize_error("ghp_abc123 and sk-projXyz789 and token=foo")
        assert "ghp_" not in result
        assert "sk-" not in result
        assert "token=" not in result
        assert result.count("[REDACTED]") == 3


# ---------------------------------------------------------------------------
# HTTP config
# ---------------------------------------------------------------------------

class TestHTTPConfig:
    """Tests for HTTP transport detection and handling."""

    def test_is_http_with_url(self):
        from tools.mcp_tool import MCPServerTask
        server = MCPServerTask("remote")
        server._config = {"url": "https://example.com/mcp"}
        assert server._is_http() is True

    def test_is_stdio_with_command(self):
        from tools.mcp_tool import MCPServerTask
        server = MCPServerTask("local")
        server._config = {"command": "npx", "args": []}
        assert server._is_http() is False

    def test_conflicting_url_and_command_warns(self):
        """Config with both url and command logs a warning and uses HTTP."""
        from tools.mcp_tool import MCPServerTask
        server = MCPServerTask("conflict")
        config = {"url": "https://example.com/mcp", "command": "npx", "args": []}
        # url takes precedence
        server._config = config
        assert server._is_http() is True

    def test_http_unavailable_raises(self):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")
        config = {"url": "https://example.com/mcp"}

        async def _test():
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", False):
                with pytest.raises(ImportError, match="HTTP transport"):
                    await server._run_http(config)

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Reconnection logic
# ---------------------------------------------------------------------------

class TestReconnection:
    """Tests for automatic reconnection behavior in MCPServerTask.run()."""

    def test_reconnect_on_disconnect(self):
        """After initial success, a connection drop triggers reconnection."""
        from tools.mcp_tool import MCPServerTask

        run_count = 0
        target_server = None

        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            nonlocal run_count, target_server
            run_count += 1
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            if run_count == 1:
                # First connection succeeds, then simulate disconnect
                self_srv.session = MagicMock()
                self_srv._tools = []
                self_srv._ready.set()
                raise ConnectionError("connection dropped")
            else:
                # Reconnection succeeds; signal shutdown so run() exits
                self_srv.session = MagicMock()
                self_srv._shutdown_event.set()
                await self_srv._shutdown_event.wait()

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await server.run({"command": "test"})

            assert run_count >= 2  # At least one reconnection attempt

        asyncio.run(_test())

    def test_no_reconnect_on_shutdown(self):
        """If shutdown is requested, don't attempt reconnection."""
        from tools.mcp_tool import MCPServerTask

        run_count = 0
        target_server = None

        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            nonlocal run_count, target_server
            run_count += 1
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            self_srv.session = MagicMock()
            self_srv._tools = []
            self_srv._ready.set()
            raise ConnectionError("connection dropped")

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server
            server._shutdown_event.set()  # Shutdown already requested

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await server.run({"command": "test"})

            # Should not retry because shutdown was set
            assert run_count == 1

        asyncio.run(_test())

    def test_no_reconnect_on_initial_failure(self):
        """First connection failure reports error immediately, no retry."""
        from tools.mcp_tool import MCPServerTask

        run_count = 0
        target_server = None

        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            nonlocal run_count, target_server
            run_count += 1
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            raise ConnectionError("cannot connect")

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await server.run({"command": "test"})

            # Only one attempt, no retry on initial failure
            assert run_count == 1
            assert server._error is not None
            assert "cannot connect" in str(server._error)

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Configurable timeouts
# ---------------------------------------------------------------------------

class TestConfigurableTimeouts:
    """Tests for configurable per-server timeouts."""

    def test_default_timeout(self):
        """Server with no timeout config gets _DEFAULT_TOOL_TIMEOUT."""
        from tools.mcp_tool import MCPServerTask, _DEFAULT_TOOL_TIMEOUT

        server = MCPServerTask("test_srv")
        assert server.tool_timeout == _DEFAULT_TOOL_TIMEOUT
        assert server.tool_timeout == 120

    def test_custom_timeout(self):
        """Server with timeout=180 in config gets 180."""
        from tools.mcp_tool import MCPServerTask

        target_server = None

        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            self_srv.session = MagicMock()
            self_srv._tools = []
            self_srv._ready.set()
            await self_srv._shutdown_event.wait()

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio):
                task = asyncio.ensure_future(
                    server.run({"command": "test", "timeout": 180})
                )
                await server._ready.wait()
                assert server.tool_timeout == 180
                server._shutdown_event.set()
                await task

        asyncio.run(_test())

    def test_timeout_passed_to_handler(self):
        """The tool handler uses the server's configured timeout."""
        from tools.mcp_tool import _make_tool_handler, _servers, MCPServerTask

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("ok", is_error=False)
        )
        server = _make_mock_server("test_srv", session=mock_session)
        server.tool_timeout = 180
        _servers["test_srv"] = server

        try:
            handler = _make_tool_handler("test_srv", "my_tool", 180)
            with patch("tools.mcp_tool._run_on_mcp_loop") as mock_run:
                mock_run.return_value = json.dumps({"result": "ok"})
                handler({})
                # Verify timeout=180 was passed
                call_kwargs = mock_run.call_args
                assert call_kwargs.kwargs.get("timeout") == 180 or \
                       (len(call_kwargs.args) > 1 and call_kwargs.args[1] == 180) or \
                       call_kwargs[1].get("timeout") == 180
        finally:
            _servers.pop("test_srv", None)


# ---------------------------------------------------------------------------
# Utility tool schemas (Resources & Prompts)
# ---------------------------------------------------------------------------

class TestUtilitySchemas:
    """Tests for _build_utility_schemas() and the schema format of utility tools."""

    def test_builds_four_utility_schemas(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("myserver")
        assert len(schemas) == 4
        names = [s["schema"]["name"] for s in schemas]
        assert "mcp_myserver_list_resources" in names
        assert "mcp_myserver_read_resource" in names
        assert "mcp_myserver_list_prompts" in names
        assert "mcp_myserver_get_prompt" in names

    def test_hyphens_sanitized_in_utility_names(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("my-server")
        names = [s["schema"]["name"] for s in schemas]
        for name in names:
            assert "-" not in name
        assert "mcp_my_server_list_resources" in names

    def test_list_resources_schema_no_required_params(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("srv")
        lr = next(s for s in schemas if s["handler_key"] == "list_resources")
        params = lr["schema"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}
        assert "required" not in params

    def test_read_resource_schema_requires_uri(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("srv")
        rr = next(s for s in schemas if s["handler_key"] == "read_resource")
        params = rr["schema"]["parameters"]
        assert "uri" in params["properties"]
        assert params["properties"]["uri"]["type"] == "string"
        assert params["required"] == ["uri"]

    def test_list_prompts_schema_no_required_params(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("srv")
        lp = next(s for s in schemas if s["handler_key"] == "list_prompts")
        params = lp["schema"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}
        assert "required" not in params

    def test_get_prompt_schema_requires_name(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("srv")
        gp = next(s for s in schemas if s["handler_key"] == "get_prompt")
        params = gp["schema"]["parameters"]
        assert "name" in params["properties"]
        assert params["properties"]["name"]["type"] == "string"
        assert "arguments" in params["properties"]
        assert params["properties"]["arguments"]["type"] == "object"
        assert params["required"] == ["name"]

    def test_schemas_have_descriptions(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("test_srv")
        for entry in schemas:
            desc = entry["schema"]["description"]
            assert desc and len(desc) > 0
            assert "test_srv" in desc


# ---------------------------------------------------------------------------
# Utility tool handlers (Resources & Prompts)
# ---------------------------------------------------------------------------

class TestUtilityHandlers:
    """Tests for the MCP Resources & Prompts handler functions."""

    def _patch_mcp_loop(self):
        """Return a patch for _run_on_mcp_loop that runs the coroutine directly."""
        def fake_run(coro, timeout=30):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        return patch("tools.mcp_tool._run_on_mcp_loop", side_effect=fake_run)

    # -- list_resources --

    def test_list_resources_success(self):
        from tools.mcp_tool import _make_list_resources_handler, _servers

        mock_resource = SimpleNamespace(
            uri="file:///tmp/test.txt", name="test.txt",
            description="A test file", mimeType="text/plain",
        )
        mock_session = MagicMock()
        mock_session.list_resources = AsyncMock(
            return_value=SimpleNamespace(resources=[mock_resource])
        )
        server = _make_mock_server("srv", session=mock_session)
        _servers["srv"] = server

        try:
            handler = _make_list_resources_handler("srv", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({}))
            assert "resources" in result
            assert len(result["resources"]) == 1
            assert result["resources"][0]["uri"] == "file:///tmp/test.txt"
            assert result["resources"][0]["name"] == "test.txt"
        finally:
            _servers.pop("srv", None)

    def test_list_resources_empty(self):
        from tools.mcp_tool import _make_list_resources_handler, _servers

        mock_session = MagicMock()
        mock_session.list_resources = AsyncMock(
            return_value=SimpleNamespace(resources=[])
        )
        server = _make_mock_server("srv", session=mock_session)
        _servers["srv"] = server

        try:
            handler = _make_list_resources_handler("srv", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({}))
            assert result["resources"] == []
        finally:
            _servers.pop("srv", None)

    def test_list_resources_disconnected(self):
        from tools.mcp_tool import _make_list_resources_handler, _servers
        _servers.pop("ghost", None)
        handler = _make_list_resources_handler("ghost", 120)
        result = json.loads(handler({}))
        assert "error" in result
        assert "not connected" in result["error"]

    # -- read_resource --

    def test_read_resource_success(self):
        from tools.mcp_tool import _make_read_resource_handler, _servers

        content_block = SimpleNamespace(text="Hello from resource")
        mock_session = MagicMock()
        mock_session.read_resource = AsyncMock(
            return_value=SimpleNamespace(contents=[content_block])
        )
        server = _make_mock_server("srv", session=mock_session)
        _servers["srv"] = server

        try:
            handler = _make_read_resource_handler("srv", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({"uri": "file:///tmp/test.txt"}))
            assert result["result"] == "Hello from resource"
            mock_session.read_resource.assert_called_once_with("file:///tmp/test.txt")
        finally:
            _servers.pop("srv", None)

    def test_read_resource_missing_uri(self):
        from tools.mcp_tool import _make_read_resource_handler, _servers

        server = _make_mock_server("srv", session=MagicMock())
        _servers["srv"] = server

        try:
            handler = _make_read_resource_handler("srv", 120)
            result = json.loads(handler({}))
            assert "error" in result
            assert "uri" in result["error"].lower()
        finally:
            _servers.pop("srv", None)

    def test_read_resource_disconnected(self):
        from tools.mcp_tool import _make_read_resource_handler, _servers
        _servers.pop("ghost", None)
        handler = _make_read_resource_handler("ghost", 120)
        result = json.loads(handler({"uri": "test://x"}))
        assert "error" in result
        assert "not connected" in result["error"]

    # -- list_prompts --

    def test_list_prompts_success(self):
        from tools.mcp_tool import _make_list_prompts_handler, _servers

        mock_prompt = SimpleNamespace(
            name="summarize", description="Summarize text",
            arguments=[
                SimpleNamespace(name="text", description="Text to summarize", required=True),
            ],
        )
        mock_session = MagicMock()
        mock_session.list_prompts = AsyncMock(
            return_value=SimpleNamespace(prompts=[mock_prompt])
        )
        server = _make_mock_server("srv", session=mock_session)
        _servers["srv"] = server

        try:
            handler = _make_list_prompts_handler("srv", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({}))
            assert "prompts" in result
            assert len(result["prompts"]) == 1
            assert result["prompts"][0]["name"] == "summarize"
            assert result["prompts"][0]["arguments"][0]["name"] == "text"
        finally:
            _servers.pop("srv", None)

    def test_list_prompts_empty(self):
        from tools.mcp_tool import _make_list_prompts_handler, _servers

        mock_session = MagicMock()
        mock_session.list_prompts = AsyncMock(
            return_value=SimpleNamespace(prompts=[])
        )
        server = _make_mock_server("srv", session=mock_session)
        _servers["srv"] = server

        try:
            handler = _make_list_prompts_handler("srv", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({}))
            assert result["prompts"] == []
        finally:
            _servers.pop("srv", None)

    def test_list_prompts_disconnected(self):
        from tools.mcp_tool import _make_list_prompts_handler, _servers
        _servers.pop("ghost", None)
        handler = _make_list_prompts_handler("ghost", 120)
        result = json.loads(handler({}))
        assert "error" in result
        assert "not connected" in result["error"]

    # -- get_prompt --

    def test_get_prompt_success(self):
        from tools.mcp_tool import _make_get_prompt_handler, _servers

        mock_msg = SimpleNamespace(
            role="assistant",
            content=SimpleNamespace(text="Here is a summary of your text."),
        )
        mock_session = MagicMock()
        mock_session.get_prompt = AsyncMock(
            return_value=SimpleNamespace(messages=[mock_msg], description=None)
        )
        server = _make_mock_server("srv", session=mock_session)
        _servers["srv"] = server

        try:
            handler = _make_get_prompt_handler("srv", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({"name": "summarize", "arguments": {"text": "hello"}}))
            assert "messages" in result
            assert len(result["messages"]) == 1
            assert result["messages"][0]["role"] == "assistant"
            assert "summary" in result["messages"][0]["content"].lower()
            mock_session.get_prompt.assert_called_once_with(
                "summarize", arguments={"text": "hello"}
            )
        finally:
            _servers.pop("srv", None)

    def test_get_prompt_missing_name(self):
        from tools.mcp_tool import _make_get_prompt_handler, _servers

        server = _make_mock_server("srv", session=MagicMock())
        _servers["srv"] = server

        try:
            handler = _make_get_prompt_handler("srv", 120)
            result = json.loads(handler({}))
            assert "error" in result
            assert "name" in result["error"].lower()
        finally:
            _servers.pop("srv", None)

    def test_get_prompt_disconnected(self):
        from tools.mcp_tool import _make_get_prompt_handler, _servers
        _servers.pop("ghost", None)
        handler = _make_get_prompt_handler("ghost", 120)
        result = json.loads(handler({"name": "test"}))
        assert "error" in result
        assert "not connected" in result["error"]

    def test_get_prompt_default_arguments(self):
        from tools.mcp_tool import _make_get_prompt_handler, _servers

        mock_session = MagicMock()
        mock_session.get_prompt = AsyncMock(
            return_value=SimpleNamespace(messages=[], description=None)
        )
        server = _make_mock_server("srv", session=mock_session)
        _servers["srv"] = server

        try:
            handler = _make_get_prompt_handler("srv", 120)
            with self._patch_mcp_loop():
                handler({"name": "test_prompt"})
            # arguments defaults to {} when not provided
            mock_session.get_prompt.assert_called_once_with(
                "test_prompt", arguments={}
            )
        finally:
            _servers.pop("srv", None)


# ---------------------------------------------------------------------------
# Utility tools registration in _discover_and_register_server
# ---------------------------------------------------------------------------

class TestUtilityToolRegistration:
    """Verify utility tools are registered alongside regular MCP tools."""

    def test_utility_tools_registered(self):
        """_discover_and_register_server registers all 4 utility tools."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()
        mock_tools = [_make_mcp_tool("read_file", "Read a file")]
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("fs", {"command": "npx", "args": []})
            )

        # Regular tool + 4 utility tools
        assert "mcp_fs_read_file" in registered
        assert "mcp_fs_list_resources" in registered
        assert "mcp_fs_read_resource" in registered
        assert "mcp_fs_list_prompts" in registered
        assert "mcp_fs_get_prompt" in registered
        assert len(registered) == 5

        # All in the registry
        all_names = mock_registry.get_all_tool_names()
        for name in registered:
            assert name in all_names

        _servers.pop("fs", None)

    def test_utility_tools_in_same_toolset(self):
        """Utility tools belong to the same mcp-{server} toolset."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = []
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            asyncio.run(
                _discover_and_register_server("myserv", {"command": "test"})
            )

        # Check that utility tools are in the right toolset
        for tool_name in ["mcp_myserv_list_resources", "mcp_myserv_read_resource",
                          "mcp_myserv_list_prompts", "mcp_myserv_get_prompt"]:
            entry = mock_registry._tools.get(tool_name)
            assert entry is not None, f"{tool_name} not found in registry"
            assert entry.toolset == "mcp-myserv"

        _servers.pop("myserv", None)

    def test_utility_tools_have_check_fn(self):
        """Utility tools have a working check_fn."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = []
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            asyncio.run(
                _discover_and_register_server("chk", {"command": "test"})
            )

        entry = mock_registry._tools.get("mcp_chk_list_resources")
        assert entry is not None
        # Server is connected, check_fn should return True
        assert entry.check_fn() is True

        # Disconnect the server
        _servers["chk"].session = None
        assert entry.check_fn() is False

        _servers.pop("chk", None)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _make_sampling_params(
    messages=None,
    max_tokens=1024,
    system_prompt=None,
    model_preferences=None,
    temperature=None,
    stop_sequences=None,
    tools=None,
    tool_choice=None,
):
    """Create a fake MCP CreateMessageRequestParams object."""
    if messages is None:
        messages = [SimpleNamespace(
            role="user",
            content=SimpleNamespace(text="Hello"),
        )]
    params = SimpleNamespace(
        messages=messages,
        maxTokens=max_tokens,
        systemPrompt=system_prompt,
        modelPreferences=model_preferences,
        temperature=temperature,
        stopSequences=stop_sequences,
    )
    if tools is not None:
        params.tools = tools
    if tool_choice is not None:
        params.toolChoice = tool_choice
    return params


def _make_llm_response(content="LLM response", model="test-model", finish_reason="stop",
                        tool_calls=None):
    """Create a fake OpenAI-style chat completion response."""
    message = SimpleNamespace(content=content)
    if tool_calls is not None:
        message.tool_calls = tool_calls
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        choices=[choice],
        model=model,
        usage=SimpleNamespace(total_tokens=42),
    )


def _make_llm_tool_response(tool_calls_data=None, model="test-model"):
    """Create a fake OpenAI-style response with tool_calls.

    Args:
        tool_calls_data: list of (id, name, arguments) tuples.
        model: model name.
    """
    if tool_calls_data is None:
        tool_calls_data = [("call_1", "get_weather", '{"city": "London"}')]
    tc_objects = []
    for call_id, name, args in tool_calls_data:
        tc_objects.append(SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=args),
        ))
    message = SimpleNamespace(content=None, tool_calls=tc_objects)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(
        choices=[choice],
        model=model,
        usage=SimpleNamespace(total_tokens=55),
    )


# ---------------------------------------------------------------------------
# Sampling callback tests
# ---------------------------------------------------------------------------

class TestSamplingHelpers:
    """Tests for sampling helper functions."""

    def test_resolve_model_config_override(self):
        """Config model override takes precedence over hints."""
        from tools.mcp_tool import _resolve_model

        prefs = SimpleNamespace(hints=[SimpleNamespace(name="gpt-4")])
        result = _resolve_model(prefs, {"model": "gemini-3-flash"})
        assert result == "gemini-3-flash"

    def test_resolve_model_server_hint(self):
        """Server model hints are resolved when no config override."""
        from tools.mcp_tool import _resolve_model

        prefs = SimpleNamespace(hints=[SimpleNamespace(name="gpt-4")])
        result = _resolve_model(prefs, {})
        assert result == "gpt-4"

    def test_resolve_model_no_preferences(self):
        """Returns None when no preferences and no config."""
        from tools.mcp_tool import _resolve_model

        result = _resolve_model(None, {})
        assert result is None

    def test_resolve_model_empty_hints(self):
        """Returns None when hints list is empty."""
        from tools.mcp_tool import _resolve_model

        prefs = SimpleNamespace(hints=[])
        result = _resolve_model(prefs, {})
        assert result is None

    def test_convert_sampling_messages_text(self):
        """Text content messages are converted correctly."""
        from tools.mcp_tool import _convert_sampling_messages

        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=SimpleNamespace(text="Hello")),
            SimpleNamespace(role="assistant", content=SimpleNamespace(text="Hi")),
        ])
        result = _convert_sampling_messages(params)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}

    def test_convert_sampling_messages_image(self):
        """Image content blocks are converted to OpenAI format."""
        from tools.mcp_tool import _convert_sampling_messages

        image_block = SimpleNamespace(
            data="base64data==",
            mimeType="image/png",
        )
        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=[
                SimpleNamespace(text="What's in this image?"),
                image_block,
            ]),
        ])
        result = _convert_sampling_messages(params)
        assert len(result) == 1
        parts = result[0]["content"]
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "text": "What's in this image?"}
        assert parts[1]["type"] == "image_url"
        assert "data:image/png;base64,base64data==" in parts[1]["image_url"]["url"]

    def test_convert_sampling_messages_fallback(self):
        """Non-text, non-list content falls back to str()."""
        from tools.mcp_tool import _convert_sampling_messages

        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=12345),
        ])
        result = _convert_sampling_messages(params)
        assert result[0]["content"] == "12345"

    def test_map_stop_reason_stop(self):
        from tools.mcp_tool import _map_stop_reason
        assert _map_stop_reason("stop") == "endTurn"

    def test_map_stop_reason_length(self):
        from tools.mcp_tool import _map_stop_reason
        assert _map_stop_reason("length") == "maxTokens"

    def test_map_stop_reason_tool_calls(self):
        from tools.mcp_tool import _map_stop_reason
        assert _map_stop_reason("tool_calls") == "toolUse"

    def test_map_stop_reason_unknown(self):
        from tools.mcp_tool import _map_stop_reason
        assert _map_stop_reason("content_filter") == "endTurn"

    def test_rate_limit_allows_under_limit(self):
        """Requests under the limit are allowed."""
        from tools.mcp_tool import _check_rate_limit, _sampling_counters

        _sampling_counters.pop("test_rl_allow", None)
        assert _check_rate_limit("test_rl_allow", 10) is True
        _sampling_counters.pop("test_rl_allow", None)

    def test_rate_limit_blocks_over_limit(self):
        """Requests over the limit are blocked."""
        from tools.mcp_tool import _check_rate_limit, _sampling_counters, _DEFAULT_SAMPLING_RATE_LIMIT

        _sampling_counters["test_rl_block"] = [time.time() for _ in range(_DEFAULT_SAMPLING_RATE_LIMIT)]
        assert _check_rate_limit("test_rl_block", _DEFAULT_SAMPLING_RATE_LIMIT) is False
        _sampling_counters.pop("test_rl_block", None)

    def test_rate_limit_expires_old_entries(self):
        """Old entries outside the window are expired."""
        from tools.mcp_tool import _check_rate_limit, _sampling_counters, _DEFAULT_SAMPLING_RATE_LIMIT

        # All timestamps from 2 minutes ago -- should be expired
        _sampling_counters["test_rl_expire"] = [
            time.time() - 120 for _ in range(_DEFAULT_SAMPLING_RATE_LIMIT)
        ]
        assert _check_rate_limit("test_rl_expire", _DEFAULT_SAMPLING_RATE_LIMIT) is True
        _sampling_counters.pop("test_rl_expire", None)

    def test_rate_limit_custom_max_rpm(self):
        """Custom max_rpm value is respected."""
        from tools.mcp_tool import _check_rate_limit, _sampling_counters

        _sampling_counters["test_rl_custom"] = [time.time() for _ in range(3)]
        assert _check_rate_limit("test_rl_custom", 3) is False  # At limit
        assert _check_rate_limit("test_rl_custom", 5) is True   # Under higher limit
        _sampling_counters.pop("test_rl_custom", None)


class TestSamplingCallback:
    """Tests for MCP sampling/createMessage support."""

    def setup_method(self):
        """Clear sampling rate limit counters between tests."""
        from tools.mcp_tool import _sampling_counters
        _sampling_counters.clear()

    def test_basic_text_sampling(self):
        """Sampling callback returns LLM response for text message."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("Hi there!")

        callback = _make_sampling_callback("test_srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "default-model")):
            result = asyncio.run(callback(None, params))

        assert result.role == "assistant"
        assert result.content.text == "Hi there!"
        assert result.model == "test-model"
        assert result.stopReason == "endTurn"

    def test_max_tokens_cap(self):
        """Max tokens capped to config limit."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {"max_tokens_cap": 256})
        params = _make_sampling_params(max_tokens=8192)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 256

    def test_max_tokens_server_under_cap(self):
        """Server requests fewer tokens than cap -- use server value."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {"max_tokens_cap": 4096})
        params = _make_sampling_params(max_tokens=100)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100

    def test_config_model_override(self):
        """Config model override takes precedence over server hints."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {"model": "my-custom-model"})
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="gpt-4")])
        params = _make_sampling_params(model_preferences=prefs)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "default")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-custom-model"

    def test_model_hint_used(self):
        """Server model hint used when no config override."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {})
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="gpt-4o")])
        params = _make_sampling_params(model_preferences=prefs)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "fallback")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_system_prompt_passed(self):
        """System prompt from server is included in LLM call."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(system_prompt="You are a helpful assistant")

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are a helpful assistant"}
        assert messages[1]["role"] == "user"

    def test_no_provider_returns_error(self):
        """Returns ErrorData when no LLM provider available."""
        from tools.mcp_tool import _make_sampling_callback

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(None, None)):
            result = asyncio.run(callback(None, params))

        assert isinstance(result, ErrorData)
        assert "No LLM provider" in result.message

    def test_rate_limit_exceeded(self):
        """Returns ErrorData when rate limit exceeded."""
        from tools.mcp_tool import _make_sampling_callback, _sampling_counters, _DEFAULT_SAMPLING_RATE_LIMIT

        callback = _make_sampling_callback("rl_test_srv", {})
        params = _make_sampling_params()

        _sampling_counters["rl_test_srv"] = [time.time() for _ in range(_DEFAULT_SAMPLING_RATE_LIMIT)]

        try:
            result = asyncio.run(callback(None, params))
            assert isinstance(result, ErrorData)
            assert "rate limit exceeded" in result.message.lower()
        finally:
            _sampling_counters.pop("rl_test_srv", None)

    def test_stop_reason_mapping_in_callback(self):
        """OpenAI finish_reason mapped to MCP stopReason in callback."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            finish_reason="length"
        )

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            result = asyncio.run(callback(None, params))

        assert result.stopReason == "maxTokens"

    def test_sampling_disabled_config(self):
        """Callback is None when sampling disabled in config."""
        from tools.mcp_tool import MCPServerTask

        target_server = None
        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            self_srv.session = MagicMock()
            self_srv._tools = []
            self_srv._ready.set()
            await self_srv._shutdown_event.wait()

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio):
                task = asyncio.ensure_future(
                    server.run({"command": "test", "sampling": {"enabled": False}})
                )
                await server._ready.wait()
                assert server._sampling_callback is None
                server._shutdown_event.set()
                await task

        asyncio.run(_test())

    def test_sampling_enabled_by_default(self):
        """Sampling callback is set when no sampling config (default enabled)."""
        from tools.mcp_tool import MCPServerTask

        target_server = None
        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            self_srv.session = MagicMock()
            self_srv._tools = []
            self_srv._ready.set()
            await self_srv._shutdown_event.wait()

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio):
                task = asyncio.ensure_future(
                    server.run({"command": "test"})
                )
                await server._ready.wait()
                assert server._sampling_callback is not None
                assert callable(server._sampling_callback)
                server._shutdown_event.set()
                await task

        asyncio.run(_test())

    def test_credential_stripping_in_response(self):
        """Credentials stripped from LLM response text."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            content="Here is your key: ghp_secrettoken123"
        )

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            result = asyncio.run(callback(None, params))

        assert "ghp_" not in result.content.text
        assert "[REDACTED]" in result.content.text

    def test_llm_api_error_returns_error_data(self):
        """API errors return ErrorData with sanitized message."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "Auth failed with key sk-secret123"
        )

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            result = asyncio.run(callback(None, params))

        assert isinstance(result, ErrorData)
        assert "Sampling LLM call failed" in result.message
        assert "sk-" not in result.message  # Credential sanitized

    def test_temperature_passthrough(self):
        """Temperature from server request passed to LLM."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(temperature=0.7)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    def test_temperature_none_not_passed(self):
        """None temperature is not included in LLM call kwargs."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(temperature=None)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "temperature" not in call_kwargs

    def test_multiple_messages(self):
        """Multi-turn conversation messages converted correctly."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(messages=[
            SimpleNamespace(role="user", content=SimpleNamespace(text="Hello")),
            SimpleNamespace(role="assistant", content=SimpleNamespace(text="Hi!")),
            SimpleNamespace(role="user", content=SimpleNamespace(text="How are you?")),
        ])

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 3
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi!"
        assert messages[2]["content"] == "How are you?"

    def test_stop_sequences_passed(self):
        """Stop sequences from server request passed to LLM."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(stop_sequences=["###", "END"])

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stop"] == ["###", "END"]

    def test_callback_passed_to_session_stdio(self):
        """ClientSession receives sampling_callback parameter for stdio transport."""
        from tools.mcp_tool import MCPServerTask

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[])
        )

        mock_read, mock_write = MagicMock(), MagicMock()
        mock_stdio_cm = MagicMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        mock_cs_cm = MagicMock()
        mock_cs_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cs_cm.__aexit__ = AsyncMock(return_value=False)

        async def _test_with_shutdown():
            with patch("tools.mcp_tool.StdioServerParameters", create=True), \
                 patch("tools.mcp_tool.stdio_client", create=True, return_value=mock_stdio_cm), \
                 patch("tools.mcp_tool.ClientSession", create=True, return_value=mock_cs_cm) as mock_cs_cls:
                server = MCPServerTask("test_srv")
                server._sampling_callback = MagicMock()
                server._shutdown_event.set()  # So it doesn't block
                await server._run_stdio({"command": "test", "args": []})

                call_kwargs = mock_cs_cls.call_args
                assert "sampling_callback" in call_kwargs.kwargs

        asyncio.run(_test_with_shutdown())

    def test_llm_call_uses_to_thread(self):
        """LLM call is offloaded via asyncio.to_thread (non-blocking)."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params()

        # Only mock to_thread; let wait_for properly await the coroutine
        # to avoid unawaited coroutine warnings.
        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")), \
             patch("tools.mcp_tool.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=_make_llm_response("threaded")) as mock_to_thread:
            result = asyncio.run(callback(None, params))

        assert result.content.text == "threaded"
        mock_to_thread.assert_called_once()

    def test_llm_timeout_returns_error_data(self):
        """LLM call timeout returns ErrorData."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()

        callback = _make_sampling_callback("srv", {"timeout": 1})
        params = _make_sampling_params()

        async def slow_to_thread(fn, **kwargs):
            await asyncio.sleep(10)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")), \
             patch("tools.mcp_tool.asyncio.to_thread", side_effect=slow_to_thread):
            result = asyncio.run(callback(None, params))

        assert isinstance(result, ErrorData)
        assert "timed out" in result.message.lower()

    def test_custom_timeout_config(self):
        """Custom timeout from config is used."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {"timeout": 60})
        params = _make_sampling_params()

        captured_timeout = None

        async def capturing_wait_for(coro, *, timeout=None):
            nonlocal captured_timeout
            captured_timeout = timeout
            return await coro  # Properly await to avoid coroutine warning

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")), \
             patch("tools.mcp_tool.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=_make_llm_response()), \
             patch("tools.mcp_tool.asyncio.wait_for",
                   side_effect=capturing_wait_for):
            asyncio.run(callback(None, params))

        # Verify wait_for was called with timeout=60
        assert captured_timeout == 60

    def test_custom_max_rpm_config(self):
        """Custom max_rpm from config is respected."""
        from tools.mcp_tool import _make_sampling_callback, _sampling_counters

        # Set rate limit to 2 RPM
        callback = _make_sampling_callback("rpm_test", {"max_rpm": 2})
        params = _make_sampling_params()

        _sampling_counters["rpm_test"] = [time.time(), time.time()]

        try:
            result = asyncio.run(callback(None, params))
            assert isinstance(result, ErrorData)
            assert "rate limit" in result.message.lower()
        finally:
            _sampling_counters.pop("rpm_test", None)

    def test_sampling_disabled_when_types_unavailable(self):
        """Sampling callback is None when _MCP_SAMPLING_TYPES is False."""
        from tools.mcp_tool import MCPServerTask

        target_server = None
        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            self_srv.session = MagicMock()
            self_srv._tools = []
            self_srv._ready.set()
            await self_srv._shutdown_event.wait()

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
                 patch("tools.mcp_tool._MCP_SAMPLING_TYPES", False):
                task = asyncio.ensure_future(
                    server.run({"command": "test"})
                )
                await server._ready.wait()
                assert server._sampling_callback is None
                server._shutdown_event.set()
                await task

        asyncio.run(_test())

    # -- Config type coercion tests ------------------------------------------

    def test_string_max_rpm_coerced_to_int(self):
        """String max_rpm value (e.g. from YAML) is coerced to int."""
        from tools.mcp_tool import _make_sampling_callback, _sampling_counters

        # Use string "2" instead of int 2
        callback = _make_sampling_callback("str_rpm_srv", {"max_rpm": "2"})
        params = _make_sampling_params()

        _sampling_counters["str_rpm_srv"] = [time.time(), time.time()]

        try:
            result = asyncio.run(callback(None, params))
            assert isinstance(result, ErrorData)
            assert "rate limit" in result.message.lower()
        finally:
            _sampling_counters.pop("str_rpm_srv", None)

    def test_string_max_tokens_cap_coerced_to_int(self):
        """String max_tokens_cap value is coerced to int."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        # Pass string "512" -- should be coerced to int(512)
        callback = _make_sampling_callback("srv", {"max_tokens_cap": "512"})
        params = _make_sampling_params(max_tokens=1024)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 512  # capped to 512

    def test_string_timeout_coerced_to_float(self):
        """String timeout value is coerced to float."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        # Pass string "45" -- should be coerced to float(45)
        callback = _make_sampling_callback("srv", {"timeout": "45"})
        params = _make_sampling_params()

        captured_timeout = None

        async def capturing_wait_for(coro, *, timeout=None):
            nonlocal captured_timeout
            captured_timeout = timeout
            return await coro

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")), \
             patch("tools.mcp_tool.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=_make_llm_response()), \
             patch("tools.mcp_tool.asyncio.wait_for",
                   side_effect=capturing_wait_for):
            asyncio.run(callback(None, params))

        assert captured_timeout == 45.0

    def test_invalid_config_values_fallback_to_defaults(self):
        """Invalid config values (e.g. 'abc') fall back to defaults."""
        from tools.mcp_tool import (
            _make_sampling_callback,
            _DEFAULT_SAMPLING_RATE_LIMIT,
            _DEFAULT_SAMPLING_MAX_TOKENS_CAP,
            _DEFAULT_SAMPLING_TIMEOUT,
            _check_rate_limit,
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("fallback_srv", {
            "max_rpm": "abc",
            "timeout": "not_a_number",
            "max_tokens_cap": [],
        })
        params = _make_sampling_params(max_tokens=99999)

        captured_timeout = None

        async def capturing_wait_for(coro, *, timeout=None):
            nonlocal captured_timeout
            captured_timeout = timeout
            return await coro

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")), \
             patch("tools.mcp_tool.asyncio.wait_for",
                   side_effect=capturing_wait_for):
            result = asyncio.run(callback(None, params))

        # Should succeed with defaults, not raise TypeError
        assert result.content.text is not None
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == _DEFAULT_SAMPLING_MAX_TOKENS_CAP
        # Verify timeout fell back to default
        assert captured_timeout == _DEFAULT_SAMPLING_TIMEOUT
        # Verify max_rpm fell back to default: fill counter to DEFAULT limit,
        # then the next callback call must hit rate limit -- proving the
        # callback actually uses _DEFAULT_SAMPLING_RATE_LIMIT (not some other value).
        from tools.mcp_tool import _sampling_counters
        _sampling_counters["fallback_srv"] = [
            time.time() for _ in range(_DEFAULT_SAMPLING_RATE_LIMIT)
        ]
        try:
            result2 = asyncio.run(callback(None, params))
            assert isinstance(result2, ErrorData)
            assert "rate limit" in result2.message.lower()
        finally:
            _sampling_counters.pop("fallback_srv", None)

    def test_zero_and_negative_config_clamped(self):
        """Zero/negative config values are clamped to minimum (1)."""
        from tools.mcp_tool import _make_sampling_callback, _sampling_counters

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        # max_rpm=0 should be clamped to 1, not block everything
        callback = _make_sampling_callback("clamp_srv", {
            "max_rpm": 0,
            "timeout": -5,
            "max_tokens_cap": -1,
        })
        params = _make_sampling_params(max_tokens=2000)

        _sampling_counters.pop("clamp_srv", None)

        captured_timeout = None

        async def capturing_wait_for(coro, *, timeout=None):
            nonlocal captured_timeout
            captured_timeout = timeout
            return await coro

        try:
            with patch("agent.auxiliary_client.get_text_auxiliary_client",
                        return_value=(mock_client, "model")), \
                 patch("tools.mcp_tool.asyncio.wait_for",
                       side_effect=capturing_wait_for):
                result = asyncio.run(callback(None, params))

            # Should succeed -- max_rpm clamped to 1, first request allowed
            assert result.content.text is not None
            # timeout clamped to 1 (minimum)
            assert captured_timeout == 1
            # max_tokens_cap clamped to 1, so min(2000, 1) = 1
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["max_tokens"] == 1
        finally:
            _sampling_counters.pop("clamp_srv", None)

    def test_nan_timeout_fallback_to_default(self):
        """NaN timeout value falls back to default."""
        from tools.mcp_tool import _make_sampling_callback, _DEFAULT_SAMPLING_TIMEOUT

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {"timeout": float("nan")})
        params = _make_sampling_params()

        captured_timeout = None

        async def capturing_wait_for(coro, *, timeout=None):
            nonlocal captured_timeout
            captured_timeout = timeout
            return await coro

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")), \
             patch("tools.mcp_tool.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=_make_llm_response()), \
             patch("tools.mcp_tool.asyncio.wait_for",
                   side_effect=capturing_wait_for):
            asyncio.run(callback(None, params))

        assert captured_timeout == _DEFAULT_SAMPLING_TIMEOUT

    def test_inf_max_rpm_fallback_to_default(self):
        """Infinity max_rpm value falls back to default."""
        from tools.mcp_tool import (
            _make_sampling_callback, _sampling_counters,
            _DEFAULT_SAMPLING_RATE_LIMIT,
        )

        callback = _make_sampling_callback("inf_rpm_srv", {"max_rpm": float("inf")})
        params = _make_sampling_params()

        # Fill counters to default rate limit -- should block if default was used
        _sampling_counters["inf_rpm_srv"] = [
            time.time() for _ in range(_DEFAULT_SAMPLING_RATE_LIMIT)
        ]

        try:
            result = asyncio.run(callback(None, params))
            assert isinstance(result, ErrorData)
            assert "rate limit" in result.message.lower()
        finally:
            _sampling_counters.pop("inf_rpm_srv", None)

    def test_unsupported_content_block_skipped(self):
        """Unknown content block types are skipped with a warning."""
        from tools.mcp_tool import _convert_sampling_messages

        unknown_block = SimpleNamespace(audio_data="raw_audio")  # no text, no data/mimeType
        text_block = SimpleNamespace(text="Hello")

        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=[text_block, unknown_block]),
        ])

        import logging as _logging
        with patch("tools.mcp_tool.logger") as mock_logger:
            result = _convert_sampling_messages(params)

        # Text block preserved, unknown block skipped
        assert len(result) == 1
        parts = result[0]["content"]
        assert len(parts) == 1
        assert parts[0] == {"type": "text", "text": "Hello"}
        mock_logger.warning.assert_called_once()
        assert "Unsupported" in mock_logger.warning.call_args[0][0]

    def test_unsupported_content_block_all_skipped_gives_empty_string(self):
        """All unknown content blocks produce empty string content."""
        from tools.mcp_tool import _convert_sampling_messages

        unknown_block = SimpleNamespace(audio_data="raw_audio")

        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=[unknown_block]),
        ])

        with patch("tools.mcp_tool.logger"):
            result = _convert_sampling_messages(params)

        assert result[0]["content"] == ""

    def test_sampling_error_helper_returns_error_data(self):
        """_sampling_error returns ErrorData when types are available."""
        from tools.mcp_tool import _sampling_error

        result = _sampling_error("test error", code=-42)
        assert isinstance(result, ErrorData)
        assert result.code == -42
        assert result.message == "test error"

    def test_sampling_error_helper_raises_without_types(self):
        """_sampling_error raises Exception when MCP types unavailable."""
        from tools.mcp_tool import _sampling_error

        with patch("tools.mcp_tool._MCP_SAMPLING_TYPES", False):
            with pytest.raises(Exception, match="fallback error"):
                _sampling_error("fallback error")

    def test_make_sampling_callback_returns_none_without_types(self):
        """_make_sampling_callback returns None when sampling types unavailable."""
        from tools.mcp_tool import _make_sampling_callback

        with patch("tools.mcp_tool._MCP_SAMPLING_TYPES", False):
            result = _make_sampling_callback("srv", {})

        assert result is None


# ---------------------------------------------------------------------------
# Phase 2: Tool content block conversion tests
# ---------------------------------------------------------------------------

class TestToolContentConversion:
    """Tests for tool_result and tool_use content block handling."""

    def test_convert_messages_tool_result_single(self):
        """Single tool_result content converts to OpenAI tool role message."""
        from tools.mcp_tool import _convert_sampling_messages

        tool_result = SimpleNamespace(
            toolUseId="call_123",
            content=[SimpleNamespace(text="Result from tool")],
        )
        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=tool_result),
        ])
        result = _convert_sampling_messages(params)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["content"] == "Result from tool"

    def test_convert_messages_tool_result_multiple(self):
        """List of tool_result blocks produces multiple tool role messages."""
        from tools.mcp_tool import _convert_sampling_messages

        results = [
            SimpleNamespace(
                toolUseId="call_1",
                content=[SimpleNamespace(text="Result 1")],
            ),
            SimpleNamespace(
                toolUseId="call_2",
                content=[SimpleNamespace(text="Result 2")],
            ),
        ]
        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=results),
        ])
        result = _convert_sampling_messages(params)
        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_2"

    def test_convert_messages_tool_use_content(self):
        """tool_use content blocks convert to assistant message with tool_calls."""
        from tools.mcp_tool import _convert_sampling_messages

        tool_use = SimpleNamespace(
            id="call_abc",
            name="get_weather",
            input={"city": "London"},
        )
        params = SimpleNamespace(messages=[
            SimpleNamespace(role="assistant", content=[tool_use]),
        ])
        result = _convert_sampling_messages(params)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "London"}

    def test_convert_messages_mixed_tool_use_and_text(self):
        """Mixed tool_use and text blocks produce correct assistant message."""
        from tools.mcp_tool import _convert_sampling_messages

        tool_use = SimpleNamespace(
            id="call_1",
            name="search",
            input={"q": "test"},
        )
        text_block = SimpleNamespace(text="Let me search for that")
        params = SimpleNamespace(messages=[
            SimpleNamespace(role="assistant", content=[text_block, tool_use]),
        ])
        result = _convert_sampling_messages(params)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me search for that"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search"

    def test_extract_tool_result_text_helper(self):
        """_extract_tool_result_text extracts text from content list."""
        from tools.mcp_tool import _extract_tool_result_text

        block = SimpleNamespace(
            content=[SimpleNamespace(text="line1"), SimpleNamespace(text="line2")],
        )
        assert _extract_tool_result_text(block) == "line1\nline2"

    def test_extract_tool_result_text_empty(self):
        """_extract_tool_result_text returns empty string for missing content."""
        from tools.mcp_tool import _extract_tool_result_text

        assert _extract_tool_result_text(SimpleNamespace(content=None)) == ""
        assert _extract_tool_result_text(SimpleNamespace()) == ""


# ---------------------------------------------------------------------------
# Phase 2: Tools/toolChoice forwarding tests
# ---------------------------------------------------------------------------

class TestToolsForwarding:
    """Tests for tools/toolChoice forwarding to LLM."""

    def setup_method(self):
        from tools.mcp_tool import _sampling_counters, _sampling_metrics, _tool_loop_counters
        _sampling_counters.clear()
        _sampling_metrics.clear()
        _tool_loop_counters.clear()

    def test_tools_forwarded_to_llm(self):
        """Server-provided tools are forwarded to the LLM call."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        server_tools = [
            SimpleNamespace(name="get_weather", description="Get weather", inputSchema={"type": "object"}),
        ]
        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(tools=server_tools)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"

    def test_tool_choice_auto(self):
        """toolChoice mode=auto maps to tool_choice=auto."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        server_tools = [SimpleNamespace(name="t", description="", inputSchema={})]
        tc = SimpleNamespace(mode="auto")
        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(tools=server_tools, tool_choice=tc)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"] == "auto"

    def test_tool_choice_required(self):
        """toolChoice mode=required maps to tool_choice=required."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        server_tools = [SimpleNamespace(name="t", description="", inputSchema={})]
        tc = SimpleNamespace(mode="required")
        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(tools=server_tools, tool_choice=tc)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"] == "required"

    def test_tool_choice_none(self):
        """toolChoice mode=none maps to tool_choice=none."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        server_tools = [SimpleNamespace(name="t", description="", inputSchema={})]
        tc = SimpleNamespace(mode="none")
        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(tools=server_tools, tool_choice=tc)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"] == "none"

    def test_no_tools_backward_compat(self):
        """No tools attribute in params -> no tools key in call_kwargs."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_tool_use_response_returns_tool_use_content(self):
        """LLM tool_calls response produces CreateMessageResult with tool_use content."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_tool_response()

        server_tools = [SimpleNamespace(name="get_weather", description="", inputSchema={})]
        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params(tools=server_tools)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            result = asyncio.run(callback(None, params))

        assert result.role == "assistant"
        assert result.stopReason == "toolUse"
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert result.content[0]["type"] == "tool_use"
        assert result.content[0]["name"] == "get_weather"
        assert result.content[0]["input"] == {"city": "London"}


# ---------------------------------------------------------------------------
# Phase 2: Audit metrics tests
# ---------------------------------------------------------------------------

class TestSamplingMetrics:
    """Tests for _sampling_metrics audit tracking."""

    def setup_method(self):
        from tools.mcp_tool import _sampling_counters, _sampling_metrics, _tool_loop_counters
        _sampling_counters.clear()
        _sampling_metrics.clear()
        _tool_loop_counters.clear()

    def test_sampling_metrics_increment_on_success(self):
        """Successful sampling increments requests counter."""
        from tools.mcp_tool import _make_sampling_callback, _sampling_metrics

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("metrics_srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        assert "metrics_srv" in _sampling_metrics
        assert _sampling_metrics["metrics_srv"]["requests"] == 1
        assert _sampling_metrics["metrics_srv"]["tokens_used"] == 42

    def test_sampling_metrics_increment_on_error(self):
        """Error during sampling increments errors counter."""
        from tools.mcp_tool import _make_sampling_callback, _sampling_metrics

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("fail")

        callback = _make_sampling_callback("err_srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        assert "err_srv" in _sampling_metrics
        assert _sampling_metrics["err_srv"]["errors"] == 1

    def test_sampling_metrics_tool_use_count(self):
        """Tool use response increments tool_use_count."""
        from tools.mcp_tool import _make_sampling_callback, _sampling_metrics

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_tool_response()

        callback = _make_sampling_callback("tu_srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            asyncio.run(callback(None, params))

        assert _sampling_metrics["tu_srv"]["tool_use_count"] == 1

    def test_get_mcp_status_includes_sampling_metrics(self):
        """get_mcp_status() includes sampling metrics when available."""
        from tools.mcp_tool import get_mcp_status, _servers, _sampling_metrics

        server = _make_mock_server("metric_srv", session=MagicMock(), tools=[])
        _servers["metric_srv"] = server
        _sampling_metrics["metric_srv"] = {
            "requests": 5, "errors": 1, "tokens_used": 200, "tool_use_count": 2,
        }

        try:
            with patch("tools.mcp_tool._load_mcp_config",
                        return_value={"metric_srv": {"command": "test"}}):
                status = get_mcp_status()

            entry = next(s for s in status if s["name"] == "metric_srv")
            assert "sampling" in entry
            assert entry["sampling"]["requests"] == 5
            assert entry["sampling"]["errors"] == 1
        finally:
            _servers.pop("metric_srv", None)
            _sampling_metrics.pop("metric_srv", None)


# ---------------------------------------------------------------------------
# Phase 2: allowed_models tests
# ---------------------------------------------------------------------------

class TestAllowedModels:
    """Tests for allowed_models whitelist governance."""

    def setup_method(self):
        from tools.mcp_tool import _sampling_counters, _sampling_metrics, _tool_loop_counters
        _sampling_counters.clear()
        _sampling_metrics.clear()
        _tool_loop_counters.clear()

    def test_allowed_models_empty_permits_any(self):
        """Empty allowed_models list permits any model."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {"allowed_models": []})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "some-model")):
            result = asyncio.run(callback(None, params))

        assert result.role == "assistant"

    def test_allowed_models_blocks_unlisted(self):
        """Model not in allowed_models returns ErrorData."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        callback = _make_sampling_callback("srv", {
            "allowed_models": ["gpt-4", "gemini-3-flash"],
        })
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "not-allowed-model")):
            result = asyncio.run(callback(None, params))

        assert isinstance(result, ErrorData)
        assert "not allowed" in result.message.lower()
        assert "not-allowed-model" in result.message

    def test_allowed_models_permits_listed(self):
        """Model in allowed_models proceeds normally."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("srv", {
            "allowed_models": ["gpt-4", "gemini-3-flash"],
        })
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "gpt-4")):
            result = asyncio.run(callback(None, params))

        assert result.role == "assistant"


# ---------------------------------------------------------------------------
# Phase 2: max_tool_rounds tests
# ---------------------------------------------------------------------------

class TestMaxToolRounds:
    """Tests for max_tool_rounds tool loop governance."""

    def setup_method(self):
        from tools.mcp_tool import _sampling_counters, _sampling_metrics, _tool_loop_counters
        _sampling_counters.clear()
        _sampling_metrics.clear()
        _tool_loop_counters.clear()

    def test_max_tool_rounds_enforced(self):
        """Exceeding max_tool_rounds returns ErrorData."""
        from tools.mcp_tool import _make_sampling_callback, _tool_loop_counters

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_tool_response()

        callback = _make_sampling_callback("loop_srv", {"max_tool_rounds": 2})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            # Round 1 -- allowed
            r1 = asyncio.run(callback(None, params))
            assert r1.stopReason == "toolUse"
            assert _tool_loop_counters.get("loop_srv") == 1

            # Round 2 -- allowed
            r2 = asyncio.run(callback(None, params))
            assert r2.stopReason == "toolUse"
            assert _tool_loop_counters.get("loop_srv") == 2

            # Round 3 -- exceeds limit
            r3 = asyncio.run(callback(None, params))
            assert isinstance(r3, ErrorData)
            assert "loop limit exceeded" in r3.message.lower()

    def test_max_tool_rounds_zero_disables(self):
        """max_tool_rounds=0 disables tool loops entirely."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_tool_response()

        callback = _make_sampling_callback("no_loop_srv", {"max_tool_rounds": 0})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            result = asyncio.run(callback(None, params))

        assert isinstance(result, ErrorData)
        assert "disabled" in result.message.lower()

    def test_max_tool_rounds_resets_on_end_turn(self):
        """Tool loop counter resets when LLM returns a normal text response."""
        from tools.mcp_tool import _make_sampling_callback, _tool_loop_counters

        mock_client = MagicMock()

        callback = _make_sampling_callback("reset_srv", {"max_tool_rounds": 5})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            # First call returns tool_calls
            mock_client.chat.completions.create.return_value = _make_llm_tool_response()
            asyncio.run(callback(None, params))
            assert _tool_loop_counters.get("reset_srv") == 1

            # Second call returns normal text -- should reset counter
            mock_client.chat.completions.create.return_value = _make_llm_response()
            asyncio.run(callback(None, params))
            assert "reset_srv" not in _tool_loop_counters

    def test_max_tool_rounds_default_5(self):
        """Default max_tool_rounds is 5."""
        from tools.mcp_tool import _make_sampling_callback, _tool_loop_counters

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_tool_response()

        # No max_tool_rounds in config -> defaults to 5
        callback = _make_sampling_callback("default_srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            # 5 rounds allowed
            for i in range(5):
                result = asyncio.run(callback(None, params))
                assert result.stopReason == "toolUse", f"Round {i+1} should succeed"

            # 6th round exceeds limit
            result = asyncio.run(callback(None, params))
            assert isinstance(result, ErrorData)
            assert "loop limit exceeded" in result.message.lower()

        _tool_loop_counters.pop("default_srv", None)


# ---------------------------------------------------------------------------
# Phase 2: log_level tests
# ---------------------------------------------------------------------------

class TestLogLevel:
    """Tests for config-driven audit log level."""

    def setup_method(self):
        from tools.mcp_tool import _sampling_counters, _sampling_metrics, _tool_loop_counters
        _sampling_counters.clear()
        _sampling_metrics.clear()
        _tool_loop_counters.clear()

    def test_log_level_config_applied(self):
        """Custom log_level changes the level of audit log messages."""
        import logging as _logging
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response()

        callback = _make_sampling_callback("log_srv", {"log_level": "debug"})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")), \
             patch("tools.mcp_tool.logger") as mock_logger:
            asyncio.run(callback(None, params))

        # logger.log should have been called with DEBUG level
        log_calls = [c for c in mock_logger.log.call_args_list if c[0][0] == _logging.DEBUG]
        assert len(log_calls) >= 1, "Expected at least one DEBUG-level log call"


# ---------------------------------------------------------------------------
# Phase 2 bug fixes: malformed args + mixed content list
# ---------------------------------------------------------------------------

class TestMalformedToolCallArgs:
    """Tests for graceful handling of malformed tool_calls arguments from LLM."""

    def setup_method(self):
        from tools.mcp_tool import _sampling_counters, _sampling_metrics, _tool_loop_counters
        _sampling_counters.clear()
        _sampling_metrics.clear()
        _tool_loop_counters.clear()

    def test_malformed_json_args_does_not_crash(self):
        """Malformed JSON in tool_calls arguments falls back to raw string."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        # Create a tool response with invalid JSON arguments
        bad_args = "{not valid json"
        mock_client.chat.completions.create.return_value = _make_llm_tool_response(
            tool_calls_data=[("call_1", "search", bad_args)]
        )

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            result = asyncio.run(callback(None, params))

        # Should NOT crash -- returns CreateMessageResult with raw string input
        assert result.role == "assistant"
        assert result.stopReason == "toolUse"
        assert result.content[0]["input"] == bad_args

    def test_valid_json_args_parsed_normally(self):
        """Valid JSON arguments are parsed as dict."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_tool_response(
            tool_calls_data=[("call_1", "search", '{"query": "test"}')]
        )

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            result = asyncio.run(callback(None, params))

        assert result.content[0]["input"] == {"query": "test"}

    def test_dict_args_passed_through(self):
        """Dict arguments (non-string) are passed through without parsing."""
        from tools.mcp_tool import _make_sampling_callback

        mock_client = MagicMock()
        # Simulate args already being a dict (some providers do this)
        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="test", arguments={"key": "val"}),
        )
        message = SimpleNamespace(content=None, tool_calls=[tc])
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[choice], model="m", usage=SimpleNamespace(total_tokens=10),
        )

        callback = _make_sampling_callback("srv", {})
        params = _make_sampling_params()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(mock_client, "model")):
            result = asyncio.run(callback(None, params))

        assert result.content[0]["input"] == {"key": "val"}


class TestMixedContentList:
    """Tests for mixed content lists (text + tool_result) in message conversion."""

    def test_mixed_text_and_tool_result(self):
        """List with both text and tool_result blocks preserves both."""
        from tools.mcp_tool import _convert_sampling_messages

        text_block = SimpleNamespace(text="Context information")
        tool_result = SimpleNamespace(
            toolUseId="call_1",
            content=[SimpleNamespace(text="Tool output")],
        )
        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=[text_block, tool_result]),
        ])
        result = _convert_sampling_messages(params)

        # Should produce 2 messages: one text, one tool
        assert len(result) == 2
        text_msgs = [m for m in result if m["role"] == "user"]
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(text_msgs) == 1
        assert text_msgs[0]["content"] == "Context information"
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["content"] == "Tool output"

    def test_mixed_list_unknown_block_warned(self):
        """Unknown block in mixed list is skipped with warning."""
        from tools.mcp_tool import _convert_sampling_messages

        tool_result = SimpleNamespace(
            toolUseId="call_1",
            content=[SimpleNamespace(text="Result")],
        )
        unknown = SimpleNamespace(audio_data="raw")
        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=[tool_result, unknown]),
        ])

        with patch("tools.mcp_tool.logger") as mock_logger:
            result = _convert_sampling_messages(params)

        # tool_result preserved, unknown skipped
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        mock_logger.warning.assert_called_once()

    def test_pure_text_image_list_still_works(self):
        """Pure text+image list (no tool_result) still works as before."""
        from tools.mcp_tool import _convert_sampling_messages

        text_block = SimpleNamespace(text="Describe this")
        image_block = SimpleNamespace(data="base64==", mimeType="image/png")
        params = SimpleNamespace(messages=[
            SimpleNamespace(role="user", content=[text_block, image_block]),
        ])
        result = _convert_sampling_messages(params)

        assert len(result) == 1
        parts = result[0]["content"]
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "text": "Describe this"}
        assert parts[1]["type"] == "image_url"
