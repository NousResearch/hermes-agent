"""Tests for the MCP (Model Context Protocol) client support.

All tests use mocks -- no real MCP servers or subprocesses are started.
"""

import asyncio
import json
import os
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
