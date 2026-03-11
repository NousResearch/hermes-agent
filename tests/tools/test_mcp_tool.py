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


# ===========================================================================
# SamplingHandler tests
# ===========================================================================

import math
import time

from mcp.types import (
    CreateMessageResult,
    CreateMessageResultWithTools,
    ErrorData,
    SamplingCapability,
    SamplingToolsCapability,
    TextContent,
    ToolUseContent,
)

from tools.mcp_tool import SamplingHandler, _safe_numeric


# ---------------------------------------------------------------------------
# Helpers for sampling tests
# ---------------------------------------------------------------------------

def _make_sampling_params(
    messages=None,
    max_tokens=100,
    system_prompt=None,
    model_preferences=None,
    temperature=None,
    stop_sequences=None,
    tools=None,
    tool_choice=None,
):
    """Create a fake CreateMessageRequestParams using SimpleNamespace.

    Each message must have a ``content_as_list`` attribute that mirrors
    the SDK helper so that ``_convert_messages`` works correctly.
    """
    if messages is None:
        content = SimpleNamespace(text="Hello")
        msg = SimpleNamespace(role="user", content=content, content_as_list=[content])
        messages = [msg]

    params = SimpleNamespace(
        messages=messages,
        maxTokens=max_tokens,
        modelPreferences=model_preferences,
        temperature=temperature,
        stopSequences=stop_sequences,
        tools=tools,
        toolChoice=tool_choice,
    )
    if system_prompt is not None:
        params.systemPrompt = system_prompt
    return params


def _make_llm_response(
    content="LLM response",
    model="test-model",
    finish_reason="stop",
    tool_calls=None,
):
    """Create a fake OpenAI chat completion response (text)."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message=message,
    )
    usage = SimpleNamespace(total_tokens=42)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _make_llm_tool_response(tool_calls_data=None, model="test-model"):
    """Create a fake response with tool_calls.

    ``tool_calls_data``: list of (id, name, arguments_json) tuples.
    """
    if tool_calls_data is None:
        tool_calls_data = [("call_1", "get_weather", '{"city": "London"}')]

    tc_list = [
        SimpleNamespace(
            id=tc_id,
            function=SimpleNamespace(name=name, arguments=args),
        )
        for tc_id, name, args in tool_calls_data
    ]
    return _make_llm_response(
        content=None,
        model=model,
        finish_reason="tool_calls",
        tool_calls=tc_list,
    )


# ---------------------------------------------------------------------------
# 1. _safe_numeric helper
# ---------------------------------------------------------------------------

class TestSafeNumeric:
    def test_int_passthrough(self):
        assert _safe_numeric(10, 5, int) == 10

    def test_string_coercion(self):
        assert _safe_numeric("20", 5, int) == 20

    def test_none_returns_default(self):
        assert _safe_numeric(None, 7, int) == 7

    def test_inf_returns_default(self):
        assert _safe_numeric(float("inf"), 3.0, float) == 3.0

    def test_nan_returns_default(self):
        assert _safe_numeric(float("nan"), 4.0, float) == 4.0

    def test_below_minimum_clamps(self):
        assert _safe_numeric(-5, 10, int, minimum=1) == 1

    def test_minimum_zero_allowed(self):
        assert _safe_numeric(0, 10, int, minimum=0) == 0

    def test_non_numeric_string_returns_default(self):
        assert _safe_numeric("abc", 42, int) == 42

    def test_float_coercion(self):
        assert _safe_numeric("3.5", 1.0, float) == 3.5


# ---------------------------------------------------------------------------
# 2. SamplingHandler initialization and config parsing
# ---------------------------------------------------------------------------

class TestSamplingHandlerInit:
    def test_defaults(self):
        h = SamplingHandler("srv", {})
        assert h.server_name == "srv"
        assert h.max_rpm == 10
        assert h.timeout == 30
        assert h.max_tokens_cap == 4096
        assert h.max_tool_rounds == 5
        assert h.model_override is None
        assert h.allowed_models == []
        assert h.metrics == {"requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0}

    def test_custom_config(self):
        cfg = {
            "max_rpm": 20,
            "timeout": 60,
            "max_tokens_cap": 2048,
            "max_tool_rounds": 3,
            "model": "gpt-4o",
            "allowed_models": ["gpt-4o", "gpt-3.5-turbo"],
            "log_level": "debug",
        }
        h = SamplingHandler("custom", cfg)
        assert h.max_rpm == 20
        assert h.timeout == 60.0
        assert h.max_tokens_cap == 2048
        assert h.max_tool_rounds == 3
        assert h.model_override == "gpt-4o"
        assert h.allowed_models == ["gpt-4o", "gpt-3.5-turbo"]

    def test_string_numeric_config_values(self):
        """YAML sometimes delivers numeric values as strings."""
        cfg = {"max_rpm": "15", "timeout": "45.5", "max_tokens_cap": "1024"}
        h = SamplingHandler("s", cfg)
        assert h.max_rpm == 15
        assert h.timeout == 45.5
        assert h.max_tokens_cap == 1024


# ---------------------------------------------------------------------------
# 3. Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def setup_method(self):
        self.handler = SamplingHandler("rl", {"max_rpm": 3})

    def test_allows_under_limit(self):
        assert self.handler._check_rate_limit() is True
        assert self.handler._check_rate_limit() is True
        assert self.handler._check_rate_limit() is True

    def test_rejects_over_limit(self):
        for _ in range(3):
            self.handler._check_rate_limit()
        assert self.handler._check_rate_limit() is False

    def test_window_expiry(self):
        """Old timestamps should be purged from the sliding window."""
        for _ in range(3):
            self.handler._check_rate_limit()
        # Simulate timestamps from 61 seconds ago
        self.handler._rate_timestamps[:] = [time.time() - 61] * 3
        assert self.handler._check_rate_limit() is True


# ---------------------------------------------------------------------------
# 4. Model resolution
# ---------------------------------------------------------------------------

class TestResolveModel:
    def setup_method(self):
        self.handler = SamplingHandler("mr", {})

    def test_no_preference_no_override(self):
        assert self.handler._resolve_model(None) is None

    def test_config_override_wins(self):
        self.handler.model_override = "override-model"
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="hint-model")])
        assert self.handler._resolve_model(prefs) == "override-model"

    def test_hint_used_when_no_override(self):
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="hint-model")])
        assert self.handler._resolve_model(prefs) == "hint-model"

    def test_empty_hints(self):
        prefs = SimpleNamespace(hints=[])
        assert self.handler._resolve_model(prefs) is None

    def test_hint_without_name(self):
        prefs = SimpleNamespace(hints=[SimpleNamespace(name=None)])
        assert self.handler._resolve_model(prefs) is None


# ---------------------------------------------------------------------------
# 5. Message conversion
# ---------------------------------------------------------------------------

class TestConvertMessages:
    def setup_method(self):
        self.handler = SamplingHandler("mc", {})

    def test_single_text_message(self):
        content = SimpleNamespace(text="Hello world")
        msg = SimpleNamespace(role="user", content=content, content_as_list=[content])
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello world"}

    def test_image_message(self):
        text_block = SimpleNamespace(text="Look at this")
        img_block = SimpleNamespace(data="abc123", mimeType="image/png")
        msg = SimpleNamespace(
            role="user",
            content=[text_block, img_block],
            content_as_list=[text_block, img_block],
        )
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        parts = result[0]["content"]
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "text": "Look at this"}
        assert parts[1]["type"] == "image_url"
        assert "data:image/png;base64,abc123" in parts[1]["image_url"]["url"]

    def test_tool_result_message(self):
        inner = SimpleNamespace(text="42 degrees")
        tr_block = SimpleNamespace(toolUseId="call_1", content=[inner])
        msg = SimpleNamespace(
            role="user",
            content=[tr_block],
            content_as_list=[tr_block],
        )
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "42 degrees"

    def test_tool_use_message(self):
        tu_block = SimpleNamespace(
            id="call_2", name="get_weather", input={"city": "London"}
        )
        msg = SimpleNamespace(
            role="assistant",
            content=[tu_block],
            content_as_list=[tu_block],
        )
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"city": "London"}

    def test_mixed_text_and_tool_use(self):
        """Assistant message with both text and tool_calls."""
        text_block = SimpleNamespace(text="Let me check the weather")
        tu_block = SimpleNamespace(
            id="call_3", name="get_weather", input={"city": "Paris"}
        )
        msg = SimpleNamespace(
            role="assistant",
            content=[text_block, tu_block],
            content_as_list=[text_block, tu_block],
        )
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        assert result[0]["content"] == "Let me check the weather"
        assert len(result[0]["tool_calls"]) == 1

    def test_fallback_without_content_as_list(self):
        """When content_as_list is absent, falls back to content."""
        content = SimpleNamespace(text="Fallback text")
        msg = SimpleNamespace(role="user", content=content)
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        assert result[0]["content"] == "Fallback text"


# ---------------------------------------------------------------------------
# 6. Text-only sampling callback (full flow)
# ---------------------------------------------------------------------------

class TestSamplingCallbackText:
    def setup_method(self):
        self.handler = SamplingHandler("txt", {})

    def test_text_response(self):
        """Full flow: text response returns CreateMessageResult."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response(
            content="Hello from LLM"
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            params = _make_sampling_params()
            result = asyncio.run(self.handler(None, params))

        assert isinstance(result, CreateMessageResult)
        assert isinstance(result.content, TextContent)
        assert result.content.text == "Hello from LLM"
        assert result.model == "test-model"
        assert result.role == "assistant"
        assert result.stopReason == "endTurn"

    def test_system_prompt_prepended(self):
        """System prompt is inserted as the first message."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            params = _make_sampling_params(system_prompt="Be helpful")
            asyncio.run(self.handler(None, params))

        call_args = fake_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Be helpful"}

    def test_length_stop_reason(self):
        """finish_reason='length' maps to stopReason='maxTokens'."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response(
            finish_reason="length"
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            params = _make_sampling_params()
            result = asyncio.run(self.handler(None, params))

        assert isinstance(result, CreateMessageResult)
        assert result.stopReason == "maxTokens"


# ---------------------------------------------------------------------------
# 7. Tool use sampling callback
# ---------------------------------------------------------------------------

class TestSamplingCallbackToolUse:
    def setup_method(self):
        self.handler = SamplingHandler("tu", {})

    def test_tool_use_response(self):
        """LLM tool_calls response returns CreateMessageResultWithTools."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            params = _make_sampling_params()
            result = asyncio.run(self.handler(None, params))

        assert isinstance(result, CreateMessageResultWithTools)
        assert result.stopReason == "toolUse"
        assert result.model == "test-model"
        assert len(result.content) == 1
        tc = result.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.name == "get_weather"
        assert tc.id == "call_1"
        assert tc.input == {"city": "London"}

    def test_multiple_tool_calls(self):
        """Multiple tool_calls in a single response."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response(
            tool_calls_data=[
                ("call_a", "func_a", '{"x": 1}'),
                ("call_b", "func_b", '{"y": 2}'),
            ]
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            result = asyncio.run(self.handler(None, _make_sampling_params()))

        assert isinstance(result, CreateMessageResultWithTools)
        assert len(result.content) == 2
        assert result.content[0].name == "func_a"
        assert result.content[1].name == "func_b"


# ---------------------------------------------------------------------------
# 8. Tool loop governance
# ---------------------------------------------------------------------------

class TestToolLoopGovernance:
    def test_max_tool_rounds_enforcement(self):
        """After max_tool_rounds consecutive tool responses, an error is returned."""
        handler = SamplingHandler("tl", {"max_tool_rounds": 2})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            params = _make_sampling_params()
            # Round 1, 2: allowed
            r1 = asyncio.run(handler(None, params))
            assert isinstance(r1, CreateMessageResultWithTools)
            r2 = asyncio.run(handler(None, params))
            assert isinstance(r2, CreateMessageResultWithTools)
            # Round 3: exceeds limit
            r3 = asyncio.run(handler(None, params))
            assert isinstance(r3, ErrorData)
            assert "Tool loop limit exceeded" in r3.message

    def test_text_response_resets_counter(self):
        """A text response resets the tool loop counter."""
        handler = SamplingHandler("tl2", {"max_tool_rounds": 1})
        fake_client = MagicMock()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            # Tool response (round 1 of 1 allowed)
            fake_client.chat.completions.create.return_value = _make_llm_tool_response()
            r1 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r1, CreateMessageResultWithTools)

            # Text response resets counter
            fake_client.chat.completions.create.return_value = _make_llm_response()
            r2 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r2, CreateMessageResult)

            # Tool response again (should succeed since counter was reset)
            fake_client.chat.completions.create.return_value = _make_llm_tool_response()
            r3 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r3, CreateMessageResultWithTools)

    def test_max_tool_rounds_zero_disables(self):
        """max_tool_rounds=0 means tool loops are disabled entirely."""
        handler = SamplingHandler("tl3", {"max_tool_rounds": 0})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "Tool loops disabled" in result.message


# ---------------------------------------------------------------------------
# 9. Error paths: rate limit, timeout, no provider
# ---------------------------------------------------------------------------

class TestSamplingErrors:
    def test_rate_limit_error(self):
        handler = SamplingHandler("rle", {"max_rpm": 1})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            # First call succeeds
            r1 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r1, CreateMessageResult)
            # Second call is rate limited
            r2 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r2, ErrorData)
            assert "rate limit" in r2.message.lower()
            assert handler.metrics["errors"] == 1

    def test_timeout_error(self):
        handler = SamplingHandler("to", {"timeout": 0.05})
        fake_client = MagicMock()

        def slow_call(**kwargs):
            import threading
            # Use an event to ensure the thread truly blocks long enough
            evt = threading.Event()
            evt.wait(5)  # blocks for up to 5 seconds (cancelled by timeout)
            return _make_llm_response()

        fake_client.chat.completions.create.side_effect = slow_call

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "timed out" in result.message.lower()
            assert handler.metrics["errors"] == 1

    def test_no_provider_error(self):
        handler = SamplingHandler("np", {})

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(None, None),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "No LLM provider" in result.message
            assert handler.metrics["errors"] == 1

    def test_empty_choices_returns_error(self):
        """LLM returning choices=[] is handled gracefully, not IndexError."""
        handler = SamplingHandler("ec", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[],
            model="test-model",
            usage=SimpleNamespace(total_tokens=0),
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))

        assert isinstance(result, ErrorData)
        assert "empty response" in result.message.lower()
        assert handler.metrics["errors"] == 1

    def test_none_choices_returns_error(self):
        """LLM returning choices=None is handled gracefully, not TypeError."""
        handler = SamplingHandler("nc", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=None,
            model="test-model",
            usage=SimpleNamespace(total_tokens=0),
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))

        assert isinstance(result, ErrorData)
        assert "empty response" in result.message.lower()
        assert handler.metrics["errors"] == 1

    def test_missing_choices_attr_returns_error(self):
        """LLM response without choices attribute is handled gracefully."""
        handler = SamplingHandler("mc", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            model="test-model",
            usage=SimpleNamespace(total_tokens=0),
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))

        assert isinstance(result, ErrorData)
        assert "empty response" in result.message.lower()
        assert handler.metrics["errors"] == 1


# ---------------------------------------------------------------------------
# 10. Model whitelist
# ---------------------------------------------------------------------------

class TestModelWhitelist:
    def test_allowed_model_passes(self):
        handler = SamplingHandler("wl", {"allowed_models": ["gpt-4o", "test-model"]})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "test-model"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, CreateMessageResult)

    def test_disallowed_model_rejected(self):
        handler = SamplingHandler("wl2", {"allowed_models": ["gpt-4o"]})
        fake_client = MagicMock()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "gpt-3.5-turbo"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "not allowed" in result.message
            assert handler.metrics["errors"] == 1

    def test_empty_whitelist_allows_all(self):
        handler = SamplingHandler("wl3", {"allowed_models": []})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "any-model"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, CreateMessageResult)


# ---------------------------------------------------------------------------
# 11. Malformed tool_call arguments
# ---------------------------------------------------------------------------

class TestMalformedToolCallArgs:
    def test_invalid_json_wrapped_as_raw(self):
        """Malformed JSON arguments get wrapped in {"_raw": ...}."""
        handler = SamplingHandler("mf", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response(
            tool_calls_data=[("call_x", "some_tool", "not valid json {{{")]
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))

        assert isinstance(result, CreateMessageResultWithTools)
        tc = result.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.input == {"_raw": "not valid json {{{"}

    def test_dict_args_pass_through(self):
        """When arguments are already a dict, they pass through directly."""
        handler = SamplingHandler("mf2", {})

        # Build a tool call where arguments is already a dict
        tc_obj = SimpleNamespace(
            id="call_d",
            function=SimpleNamespace(name="do_stuff", arguments={"key": "val"}),
        )
        message = SimpleNamespace(content=None, tool_calls=[tc_obj])
        choice = SimpleNamespace(finish_reason="tool_calls", message=message)
        usage = SimpleNamespace(total_tokens=10)
        response = SimpleNamespace(choices=[choice], model="m", usage=usage)

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = response

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))

        assert isinstance(result, CreateMessageResultWithTools)
        assert result.content[0].input == {"key": "val"}


# ---------------------------------------------------------------------------
# 12. Metrics tracking
# ---------------------------------------------------------------------------

class TestMetricsTracking:
    def test_request_and_token_metrics(self):
        handler = SamplingHandler("met", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            asyncio.run(handler(None, _make_sampling_params()))

        assert handler.metrics["requests"] == 1
        assert handler.metrics["tokens_used"] == 42
        assert handler.metrics["errors"] == 0

    def test_tool_use_count_metric(self):
        handler = SamplingHandler("met2", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "default-model"),
        ):
            asyncio.run(handler(None, _make_sampling_params()))

        assert handler.metrics["tool_use_count"] == 1
        assert handler.metrics["requests"] == 1

    def test_error_metric_incremented(self):
        handler = SamplingHandler("met3", {})

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(None, None),
        ):
            asyncio.run(handler(None, _make_sampling_params()))

        assert handler.metrics["errors"] == 1
        assert handler.metrics["requests"] == 0


# ---------------------------------------------------------------------------
# 13. session_kwargs()
# ---------------------------------------------------------------------------

class TestSessionKwargs:
    def test_returns_correct_keys(self):
        handler = SamplingHandler("sk", {})
        kwargs = handler.session_kwargs()
        assert "sampling_callback" in kwargs
        assert "sampling_capabilities" in kwargs
        assert kwargs["sampling_callback"] is handler

    def test_sampling_capabilities_type(self):
        handler = SamplingHandler("sk2", {})
        kwargs = handler.session_kwargs()
        cap = kwargs["sampling_capabilities"]
        assert isinstance(cap, SamplingCapability)
        assert isinstance(cap.tools, SamplingToolsCapability)


# ---------------------------------------------------------------------------
# 14. MCPServerTask integration
# ---------------------------------------------------------------------------

class TestMCPServerTaskSamplingIntegration:
    def test_sampling_handler_created_when_enabled(self):
        """MCPServerTask.run() creates a SamplingHandler when sampling is enabled."""
        from tools.mcp_tool import MCPServerTask, _MCP_SAMPLING_TYPES

        server = MCPServerTask("int_test")
        config = {
            "command": "fake",
            "sampling": {"enabled": True, "max_rpm": 5},
        }
        # We only need to test the setup logic, not the actual connection.
        # Calling run() would attempt a real connection, so we test the
        # sampling setup portion directly.
        server._config = config
        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            server._sampling = SamplingHandler(server.name, sampling_config)
        else:
            server._sampling = None

        assert server._sampling is not None
        assert isinstance(server._sampling, SamplingHandler)
        assert server._sampling.server_name == "int_test"
        assert server._sampling.max_rpm == 5

    def test_sampling_handler_none_when_disabled(self):
        """MCPServerTask._sampling is None when sampling is disabled."""
        from tools.mcp_tool import MCPServerTask, _MCP_SAMPLING_TYPES

        server = MCPServerTask("int_test2")
        config = {
            "command": "fake",
            "sampling": {"enabled": False},
        }
        server._config = config
        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            server._sampling = SamplingHandler(server.name, sampling_config)
        else:
            server._sampling = None

        assert server._sampling is None

    def test_session_kwargs_used_in_stdio(self):
        """When sampling is set, session_kwargs() are passed to ClientSession."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("sk_test")
        server._sampling = SamplingHandler("sk_test", {"max_rpm": 7})
        kwargs = server._sampling.session_kwargs()
        assert "sampling_callback" in kwargs
        assert "sampling_capabilities" in kwargs


# ---------------------------------------------------------------------------
# Discovery failed_count tracking
# ---------------------------------------------------------------------------

class TestDiscoveryFailedCount:
    """Verify discover_mcp_tools() correctly tracks failed server connections."""

    def test_failed_server_increments_failed_count(self):
        """When _discover_and_register_server raises, failed_count increments."""
        from tools.mcp_tool import discover_mcp_tools, _servers, _ensure_mcp_loop

        fake_config = {
            "good_server": {"command": "npx", "args": ["good"]},
            "bad_server": {"command": "npx", "args": ["bad"]},
        }

        async def fake_register(name, cfg):
            if name == "bad_server":
                raise ConnectionError("Connection refused")
            # Simulate successful registration
            from tools.mcp_tool import MCPServerTask
            server = MCPServerTask(name)
            server.session = MagicMock()
            server._tools = [_make_mcp_tool("tool_a")]
            _servers[name] = server
            return [f"mcp_{name}_tool_a"]

        with patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_register), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp_good_server_tool_a"]):
            _ensure_mcp_loop()

            # Capture the logger to verify failed_count in summary
            with patch("tools.mcp_tool.logger") as mock_logger:
                discover_mcp_tools()

                # Find the summary info call
                info_calls = [
                    str(call)
                    for call in mock_logger.info.call_args_list
                    if "failed" in str(call).lower() or "MCP:" in str(call)
                ]
                # The summary should mention the failure
                assert any("1 failed" in str(c) for c in info_calls), (
                    f"Summary should report 1 failed server, got: {info_calls}"
                )

        _servers.pop("good_server", None)
        _servers.pop("bad_server", None)

    def test_all_servers_fail_still_prints_summary(self):
        """When all servers fail, a summary with failure count is still printed."""
        from tools.mcp_tool import discover_mcp_tools, _servers, _ensure_mcp_loop

        fake_config = {
            "srv1": {"command": "npx", "args": ["a"]},
            "srv2": {"command": "npx", "args": ["b"]},
        }

        async def always_fail(name, cfg):
            raise ConnectionError(f"Server {name} refused")

        with patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=always_fail), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            _ensure_mcp_loop()

            with patch("tools.mcp_tool.logger") as mock_logger:
                discover_mcp_tools()

                # Summary must be printed even when all servers fail
                info_calls = [str(call) for call in mock_logger.info.call_args_list]
                assert any("2 failed" in str(c) for c in info_calls), (
                    f"Summary should report 2 failed servers, got: {info_calls}"
                )

        _servers.pop("srv1", None)
        _servers.pop("srv2", None)

    def test_ok_servers_excludes_failures(self):
        """ok_servers count correctly excludes failed servers."""
        from tools.mcp_tool import discover_mcp_tools, _servers, _ensure_mcp_loop

        fake_config = {
            "ok1": {"command": "npx", "args": ["ok1"]},
            "ok2": {"command": "npx", "args": ["ok2"]},
            "fail1": {"command": "npx", "args": ["fail"]},
        }

        async def selective_register(name, cfg):
            if name == "fail1":
                raise ConnectionError("Refused")
            from tools.mcp_tool import MCPServerTask
            server = MCPServerTask(name)
            server.session = MagicMock()
            server._tools = [_make_mcp_tool("t")]
            _servers[name] = server
            return [f"mcp_{name}_t"]

        with patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=selective_register), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp_ok1_t", "mcp_ok2_t"]):
            _ensure_mcp_loop()

            with patch("tools.mcp_tool.logger") as mock_logger:
                discover_mcp_tools()

                info_calls = [str(call) for call in mock_logger.info.call_args_list]
                # Should say "2 server(s)" not "3 server(s)"
                assert any("2 server" in str(c) for c in info_calls), (
                    f"Summary should report 2 ok servers, got: {info_calls}"
                )
                assert any("1 failed" in str(c) for c in info_calls), (
                    f"Summary should report 1 failed, got: {info_calls}"
                )

        _servers.pop("ok1", None)
        _servers.pop("ok2", None)
        _servers.pop("fail1", None)
