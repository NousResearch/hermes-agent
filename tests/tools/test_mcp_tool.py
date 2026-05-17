"""Tests for the MCP (Model Context Protocol) client support.

All tests use mocks -- no real MCP servers or subprocesses are started.
"""

import asyncio
import json
import os
import threading
import time
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

    def test_object_schema_without_properties_gets_normalized(self):
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(
            name="ask",
            description="Ask Crawl4AI",
            input_schema={"type": "object"},
        )
        schema = _convert_mcp_schema("crawl4ai", mcp_tool)

        assert schema["parameters"] == {"type": "object", "properties": {}}

    def test_definitions_refs_are_rewritten_to_defs(self):
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(
            name="submit",
            description="Submit a payload",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"$ref": "#/definitions/Payload"},
                },
                "required": ["input"],
                "definitions": {
                    "Payload": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    }
                },
            },
        )

        schema = _convert_mcp_schema("forms", mcp_tool)

        assert schema["parameters"]["properties"]["input"]["$ref"] == "#/$defs/Payload"
        assert "$defs" in schema["parameters"]
        assert "definitions" not in schema["parameters"]

    def test_nested_definition_refs_are_rewritten_recursively(self):
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(
            name="nested",
            description="Nested schema",
            input_schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/Entry"},
                    },
                },
                "definitions": {
                    "Entry": {
                        "type": "object",
                        "properties": {
                            "child": {"$ref": "#/definitions/Child"},
                        },
                    },
                    "Child": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                        },
                    },
                },
            },
        )

        schema = _convert_mcp_schema("forms", mcp_tool)

        assert schema["parameters"]["properties"]["items"]["items"]["$ref"] == "#/$defs/Entry"
        assert schema["parameters"]["$defs"]["Entry"]["properties"]["child"]["$ref"] == "#/$defs/Child"

    def test_missing_type_on_object_is_coerced(self):
        """Schemas that describe an object but omit ``type`` get type='object'."""
        from tools.mcp_tool import _normalize_mcp_input_schema

        schema = _normalize_mcp_input_schema({
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        })

        assert schema["type"] == "object"
        assert schema["properties"]["q"]["type"] == "string"
        assert schema["required"] == ["q"]

    def test_null_type_on_object_is_coerced(self):
        """type: None should be treated like missing type (common MCP server bug)."""
        from tools.mcp_tool import _normalize_mcp_input_schema

        schema = _normalize_mcp_input_schema({
            "type": None,
            "properties": {"x": {"type": "integer"}},
        })

        assert schema["type"] == "object"

    def test_required_pruned_when_property_missing(self):
        """Gemini 400s on required names that don't exist in properties."""
        from tools.mcp_tool import _normalize_mcp_input_schema

        schema = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", "ghost", "phantom"],
        })

        assert schema["required"] == ["a"]

    def test_required_removed_when_all_names_dangle(self):
        from tools.mcp_tool import _normalize_mcp_input_schema

        schema = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {},
            "required": ["ghost"],
        })

        assert "required" not in schema

    def test_required_pruning_applies_recursively_inside_nested_objects(self):
        """Nested object schemas also get required pruning."""
        from tools.mcp_tool import _normalize_mcp_input_schema

        schema = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {"field": {"type": "string"}},
                    "required": ["field", "missing"],
                },
            },
        })

        assert schema["properties"]["filter"]["required"] == ["field"]

    def test_object_in_array_items_gets_properties_filled(self):
        """Array-item object schemas without properties get an empty dict."""
        from tools.mcp_tool import _normalize_mcp_input_schema

        schema = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
        })

        assert schema["properties"]["items"]["items"]["properties"] == {}

    def test_optional_nullable_field_is_collapsed_to_non_null_schema(self):
        """Anthropic rejects MCP/Pydantic anyOf-null optional parameter schemas."""
        from tools.mcp_tool import _normalize_mcp_input_schema

        schema = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "workdir": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Optional working directory",
                },
            },
            "required": ["command"],
        })

        assert schema["properties"]["workdir"] == {
            "type": "string",
            "nullable": True,
            "default": None,
            "description": "Optional working directory",
        }
        assert schema["required"] == ["command"]

    def test_nested_nullable_array_items_are_collapsed(self):
        from tools.mcp_tool import _normalize_mcp_input_schema

        schema = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {
                "filters": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {"field": {"type": "string"}},
                            },
                            {"type": "null"},
                        ]
                    },
                }
            },
        })

        assert schema["properties"]["filters"]["items"] == {
            "type": "object",
            "properties": {"field": {"type": "string"}},
            "nullable": True,
        }

    def test_convert_mcp_schema_survives_missing_inputschema_attribute(self):
        """A Tool object without .inputSchema must not crash registration."""
        import types

        from tools.mcp_tool import _convert_mcp_schema

        bare_tool = types.SimpleNamespace(name="probe", description="Probe")
        schema = _convert_mcp_schema("srv", bare_tool)

        assert schema["name"] == "mcp_srv_probe"
        assert schema["parameters"] == {"type": "object", "properties": {}}

    def test_convert_mcp_schema_with_none_inputschema(self):
        """Tool with inputSchema=None produces a valid empty object schema."""
        import types

        from tools.mcp_tool import _convert_mcp_schema

        # Note: _make_mcp_tool(input_schema=None) falls back to a default —
        # build the namespace directly so .inputSchema really is None.
        mcp_tool = types.SimpleNamespace(name="probe", description="Probe", inputSchema=None)
        schema = _convert_mcp_schema("srv", mcp_tool)

        assert schema["parameters"] == {"type": "object", "properties": {}}

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
# MCP loop runner
# ---------------------------------------------------------------------------

class TestRunOnMcpLoop:
    def test_scheduler_failure_closes_factory_coroutine(self):
        """If run_coroutine_threadsafe raises, the factory's coroutine is closed."""
        import gc
        import warnings
        import tools.mcp_tool as mcp

        created = {"coro": None}

        async def _sample():
            return "ok"

        def factory():
            created["coro"] = _sample()
            return created["coro"]

        fake_loop = MagicMock()
        fake_loop.is_running.return_value = True

        with patch.object(mcp, "_mcp_loop", fake_loop):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch(
                    "agent.async_utils.asyncio.run_coroutine_threadsafe",
                    side_effect=RuntimeError("scheduler down"),
                ):
                    with pytest.raises(RuntimeError):
                        mcp._run_on_mcp_loop(factory)
                gc.collect()

        assert created["coro"] is not None
        assert created["coro"].cr_frame is None
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_sample" in str(w.message)
        ]
        assert runtime_warnings == []

    def test_dead_loop_closes_passed_coroutine(self):
        """If loop is None, a passed coroutine (not factory) is closed."""
        import gc
        import warnings
        import tools.mcp_tool as mcp

        async def _sample():
            return "ok"

        coro = _sample()
        with patch.object(mcp, "_mcp_loop", None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError, match="not running"):
                    mcp._run_on_mcp_loop(coro)
                gc.collect()

        assert coro.cr_frame is None
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_sample" in str(w.message)
        ]
        assert runtime_warnings == []


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

class TestToolHandler:
    """Tool handlers are sync functions that schedule work on the MCP loop."""

    def _patch_mcp_loop(self, coro_side_effect=None):
        """Return a patch for _run_on_mcp_loop that runs the coroutine directly."""
        def fake_run(coro_or_factory, timeout=30):
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)
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

    def test_interrupted_call_returns_interrupted_error(self):
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        server = _make_mock_server("test_srv", session=mock_session)
        _servers["test_srv"] = server

        try:
            handler = _make_tool_handler("test_srv", "greet", 120)
            def _interrupting_run(coro_or_factory, timeout=30):
                coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                coro.close()
                raise InterruptedError("User sent a new message")
            with patch(
                "tools.mcp_tool._run_on_mcp_loop",
                side_effect=_interrupting_run,
            ):
                result = json.loads(handler({}))
            assert result == {"error": "MCP call interrupted: user sent a new message"}
        finally:
            _servers.pop("test_srv", None)


class TestRunOnMCPLoopInterrupts:
    def test_interrupt_cancels_waiting_mcp_call(self):
        import tools.mcp_tool as mcp_mod
        from tools.interrupt import set_interrupt

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()

        cancelled = threading.Event()

        async def _slow_call():
            try:
                await asyncio.sleep(5)
                return "done"
            except asyncio.CancelledError:
                cancelled.set()
                raise

        old_loop = mcp_mod._mcp_loop
        old_thread = mcp_mod._mcp_thread
        mcp_mod._mcp_loop = loop
        mcp_mod._mcp_thread = thread

        waiter_tid = threading.current_thread().ident

        def _interrupt_soon():
            time.sleep(0.2)
            set_interrupt(True, waiter_tid)

        interrupter = threading.Thread(target=_interrupt_soon, daemon=True)
        interrupter.start()

        try:
            with pytest.raises(InterruptedError, match="User sent a new message"):
                mcp_mod._run_on_mcp_loop(_slow_call(), timeout=2)

            deadline = time.time() + 2
            while time.time() < deadline and not cancelled.is_set():
                time.sleep(0.05)
            assert cancelled.is_set()
        finally:
            set_interrupt(False, waiter_tid)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
            loop.close()
            mcp_mod._mcp_loop = old_loop
            mcp_mod._mcp_thread = old_thread

    def test_timeout_reports_elapsed_and_configured_timeout(self):
        import tools.mcp_tool as mcp_mod

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()

        cancelled = threading.Event()

        async def _slow_call():
            try:
                await asyncio.sleep(5)
                return "done"
            except asyncio.CancelledError:
                cancelled.set()
                raise

        old_loop = mcp_mod._mcp_loop
        old_thread = mcp_mod._mcp_thread
        mcp_mod._mcp_loop = loop
        mcp_mod._mcp_thread = thread

        try:
            with pytest.raises(TimeoutError, match=r"MCP call timed out after .*configured timeout: 0.2s"):
                mcp_mod._run_on_mcp_loop(_slow_call(), timeout=0.2)

            deadline = time.time() + 2
            while time.time() < deadline and not cancelled.is_set():
                time.sleep(0.05)
            assert cancelled.is_set()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
            loop.close()
            mcp_mod._mcp_loop = old_loop
            mcp_mod._mcp_thread = old_thread


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

    def test_toolset_resolves_live_from_registry(self):
        """MCP toolsets resolve through the live registry without TOOLSETS mutation."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask
        from toolsets import resolve_toolset, validate_toolset

        mock_registry = ToolRegistry()
        mock_tools = [_make_mcp_tool("ping", "Ping")]
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            asyncio.run(
                _discover_and_register_server("myserver", {"command": "test"})
            )

            assert validate_toolset("myserver") is True
            assert validate_toolset("mcp-myserver") is True
            assert "mcp_myserver_ping" in resolve_toolset("myserver")
            assert "mcp_myserver_ping" in resolve_toolset("mcp-myserver")

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

    def test_refresh_tools_deregisters_removed_tools(self):
        """Dynamic refresh removes stale registry entries for deleted tools."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import MCPServerTask

        mock_registry = ToolRegistry()
        server = MCPServerTask("srv")
        server._config = {"command": "test"}
        server._tools = [_make_mcp_tool("old"), _make_mcp_tool("keep")]
        server._registered_tool_names = ["mcp_srv_old", "mcp_srv_keep"]
        server.session = MagicMock()
        server.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_make_mcp_tool("keep"), _make_mcp_tool("new")])
        )

        with patch("tools.registry.registry", mock_registry):
            mock_registry.register(
                name="mcp_srv_old",
                toolset="mcp-srv",
                schema={"name": "mcp_srv_old", "description": "Old"},
                handler=lambda *_args, **_kwargs: "{}",
            )
            mock_registry.register(
                name="mcp_srv_keep",
                toolset="mcp-srv",
                schema={"name": "mcp_srv_keep", "description": "Keep"},
                handler=lambda *_args, **_kwargs: "{}",
            )

            asyncio.run(server._refresh_tools())

            names = mock_registry.get_all_tool_names()
            assert "mcp_srv_old" not in names
            assert "mcp_srv_keep" in names
            assert "mcp_srv_new" in names
            assert set(server._registered_tool_names) == {
                "mcp_srv_keep",
                "mcp_srv_new",
                "mcp_srv_list_resources",
                "mcp_srv_read_resource",
                "mcp_srv_list_prompts",
                "mcp_srv_get_prompt",
            }

    def test_schedule_tools_refresh_keeps_task_until_done(self):
        """Background refresh tasks are strongly referenced and then discarded."""
        from tools.mcp_tool import MCPServerTask

        async def _test():
            started = asyncio.Event()
            finish = asyncio.Event()
            server = MCPServerTask("srv")

            async def fake_refresh(_server):
                started.set()
                await finish.wait()

            with patch.object(MCPServerTask, "_refresh_tools", new=fake_refresh):
                server._schedule_tools_refresh()

                await started.wait()
                assert len(server._pending_refresh_tasks) == 1
                task = next(iter(server._pending_refresh_tasks))
                assert not task.done()

                finish.set()
                await task
                await asyncio.sleep(0)
                assert server._pending_refresh_tasks == set()

        asyncio.run(_test())

    def test_shutdown_cancels_pending_refresh_tasks(self):
        """shutdown() cancels in-flight background refresh tasks."""
        from tools.mcp_tool import MCPServerTask

        async def _test():
            started = asyncio.Event()
            cancelled = asyncio.Event()
            server = MCPServerTask("srv")

            async def fake_refresh(_server):
                started.set()
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            with patch.object(MCPServerTask, "_refresh_tools", new=fake_refresh):
                server._schedule_tools_refresh()
                await started.wait()

                await server.shutdown()

            assert cancelled.is_set()
            assert server._pending_refresh_tasks == set()

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
    def test_mcp_tools_resolve_through_server_aliases(self):
        """Discovered MCP tools resolve through raw server-name aliases."""
        from tools.mcp_tool import MCPServerTask
        from tools.registry import ToolRegistry
        from toolsets import resolve_toolset, validate_toolset

        mock_tools = [_make_mcp_tool("list_files", "List files")]
        mock_session = MagicMock()
        mock_registry = ToolRegistry()

        fresh_servers = {}

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        fake_config = {"fs": {"command": "npx", "args": []}}

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._servers", fresh_servers), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()

            assert "mcp_fs_list_files" in result
            assert validate_toolset("fs") is True
            assert validate_toolset("mcp-fs") is True
            assert "mcp_fs_list_files" in resolve_toolset("fs")
            assert "mcp_fs_list_files" in resolve_toolset("mcp-fs")

    def test_server_toolset_skips_builtin_collision(self):
        """MCP raw aliases never overwrite a built-in toolset name."""
        from tools.mcp_tool import MCPServerTask
        from tools.registry import ToolRegistry
        from toolsets import resolve_toolset, validate_toolset

        mock_tools = [_make_mcp_tool("run", "Run command")]
        mock_session = MagicMock()
        fresh_servers = {}
        mock_registry = ToolRegistry()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        fake_toolsets = {
            "hermes-cli": {"tools": ["terminal"], "description": "CLI", "includes": []},
            # Built-in toolset named "terminal" — must not be overwritten
            "terminal": {"tools": ["terminal"], "description": "Terminal tools", "includes": []},
        }
        fake_config = {"terminal": {"command": "npx", "args": []}}

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._servers", fresh_servers), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry), \
             patch("toolsets.TOOLSETS", fake_toolsets):
            from tools.mcp_tool import discover_mcp_tools
            discover_mcp_tools()

            assert fake_toolsets["terminal"]["description"] == "Terminal tools"
            assert "mcp_terminal_run" not in resolve_toolset("terminal")
            assert validate_toolset("mcp-terminal") is True
            assert "mcp_terminal_run" in resolve_toolset("mcp-terminal")

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

    def test_shutdown_deregisters_registered_tools(self):
        """shutdown_mcp_servers removes MCP tools and their raw alias."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import MCPServerTask, shutdown_mcp_servers, _servers
        from tools.registry import registry
        from toolsets import resolve_toolset, validate_toolset

        _servers.clear()
        registry.register(
            name="mcp_test_ping",
            toolset="mcp-test",
            schema={
                "name": "mcp_test_ping",
                "description": "Ping",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda *_args, **_kwargs: "{}",
        )
        registry.register_toolset_alias("test", "mcp-test")

        server = MCPServerTask("test")
        server._registered_tool_names = ["mcp_test_ping"]
        _servers["test"] = server

        mcp_mod._ensure_mcp_loop()
        try:
            assert validate_toolset("test") is True
            assert "mcp_test_ping" in resolve_toolset("test")
            shutdown_mcp_servers()
        finally:
            mcp_mod._mcp_loop = None
            mcp_mod._mcp_thread = None

        assert "mcp_test_ping" not in registry.get_all_tool_names()
        assert validate_toolset("test") is False

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

    def test_http_seeds_initial_protocol_header(self):
        from tools.mcp_tool import LATEST_PROTOCOL_VERSION, MCPServerTask

        server = MCPServerTask("remote")
        captured = {}

        class DummyAsyncClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class DummyTransportCtx:
            async def __aenter__(self):
                return MagicMock(), MagicMock(), (lambda: None)

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class DummySession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def initialize(self):
                return None

        class DummyLegacyTransportCtx:
            def __init__(self, **kwargs):
                captured["legacy_headers"] = kwargs.get("headers")

            async def __aenter__(self):
                return MagicMock(), MagicMock(), (lambda: None)

            async def __aexit__(self, exc_type, exc, tb):
                return False

        async def _discover_tools(self):
            self._shutdown_event.set()

        async def _run(config, *, new_http):
            captured.clear()
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
                 patch("tools.mcp_tool._MCP_NEW_HTTP", new_http), \
                 patch("httpx.AsyncClient", DummyAsyncClient), \
                 patch("tools.mcp_tool.streamable_http_client", return_value=DummyTransportCtx()), \
                 patch("tools.mcp_tool.streamablehttp_client", side_effect=lambda url, **kwargs: DummyLegacyTransportCtx(**kwargs)), \
                 patch("tools.mcp_tool.ClientSession", DummySession), \
                 patch.object(MCPServerTask, "_discover_tools", _discover_tools):
                await server._run_http(config)

        asyncio.run(_run({"url": "https://example.com/mcp"}, new_http=True))
        assert captured["headers"]["mcp-protocol-version"] == LATEST_PROTOCOL_VERSION

        asyncio.run(_run({
            "url": "https://example.com/mcp",
            "headers": {"mcp-protocol-version": "custom-version"},
        }, new_http=True))
        assert captured["headers"]["mcp-protocol-version"] == "custom-version"

        asyncio.run(_run({
            "url": "https://example.com/mcp",
            "headers": {"MCP-Protocol-Version": "custom-version"},
        }, new_http=True))
        assert captured["headers"]["MCP-Protocol-Version"] == "custom-version"
        assert "mcp-protocol-version" not in captured["headers"]

        asyncio.run(_run({"url": "https://example.com/mcp"}, new_http=False))
        assert captured["legacy_headers"]["mcp-protocol-version"] == LATEST_PROTOCOL_VERSION

        asyncio.run(_run({
            "url": "https://example.com/mcp",
            "headers": {"MCP-Protocol-Version": "custom-version"},
        }, new_http=False))
        assert captured["legacy_headers"]["MCP-Protocol-Version"] == "custom-version"
        assert "mcp-protocol-version" not in captured["legacy_headers"]


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
        """First connection failure retries up to _MAX_INITIAL_CONNECT_RETRIES times.

        Before the MCP resilience fix, initial failures gave up immediately.
        Now they retry with backoff to handle transient DNS/network blips.
        """
        from tools.mcp_tool import MCPServerTask, _MAX_INITIAL_CONNECT_RETRIES

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

            # Now retries up to _MAX_INITIAL_CONNECT_RETRIES before giving up
            assert run_count == _MAX_INITIAL_CONNECT_RETRIES + 1
            assert server._error is not None
            assert "cannot connect" in str(server._error)

        asyncio.run(_test())

    def test_initial_oauth_failure_does_not_retry(self):
        """Initial OAuth failures stop immediately to avoid repeated browser prompts."""
        from tools.mcp_tool import MCPServerTask

        run_count = 0
        target_server = None
        oauth_error = RuntimeError("Token exchange failed (400): Unknown client_id")

        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            nonlocal run_count, target_server
            run_count += 1
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            raise oauth_error

        async def _test():
            nonlocal target_server
            server = MCPServerTask("oauth_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
                 patch("tools.mcp_tool._is_auth_error", return_value=True), \
                 patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await server.run({"command": "test"})

            assert run_count == 1
            assert server._error is oauth_error
            assert server._ready.is_set()
            assert mock_sleep.await_count == 0

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
                def fake_run(coro, timeout=30):
                    coro.close()
                    return json.dumps({"result": "ok"})

                mock_run.side_effect = fake_run
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
        def fake_run(coro_or_factory, timeout=30):
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)
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

class _CompatType:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


try:
    from mcp.types import (
        CreateMessageResult,
        ErrorData,
        SamplingCapability,
        TextContent,
    )
except ImportError:
    CreateMessageResult = _CompatType
    ErrorData = _CompatType
    SamplingCapability = _CompatType
    TextContent = _CompatType

try:
    from mcp.types import CreateMessageResultWithTools
except ImportError:
    CreateMessageResultWithTools = _CompatType

try:
    from mcp.types import SamplingToolsCapability
except ImportError:
    SamplingToolsCapability = _CompatType

try:
    from mcp.types import ToolUseContent
except ImportError:
    ToolUseContent = _CompatType

from tools.mcp_tool import (
    CreateMessageResultWithTools,
    SamplingHandler,
    SamplingToolsCapability,
    ToolUseContent,
    _safe_numeric,
)


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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ) as mock_call:
            params = _make_sampling_params(system_prompt="Be helpful")
            asyncio.run(self.handler(None, params))

        call_args = mock_call.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Be helpful"}

    def test_server_tools_with_object_schema_are_normalized(self):
        """Server-provided tools should gain empty properties for object schemas."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()
        server_tool = SimpleNamespace(
            name="ask",
            description="Ask Crawl4AI",
            inputSchema={"type": "object"},
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ) as mock_call:
            params = _make_sampling_params(tools=[server_tool])
            asyncio.run(self.handler(None, params))

        tools = mock_call.call_args.kwargs["tools"]
        assert tools == [{
            "type": "function",
            "function": {
                "name": "ask",
                "description": "Ask Crawl4AI",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

    def test_length_stop_reason(self):
        """finish_reason='length' maps to stopReason='maxTokens'."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response(
            finish_reason="length"
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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

        # Use a list to hold the current response, so the side_effect can
        # pick up changes between calls.
        responses = [_make_llm_tool_response()]

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=lambda **kw: responses[0],
        ):
            # Tool response (round 1 of 1 allowed)
            r1 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r1, CreateMessageResultWithTools)

            # Text response resets counter
            responses[0] = _make_llm_response()
            r2 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r2, CreateMessageResult)

            # Tool response again (should succeed since counter was reset)
            responses[0] = _make_llm_tool_response()
            r3 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r3, CreateMessageResultWithTools)

    def test_max_tool_rounds_zero_disables(self):
        """max_tool_rounds=0 means tool loops are disabled entirely."""
        handler = SamplingHandler("tl3", {"max_tool_rounds": 0})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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

        def slow_call(**kwargs):
            import threading
            evt = threading.Event()
            evt.wait(5)  # blocks for up to 5 seconds (cancelled by timeout)
            return _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=slow_call,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "timed out" in result.message.lower()
            assert handler.metrics["errors"] == 1

    def test_no_provider_error(self):
        handler = SamplingHandler("np", {})

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("No LLM provider configured"),
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, CreateMessageResult)

    def test_disallowed_model_rejected(self):
        handler = SamplingHandler("wl2", {"allowed_models": ["gpt-4o"], "model": "test-model"})
        fake_client = MagicMock()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
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
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            asyncio.run(handler(None, _make_sampling_params()))

        assert handler.metrics["tool_use_count"] == 1
        assert handler.metrics["requests"] == 1

    def test_error_metric_incremented(self):
        handler = SamplingHandler("met3", {})

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("No LLM provider configured"),
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


class TestMCPSelectiveToolLoading:
    """Tests for per-server MCP filtering and utility tool policies."""

    def _make_server(self, name, tool_names, session=None):
        server = _make_mock_server(
            name,
            session=session or SimpleNamespace(),
            tools=[_make_mcp_tool(n, n) for n in tool_names],
        )
        return server

    def _run_discover(self, name, tool_names, config, session=None):
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers

        mock_registry = ToolRegistry()
        server = self._make_server(name, tool_names, session=session)

        async def fake_connect(_name, _config):
            return server

        async def run():
            with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
                 patch("tools.registry.registry", mock_registry), \
                 patch("toolsets.create_custom_toolset"):
                return await _discover_and_register_server(name, config)

        try:
            registered = asyncio.run(run())
        finally:
            _servers.pop(name, None)
        return registered, mock_registry

    def test_include_takes_precedence_over_exclude(self):
        config = {
            "url": "https://mcp.example.com",
            "tools": {
                "include": ["create_service"],
                "exclude": ["create_service", "delete_service"],
            },
        }
        registered, _ = self._run_discover(
            "ink",
            ["create_service", "delete_service", "list_services"],
            config,
            session=SimpleNamespace(),
        )
        assert registered == ["mcp_ink_create_service"]

    def test_exclude_filter_registers_all_except_listed_tools(self):
        config = {
            "url": "https://mcp.example.com",
            "tools": {"exclude": ["delete_service"]},
        }
        registered, _ = self._run_discover(
            "ink_exclude",
            ["create_service", "delete_service", "list_services"],
            config,
            session=SimpleNamespace(),
        )
        assert registered == [
            "mcp_ink_exclude_create_service",
            "mcp_ink_exclude_list_services",
        ]

    def test_include_filter_skips_utility_tools_without_capabilities(self):
        config = {
            "url": "https://mcp.example.com",
            "tools": {"include": ["create_service"]},
        }
        registered, mock_registry = self._run_discover(
            "ink_no_caps",
            ["create_service", "delete_service"],
            config,
            session=SimpleNamespace(),
        )
        assert registered == ["mcp_ink_no_caps_create_service"]
        assert set(mock_registry.get_all_tool_names()) == {"mcp_ink_no_caps_create_service"}

    def test_no_filter_registers_all_server_tools_when_no_utilities_supported(self):
        registered, _ = self._run_discover(
            "ink_no_filter",
            ["create_service", "delete_service", "list_services"],
            {"url": "https://mcp.example.com"},
            session=SimpleNamespace(),
        )
        assert registered == [
            "mcp_ink_no_filter_create_service",
            "mcp_ink_no_filter_delete_service",
            "mcp_ink_no_filter_list_services",
        ]

    def test_resources_and_prompts_can_be_disabled_explicitly(self):
        session = SimpleNamespace(
            list_resources=AsyncMock(),
            read_resource=AsyncMock(),
            list_prompts=AsyncMock(),
            get_prompt=AsyncMock(),
        )
        config = {
            "url": "https://mcp.example.com",
            "tools": {
                "resources": False,
                "prompts": False,
            },
        }
        registered, _ = self._run_discover(
            "ink_disabled_utils",
            ["create_service"],
            config,
            session=session,
        )
        assert registered == ["mcp_ink_disabled_utils_create_service"]

    def test_registers_only_utility_tools_supported_by_server_capabilities(self):
        session = SimpleNamespace(
            list_resources=AsyncMock(return_value=SimpleNamespace(resources=[])),
            read_resource=AsyncMock(return_value=SimpleNamespace(contents=[])),
        )
        registered, _ = self._run_discover(
            "ink_resources_only",
            ["create_service"],
            {"url": "https://mcp.example.com"},
            session=session,
        )
        assert "mcp_ink_resources_only_create_service" in registered
        assert "mcp_ink_resources_only_list_resources" in registered
        assert "mcp_ink_resources_only_read_resource" in registered
        assert "mcp_ink_resources_only_list_prompts" not in registered
        assert "mcp_ink_resources_only_get_prompt" not in registered

    def test_existing_tool_names_reflect_registered_subset(self):
        from tools.mcp_tool import _existing_tool_names, _servers, _discover_and_register_server
        from tools.registry import ToolRegistry

        mock_registry = ToolRegistry()
        server = self._make_server(
            "ink_existing",
            ["create_service", "delete_service"],
            session=SimpleNamespace(),
        )

        async def fake_connect(_name, _config):
            return server

        async def run():
            with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
                 patch.dict("tools.mcp_tool._servers", {}, clear=True), \
                 patch("tools.registry.registry", mock_registry), \
                 patch("toolsets.create_custom_toolset"):
                registered = await _discover_and_register_server(
                    "ink_existing",
                    {"url": "https://mcp.example.com", "tools": {"include": ["create_service"]}},
                )
                return registered, _existing_tool_names()

        try:
            registered, existing = asyncio.run(run())
            assert registered == ["mcp_ink_existing_create_service"]
            assert existing == ["mcp_ink_existing_create_service"]
        finally:
            _servers.pop("ink_existing", None)

    def test_no_toolset_created_when_everything_is_filtered_out(self):
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers

        mock_registry = ToolRegistry()
        server = self._make_server("ink_none", ["create_service"], session=SimpleNamespace())
        mock_create = MagicMock()

        async def fake_connect(_name, _config):
            return server

        async def run():
            with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
                 patch("tools.registry.registry", mock_registry), \
                 patch("toolsets.create_custom_toolset", mock_create):
                return await _discover_and_register_server(
                    "ink_none",
                    {
                        "url": "https://mcp.example.com",
                        "tools": {
                            "include": ["missing_tool"],
                            "resources": False,
                            "prompts": False,
                        },
                    },
                )

        try:
            registered = asyncio.run(run())
            assert registered == []
            mock_create.assert_not_called()
            assert mock_registry.get_all_tool_names() == []
        finally:
            _servers.pop("ink_none", None)

    def test_enabled_false_skips_connection_attempt(self):
        from tools.mcp_tool import discover_mcp_tools

        connect_called = []

        async def fake_connect(name, config):
            connect_called.append(name)
            return self._make_server(name, ["create_service"])

        fake_config = {
            "ink": {
                "url": "https://mcp.example.com",
                "enabled": False,
            }
        }
        fake_toolsets = {
            "hermes-cli": {"tools": [], "description": "CLI", "includes": []},
        }

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._servers", {}), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("toolsets.TOOLSETS", fake_toolsets):
            result = discover_mcp_tools()

        assert connect_called == []
        assert result == []


# ---------------------------------------------------------------------------
# Tool name collision protection
# ---------------------------------------------------------------------------

class TestRegistryCollisionWarning:
    """registry.register() warns when a tool name is overwritten by a different toolset."""

    def test_overwrite_different_toolset_logs_warning(self, caplog):
        """Overwriting a tool from a different toolset is REJECTED with an error."""
        from tools.registry import ToolRegistry
        import logging

        reg = ToolRegistry()
        schema = {"name": "my_tool", "description": "test", "parameters": {"type": "object", "properties": {}}}
        handler = lambda args, **kw: "{}"

        reg.register(name="my_tool", toolset="builtin", schema=schema, handler=handler)

        with caplog.at_level(logging.ERROR, logger="tools.registry"):
            reg.register(name="my_tool", toolset="mcp-ext", schema=schema, handler=handler)

        assert any("rejected" in r.message.lower() for r in caplog.records)
        assert any("builtin" in r.message and "mcp-ext" in r.message for r in caplog.records)
        # The original tool should still be from 'builtin', not overwritten
        assert reg.get_toolset_for_tool("my_tool") == "builtin"

    def test_overwrite_same_toolset_no_warning(self, caplog):
        """Re-registering within the same toolset is silent (e.g. reconnect)."""
        from tools.registry import ToolRegistry
        import logging

        reg = ToolRegistry()
        schema = {"name": "my_tool", "description": "test", "parameters": {"type": "object", "properties": {}}}
        handler = lambda args, **kw: "{}"

        reg.register(name="my_tool", toolset="mcp-server", schema=schema, handler=handler)

        with caplog.at_level(logging.WARNING, logger="tools.registry"):
            reg.register(name="my_tool", toolset="mcp-server", schema=schema, handler=handler)

        assert not any("collision" in r.message.lower() for r in caplog.records)


class TestMCPBuiltinCollisionGuard:
    """MCP tools that collide with built-in tool names are skipped."""

    def test_mcp_tool_skipped_when_builtin_exists(self):
        """An MCP tool whose prefixed name collides with a built-in is skipped."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()

        # Pre-register a "built-in" tool with the name that the MCP tool would produce.
        # Server "abc", tool "search" → mcp_abc_search
        builtin_schema = {
            "name": "mcp_abc_search",
            "description": "A hypothetical built-in",
            "parameters": {"type": "object", "properties": {}},
        }
        mock_registry.register(
            name="mcp_abc_search", toolset="web",
            schema=builtin_schema, handler=lambda a, **k: "{}",
        )

        mock_tools = [_make_mcp_tool("search", "Search the web")]
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("abc", {"command": "test", "args": []})
            )

        # The MCP tool should have been skipped — built-in preserved.
        assert "mcp_abc_search" not in registered
        assert mock_registry.get_toolset_for_tool("mcp_abc_search") == "web"

        _servers.pop("abc", None)

    def test_mcp_tool_registered_when_no_builtin_collision(self):
        """MCP tools register normally when there's no collision."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()
        mock_tools = [_make_mcp_tool("web_search", "Search the web")]
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("minimax", {"command": "test", "args": []})
            )

        assert "mcp_minimax_web_search" in registered
        assert mock_registry.get_toolset_for_tool("mcp_minimax_web_search") == "mcp-minimax"

        _servers.pop("minimax", None)

    def test_mcp_tool_allowed_when_collision_is_another_mcp(self):
        """Collision between two MCP toolsets is allowed (last wins)."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()

        # Pre-register an MCP tool from a different server.
        mcp_schema = {
            "name": "mcp_srv_do_thing",
            "description": "From another MCP server",
            "parameters": {"type": "object", "properties": {}},
        }
        mock_registry.register(
            name="mcp_srv_do_thing", toolset="mcp-old",
            schema=mcp_schema, handler=lambda a, **k: "{}",
        )

        mock_tools = [_make_mcp_tool("do_thing", "Do a thing")]
        mock_session = MagicMock()

        async def fake_connect(name, config):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("srv", {"command": "test", "args": []})
            )

        # MCP-to-MCP collision is allowed — the new server wins.
        assert "mcp_srv_do_thing" in registered
        assert mock_registry.get_toolset_for_tool("mcp_srv_do_thing") == "mcp-srv"

        _servers.pop("srv", None)


# ---------------------------------------------------------------------------
# sanitize_mcp_name_component
# ---------------------------------------------------------------------------


class TestSanitizeMcpNameComponent:
    """Verify sanitize_mcp_name_component handles all edge cases."""

    def test_hyphens_replaced(self):
        from tools.mcp_tool import sanitize_mcp_name_component
        assert sanitize_mcp_name_component("my-server") == "my_server"

    def test_dots_replaced(self):
        from tools.mcp_tool import sanitize_mcp_name_component
        assert sanitize_mcp_name_component("ai.exa") == "ai_exa"

    def test_slashes_replaced(self):
        from tools.mcp_tool import sanitize_mcp_name_component
        assert sanitize_mcp_name_component("ai.exa/exa") == "ai_exa_exa"

    def test_mixed_special_characters(self):
        from tools.mcp_tool import sanitize_mcp_name_component
        assert sanitize_mcp_name_component("@scope/my-pkg.v2") == "_scope_my_pkg_v2"

    def test_alphanumeric_and_underscores_preserved(self):
        from tools.mcp_tool import sanitize_mcp_name_component
        assert sanitize_mcp_name_component("my_server_123") == "my_server_123"

    def test_empty_string(self):
        from tools.mcp_tool import sanitize_mcp_name_component
        assert sanitize_mcp_name_component("") == ""

    def test_none_returns_empty(self):
        from tools.mcp_tool import sanitize_mcp_name_component
        assert sanitize_mcp_name_component(None) == ""

    def test_slash_in_convert_mcp_schema(self):
        """Server names with slashes produce valid tool names via _convert_mcp_schema."""
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(name="search")
        schema = _convert_mcp_schema("ai.exa/exa", mcp_tool)
        assert schema["name"] == "mcp_ai_exa_exa_search"
        # Must match Anthropic's pattern: ^[a-zA-Z0-9_-]{1,128}$
        import re
        assert re.match(r"^[a-zA-Z0-9_-]{1,128}$", schema["name"])

    def test_slash_in_build_utility_schemas(self):
        """Server names with slashes produce valid utility tool names."""
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("ai.exa/exa")
        for s in schemas:
            name = s["schema"]["name"]
            assert "/" not in name
            assert "." not in name

    def test_slash_in_server_alias_resolution(self):
        """Server names with slashes resolve through their live MCP alias."""
        from tools.registry import ToolRegistry
        from toolsets import resolve_toolset, validate_toolset

        reg = ToolRegistry()
        reg.register(
            name="mcp_ai_exa_exa_search",
            toolset="mcp-ai.exa/exa",
            schema={"name": "mcp_ai_exa_exa_search", "description": "Search", "parameters": {"type": "object", "properties": {}}},
            handler=lambda *_args, **_kwargs: "{}",
        )
        reg.register_toolset_alias("ai.exa/exa", "mcp-ai.exa/exa")

        with patch("tools.registry.registry", reg):
            assert validate_toolset("ai.exa/exa") is True
            assert "mcp_ai_exa_exa_search" in resolve_toolset("ai.exa/exa")


# ---------------------------------------------------------------------------
# register_mcp_servers public API
# ---------------------------------------------------------------------------


class TestRegisterMcpServers:
    """Verify the new register_mcp_servers() public API."""

    def test_empty_servers_returns_empty(self):
        from tools.mcp_tool import register_mcp_servers

        with patch("tools.mcp_tool._MCP_AVAILABLE", True):
            result = register_mcp_servers({})
        assert result == []

    def test_mcp_not_available_returns_empty(self):
        from tools.mcp_tool import register_mcp_servers

        with patch("tools.mcp_tool._MCP_AVAILABLE", False):
            result = register_mcp_servers({"srv": {"command": "test"}})
        assert result == []

    def test_skips_already_connected_servers(self):
        from tools.mcp_tool import register_mcp_servers, _servers

        mock_server = _make_mock_server("existing")
        _servers["existing"] = mock_server

        try:
            with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._existing_tool_names", return_value=["mcp_existing_tool"]):
                result = register_mcp_servers({"existing": {"command": "test"}})
            assert result == ["mcp_existing_tool"]
        finally:
            _servers.pop("existing", None)

    def test_skips_disabled_servers(self):
        from tools.mcp_tool import register_mcp_servers, _servers

        try:
            with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._existing_tool_names", return_value=[]):
                result = register_mcp_servers({"srv": {"command": "test", "enabled": False}})
            assert result == []
        finally:
            _servers.pop("srv", None)

    def test_connects_new_servers(self):
        from tools.mcp_tool import register_mcp_servers, _servers, _ensure_mcp_loop

        fake_config = {"my_server": {"command": "npx", "args": ["test"]}}

        async def fake_register(name, cfg):
            server = _make_mock_server(name)
            server._registered_tool_names = ["mcp_my_server_tool1"]
            _servers[name] = server
            return ["mcp_my_server_tool1"]

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_register), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp_my_server_tool1"]):
            _ensure_mcp_loop()
            result = register_mcp_servers(fake_config)

        assert "mcp_my_server_tool1" in result
        _servers.pop("my_server", None)

    def test_logs_summary_on_success(self):
        from tools.mcp_tool import register_mcp_servers, _servers, _ensure_mcp_loop

        fake_config = {"srv": {"command": "npx", "args": ["test"]}}

        async def fake_register(name, cfg):
            server = _make_mock_server(name)
            server._registered_tool_names = ["mcp_srv_t1", "mcp_srv_t2"]
            _servers[name] = server
            return ["mcp_srv_t1", "mcp_srv_t2"]

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_register), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp_srv_t1", "mcp_srv_t2"]):
            _ensure_mcp_loop()

            with patch("tools.mcp_tool.logger") as mock_logger:
                register_mcp_servers(fake_config)

                info_calls = [str(c) for c in mock_logger.info.call_args_list]
                assert any("2 tool(s)" in c and "1 server(s)" in c for c in info_calls), (
                    f"Summary should report 2 tools from 1 server, got: {info_calls}"
                )

        _servers.pop("srv", None)


# ---------------------------------------------------------------------------
# Tests for parallel tool call support (port from openai/codex#17667)
# ---------------------------------------------------------------------------

class TestMcpParallelToolCalls:
    """Tests for the supports_parallel_tool_calls config option."""

    def test_is_mcp_tool_parallel_safe_non_mcp_tool(self):
        """Non-MCP tool names always return False."""
        from tools.mcp_tool import is_mcp_tool_parallel_safe
        assert is_mcp_tool_parallel_safe("web_search") is False
        assert is_mcp_tool_parallel_safe("read_file") is False
        assert is_mcp_tool_parallel_safe("terminal") is False
        assert is_mcp_tool_parallel_safe("") is False

    def test_is_mcp_tool_parallel_safe_no_servers(self):
        """MCP tool from unknown server returns False."""
        from tools.mcp_tool import is_mcp_tool_parallel_safe, _parallel_safe_servers, _lock
        with _lock:
            _parallel_safe_servers.clear()
        assert is_mcp_tool_parallel_safe("mcp_docs_search") is False

    def test_is_mcp_tool_parallel_safe_with_flag(self):
        """MCP tool from a parallel-safe server returns True."""
        from tools.mcp_tool import is_mcp_tool_parallel_safe, _parallel_safe_servers, _lock
        with _lock:
            _parallel_safe_servers.add("docs")
        try:
            assert is_mcp_tool_parallel_safe("mcp_docs_search") is True
            assert is_mcp_tool_parallel_safe("mcp_docs_read_file") is True
            # Different server should be False
            assert is_mcp_tool_parallel_safe("mcp_github_list_repos") is False
        finally:
            with _lock:
                _parallel_safe_servers.discard("docs")

    def test_is_mcp_tool_parallel_safe_server_with_underscores(self):
        """Server names containing underscores are correctly matched."""
        from tools.mcp_tool import is_mcp_tool_parallel_safe, _parallel_safe_servers, _lock
        with _lock:
            _parallel_safe_servers.add("my_server")
        try:
            assert is_mcp_tool_parallel_safe("mcp_my_server_query") is True
        finally:
            with _lock:
                _parallel_safe_servers.discard("my_server")

    def test_is_mcp_tool_parallel_safe_no_tool_suffix(self):
        """Tool name that is just 'mcp_{server}' without a tool part returns False."""
        from tools.mcp_tool import is_mcp_tool_parallel_safe, _parallel_safe_servers, _lock
        with _lock:
            _parallel_safe_servers.add("docs")
        try:
            # "mcp_docs" has no tool part after the server name
            assert is_mcp_tool_parallel_safe("mcp_docs") is False
            # "mcp_docs_" has empty tool part
            assert is_mcp_tool_parallel_safe("mcp_docs_") is False
        finally:
            with _lock:
                _parallel_safe_servers.discard("docs")

    def test_register_mcp_servers_tracks_parallel_flag(self):
        """register_mcp_servers populates _parallel_safe_servers from config."""
        from tools.mcp_tool import (
            register_mcp_servers, _parallel_safe_servers, _lock,
            sanitize_mcp_name_component,
        )
        fake_config = {
            "parallel_srv": {
                "command": "echo",
                "supports_parallel_tool_calls": True,
            },
            "serial_srv": {
                "command": "echo",
                "supports_parallel_tool_calls": False,
            },
            "default_srv": {
                "command": "echo",
                # no supports_parallel_tool_calls key
            },
        }
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop"), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            register_mcp_servers(fake_config)

        with _lock:
            assert sanitize_mcp_name_component("parallel_srv") in _parallel_safe_servers
            assert sanitize_mcp_name_component("serial_srv") not in _parallel_safe_servers
            assert sanitize_mcp_name_component("default_srv") not in _parallel_safe_servers
            # Cleanup
            _parallel_safe_servers.discard(sanitize_mcp_name_component("parallel_srv"))

    def test_register_mcp_servers_removes_parallel_flag_on_toggle(self):
        """Toggling supports_parallel_tool_calls to false removes server from the set."""
        from tools.mcp_tool import (
            register_mcp_servers, _parallel_safe_servers, _lock,
            sanitize_mcp_name_component,
        )

        # First registration: parallel enabled
        config_on = {
            "toggle_srv": {
                "command": "echo",
                "supports_parallel_tool_calls": True,
            },
        }
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop"), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            register_mcp_servers(config_on)
        with _lock:
            assert sanitize_mcp_name_component("toggle_srv") in _parallel_safe_servers

        # Second registration: parallel disabled
        config_off = {
            "toggle_srv": {
                "command": "echo",
                "supports_parallel_tool_calls": False,
            },
        }
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop"), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            register_mcp_servers(config_off)
        with _lock:
            assert sanitize_mcp_name_component("toggle_srv") not in _parallel_safe_servers
