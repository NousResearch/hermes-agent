"""Tests for the MCP (Model Context Protocol) client support.

All tests use mocks -- no real MCP servers or subprocesses are started.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stop_reason(result):
    """Read a sampling result's stop reason across the mcp 1.x -> 2.x rename.

    ``CreateMessageResult.stopReason`` became ``.stop_reason`` in mcp 2.0
    (camelCase survives only as the serialization alias, which pydantic does
    not expose to attribute access).
    """
    from tools.mcp_tool import mcp_field

    return mcp_field(result, "stop_reason", "stopReason")


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


class TestFilterMCPChildren:
    def test_filters_gateway_children_by_argv_marker(self, monkeypatch):
        """Non-MCP children start with an interpreter/binary, not the marker."""
        import sys

        import tools.mcp_tool as mcp_tool

        cmdlines = {
            101: [
                "/usr/bin/python3",
                "-m",
                "tui_gateway.slash_worker",
                "--session-key",
                "abc",
            ],
            102: [
                "/usr/bin/java",
                "-jar",
                "/opt/jdtls/plugins/org.eclipse.equinox.launcher_1.7.0.jar",
            ],
            103: ["/usr/bin/node", "server.js"],
        }

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid

            def cmdline(self):
                return cmdlines[self.pid]

        fake_psutil = SimpleNamespace(
            Process=FakeProcess,
            NoSuchProcess=ProcessLookupError,
            AccessDenied=PermissionError,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        assert mcp_tool._filter_mcp_children({101, 102, 103}) == {103}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadMCPConfig:

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

    def test_portable_servers_merge_after_native_interpolation(self):
        native = {"native": {"command": "node", "args": ["${PORT}"]}}
        portable = {
            "agent-plugin-demo__worker": {
                "command": "python",
                "args": ["${UNKNOWN}"],
                "cwd": "/plugin",
            }
        }
        manager = SimpleNamespace(get_portable_mcp_servers=lambda: portable)
        with (
            patch("hermes_cli.config.load_config", return_value={"mcp_servers": native}),
            patch("hermes_cli.plugins.discover_plugins"),
            patch("hermes_cli.plugins.get_plugin_manager", return_value=manager),
            patch.dict(os.environ, {"PORT": "3000"}),
        ):
            from tools.mcp_tool import _load_mcp_config

            result = _load_mcp_config()

        assert result["native"]["args"] == ["3000"]
        assert result["agent-plugin-demo__worker"]["args"] == ["${UNKNOWN}"]

    def test_portable_server_resolves_through_real_plugin_discovery(
        self, tmp_path, monkeypatch
    ):
        import json
        import yaml
        from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1
        from hermes_cli import plugins as plugins_mod

        home = tmp_path / "home"
        plugin = home / "plugins" / "portable"
        plugin.mkdir(parents=True)
        (plugin / "plugin.json").write_text(
            json.dumps({"$schema": PLUGIN_SCHEMA_V1, "name": "portable.test"})
        )
        (plugin / "mcp.json").write_text(
            json.dumps(
                {
                    "$schema": MCP_SCHEMA_V1,
                    "mcpServers": {
                        "worker": {"type": "stdio", "command": "python"}
                    },
                }
            )
        )
        home.mkdir(exist_ok=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": ["portable.test"]}})
        )
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
        monkeypatch.setattr(plugins_mod, "_plugin_manager", None)

        from tools.mcp_tool import _load_mcp_config

        result = _load_mcp_config()

        [server] = result.values()
        assert server["command"] == "python"
        assert server["cwd"] == str(plugin.resolve())
        assert server["env"]["PLUGIN_ROOT"] == str(plugin.resolve())
        assert server["env"]["PLUGIN_DATA"].startswith(str(home / "plugin-data"))
        assert "agent_plugin" not in server


class TestMCPParallelSafetyProvenance:
    def test_parallel_safe_servers_keep_exact_raw_names(self, monkeypatch):
        import tools.mcp_tool as mcp_tool

        first = SimpleNamespace(session=object(), _registered_tool_names=[])
        second = SimpleNamespace(session=object(), _registered_tool_names=[])

        with mcp_tool._lock:
            saved_servers = dict(mcp_tool._servers)
            saved_parallel = set(mcp_tool._parallel_safe_servers)
            mcp_tool._servers.clear()
            mcp_tool._servers.update({"foo-bar": first, "foo_bar": second})
            mcp_tool._parallel_safe_servers.clear()

        try:
            monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
            monkeypatch.setattr(
                mcp_tool, "_filter_suspicious_mcp_servers", lambda servers: servers
            )
            mcp_tool.register_mcp_servers(
                {
                    "foo-bar": {"supports_parallel_tool_calls": True},
                    "foo_bar": {"supports_parallel_tool_calls": False},
                }
            )
            with mcp_tool._lock:
                assert "foo-bar" in mcp_tool._parallel_safe_servers
                assert "foo_bar" not in mcp_tool._parallel_safe_servers
        finally:
            with mcp_tool._lock:
                mcp_tool._servers.clear()
                mcp_tool._servers.update(saved_servers)
                mcp_tool._parallel_safe_servers.clear()
                mcp_tool._parallel_safe_servers.update(saved_parallel)

    def test_tool_provenance_keeps_exact_raw_server_names(self):
        import tools.mcp_tool as mcp_tool

        first_tool = "mcp__foo_bar__first"
        second_tool = "mcp__foo_bar__second"
        with mcp_tool._lock:
            saved_map = dict(mcp_tool._mcp_tool_server_names)
            saved_parallel = set(mcp_tool._parallel_safe_servers)
            mcp_tool._mcp_tool_server_names.clear()
            mcp_tool._parallel_safe_servers.clear()
            mcp_tool._parallel_safe_servers.add("foo-bar")

        try:
            mcp_tool._track_mcp_tool_server(first_tool, "foo-bar")
            mcp_tool._track_mcp_tool_server(second_tool, "foo_bar")

            assert mcp_tool.is_mcp_tool_parallel_safe(first_tool) is True
            assert mcp_tool.is_mcp_tool_parallel_safe(second_tool) is False
            assert mcp_tool.get_registered_mcp_server_names() == {
                "foo-bar",
                "foo_bar",
            }
        finally:
            with mcp_tool._lock:
                mcp_tool._mcp_tool_server_names.clear()
                mcp_tool._mcp_tool_server_names.update(saved_map)
                mcp_tool._parallel_safe_servers.clear()
                mcp_tool._parallel_safe_servers.update(saved_parallel)

class TestMCPStatus:
    def test_status_distinguishes_configured_connecting_failed_and_disabled(
        self, monkeypatch
    ):
        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(
            mcp_tool,
            "_load_mcp_config",
            lambda: {
                "configured": {"command": "docker", "args": ["mcp", "gateway", "run"]},
                "connecting": {"command": "slow-mcp"},
                "failed": {"command": "bad-mcp"},
                "disabled": {"command": "off-mcp", "enabled": False},
            },
        )
        with mcp_tool._lock:
            saved_servers = dict(mcp_tool._servers)
            saved_connecting = set(mcp_tool._server_connecting)
            saved_errors = dict(mcp_tool._server_connect_errors)
            mcp_tool._servers.clear()
            mcp_tool._server_connecting.clear()
            mcp_tool._server_connect_errors.clear()
            mcp_tool._server_connecting.add("connecting")
            mcp_tool._server_connect_errors["failed"] = "Connection closed"

        try:
            statuses = {
                entry["name"]: entry
                for entry in mcp_tool.get_mcp_status()
            }
        finally:
            with mcp_tool._lock:
                mcp_tool._servers.clear()
                mcp_tool._servers.update(saved_servers)
                mcp_tool._server_connecting.clear()
                mcp_tool._server_connecting.update(saved_connecting)
                mcp_tool._server_connect_errors.clear()
                mcp_tool._server_connect_errors.update(saved_errors)

        assert statuses["configured"]["status"] == "configured"
        assert statuses["configured"]["connected"] is False
        assert statuses["configured"]["disabled"] is False
        assert statuses["connecting"]["status"] == "connecting"
        assert statuses["failed"]["status"] == "failed"
        assert statuses["failed"]["error"] == "Connection closed"
        assert statuses["disabled"]["status"] == "disabled"
        assert statuses["disabled"]["disabled"] is True


class TestLifecycleConfig:
    def test_get_lifecycle_seconds_accepts_top_level_and_nested_values(self):
        from tools.mcp_tool import _get_lifecycle_seconds

        assert (
            _get_lifecycle_seconds(
                {"idle_timeout_seconds": "3.5"},
                "idle_timeout_seconds",
            )
            == 3.5
        )
        assert _get_lifecycle_seconds(
            {"lifecycle": {"max_lifetime_seconds": 42}},
            "max_lifetime_seconds",
        ) == 42.0

    def test_get_lifecycle_seconds_ignores_invalid_values(self, caplog):
        from tools.mcp_tool import _get_lifecycle_seconds

        assert (
            _get_lifecycle_seconds(
                {"idle_timeout_seconds": "soon"},
                "idle_timeout_seconds",
            )
            is None
        )
        assert (
            _get_lifecycle_seconds(
                {"idle_timeout_seconds": -1},
                "idle_timeout_seconds",
            )
            is None
        )

        messages = [record.getMessage() for record in caplog.records]
        assert any("must be a number of seconds" in msg for msg in messages)
        assert any("must be positive" in msg for msg in messages)


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

class TestSchemaConversion:
    def test_converts_mcp_tool_to_hermes_schema(self):
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(name="read_file", description="Read a file")
        schema = _convert_mcp_schema("filesystem", mcp_tool)

        assert schema["name"] == "mcp__filesystem__read_file"
        assert schema["description"] == "Read a file"
        assert "properties" in schema["parameters"]

    def test_definitions_as_property_name_is_preserved(self):
        """A tool parameter literally named ``definitions`` must not be renamed.

        Regression: the rewrite that promotes the legacy ``definitions``
        meta-keyword to ``$defs`` used to fire for *any* key named
        ``definitions`` anywhere in the tree, including inside ``properties``
        dicts. That turned user-facing parameter names into ``$defs``, which
        Anthropic and OpenAI both reject because ``$`` is not in the
        ``^[a-zA-Z0-9_.-]{1,64}$`` property-name pattern. Real-world repro: a
        CI/pipelines MCP tool whose ``definitions`` parameter is an array of
        pipeline-definition IDs.
        """
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(
            name="pipelines_build",
            description="List pipeline builds",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "definitions": {
                        "description": "Array of build definition IDs to filter builds.",
                    },
                    "top": {"type": "integer"},
                },
            },
        )

        schema = _convert_mcp_schema("pipelines", mcp_tool)

        props = schema["parameters"]["properties"]
        assert "definitions" in props, "user-facing property name was renamed away"
        assert "$defs" not in props, "user-facing property name was rewritten to $defs"
        # And the meta-keyword promotion didn't happen at the root either,
        # because there was no `definitions` meta-keyword to promote.
        assert "$defs" not in schema["parameters"]
        assert "definitions" not in schema["parameters"]


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

    def test_hyphens_sanitized_to_underscores(self):
        """Hyphens in tool/server names are replaced with underscores for LLM compat."""
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(name="get-sum")
        schema = _convert_mcp_schema("my-server", mcp_tool)

        assert schema["name"] == "mcp__my_server__get_sum"
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


    def test_recycled_stdio_server_remains_available_for_lazy_reconnect(self):
        from tools.mcp_tool import _make_check_fn, _servers

        server = _make_mock_server("test_server", session=None)
        server._config = {"command": "npx"}
        server._recycled_reason = "idle_timeout_seconds"
        _servers["test_server"] = server
        try:
            check = _make_check_fn("test_server")
            assert check() is True
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


    def test_recycled_stdio_server_reconnects_lazily_on_tool_call(self):
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("reconnected", is_error=False)
        )
        server = _make_mock_server("test_srv", session=None)
        server._config = {"command": "npx"}
        server._recycled_reason = "idle_timeout_seconds"
        _servers["test_srv"] = server

        def fake_lazy_reconnect(server_name, srv):
            assert server_name == "test_srv"
            assert srv is server
            srv.session = mock_session
            srv._recycled_reason = None
            return True

        try:
            handler = _make_tool_handler("test_srv", "greet", 120)
            with patch("tools.mcp_tool._request_lazy_reconnect", side_effect=fake_lazy_reconnect) as reconnect, \
                 self._patch_mcp_loop():
                result = json.loads(handler({"name": "world"}))
            assert result["result"] == "reconnected"
            reconnect.assert_called_once()
            mock_session.call_tool.assert_called_once_with("greet", arguments={"name": "world"})
        finally:
            _servers.pop("test_srv", None)


class TestRunOnMCPLoopInterrupts:
    @staticmethod
    def _run_with_future(mcp_mod, future):
        loop = MagicMock()
        loop.is_running.return_value = True

        async def _unused_call():
            return "unused"

        def _schedule(coro, scheduled_loop, **_kwargs):
            assert scheduled_loop is loop
            coro.close()
            return future

        with patch.object(mcp_mod, "_mcp_loop", loop):
            with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=_schedule):
                return mcp_mod._run_on_mcp_loop(_unused_call(), timeout=1)

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
            time.sleep(0.02)
            set_interrupt(True, waiter_tid)

        interrupter = threading.Thread(target=_interrupt_soon, daemon=True)
        interrupter.start()

        try:
            with pytest.raises(InterruptedError, match="User sent a new message"):
                mcp_mod._run_on_mcp_loop(_slow_call(), timeout=10)

            deadline = time.time() + 2
            while time.time() < deadline and not cancelled.is_set():
                time.sleep(0.01)
            assert cancelled.is_set()
        finally:
            set_interrupt(False, waiter_tid)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10)
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
            # 0.1s is the floor the MCP loop clamps short timeouts to.
            with pytest.raises(TimeoutError, match=r"MCP call timed out after .*configured timeout: 0.1s"):
                mcp_mod._run_on_mcp_loop(_slow_call(), timeout=0.1)

            deadline = time.time() + 2
            while time.time() < deadline and not cancelled.is_set():
                time.sleep(0.01)
            assert cancelled.is_set()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10)
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

        assert "mcp__fs__read_file" in registered
        assert "mcp__fs__write_file" in registered
        assert "mcp__fs__read_file" in mock_registry.get_all_tool_names()
        assert "mcp__fs__write_file" in mock_registry.get_all_tool_names()

        _servers.pop("fs", None)


    def test_same_server_normalization_collision_skips_all_ambiguous_tools(self, caplog):
        from tools.mcp_tool import _register_server_tools
        from tools.registry import ToolRegistry

        registry = ToolRegistry()
        server = _make_mock_server(
            "srv",
            session=MagicMock(),
            tools=[
                _make_mcp_tool("read-file"),
                _make_mcp_tool("read_file"),
                _make_mcp_tool("safe_tool"),
            ],
        )
        config = {"tools": {"resources": False, "prompts": False}}

        with patch("tools.registry.registry", registry), \
             patch("tools.mcp_tool._track_mcp_tool_server"), \
             caplog.at_level(logging.ERROR, logger="tools.mcp_tool"):
            registered = _register_server_tools("srv", server, config)

        assert registered == ["mcp__srv__safe_tool"]
        assert registry.get_entry("mcp__srv__read_file") is None
        assert registry.get_entry("mcp__srv__safe_tool") is not None
        assert any(
            "name normalization collision" in record.message
            and "tool 'read-file'" in record.message
            and "tool 'read_file'" in record.message
            for record in caplog.records
        )

    def test_native_tool_wins_over_generated_utility_on_collision(self, caplog):
        """A server-native tool named `read_resource` must survive its collision
        with the generated `read_resource` utility (#87112).

        Before the fix the collision handler treated the pair as ambiguous and
        skipped BOTH, so the server's own tool vanished on every boot. The
        generated utility is only sugar for servers that lack such a tool, so
        the native tool wins and the utility is dropped.
        """
        from tools.mcp_tool import _register_server_tools
        from tools.registry import ToolRegistry

        registry = ToolRegistry()
        server = _make_mock_server(
            "srv",
            session=MagicMock(),
            tools=[
                _make_mcp_tool("read_resource", "Native read-resource tool"),
                _make_mcp_tool("safe_tool"),
            ],
        )
        # Resources enabled (default) so the read_resource utility is generated
        # and collides; prompts disabled to keep the candidate set focused.
        config = {"tools": {"prompts": False}}

        with patch("tools.registry.registry", registry), \
             patch("tools.mcp_tool._track_mcp_tool_server"), \
             caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
            registered = _register_server_tools("srv", server, config)

        # The native tool is registered (before the fix it was dropped) and it
        # is the server's tool, not the utility stub.
        assert "mcp__srv__read_resource" in registered
        entry = registry.get_entry("mcp__srv__read_resource")
        assert entry is not None
        assert entry.description == "Native read-resource tool"
        assert "mcp__srv__safe_tool" in registered

        # The collision was resolved in favour of the native tool, not skipped
        # as ambiguous.
        assert not any(
            "name normalization collision" in record.message
            and "mcp__srv__read_resource" in record.message
            for record in caplog.records
        )
        assert any(
            record.levelno == logging.INFO
            and "keeping the native tool and dropping the utility" in record.message
            and "read_resource" in record.message
            for record in caplog.records
        )

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
            with patch("tools.mcp_tool.StdioServerParameters") as params, p_stdio, p_cs:
                server = MCPServerTask("test_srv")
                await server.start(
                    {"command": "npx", "args": ["-y", "test"], "cwd": "/plugin"}
                )

                assert server.session is mock_session
                assert len(server._tools) == 1
                assert server._tools[0].name == "echo"
                mock_session.initialize.assert_called_once()
                assert params.call_args.kwargs["cwd"] == "/plugin"

                await server.shutdown()
                assert server.session is None

        asyncio.run(_test())

    def test_start_preserves_native_default_cwd(self):
        from tools.mcp_tool import MCPServerTask

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
        p_stdio, p_cs, _, _ = self._mock_stdio_and_session(mock_session)

        async def _test():
            with patch("tools.mcp_tool.StdioServerParameters") as params, p_stdio, p_cs:
                server = MCPServerTask("native")
                await server.start({"command": "npx", "args": ["-y", "test"]})
                assert params.call_args.kwargs["cwd"] is None
                await server.shutdown()

        asyncio.run(_test())


    def test_stdio_recycle_deadline_pauses_while_rpc_active(self):
        from tools.mcp_tool import MCPServerTask

        async def _test():
            server = MCPServerTask("srv")
            server._config = {"command": "npx"}
            server._idle_timeout_seconds = 0.01
            server._last_tool_call_at = time.monotonic() - 1.0

            async with server._rpc_lock:
                assert server._stdio_recycle_reason() is None
                assert server._next_stdio_recycle_deadline() is None

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

            assert "mcp__fs__list_files" in result
            assert validate_toolset("fs") is True
            assert validate_toolset("mcp-fs") is True
            assert "mcp__fs__list_files" in resolve_toolset("fs")
            assert "mcp__fs__list_files" in resolve_toolset("mcp-fs")

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
            assert "mcp__good__ping" in result1
            assert "mcp__broken__ping" not in result1
            first_attempts = call_count

            # "Fix" the broken server
            broken_fixed = True
            call_count = 0

            # The failed server is now serving a post-failure backoff
            # (#50394: prevents a tight re-spawn storm across the frequent
            # per-worker-session discovery passes). Expire that cooldown to
            # simulate the retry window having elapsed.
            import tools.mcp_tool as _mcp_mod
            _mcp_mod._server_connect_retry_after.pop("broken", None)

            # Next call after the cooldown: should retry broken, skip good
            result2 = discover_mcp_tools()
            assert "mcp__good__ping" in result2
            assert "mcp__broken__ping" in result2
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

# ---------------------------------------------------------------------------
# Shutdown (public API)
# ---------------------------------------------------------------------------

class TestShutdown:

    def test_shutdown_drains_parked_server_after_bounded_wait_expires(self):
        """The public shutdown path drains a parked server if graceful shutdown stalls.

        This exercises the production ownership path: a real ``MCPServerTask``
        is registered, its parked waiter owns child tasks on the shared loop,
        and the bounded wait for the scheduled shutdown expires. The loop owner
        must still cancel and drain that waiter before closing the loop.
        """
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import MCPServerTask, shutdown_mcp_servers

        shutdown_started = threading.Event()
        parked_task_done = threading.Event()
        scheduled_shutdown = {}
        schedule_count = 0

        class StalledShutdownServer(MCPServerTask):
            async def shutdown(self):
                shutdown_started.set()
                await asyncio.Event().wait()

        server = StalledShutdownServer("parked")
        with mcp_mod._lock:
            mcp_mod._servers.clear()
            mcp_mod._server_connecting.clear()
        mcp_mod._ensure_mcp_loop()
        with mcp_mod._lock:
            loop = mcp_mod._mcp_loop
        assert loop is not None

        async def install_parked_waiter():
            task = asyncio.create_task(server._wait_for_reconnect_or_shutdown())
            server._task = task
            task.add_done_callback(lambda _task: parked_task_done.set())
            await asyncio.sleep(0)
            return task

        parked_task = asyncio.run_coroutine_threadsafe(
            install_parked_waiter(), loop
        ).result(timeout=2)
        with mcp_mod._lock:
            mcp_mod._servers[server.name] = server

        def schedule_then_report_timeout(coro, target_loop, **_kwargs):
            nonlocal schedule_count
            schedule_count += 1
            future = asyncio.run_coroutine_threadsafe(coro, target_loop)
            if schedule_count > 1:
                return future

            scheduled_shutdown["future"] = future

            class TimedOutFuture:
                def result(self, timeout):
                    assert timeout > 0
                    assert shutdown_started.wait(timeout=2)
                    raise TimeoutError("simulated bounded MCP shutdown timeout")

            return TimedOutFuture()

        try:
            with patch(
                "agent.async_utils.safe_schedule_threadsafe",
                side_effect=schedule_then_report_timeout,
            ):
                shutdown_mcp_servers()

            assert loop.is_closed()
            assert parked_task_done.is_set(), (
                "parked MCPServerTask was not drained before its loop closed"
            )
            assert parked_task.done()
            assert scheduled_shutdown["future"].done()
            assert schedule_count == 2
        finally:
            with mcp_mod._lock:
                mcp_mod._servers.clear()
                mcp_mod._server_connecting.clear()
            mcp_mod._stop_mcp_loop()

    def test_shutdown_deregisters_registered_tools(self):
        """shutdown_mcp_servers removes MCP tools and their raw alias."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import MCPServerTask, shutdown_mcp_servers, _servers
        from tools.registry import registry
        from toolsets import resolve_toolset, validate_toolset

        _servers.clear()
        registry.register(
            name="mcp__test__ping",
            toolset="mcp-test",
            schema={
                "name": "mcp__test__ping",
                "description": "Ping",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda *_args, **_kwargs: "{}",
        )
        registry.register_toolset_alias("test", "mcp-test")

        server = MCPServerTask("test")
        server._registered_tool_names = ["mcp__test__ping"]
        _servers["test"] = server

        mcp_mod._ensure_mcp_loop()
        try:
            assert validate_toolset("test") is True
            assert "mcp__test__ping" in resolve_toolset("test")
            shutdown_mcp_servers()
        finally:
            mcp_mod._mcp_loop = None
            mcp_mod._mcp_thread = None

        assert "mcp__test__ping" not in registry.get_all_tool_names()
        assert validate_toolset("test") is False

    def test_shutdown_is_parallel(self):
        """Multiple servers are shut down in parallel via asyncio.gather."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import shutdown_mcp_servers, _servers
        import time

        _servers.clear()

        # 4 servers each taking 50ms to shut down
        delay = 0.05
        for i in range(4):
            mock_server = MagicMock()
            mock_server.name = f"srv_{i}"
            async def slow_shutdown():
                await asyncio.sleep(delay)
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
        # Parallel: ~1 delay, not 4. Margin covers scheduling jitter but stays
        # well under the serial total.
        assert elapsed < delay * 3, (
            f"Shutdown took {elapsed:.3f}s, expected ~{delay}s (parallel)"
        )


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

    def test_secret_source_injected_vars_are_passed(self, monkeypatch):
        """Vars tagged by an external secret source (Bitwarden/1Password) are
        deliberately allowed for MCP stdio servers."""
        from hermes_cli import env_loader
        from tools.mcp_tool import _build_safe_env

        monkeypatch.setitem(env_loader._SECRET_SOURCES, "ALPACA_API_KEY", "bitwarden")
        monkeypatch.setitem(env_loader._SECRET_SOURCES, "NOTION_TOKEN", "onepassword")
        fake_env = {
            "PATH": "/usr/bin",
            "ALPACA_API_KEY": "from-bws-key",
            "NOTION_TOKEN": "from-op",
            "UNTRACKED_SECRET_KEY": "still-filtered",
        }
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)

        assert result["PATH"] == "/usr/bin"
        assert result["ALPACA_API_KEY"] == "from-bws-key"
        assert result["NOTION_TOKEN"] == "from-op"
        assert "UNTRACKED_SECRET_KEY" not in result

    def test_windows_location_vars_passed_without_secrets(self):
        """Windows launcher tools need location vars, but secrets stay filtered."""
        from tools.mcp_tool import _build_safe_env

        fake_env = {
            "PATH": r"C:\Windows\System32",
            "ProgramFiles": r"C:\Program Files",
            "ProgramData": r"C:\ProgramData",
            "ProgramW6432": r"C:\Program Files",
            "LOCALAPPDATA": r"C:\Users\alice\AppData\Local",
            "APPDATA": r"C:\Users\alice\AppData\Roaming",
            "USERPROFILE": r"C:\Users\alice",
            "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "OPENAI_API_KEY": "sk-proj-abc123",
        }
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)

        assert result["ProgramFiles"] == r"C:\Program Files"
        assert result["ProgramData"] == r"C:\ProgramData"
        assert result["ProgramW6432"] == r"C:\Program Files"
        assert result["LOCALAPPDATA"].endswith("Local")
        assert result["APPDATA"].endswith("Roaming")
        assert result["USERPROFILE"] == r"C:\Users\alice"
        assert "GITHUB_TOKEN" not in result
        assert "OPENAI_API_KEY" not in result


# ---------------------------------------------------------------------------
# _sanitize_error
# ---------------------------------------------------------------------------

class TestSanitizeError:
    """Tests for _sanitize_error() credential stripping."""

    def test_strips_credentials(self):
        from tools.mcp_tool import _sanitize_error

        for text, expected in (
            ("Error with ghp_abc123def456", "Error with [REDACTED]"),
            ("key sk-projABC123xyz", "key [REDACTED]"),
            ("Authorization: Bearer eyJabc123def", "Authorization: [REDACTED]"),
            ("url?token=secret123", "url?[REDACTED]"),
        ):
            assert _sanitize_error(text) == expected, text

        # Several credentials in one message are all masked.
        multi = _sanitize_error("ghp_abc123 and sk-projXyz789 and token=foo")
        assert "ghp_" not in multi and "sk-" not in multi and "token=" not in multi
        assert multi.count("[REDACTED]") == 3

    def test_no_credentials_unchanged(self):
        from tools.mcp_tool import _sanitize_error
        result = _sanitize_error("normal error message")
        assert result == "normal error message"

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

    def test_http_unavailable_raises(self):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")
        config = {"url": "https://example.com/mcp"}

        async def _test():
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", False):
                with pytest.raises(ImportError, match="HTTP transport"):
                    await server._run_http(config)

        asyncio.run(_test())

    def test_stdio_unavailable_raises_importerror_not_nameerror(self):
        """Regression test for #30904.

        When the mcp SDK isn't installed, ``_run_stdio`` previously leaked a
        bare ``NameError: name 'StdioServerParameters' is not defined``. The
        gate now raises a clear ``ImportError`` with install instructions,
        mirroring ``_run_http``'s behaviour when the HTTP transport is
        unavailable.
        """
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("local")
        config = {"command": "python3", "args": ["/tmp/echo.py"]}

        async def _test():
            with patch("tools.mcp_tool._MCP_AVAILABLE", False):
                with pytest.raises(ImportError, match=r"mcp.*SDK"):
                    await server._run_stdio(config)

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

    def test_reconnect_failures_after_initial_success_use_reconnect_budget(self):
        """A server that already registered tools must not be re-classified
        as "never connected" just because a later reconnect attempt fails
        while ``_ready`` is momentarily clear (#94654).

        ``_MAX_INITIAL_CONNECT_RETRIES`` is 3 and ``_MAX_RECONNECT_RETRIES``
        is 5. Four consecutive post-success reconnect failures exceed the
        (buggy) initial-connect ladder but not the reconnect budget, so this
        distinguishes the two code paths.
        """
        from tools.mcp_tool import MCPServerTask

        run_count = 0
        target_server = None

        async def patched_run_stdio(self_srv, config):
            nonlocal run_count, target_server
            run_count += 1
            if target_server is not self_srv:
                return None
            if run_count == 1:
                # Initial connection succeeds and registers tools. Setting
                # ``_ever_connected`` mirrors what the real ``_run_stdio``
                # does right after a successful ``_discover_tools()`` call;
                # guarded because the pre-fix class has no such slot.
                self_srv.session = MagicMock()
                self_srv._tools = []
                self_srv._ready.set()
                try:
                    self_srv._ever_connected = True
                except AttributeError:
                    pass
                return "reconnect"
            if run_count <= 5:
                # Four consecutive failures on later reconnect attempts --
                # a flapping transport, not a server that never connected.
                raise ConnectionError("reconnect attempt failed")
            self_srv._shutdown_event.set()
            await self_srv._shutdown_event.wait()

        async def patched_wait_parked(self_srv, timeout=None):
            # If the code under test parks early, unblock immediately
            # instead of waiting out the real park interval.
            self_srv._shutdown_event.set()
            return "shutdown"

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
                 patch.object(
                     MCPServerTask, "_wait_for_reconnect_or_shutdown",
                     patched_wait_parked,
                 ), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await server.run({"command": "test"})

            assert server._was_parked is False, (
                "server parked after only 4 reconnect failures following a "
                "successful initial connection -- it was misclassified as "
                "never having connected and re-entered the initial-connect "
                "ladder instead of the (larger) reconnect budget"
            )

        asyncio.run(_test())

    def test_preflight_probe_runs_on_initial_http_connect(self):
        """The content-type preflight probe fires on the first HTTP connect."""
        from tools.mcp_tool import MCPServerTask

        target_server = None
        probe = AsyncMock()

        original_run_http = MCPServerTask._run_http

        async def patched_run_http(self_srv, config):
            if target_server is not self_srv:
                return await original_run_http(self_srv, config)
            # First connect succeeds; signal shutdown so run() exits cleanly.
            self_srv.session = MagicMock()
            self_srv._tools = []
            self_srv._ready.set()
            self_srv._shutdown_event.set()
            await self_srv._shutdown_event.wait()

        async def _test():
            nonlocal target_server
            server = MCPServerTask("http_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_http", patched_run_http), \
                 patch.object(MCPServerTask, "_preflight_content_type", probe), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await server.run({"url": "https://example.com/mcp"})

            # Probe ran exactly once on the initial (pre-_ready) connect.
            assert probe.await_count == 1

        asyncio.run(_test())

# ---------------------------------------------------------------------------
# Configurable timeouts
# ---------------------------------------------------------------------------

class TestConfigurableTimeouts:
    """Tests for configurable per-server timeouts."""

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
        from tools.mcp_tool import _make_tool_handler, _servers

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
        assert "mcp__myserver__list_resources" in names
        assert "mcp__myserver__read_resource" in names
        assert "mcp__myserver__list_prompts" in names
        assert "mcp__myserver__get_prompt" in names

    def test_read_resource_schema_requires_uri(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("srv")
        rr = next(s for s in schemas if s["handler_key"] == "read_resource")
        params = rr["schema"]["parameters"]
        assert "uri" in params["properties"]
        assert params["properties"]["uri"]["type"] == "string"
        assert params["required"] == ["uri"]

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


    # -- read_resource --


    # -- list_prompts --

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
        assert "mcp__fs__read_file" in registered
        assert "mcp__fs__list_resources" in registered
        assert "mcp__fs__read_resource" in registered
        assert "mcp__fs__list_prompts" in registered
        assert "mcp__fs__get_prompt" in registered
        assert len(registered) == 5

        # All in the registry
        all_names = mock_registry.get_all_tool_names()
        for name in registered:
            assert name in all_names

        _servers.pop("fs", None)

# ===========================================================================
# SamplingHandler tests
# ===========================================================================


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
    def test_coercion_clamping_and_fallbacks(self):
        # (value, default, caster, kwargs, expected)
        cases = [
            (10, 5, int, {}, 10),
            ("20", 5, int, {}, 20),
            ("3.5", 1.0, float, {}, 3.5),
            (None, 7, int, {}, 7),
            ("abc", 42, int, {}, 42),
            (float("inf"), 3.0, float, {}, 3.0),
            (float("nan"), 4.0, float, {}, 4.0),
            (-5, 10, int, {"minimum": 1}, 1),
            (0, 10, int, {"minimum": 0}, 0),
        ]
        for value, default, caster, kwargs, expected in cases:
            assert _safe_numeric(value, default, caster, **kwargs) == expected, value

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

# ---------------------------------------------------------------------------
# 3. Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def setup_method(self):
        self.handler = SamplingHandler("rl", {"max_rpm": 3})

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

    def test_config_override_wins(self):
        self.handler.model_override = "override-model"
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="hint-model")])
        assert self.handler._resolve_model(prefs) == "override-model"

    def test_hint_used_when_no_override(self):
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="hint-model")])
        assert self.handler._resolve_model(prefs) == "hint-model"

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
        assert _stop_reason(result) == "endTurn"

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
        assert _stop_reason(result) == "toolUse"
        assert result.model == "test-model"
        assert len(result.content) == 1
        tc = result.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.name == "get_weather"
        assert tc.id == "call_1"
        assert tc.input == {"city": "London"}

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
        # Config values clamp to a 1s floor (_safe_numeric minimum), so set the
        # attribute directly to exercise the timeout branch without a 1s wait.
        handler = SamplingHandler("to", {})
        handler.timeout = 0.05

        def slow_call(**kwargs):
            import threading
            evt = threading.Event()
            # Outlives the 0.05s handler timeout, but short enough that the
            # abandoned worker thread doesn't stall loop shutdown.
            evt.wait(0.15)
            return _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=slow_call,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "timed out" in result.message.lower()
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
            return [f"mcp__{name}__tool_a"]

        with patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_register), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp__good_server__tool_a"]):
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
            return [f"mcp__{name}__t"]

        with patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=selective_register), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp__ok1__t", "mcp__ok2__t"]):
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
        assert registered == ["mcp__ink__create_service"]

    def test_empty_include_registers_nothing(self):
        """include: [] is an explicit empty whitelist, not "no filter".

        The install checklist writes include: [] when the user unchecks
        every tool ("contributes nothing until reconfigured") — the next
        session must not register the full tool surface.
        """
        config = {
            "url": "https://mcp.example.com",
            "tools": {"include": []},
        }
        registered, _ = self._run_discover(
            "ink",
            ["create_service", "delete_service", "list_services"],
            config,
            session=SimpleNamespace(),
        )
        assert registered == []

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
            "name": "mcp__abc__search",
            "description": "A hypothetical built-in",
            "parameters": {"type": "object", "properties": {}},
        }
        mock_registry.register(
            name="mcp__abc__search", toolset="web",
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
        assert "mcp__abc__search" not in registered
        assert mock_registry.get_toolset_for_tool("mcp__abc__search") == "web"

        _servers.pop("abc", None)

    def test_mcp_tool_rejected_when_collision_is_another_mcp(self):
        """Cross-server MCP collisions preserve the existing owner."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()

        # Pre-register an MCP tool from a different server.
        mcp_schema = {
            "name": "mcp__srv__do_thing",
            "description": "From another MCP server",
            "parameters": {"type": "object", "properties": {}},
        }
        mock_registry.register(
            name="mcp__srv__do_thing", toolset="mcp-old",
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

        # Cross-server MCP collisions fail closed: the existing owner stays active.
        assert "mcp__srv__do_thing" not in registered
        entry = mock_registry.get_entry("mcp__srv__do_thing")
        assert entry is not None
        assert entry.toolset == "mcp-old"
        assert entry.schema["description"] == "From another MCP server"
        assert mock_registry.get_toolset_for_tool("mcp__srv__do_thing") == "mcp-old"

        _servers.pop("srv", None)


# ---------------------------------------------------------------------------
# sanitize_mcp_name_component
# ---------------------------------------------------------------------------


class TestSanitizeMcpNameComponent:
    """Verify sanitize_mcp_name_component handles all edge cases."""

    def test_hyphens_replaced(self):
        from tools.mcp_tool import sanitize_mcp_name_component
        assert sanitize_mcp_name_component("my-server") == "my_server"


    def test_slash_in_server_alias_resolution(self):
        """Server names with slashes resolve through their live MCP alias."""
        from tools.registry import ToolRegistry
        from toolsets import resolve_toolset, validate_toolset

        reg = ToolRegistry()
        reg.register(
            name="mcp__ai_exa_exa__search",
            toolset="mcp-ai.exa/exa",
            schema={"name": "mcp__ai_exa_exa__search", "description": "Search", "parameters": {"type": "object", "properties": {}}},
            handler=lambda *_args, **_kwargs: "{}",
        )
        reg.register_toolset_alias("ai.exa/exa", "mcp-ai.exa/exa")

        with patch("tools.registry.registry", reg):
            assert validate_toolset("ai.exa/exa") is True
            assert "mcp__ai_exa_exa__search" in resolve_toolset("ai.exa/exa")


# ---------------------------------------------------------------------------
# register_mcp_servers public API
# ---------------------------------------------------------------------------


class TestRegisterMcpServers:
    """Verify the new register_mcp_servers() public API."""

    def test_mcp_not_available_returns_empty(self):
        from tools.mcp_tool import register_mcp_servers

        with patch("tools.mcp_tool._MCP_AVAILABLE", False):
            result = register_mcp_servers({"srv": {"command": "test"}})
        assert result == []


    def test_connects_new_servers(self):
        from tools.mcp_tool import register_mcp_servers, _servers, _ensure_mcp_loop

        fake_config = {"my_server": {"command": "npx", "args": ["test"]}}

        async def fake_register(name, cfg):
            server = _make_mock_server(name)
            server._registered_tool_names = ["mcp__my_server__tool1"]
            _servers[name] = server
            return ["mcp__my_server__tool1"]

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_register), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp__my_server__tool1"]):
            _ensure_mcp_loop()
            result = register_mcp_servers(fake_config)

        assert "mcp__my_server__tool1" in result
        _servers.pop("my_server", None)

    def test_skips_servers_already_connecting(self):
        """Servers in _server_connecting must not be spawned again (#58862)."""
        from tools.mcp_tool import (
            register_mcp_servers, _servers, _server_connecting, _ensure_mcp_loop,
        )

        fake_config = {"my_srv": {"command": "npx", "args": ["test"]}}

        # Simulate a prior call that started connecting but hasn't finished
        _server_connecting.add("my_srv")
        connect_calls = []

        async def fake_register(name, cfg):
            connect_calls.append(name)
            server = _make_mock_server(name)
            server._registered_tool_names = [f"mcp_{name}_tool"]
            _servers[name] = server
            return [f"mcp_{name}_tool"]

        try:
            with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_register), \
                 patch("tools.mcp_tool._existing_tool_names", return_value=[]), \
                 patch("tools.mcp_tool._connect_cooldown_active", return_value=False):
                _ensure_mcp_loop()
                result = register_mcp_servers(fake_config)

            # Should NOT have attempted to connect my_srv again
            assert connect_calls == [], (
                f"Server already in _server_connecting should be skipped, "
                f"but connect was called for: {connect_calls}"
            )
            assert result == []
        finally:
            _server_connecting.discard("my_srv")
            _servers.pop("my_srv", None)

    def test_clears_stale_connecting_on_timeout(self):
        """Stale entries in _server_connecting are cleaned up after timeout (#58862)."""
        from tools.mcp_tool import (
            register_mcp_servers, _servers, _server_connecting,
            _server_connect_errors, _ensure_mcp_loop,
        )

        fake_config = {
            "srv_a": {"command": "npx", "args": ["a"]},
            "srv_b": {"command": "npx", "args": ["b"]},
        }

        # Simulate that srv_a is already connecting from another call
        _server_connecting.add("srv_a")

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._run_on_mcp_loop", side_effect=TimeoutError("timed out")), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]), \
             patch("tools.mcp_tool._connect_cooldown_active", return_value=False):
            _ensure_mcp_loop()

            with pytest.raises(TimeoutError):
                register_mcp_servers(fake_config)

        # After timeout, srv_b (which was in new_servers and added to _server_connecting)
        # should have been cleaned up from _server_connecting.
        # srv_a should remain since it was added externally and not part of new_servers.
        assert "srv_b" not in _server_connecting, (
            "Stale server added during this call should have been removed from "
            "_server_connecting after timeout"
        )
        # Cleanup
        _server_connecting.discard("srv_a")
        _servers.pop("srv_a", None)
        _servers.pop("srv_b", None)

# ---------------------------------------------------------------------------
# Tests for parallel tool call support (port from openai/codex#17667)
# ---------------------------------------------------------------------------

class TestMcpParallelToolCalls:
    """Tests for the supports_parallel_tool_calls config option."""

    def test_is_mcp_tool_parallel_safe_with_flag(self):
        """MCP tool from a parallel-safe server returns True."""
        from tools.mcp_tool import (
            is_mcp_tool_parallel_safe, _mcp_tool_server_names,
            _parallel_safe_servers, _lock,
        )
        with _lock:
            _parallel_safe_servers.add("docs")
            _mcp_tool_server_names["mcp__docs__search"] = "docs"
            _mcp_tool_server_names["mcp__docs__read_file"] = "docs"
            _mcp_tool_server_names["mcp__github__list_repos"] = "github"
        try:
            assert is_mcp_tool_parallel_safe("mcp__docs__search") is True
            assert is_mcp_tool_parallel_safe("mcp__docs__read_file") is True
            # Different server should be False
            assert is_mcp_tool_parallel_safe("mcp__github__list_repos") is False
        finally:
            with _lock:
                _parallel_safe_servers.discard("docs")
                _mcp_tool_server_names.pop("mcp__docs__search", None)
                _mcp_tool_server_names.pop("mcp__docs__read_file", None)
                _mcp_tool_server_names.pop("mcp__github__list_repos", None)


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

# ---------------------------------------------------------------------------
# Cross-process MCP discovery lock (issue #62771)
# ---------------------------------------------------------------------------


class TestMCPDiscoveryCrossProcessLock:
    """Tests for the cross-process MCP discovery guard in discover_mcp_tools()."""

    @staticmethod
    def _lock_exclusive(fh):
        """Lock a file handle exclusively, cross-platform.

        Mirrors production _try_acquire_mcp_discovery_lock: fcntl on POSIX,
        portalocker on Windows (portalocker only ships on win32 installs).
        """
        if sys.platform == "win32":
            import portalocker

            self._lock_exclusive(fh)
        else:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @pytest.fixture(autouse=True)
    def _fast_retries(self):
        """Override retry constants so tests are fast."""
        import tools.mcp_tool as mcp_tool
        orig_max = mcp_tool._MCP_DISCOVERY_LOCK_MAX_RETRIES
        orig_delay = mcp_tool._MCP_DISCOVERY_LOCK_RETRY_DELAY_S
        mcp_tool._MCP_DISCOVERY_LOCK_MAX_RETRIES = 3
        mcp_tool._MCP_DISCOVERY_LOCK_RETRY_DELAY_S = 0.01
        yield
        mcp_tool._MCP_DISCOVERY_LOCK_MAX_RETRIES = orig_max
        mcp_tool._MCP_DISCOVERY_LOCK_RETRY_DELAY_S = orig_delay

    def test_lock_acquired_path(self, tmp_path):
        """Lock acquired -> discovery runs normally, lock released at end."""
        from tools.mcp_tool import (
            _LockCookie,
            discover_mcp_tools,
        )

        lock_file = tmp_path / ".mcp-discovery.lock"
        fh = open(lock_file, "w", encoding="utf-8")
        cookie = _LockCookie(fh)

        def mock_acquire():
            return cookie

        mock_config = {"test_srv": {"command": "echo", "enabled": True}}
        with patch.object(cookie, "release", wraps=cookie.release) as release_spy:
            with patch("tools.mcp_tool._try_acquire_mcp_discovery_lock", mock_acquire), \
                 patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._load_mcp_config", return_value=mock_config), \
                 patch("tools.mcp_tool.register_mcp_servers", return_value=["mcp__test_srv__ping"]) as reg_spy:
                result = discover_mcp_tools()
            assert result == ["mcp__test_srv__ping"]
            release_spy.assert_called_once()

    def test_lock_held_retries_exhausted_fallback(self):
        """All retry attempts see lock held -> runs discovery unguarded."""
        from tools.mcp_tool import (
            _LOCK_UNAVAILABLE,
            discover_mcp_tools,
            _MCP_DISCOVERY_LOCK_MAX_RETRIES,
        )

        mock_config = {"test_srv": {"command": "echo", "enabled": True}}
        # Every attempt returns None (lock held)
        with patch("tools.mcp_tool._try_acquire_mcp_discovery_lock", return_value=None), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=mock_config), \
             patch("tools.mcp_tool.register_mcp_servers") as reg_spy, \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            result = discover_mcp_tools()
        # Must still run local discovery
        reg_spy.assert_called_once_with(mock_config)

    def test_posix_flock_acquire_and_release(self):
        """_acquire_lock_on_fh uses fcntl.flock on POSIX."""
        import sys
        import tempfile
        from unittest.mock import MagicMock

        mock_fcntl = MagicMock()
        mock_fcntl.LOCK_EX = 2
        mock_fcntl.LOCK_NB = 4

        with tempfile.NamedTemporaryFile(prefix="mcp-lock-", suffix=".tmp", delete=False) as tf:
            lock_path = tf.name

        try:
            fh = open(lock_path, "w", encoding="utf-8")
            with patch.dict("sys.modules", {"fcntl": mock_fcntl}), \
                 patch("tools.mcp_tool.os.name", "posix"):
                from tools.mcp_tool import _acquire_lock_on_fh
                result = _acquire_lock_on_fh(fh)
            assert result is True
            mock_fcntl.flock.assert_called_once_with(
                fh.fileno(), mock_fcntl.LOCK_EX | mock_fcntl.LOCK_NB
            )
            fh.close()
        finally:
            try:
                os.unlink(lock_path)
            except Exception:
                pass


class TestRedirectHeaderStripper:
    """Cross-origin redirect header boundary (portable Agent Plugins v1)."""

    def _make_response(self, next_headers):
        import httpx

        next_request = httpx.Request(
            "GET", "https://other.example.test/mcp", headers=next_headers
        )
        response = SimpleNamespace(
            is_redirect=True,
            next_request=next_request,
        )
        return response, next_request

    def test_default_strips_only_authorization(self):
        import httpx

        from tools.mcp_tool import _make_redirect_header_stripper

        hook = _make_redirect_header_stripper(
            httpx.URL("https://origin.example.test/mcp")
        )
        response, next_request = self._make_response(
            {"Authorization": "Bearer x", "X-Tenant": "t"}
        )
        asyncio.run(hook(response))
        assert "authorization" not in next_request.headers
        assert next_request.headers["x-tenant"] == "t"

    def test_strict_strips_configured_headers_cross_origin(self):
        import httpx

        from tools.mcp_tool import _make_redirect_header_stripper

        hook = _make_redirect_header_stripper(
            httpx.URL("https://origin.example.test/mcp"),
            strict=True,
            configured_header_names={"x-tenant"},
        )
        response, next_request = self._make_response(
            {"Authorization": "Bearer x", "X-Tenant": "t", "Accept": "a"}
        )
        asyncio.run(hook(response))
        assert "authorization" not in next_request.headers
        assert "x-tenant" not in next_request.headers
        # Client-generated headers unrelated to package config survive.
        assert next_request.headers["accept"] == "a"

    def test_same_origin_redirect_keeps_headers(self):
        import httpx

        from tools.mcp_tool import _make_redirect_header_stripper

        hook = _make_redirect_header_stripper(
            httpx.URL("https://origin.example.test/mcp"),
            strict=True,
            configured_header_names={"x-tenant"},
        )
        next_request = httpx.Request(
            "GET",
            "https://origin.example.test/other",
            headers={"Authorization": "Bearer x", "X-Tenant": "t"},
        )
        response = SimpleNamespace(is_redirect=True, next_request=next_request)
        asyncio.run(hook(response))
        assert next_request.headers["authorization"] == "Bearer x"
        assert next_request.headers["x-tenant"] == "t"
