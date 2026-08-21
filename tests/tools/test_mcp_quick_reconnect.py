"""Tests for MCP quick-reconnect tool refresh.

When an MCP server loses its connection briefly and reconnects within the
retry budget (no parking), ``_register_discovered_tools_if_needed()``
returns early — it sees ``_registered_tool_names`` already populated and
skips re-registration.  The tool registry becomes stale until a manual
``/reload-mcp`` or until the server exhausts the retry budget and goes
through the parked-reconnect path (which deregisters and re-registers).

This is distinct from the parked-reconnect fix in #68659 / #67187.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _register_server_tools
from tools.registry import ToolRegistry


def _make_mcp_tool(name: str, desc: str = "", schema=None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=desc,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _make_task(server_name: str = "test") -> MCPServerTask:
    """Create an MCPServerTask pre-wired for unit tests."""
    task = MCPServerTask.__new__(MCPServerTask)
    task.name = server_name
    task.session = MagicMock()
    task._tools = []
    task._registered_tool_names = []
    task._ready = asyncio.Event()
    task._ready.set()
    task._refresh_lock = asyncio.Lock()
    task._rpc_lock = asyncio.Lock()
    task._config = {}
    task._ping_unsupported = False
    task._reconnect_retries = 0
    task._reconnect_event = asyncio.Event()
    task._shutdown_event = asyncio.Event()
    task._session_proven = True
    task._was_parked = False
    task._error = None
    task._task = None
    task._sampling = None
    task._elicitation = None
    task._auth_type = ""
    task._pending_refresh_tasks = set()
    task.initialize_result = None  # → _advertises_tools() returns True
    task.tool_timeout = 30
    task._list_cache_meta = {}
    return task


@pytest.fixture(autouse=True)
def _patch_advertises_tools(monkeypatch):
    """Force _advertises_tools() to always return True for tests.

    MCPServerTask uses __slots__, so monkeypatching on the *instance*
    raises AttributeError.  Patching on the class works because the
    descriptor lookup finds our replacement before __slots__.
    """
    monkeypatch.setattr(
        MCPServerTask, "_advertises_tools",
        lambda self: True,
        raising=False,
    )


@pytest.fixture
def isolated_registry(monkeypatch):
    """A fresh ToolRegistry per test to prevent cross-test pollution."""
    reg = ToolRegistry()
    import tools.registry
    monkeypatch.setattr(tools.registry, "registry", reg)
    return reg


def _schema_of(reg: ToolRegistry, name: str) -> dict:
    """Extract the schema dict for a registered tool by name."""
    entry = reg._tools.get(name)
    if entry is None:
        return None
    return entry.schema


def _description_of(reg: ToolRegistry, name: str) -> str:
    entry = reg._tools.get(name)
    if entry is None:
        return None
    return entry.description


# ---------------------------------------------------------------------------
# Schema/Metadata contract — the name-only comparison gap
# ---------------------------------------------------------------------------

class TestSchemaContract:

    @pytest.mark.asyncio
    async def test_changed_schema_on_reconnect(self, isolated_registry):
        """Same tool name, changed inputSchema — must be updated.
        _register_server_tools wraps the raw schema in a 'parameters' key."""
        old_schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        new_schema = {"type": "object", "properties": {"b": {"type": "integer"}}}

        task = _make_task(server_name="srv")
        task._registered_tool_names = ["mcp__srv__tool_x"]
        isolated_registry.register(
            name="mcp__srv__tool_x", toolset="mcp-srv", schema=old_schema,
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="old desc", emoji="",
        )
        task._tools = [_make_mcp_tool("tool_x", schema=new_schema)]
        task.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(
                tools=[_make_mcp_tool("tool_x", schema=new_schema)]
            )
        )
        task.session.get_server_version = MagicMock(return_value="1.0")

        await task._register_discovered_tools_if_needed()

        stored = _schema_of(isolated_registry, "mcp__srv__tool_x")
        assert stored is not None, "Tool_x must still be registered"
        # The schema is wrapped in a 'parameters' key by _register_server_tools
        params = stored.get("parameters", stored)
        assert params == new_schema, (
            f"Schema must be updated from {old_schema} to {new_schema}, "
            f"got params={params}"
        )

    @pytest.mark.asyncio
    async def test_changed_description_on_reconnect(self, isolated_registry):
        """Same tool name, changed description — must be updated."""
        task = _make_task(server_name="srv")
        task._registered_tool_names = ["mcp__srv__tool_x"]
        isolated_registry.register(
            name="mcp__srv__tool_x", toolset="mcp-srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="old description", emoji="",
        )
        task._tools = [_make_mcp_tool("tool_x", desc="new description")]
        task.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(
                tools=[_make_mcp_tool("tool_x", desc="new description")]
            )
        )
        task.session.get_server_version = MagicMock(return_value="1.0")

        await task._register_discovered_tools_if_needed()

        stored = _description_of(isolated_registry, "mcp__srv__tool_x")
        assert stored == "new description", (
            f"Description must be updated, got {stored!r}"
        )

    @pytest.mark.asyncio
    async def test_identical_tool_contract_is_stable(self, isolated_registry):
        """Identical full tool contract (name + schema + desc) — no duplicate
        server tool entries, no registry corruption.
        Note: _refresh_tools() -> _register_server_tools() also registers
        utility tools (list_resources, get_prompt, etc.) when the server
        advertises those capabilities, so the total tool count may grow."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        task = _make_task(server_name="srv")
        task._registered_tool_names = ["mcp__srv__tool_a"]
        isolated_registry.register(
            name="mcp__srv__tool_a", toolset="mcp-srv", schema=schema,
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="stable", emoji="",
        )
        task._tools = [_make_mcp_tool("tool_a", desc="stable", schema=schema)]
        task.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(
                tools=[_make_mcp_tool("tool_a", desc="stable", schema=schema)]
            )
        )
        task.session.get_server_version = MagicMock(return_value="1.0")

        await task._register_discovered_tools_if_needed()

        # Verify: tool_a still exists exactly once, with correct schema.
        all_tools = isolated_registry.get_all_tool_names()
        assert "mcp__srv__tool_a" in all_tools
        # Count occurrences of the server tool name
        tool_a_count = sum(1 for t in all_tools if t == "mcp__srv__tool_a")
        assert tool_a_count == 1, (
            f"tool_a must appear exactly once, got {tool_a_count}"
        )
        assert _schema_of(isolated_registry, "mcp__srv__tool_a") is not None

    @pytest.mark.asyncio
    async def test_new_tool_registered(self, isolated_registry):
        """New tool appears after quick reconnect."""
        task = _make_task(server_name="srv")
        task._registered_tool_names = ["mcp__srv__tool_a"]
        isolated_registry.register(
            name="mcp__srv__tool_a", toolset="mcp-srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )
        task._tools = [_make_mcp_tool("tool_b")]
        task.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_make_mcp_tool("tool_b")])
        )
        task.session.get_server_version = MagicMock(return_value="1.0")

        await task._register_discovered_tools_if_needed()

        assert "mcp__srv__tool_b" in isolated_registry.get_all_tool_names()

    @pytest.mark.asyncio
    async def test_removed_tool_deregistered(self, isolated_registry):
        """Removed tool must be deregistered after quick reconnect."""
        task = _make_task(server_name="srv")
        task._registered_tool_names = ["mcp__srv__tool_a", "mcp__srv__tool_b"]
        for name in ["mcp__srv__tool_a", "mcp__srv__tool_b"]:
            isolated_registry.register(
                name=name, toolset="mcp-srv", schema={},
                handler=lambda x: x, check_fn=lambda: True, is_async=False,
                description="", emoji="",
            )
        task._tools = [_make_mcp_tool("tool_b")]
        task.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_make_mcp_tool("tool_b")])
        )
        task.session.get_server_version = MagicMock(return_value="1.0")

        await task._register_discovered_tools_if_needed()

        assert "mcp__srv__tool_a" not in isolated_registry.get_all_tool_names()

    @pytest.mark.asyncio
    async def test_include_exclude_filter_no_phantom(self, isolated_registry):
        """Include/exclude filters must not create phantom tools on reconnect."""
        task = _make_task(server_name="srv")
        task._config = {"tools": {"include": ["tool_a"]}}
        task._registered_tool_names = ["mcp__srv__tool_a"]
        isolated_registry.register(
            name="mcp__srv__tool_a", toolset="mcp-srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )
        # Server now has tool_a AND tool_b, but filter excludes tool_b
        task._tools = [
            _make_mcp_tool("tool_a"),
            _make_mcp_tool("tool_b"),
        ]
        task.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[
                _make_mcp_tool("tool_a"),
                _make_mcp_tool("tool_b"),
            ])
        )
        task.session.get_server_version = MagicMock(return_value="1.0")

        await task._register_discovered_tools_if_needed()

        all_tools = isolated_registry.get_all_tool_names()
        assert "mcp__srv__tool_a" in all_tools
        assert "mcp__srv__tool_b" not in all_tools, (
            "Filtered tool_b must not appear as phantom"
        )

    @pytest.mark.asyncio
    async def test_two_servers_same_tool_name_namespaced(self, isolated_registry):
        """Two servers with same raw tool name keep namespaced isolation."""
        schema_a = {"type": "object", "properties": {"a": {"type": "string"}}}
        schema_b = {"type": "object", "properties": {"b": {"type": "integer"}}}

        # Server A: tool_x with schema_a
        task_a = _make_task(server_name="srv_a")
        task_a._registered_tool_names = ["mcp__srv_a__tool_x"]
        isolated_registry.register(
            name="mcp__srv_a__tool_x", toolset="mcp-srv_a", schema=schema_a,
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )

        # Server B: tool_x with schema_b
        task_b = _make_task(server_name="srv_b")
        task_b._tools = [_make_mcp_tool("tool_x")]
        await task_b._register_discovered_tools_if_needed()
        task_b._registered_tool_names = ["mcp__srv_b__tool_x"]
        task_b.session = MagicMock()
        task_b.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_make_mcp_tool("tool_x")])
        )
        task_b.session.get_server_version = MagicMock(return_value="1.0")

        # Quick reconnect on server A — tool_x schema changed
        task_a._tools = [_make_mcp_tool("tool_x", schema=schema_b)]
        task_a.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(
                tools=[_make_mcp_tool("tool_x", schema=schema_b)]
            )
        )
        task_a.session.get_server_version = MagicMock(return_value="1.0")

        await task_a._register_discovered_tools_if_needed()

        # Both namespaces must exist
        schema_a_stored = _schema_of(isolated_registry, "mcp__srv_a__tool_x")
        schema_b_stored = _schema_of(isolated_registry, "mcp__srv_b__tool_x")
        assert schema_a_stored is not None, "srv_a must still have tool_x"
        assert schema_b_stored is not None, "srv_b must still have tool_x"
        # The schemas must not interfere
        assert "mcp__srv_b__tool_x" in isolated_registry.get_all_tool_names()

    @pytest.mark.asyncio
    async def test_first_discovery_still_works(self, isolated_registry):
        """First-time discovery (empty _registered_tool_names) must work."""
        task = _make_task(server_name="srv")
        task._tools = [_make_mcp_tool("tool_a")]
        await task._register_discovered_tools_if_needed()
        assert "mcp__srv__tool_a" in isolated_registry.get_all_tool_names()


# ---------------------------------------------------------------------------
# Regression tests — existing behaviour remains intact
# ---------------------------------------------------------------------------

class TestRefreshToolsPreserved:

    @pytest.mark.asyncio
    async def test_refresh_tools_still_works(self, isolated_registry):
        """_refresh_tools (notification path) must still function."""
        task = _make_task(server_name="srv")
        task._registered_tool_names = ["mcp__srv__old_tool"]
        isolated_registry.register(
            name="mcp__srv__old_tool", toolset="mcp-srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )
        new_tool = _make_mcp_tool("new_tool")
        task.session = MagicMock()
        task.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[new_tool])
        )
        task.session.get_server_version = MagicMock(return_value="1.0")

        await task._refresh_tools()

        assert "mcp__srv__old_tool" not in isolated_registry.get_all_tool_names()
        assert "mcp__srv__new_tool" in isolated_registry.get_all_tool_names()


class TestMultipleServers:

    @pytest.mark.asyncio
    async def test_servers_dont_interfere(self, isolated_registry):
        """Multiple servers must not interfere during reconnect."""
        task_a = _make_task(server_name="srv_a")
        task_a._registered_tool_names = ["mcp__srv_a__tool_x"]
        isolated_registry.register(
            name="mcp__srv_a__tool_x", toolset="mcp-srv_a", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )

        task_b = _make_task(server_name="srv_b")
        task_b._tools = [_make_mcp_tool("tool_y")]
        await task_b._register_discovered_tools_if_needed()
        task_b._registered_tool_names = ["mcp__srv_b__tool_y"]
        task_b.session = MagicMock()
        task_b.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_make_mcp_tool("tool_y")])
        )
        task_b.session.get_server_version = MagicMock(return_value="1.0")

        task_a._tools = [_make_mcp_tool("tool_z")]
        task_a.session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_make_mcp_tool("tool_z")])
        )
        task_a.session.get_server_version = MagicMock(return_value="1.0")
        await task_a._register_discovered_tools_if_needed()

        all_tools = isolated_registry.get_all_tool_names()
        assert "mcp__srv_b__tool_y" in all_tools
        assert "mcp__srv_a__tool_z" in all_tools