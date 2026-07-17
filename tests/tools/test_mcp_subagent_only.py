"""Tests for MCP ``scope: subagent_only`` visibility (Lane E design).

The design funnels every MAIN agent and every delegated child through one
registry choke point: ``ToolRegistry.get_definitions``. ``subagent_only`` tools
are withheld there unless ``include_subagent_only=True`` is passed (which the
delegated-child path does via agent_init.py:1214, keyed on
``agent.platform == "subagent"``).

These tests pin that behavior at the registry level, at the
``get_tool_definitions`` cache-key level, and at the
``_register_server_tools`` config-interpretation level.
"""
import pytest
from unittest.mock import MagicMock

from tools.registry import ToolRegistry, registry
from tools import mcp_tool as mcp_tool_mod
from tools.mcp_tool import _normalize_mcp_scope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_registry():
    """Return a ToolRegistry isolated from the global singleton."""
    reg = ToolRegistry.__new__(ToolRegistry)
    # Re-run the real __init__ against private attrs without touching the
    # process-wide singleton used by the rest of the agent.
    ToolRegistry.__init__(reg)
    return reg


def _reg_tool(reg, name, scope="main", toolset="ts", check_fn=None):
    reg.register(
        name=name,
        toolset=toolset,
        schema={"name": name, "description": f"{name} tool"},
        handler=lambda x: x,
        check_fn=check_fn,
        description=f"{name} tool",
        scope=scope,
    )


class _FakeMcpTool:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description


class _FakeServer:
    """Minimal stand-in for MCPServerTask for _register_server_tools."""
    def __init__(self, tools, tool_timeout=30):
        self._tools = tools
        self.tool_timeout = tool_timeout
        self.name = "fakesrv"
        self.session = MagicMock()  # _select_utility_schemas legacy fallback
        self.initialize_result = None  # capabilities unknown → utilities skipped


# ---------------------------------------------------------------------------
# Registry-level scope guard (the choke point)
# ---------------------------------------------------------------------------

def test_scope_default_visible_to_main_and_child():
    reg = _fresh_registry()
    _reg_tool(reg, "mcp__srv__t")
    names = {d["function"]["name"] for d in reg.get_definitions({"mcp__srv__t"})}
    assert "mcp__srv__t" in names
    # Child path (include_subagent_only=True) is a no-op for "main" scope.
    names_child = {d["function"]["name"] for d in reg.get_definitions(
        {"mcp__srv__t"}, include_subagent_only=True)}
    assert "mcp__srv__t" in names_child


def test_scope_server_subagent_only_absent_from_main_present_in_child():
    reg = _fresh_registry()
    _reg_tool(reg, "mcp__srv__secret", scope="subagent_only")
    main = {d["function"]["name"] for d in reg.get_definitions({"mcp__srv__secret"})}
    assert "mcp__srv__secret" not in main
    child = {d["function"]["name"] for d in reg.get_definitions(
        {"mcp__srv__secret"}, include_subagent_only=True)}
    assert "mcp__srv__secret" in child


def test_scope_mixed_visibility():
    """MAIN sees only the main-scoped tool; child sees both."""
    reg = _fresh_registry()
    _reg_tool(reg, "mcp__srv__pub", scope="main")
    _reg_tool(reg, "mcp__srv__priv", scope="subagent_only")
    main = {d["function"]["name"] for d in reg.get_definitions(
        {"mcp__srv__pub", "mcp__srv__priv"})}
    assert main == {"mcp__srv__pub"}
    child = {d["function"]["name"] for d in reg.get_definitions(
        {"mcp__srv__pub", "mcp__srv__priv"}, include_subagent_only=True)}
    assert child == {"mcp__srv__pub", "mcp__srv__priv"}


def test_scope_unknown_treated_as_main():
    """A scope value other than the two allowed ones is normalized to 'main'
    at registration (never silently hidden)."""
    reg = _fresh_registry()
    # _normalize_mcp_scope is applied by callers (mcp_tool); here we assert the
    # registry itself only ever sees valid scopes by virtue of the helper.
    assert _normalize_mcp_scope("bogus", "where") == "main"
    assert _normalize_mcp_scope(None, "where") == "main"
    assert _normalize_mcp_scope("", "where") == "main"
    assert _normalize_mcp_scope("subagent_only", "where") == "subagent_only"


def test_scope_reflected_in_toolentry():
    reg = _fresh_registry()
    _reg_tool(reg, "mcp__srv__x", scope="subagent_only")
    assert reg.get_entry("mcp__srv__x").scope == "subagent_only"
    _reg_tool(reg, "mcp__srv__y")
    assert reg.get_entry("mcp__srv__y").scope == "main"


# ---------------------------------------------------------------------------
# get_tool_definitions cache-key isolation (no MAIN leakage)
# ---------------------------------------------------------------------------

def test_get_tool_definitions_cache_key_isolation(monkeypatch):
    """include_subagent_only=False and True must produce distinct cache
    entries so a MAIN call never returns child-scoped (subagent_only) tools."""
    from model_tools import get_tool_definitions

    reg = _fresh_registry()
    _reg_tool(reg, "mcp__srv__childonly", scope="subagent_only", toolset="mcp-srv")
    _reg_tool(reg, "mcp__srv__pub", scope="main", toolset="mcp-srv")
    # register_toolset_alias wires the raw server name so resolve_toolset("mcp-srv")
    # can resolve it (mirrors _register_server_tools).
    reg.register_toolset_alias("srv", "mcp-srv")

    captured = {}
    orig = reg.get_definitions
    def spy(names, quiet=False, include_subagent_only=False):
        captured.setdefault("seen", []).append(include_subagent_only)
        return orig(names, quiet=quiet, include_subagent_only=include_subagent_only)
    monkeypatch.setattr(reg, "get_definitions", spy)

    # Both toolset resolution and the registry lookup read the module-global
    # `registry`: toolsets.py does `from tools.registry import registry` (so
    # patching tools.registry.registry covers it) and model_tools.py bound
    # `registry` at import time (so patch model_tools.registry too).
    monkeypatch.setattr("tools.registry.registry", reg)
    monkeypatch.setattr("model_tools.registry", reg)

    main_defs = get_tool_definitions(
        enabled_toolsets=["mcp-srv"], quiet_mode=True, include_subagent_only=False)
    child_defs = get_tool_definitions(
        enabled_toolsets=["mcp-srv"], quiet_mode=True, include_subagent_only=True)

    main_names = {d["function"]["name"] for d in main_defs}
    child_names = {d["function"]["name"] for d in child_defs}
    assert "mcp__srv__childonly" not in main_names
    assert "mcp__srv__childonly" in child_names
    assert captured["seen"] == [False, True]


# ---------------------------------------------------------------------------
# _register_server_tools scope interpretation
# ---------------------------------------------------------------------------

def test_register_server_tools_server_level_scope():
    reg = _fresh_registry()
    server = _FakeServer([_FakeMcpTool("danger", "danger op")])
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("tools.registry.registry", reg)
        mcp_tool_mod._register_server_tools(
            "priv", server, {"scope": "subagent_only", "command": "x"})
    entry = reg.get_entry("mcp__priv__danger")
    assert entry is not None
    assert entry.scope == "subagent_only"


def test_register_server_tools_default_scope_is_main():
    reg = _fresh_registry()
    server = _FakeServer([_FakeMcpTool("op", "op")])
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("tools.registry.registry", reg)
        mcp_tool_mod._register_server_tools("plain", server, {"command": "x"})
    assert reg.get_entry("mcp__plain__op").scope == "main"


def test_register_server_tools_per_tool_scope_mixed():
    """tools.include + tools.scope restricts scope to the included tool set;
    the included tools become subagent_only while the server-level default
    (no tools.scope) leaves a non-included, non-registered tool out."""
    reg = _fresh_registry()
    server = _FakeServer([
        _FakeMcpTool("raw_query", "raw"),
        _FakeMcpTool("safe_op", "safe"),
    ])
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("tools.registry.registry", reg)
        mcp_tool_mod._register_server_tools(
            "mixed", server,
            {"command": "x", "tools": {"include": ["raw_query"],
                                       "scope": "subagent_only"}})
    # Included tool inherits the per-tool (tools.scope) scope.
    assert reg.get_entry("mcp__mixed__raw_query").scope == "subagent_only"
    # Non-included tool is not registered at all (include excludes it) — the
    # server-level default scope ("main") would apply if it were selected.
    assert reg.get_entry("mcp__mixed__safe_op") is None


def test_register_server_tools_unknown_scope_defaults_main(caplog):
    reg = _fresh_registry()
    server = _FakeServer([_FakeMcpTool("op", "op")])
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("tools.registry.registry", reg)
        mcp_tool_mod._register_server_tools(
            "bad", server, {"command": "x", "scope": "not_a_real_scope"})
    # Unknown scope is normalized to "main" (visible to MAIN), never hidden.
    assert reg.get_entry("mcp__bad__op").scope == "main"


def test_register_server_tools_utility_inherits_server_scope():
    """Utility tools (resources/prompts) inherit the server-level scope."""
    reg = _fresh_registry()
    server = _FakeServer([_FakeMcpTool("op", "op")])
    # Advertise resources capability so a utility tool is selected, then
    # verify it inherits the server-level scope.
    caps = type("Caps", (), {"resources": object(), "prompts": None})()
    server.initialize_result = type("Init", (), {"capabilities": caps})()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("tools.registry.registry", reg)
        mcp_tool_mod._register_server_tools(
            "util", server,
            {"command": "x", "scope": "subagent_only",
             "tools": {"resources": True}})
    assert reg.get_entry("mcp__util__op").scope == "subagent_only"
    # list_resources utility (if registered) inherits the same server scope.
    util_entry = reg.get_entry("mcp__util__list_resources")
    if util_entry is not None:
        assert util_entry.scope == "subagent_only"


# ---------------------------------------------------------------------------
# Dynamic refresh retains scope (re-registers from the same config)
# ---------------------------------------------------------------------------

def test_dynamic_refresh_retains_scope(monkeypatch):
    """_refresh_tools re-calls _register_server_tools with self._config, so
    scope must survive a tools/list_changed refresh."""
    reg = _fresh_registry()
    server = _FakeServer([_FakeMcpTool("danger", "danger op")])
    cfg = {"scope": "subagent_only", "command": "x"}

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("tools.registry.registry", reg)
        mcp_tool_mod._register_server_tools("dyn", server, cfg)
        # Simulate a refresh: new tool list, same config.
        server._tools = [_FakeMcpTool("danger", "danger op v2")]
        mcp_tool_mod._register_server_tools("dyn", server, cfg)

    # Deregister-reregister keeps the name; scope still subagent_only.
    entry = reg.get_entry("mcp__dyn__danger")
    assert entry is not None
    assert entry.scope == "subagent_only"
    # And MAIN still does not see it.
    assert "mcp__dyn__danger" not in {d["function"]["name"] for d in reg.get_definitions({"mcp__dyn__danger"})}
