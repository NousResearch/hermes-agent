"""A lazy restart must not reuse a manifest filtered by a different tool selection."""

from types import SimpleNamespace

import pytest
from mcp.types import Tool

from tools import mcp_tool as mcp
from tools import mcp_tool_discovery as discovery
from tools import mcp_tool_registration as registration
from tools.registry import ToolRegistry


@pytest.fixture
def server(monkeypatch):
    for attr in (
        "_servers", "_lazy_server_configs", "_lazy_server_fingerprints",
        "_lazy_server_tool_names", "_mcp_tool_server_names", "_server_trust_levels",
        "_tool_read_only_hints", "_server_tool_scopes",
    ):
        monkeypatch.setattr(mcp, attr, {})
    monkeypatch.setattr(mcp, "_server_connecting", set())
    monkeypatch.setattr("tools.registry.registry", ToolRegistry())
    srv = mcp.MCPServerTask("selection_cache")
    srv._tools = [Tool(name="lookup", inputSchema={"type": "object"})]
    srv.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(
        resources=SimpleNamespace(), prompts=SimpleNamespace(),
    ))
    return srv


def _restart_with_selection(server, old_tools, new_tools, monkeypatch):
    old_config = {"command": "test-server", "lazy": True, "tools": old_tools}
    new_config = {**old_config, "tools": new_tools}
    assert registration._register_server_tools(server.name, server, old_config)
    registry = ToolRegistry()
    monkeypatch.setattr("tools.registry.registry", registry)
    eager, _, _ = discovery._register_lazy_from_cache({server.name: new_config})
    if server.name in eager:
        registration._register_server_tools(server.name, server, eager[server.name])
    names = registry.get_tool_names_for_toolset(f"mcp-{server.name}")
    monkeypatch.setattr("tools.registry.registry", ToolRegistry())
    assert discovery._register_lazy_from_cache({server.name: new_config}) == ({}, len(names), 1)
    return set(names)


@pytest.mark.parametrize("family,utilities", [
    ("resources", {"list_resources", "read_resource"}),
    ("prompts", {"list_prompts", "get_prompt"}),
])
@pytest.mark.parametrize("enabled", [True, False])
def test_utility_selection_is_refreshed_on_lazy_restart(server, monkeypatch, family, utilities, enabled):
    names = _restart_with_selection(server, {family: not enabled}, {family: enabled}, monkeypatch)
    selected = {f"mcp__{server.name}__{name}" for name in utilities}
    assert selected <= names if enabled else selected.isdisjoint(names)
    assert f"mcp__{server.name}__lookup" in names


@pytest.mark.parametrize("enable_native", [True, False])
def test_empty_include_is_distinct_from_no_include_on_lazy_restart(server, monkeypatch, enable_native):
    old_tools, new_tools = ({"include": []}, {}) if enable_native else ({}, {"include": []})
    names = _restart_with_selection(server, old_tools, new_tools, monkeypatch)
    assert (f"mcp__{server.name}__lookup" in names) is enable_native
