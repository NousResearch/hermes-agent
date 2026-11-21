"""Regression: a plugin-provided tool must never be hidden by Tool Search.

Finding (group bot, 2026-11-06): with ``tools.tool_search.enabled`` on, the live request
carried 9 core tools plus the three bridge tools and NOTHING else — ``render_chart``,
``send_sticker``, ``recent_messages`` and the rest of the plugin surface were gone from the
model-visible array. The model then called ``tool_search`` with ``"create image chart plot
file"``, the lexical catalogue returned ``matches: []``, and the model concluded the chart
capability did not exist. A capability the model cannot see is a capability it does not have.

Policy after the fix: deferral is for MCP tools (unbounded catalogues) and for the curated
``defer`` list. A plugin registers a handful of tools for THIS installation, so they stay in
the array; an operator who wants one behind the bridge names it in ``defer`` and Tool Search
then indexes and returns it like any other deferred tool. Both doors are covered here.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "", properties: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


def _register(name: str, toolset: str, description: str = "Probe tool.") -> None:
    """Register *name* in the live registry under *toolset* (what a plugin's ctx does)."""
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    registry.register(
        name=name,
        handler=lambda args, **kwargs: "{}",
        schema=_td(name, description)["function"],
        toolset=toolset,
    )


@pytest.fixture
def plugin_tool() -> str:
    """A plugin tool exactly like ``render_chart``: registered under the ``chat`` toolset."""
    name = "probe_render_chart"
    _register(name, "chat", "Render a chart from data points to a PNG image attached to the reply.")
    return name


class TestPluginToolStaysInTheModelSurface:
    def test_plugin_tool_is_visible_with_tool_search_enabled(self, plugin_tool):
        """The defect: assembly deleted the plugin tool from the array."""
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        defs = [
            _td("terminal", "Run a shell command"),
            _td("computer_use", "Drive the OS"),  # curated defer -> the bridge is active
            _td(plugin_tool, "Render a chart from data points to a PNG image."),
        ]
        assembled = assemble_tool_defs(
            defs, context_length=200_000, config=ToolSearchConfig.from_raw({"enabled": "on"}))

        names = [td["function"]["name"] for td in assembled.tool_defs]
        assert assembled.activated is True, "the curated defer list must still activate the bridge"
        assert plugin_tool in names, f"plugin tool dropped from the model surface: {names}"
        assert "tool_search" in names
        assert "computer_use" not in names  # the curated core tool really did defer

    def test_plugin_tool_is_not_deferrable(self, plugin_tool):
        from tools.tool_search import is_deferrable_tool_name

        assert is_deferrable_tool_name(plugin_tool) is False
        assert is_deferrable_tool_name(plugin_tool, frozenset({"terminal"})) is False

    def test_bridge_door_names_the_direct_tool(self, plugin_tool):
        """``tool_describe`` on a directly-listed plugin tool must say so, not spin."""
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe

        defs = [_td("terminal", "Run a shell command"), _td(plugin_tool, "Render a chart.")]
        payload = json.loads(dispatch_tool_describe(
            {"names": [plugin_tool]}, current_tool_defs=defs,
            config=ToolSearchConfig.from_raw({"enabled": "on"})))

        assert plugin_tool not in payload.get("tools", {})
        assert "directly-listed tool" in payload["errors"][plugin_tool]

    def test_mcp_tools_still_defer(self):
        """The fix must not turn Tool Search off: MCP catalogues still go behind the bridge."""
        from tools.tool_search import is_deferrable_tool_name
        from toolsets import _HERMES_CORE_TOOLS

        name = "probe_mcp_record_create"
        _register(name, "mcp-probe-catalog", "Create a record in the connected catalogue.")

        assert is_deferrable_tool_name(name) is True
        assert name not in _HERMES_CORE_TOOLS


class TestOperatorCanStillDeferAPluginTool:
    def test_named_plugin_tool_defers_and_is_found_by_search(self, plugin_tool, monkeypatch):
        """The opt-in door: ``defer: [<plugin tool>]`` hides it AND the bridge indexes it."""
        from tools import tool_search as ts

        cfg = ts.ToolSearchConfig.from_raw({"enabled": "on", "defer": [plugin_tool]})
        # The bridge reads the session's deferral set from config (the same seam production
        # uses); point it at this test's override.
        monkeypatch.setattr(ts, "load_config_readonly", lambda: cfg)
        defs = [_td("terminal", "Run a shell command"), _td(plugin_tool, "Render a chart to a PNG.")]

        assembled = ts.assemble_tool_defs(defs, context_length=200_000, config=cfg)
        names = [td["function"]["name"] for td in assembled.tool_defs]
        assert plugin_tool not in names
        assert {"tool_search", "tool_describe", "tool_call"} <= set(names)

        payload = json.loads(ts.dispatch_tool_search(
            {"queries": [plugin_tool]}, current_tool_defs=defs, config=cfg))
        assert payload["results"][0]["matches"] == [plugin_tool]
        assert payload["total_available"] == 1
        described = json.loads(ts.dispatch_tool_describe(
            {"names": [plugin_tool]}, current_tool_defs=defs, config=cfg))
        assert described["tools"][plugin_tool]["parameters"]["type"] == "object"
