"""The Kanban lifecycle tools are core tools, so they never defer.

``toolsets._HERMES_CORE_TOOLS`` lists every Kanban tool a worker may call, and
``tools/tool_search.py`` treats a tool absent from that list as deferrable: with
Tool Search on, the model sees a catalog stub instead of the tool's schema and
has to spend a ``tool_search`` round-trip to reach it.

``kanban_schedule`` was the only Kanban tool missing from the list, so a worker
waiting on a timed wait had to search for the one tool that parks its own card.
Regression for #153430.
"""

from __future__ import annotations

import pytest


def test_every_kanban_tool_is_a_core_tool():
    """Every registered Kanban tool is core-listed, so none of them defers."""
    from tools.registry import discover_builtin_tools, registry
    from toolsets import _HERMES_CORE_TOOLS

    discover_builtin_tools()
    kanban_tools = set(registry.get_tool_names_for_toolset("kanban"))
    assert kanban_tools, "the kanban toolset registers tools"
    assert kanban_tools <= set(_HERMES_CORE_TOOLS), (
        "Kanban tools missing from _HERMES_CORE_TOOLS defer behind the tool-search "
        f"bridge: {sorted(kanban_tools - set(_HERMES_CORE_TOOLS))}"
    )


@pytest.mark.parametrize("name", ["kanban_schedule", "kanban_block", "kanban_complete"])
def test_kanban_tools_are_never_deferrable(name):
    """Core tools are directly callable; the deferral set is the curated list."""
    from tools.registry import discover_builtin_tools
    from tools.tool_search import is_deferrable_tool_name

    discover_builtin_tools()
    assert not is_deferrable_tool_name(name), name


def test_schedule_reaches_a_worker_schema_directly(monkeypatch):
    """A worker's schema carries ``kanban_schedule`` itself, not a catalog stub."""
    from tools.registry import discover_builtin_tools, invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    discover_builtin_tools()
    invalidate_check_fn_cache()
    names = {
        td["function"]["name"]
        for td in registry.get_definitions(set(resolve_toolset("kanban")), quiet=True)
    }
    assert "kanban_schedule" in names


def test_schedule_stays_out_of_a_normal_chat_schema():
    """Core-listing a Kanban tool must not expose it outside a Kanban session:
    the ``check_fn`` gate still decides who sees it."""
    from tools.registry import discover_builtin_tools, invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    discover_builtin_tools()
    invalidate_check_fn_cache()
    names = {
        td["function"]["name"]
        for td in registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    }
    assert not {n for n in names if n.startswith("kanban_")}
