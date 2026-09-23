from __future__ import annotations

import json
import sys

from hermes_platform import declaration
from tools.registry import registry


def _schema(name):
    return {"name": name, "description": "fixture", "parameters": {"type": "object", "properties": {}}}


def _decl(tmp_path):
    missing = tmp_path / "missing-app"
    return declaration.parse_declaration(
        "Example App",
        {sys.platform: {"presence": "executable", "location": str(missing)}},
        {"app": True},
        where="test",
    )


def test_hidden_declared_server_appears_in_listing_and_empty_summary(tmp_path, monkeypatch):
    import hermes_cli.agent_plugins as agent_plugins
    from tools.tool_search import ToolSearchConfig, dispatch_tool_search
    from tools.tool_search_catalog import build_catalog_listing_with_form

    server = "example-hidden"
    tool_name = "mcp__example-hidden__inspect"
    declaration.register(server, _decl(tmp_path))
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {"kind": "static"}, raising=False)
    registry.register(
        name=tool_name,
        toolset=f"mcp-{server}",
        schema=_schema(tool_name),
        handler=lambda args: "{}",
        check_fn=lambda: False,
    )
    try:
        first = build_catalog_listing_with_form([], max_tokens=4000)
        second = build_catalog_listing_with_form([], max_tokens=4000)
        result = json.loads(dispatch_tool_search(
            {"queries": ["inspect"]},
            current_tool_defs=[],
            config=ToolSearchConfig(enabled="on", threshold_pct=50.0, search_default_limit=10, max_search_limit=50, defer_tools=frozenset({tool_name})),
        ))
    finally:
        registry.deregister(tool_name)
        declaration.unregister(server)
    assert first == second
    assert first[1] == "full"
    assert "example-hidden (1 tools unavailable: Example App is not installed." in first[0]
    source = result["results"][0]["available_sources"][0]
    assert source["name"] == server and source["tool_count"] == 1
    assert "Example App is not installed." in source["unavailable"]


def test_hidden_undeclared_server_is_absent(monkeypatch):
    from tools.tool_search_catalog import build_catalog_listing_with_form

    tool_name = "mcp__plain-hidden__inspect"
    registry.register(
        name=tool_name,
        toolset="mcp-plain-hidden",
        schema=_schema(tool_name),
        handler=lambda args: "{}",
        check_fn=lambda: False,
    )
    try:
        listing, form = build_catalog_listing_with_form([], max_tokens=4000)
    finally:
        registry.deregister(tool_name)
    assert (listing, form) == (None, "none")
