"""Contract: `tools.tool_search.keep` from a real ``config.yaml`` through dispatch.

The unit tests exercise the classifier directly; this one pins the wiring the
repo rules care about: YAML on disk -> the loader the runtime uses -> the tool
surface -> ``handle_function_call`` (direct call, search/describe, bridge).
"""

from __future__ import annotations

import json

from hermes_cli.config import atomic_config_write, get_config_path


def test_keep_from_real_yaml_through_production_dispatch():
    import model_tools
    from tools import tool_search as ts
    from tools.registry import registry

    kept, cold = "review2_plugin_kept", "review2_plugin_cold"
    toolset = "review2-local-plugin"
    for name in (kept, cold):
        schema = {"name": name, "description": "review2 plugin tool",
                  "parameters": {"type": "object", "properties": {}}}
        registry.register(name=name, toolset=toolset, schema=schema,
                          handler=lambda args, **kw: json.dumps({"ok": True}))

    # `keep` must outrank an explicit `defer` entry in the same document.
    atomic_config_write(get_config_path(),
                        {"tools": {"tool_search": {"defer": [kept], "keep": [kept]}}})
    assert ts.load_config_readonly().effective_keep_tools == {kept}

    visible = model_tools.get_tool_definitions(enabled_toolsets=[toolset], quiet_mode=True)
    assert kept in ts.session_tool_names(visible)
    assert cold not in ts.session_tool_names(visible)

    def call(name, args):
        return json.loads(model_tools.handle_function_call(
            name, args, enabled_toolsets=[toolset]))

    found = call("tool_search", {"queries": ["review2"]})
    assert kept not in found["tools"] and cold in found["tools"]

    described = call("tool_describe", {"names": [kept, cold]})
    assert kept in described["errors"] and cold in described["tools"]

    rejected = call("tool_call", {"calls": [{"name": kept, "arguments": {}}]})
    assert "error" in rejected and "Call it directly" in rejected["error"]

    assert call(kept, {})["ok"] is True
    assert call("tool_call", {"calls": [{"name": cold, "arguments": {}}]})["ok"] is True
