"""Direct recovery tools remain callable without changing sibling deferral."""

import json

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.registry import registry
from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig, assemble_tool_defs, is_deferrable_tool_name


def _schema(name):
    return {"name": name, "description": "Recovery test tool",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}


@pytest.mark.parametrize("registration", ["registry", "plugin"])
@pytest.mark.parametrize("prefix", ["recovery_", "mcp_recovery_"])
def test_registration_keeps_only_the_selected_tool_direct(registration, prefix, tmp_path, monkeypatch):
    home = str(tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", home)
    direct, sibling, unavailable = (prefix + suffix for suffix in ("attest", "inspect", "offline"))
    toolset = "mcp-recovery-test" if prefix.startswith("mcp_") else "recovery-test"
    ctx = PluginContext(PluginManifest(name="recovery-test", source="user"), PluginManager(scope_key=home))
    calls = []

    def handler(args, **kwargs):
        calls.append(args)
        return json.dumps({"reason": args["reason"]})

    def register(name, **kwargs):
        if registration == "plugin":
            return ctx.register_tool(name=name, toolset=toolset, schema=_schema(name), handler=handler, **kwargs)
        return registry.register(name=name, toolset=toolset, schema=_schema(name), handler=handler, scope=home, **kwargs)

    try:
        register(direct, never_defer=True)
        register(sibling)
        register(unavailable, never_defer=True, check_fn=lambda: False)
        assert registry.get_entry(sibling).never_defer is False
        assert not is_deferrable_tool_name(direct, frozenset({direct, sibling}))
        assert is_deferrable_tool_name(sibling)
        definitions = registry.get_definitions({direct, sibling, unavailable}, quiet=True)
        assembled = assemble_tool_defs(definitions, context_length=200_000,
                                       config=ToolSearchConfig.from_raw({"enabled": "on", "defer": [direct, sibling]}))
        names = {td["function"]["name"] for td in assembled.tool_defs}
        assert assembled.activated
        assert names == BRIDGE_TOOL_NAMES | {direct}
        assert json.loads(registry.dispatch(direct, {"reason": "Nothing to record"})) == {"reason": "Nothing to record"}
        assert calls == [{"reason": "Nothing to record"}]
    finally:
        for name in (direct, sibling, unavailable):
            registry.deregister(name, scope=home)


def test_direct_declaration_is_scoped_and_restored_with_registration(tmp_path, monkeypatch):
    homes = [str(tmp_path / profile) for profile in ("a", "b")]
    name = "recovery_profile_attest"
    original = None
    try:
        for home, direct in zip(homes, (True, False)):
            registry.register(name=name, toolset="recovery-profile", schema=_schema(name),
                              handler=lambda args, **kwargs: "{}", scope=home, never_defer=direct)
        for home, expected in ((homes[0], False), (homes[1], True), (homes[0], False)):
            monkeypatch.setenv("HERMES_HOME", home)
            assert is_deferrable_tool_name(name) is expected
        original = registry.get_entry(name)
        registry.register(name=name, toolset="recovery-profile", schema=_schema(name),
                          handler=lambda args, **kwargs: "{}", scope=homes[0])
        replacement = registry.get_entry(name)
        assert is_deferrable_tool_name(name)
        assert registry.restore_registration(name, replacement, original, scope=homes[0])
        assert not is_deferrable_tool_name(name)
    finally:
        for home in homes:
            registry.deregister(name, scope=home)
