"""A schema that closes the object refuses undeclared arguments before the call, for any tool.

zod/pydantic-backed MCP servers advertise ``additionalProperties: false`` yet strip unknown keys at
parse time, so a model that wrapped the real arguments in an ``arguments`` field got a successful
default result and kept the shape all session (RooCodeInc/Roomote#2695). Open schemas keep
admitting extra keys: a third-party ``properties`` list is no promise about what the handler takes.
"""

import json

import pytest

from hermes_agent import model_tools
from hermes_agent.tools.registry import registry


@pytest.fixture
def probe_tool():
    seen: list[dict] = []
    names: list[str] = []

    def handler(args, **kwargs):
        seen.append(dict(args))
        return json.dumps({"ok": True, "query": args.get("query", "<default>")})

    def register(name, toolset="mcp-probe", properties=True, **parameters_extra):
        parameters = {"type": "object", **({"properties": {"query": {"type": "string"}}} if properties else {}),
                      **parameters_extra}
        registry.register(name=name, toolset=toolset, handler=handler,
                          schema={"name": name, "description": "probe", "parameters": parameters})
        names.append(name)
        return seen

    yield register
    for name in names:
        registry.deregister(name)


def test_closed_schema_refuses_unknown_keys_before_the_handler_runs(probe_tool):
    seen = probe_tool("mcp__probe__closed", additionalProperties=False)

    result = json.loads(model_tools.handle_function_call("mcp__probe__closed", {"arguments": {"query": "hello"}}))

    assert seen == [], "handler must not run on a malformed call"
    assert "arguments" in result["error"] and "Valid parameters: query" in result["error"]
    assert "Do not wrap the arguments" in result["error"]
    assert json.loads(model_tools.handle_function_call("mcp__probe__closed", {"query": "hello"})) == {
        "ok": True, "query": "hello"}

    # A plugin's closed no-argument schema may omit ``properties``: that declares none.
    seen.clear()
    probe_tool("plugin_probe_noargs", toolset="probe-plugin", properties=False, additionalProperties=False)
    refused = json.loads(model_tools.handle_function_call("plugin_probe_noargs", {"arguments": {"x": 1}}))
    assert seen == [] and "Valid parameters: none" in refused["error"]
    assert json.loads(model_tools.handle_function_call("plugin_probe_noargs", {}))["ok"] is True


def test_open_schema_still_admits_undeclared_keys(probe_tool):
    seen = probe_tool("mcp__probe__open")  # no additionalProperties: JSON Schema default is open

    ok = json.loads(model_tools.handle_function_call("mcp__probe__open", {"query": "q", "cross_profile": True}))

    assert ok == {"ok": True, "query": "q"} and seen == [{"query": "q", "cross_profile": True}]
