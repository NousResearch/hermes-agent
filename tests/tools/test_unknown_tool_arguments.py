"""Undeclared top-level tool arguments are rejected pre-dispatch when the schema closes the object.

zod/pydantic-backed MCP servers advertise ``additionalProperties: false`` yet strip unknown keys at
parse time, so a model that wrapped the real arguments in an ``arguments`` field got a successful
default result and kept the shape all session (RooCodeInc/Roomote#2695; same class as #115641).
Open schemas keep admitting extra keys: hermes' own tools rely on that for internal/legacy keys.
"""

import json

import pytest

import model_tools
from tools.registry import registry


def _schema(name: str, **parameters_extra):
    return {"name": name, "description": "probe",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, **parameters_extra}}


@pytest.fixture
def probe_tool():
    seen: list[dict] = []

    def handler(args, **kwargs):
        seen.append(dict(args))
        return json.dumps({"ok": True, "query": args.get("query", "<default>")})

    def register(name, **parameters_extra):
        registry.register(name=name, toolset="mcp-probe", handler=handler, schema=_schema(name, **parameters_extra))
        return seen

    yield register
    for name in ("mcp__probe__closed", "mcp__probe__open"):
        registry.deregister(name)


def test_closed_schema_rejects_unknown_keys_before_the_handler_runs(probe_tool):
    seen = probe_tool("mcp__probe__closed", additionalProperties=False)

    result = json.loads(model_tools.handle_function_call("mcp__probe__closed", {"arguments": {"query": "hello"}}))

    assert seen == [], "handler must not run on a malformed call"
    assert "'arguments'" in result["error"] and "accepts: query" in result["error"]
    assert "Do not wrap the arguments" in result["error"]  # names the fix, not just the failure

    # Declared keys only: dispatched normally.
    ok = json.loads(model_tools.handle_function_call("mcp__probe__closed", {"query": "hello"}))
    assert ok == {"ok": True, "query": "hello"} and seen == [{"query": "hello"}]


def test_open_schema_still_admits_undeclared_keys(probe_tool):
    seen = probe_tool("mcp__probe__open")  # no additionalProperties: JSON Schema default is open

    ok = json.loads(model_tools.handle_function_call("mcp__probe__open", {"query": "q", "cross_profile": True}))

    assert ok["ok"] is True and seen == [{"query": "q", "cross_profile": True}]
