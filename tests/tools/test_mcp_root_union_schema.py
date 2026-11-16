"""Tools whose input schema is a ROOT union (zod ``z.union``/``discriminatedUnion``) stay usable.

Before: a bare ``{"anyOf": [...]}`` root made the SDK reject the whole ``tools/list`` page, so the
server connected with zero tools; with ``"type": "object"`` beside the union, the provider-side
strip of root combinators left the model ``properties: {}``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap

import pytest

pytest.importorskip("mcp")

from tools.mcp_tool_schema import _convert_mcp_schema, list_tools_repairing_schemas
from tools.schema_sanitizer import sanitize_tool_schemas, unrename_tool_args

_UNION = {
    "anyOf": [
        {"type": "object", "properties": {"action": {"type": "string", "const": "get"}, "id": {"type": "string"}},
         "required": ["action", "id"], "additionalProperties": False},
        {"type": "object", "properties": {"action": {"type": "string", "const": "search"},
                                          "query": {"type": "string"}},
         "required": ["action", "query"], "additionalProperties": False},
    ],
}

_SERVER = textwrap.dedent("""
    import json, sys
    SCHEMA = json.loads(sys.argv[1])
    for line in sys.stdin:
        msg = json.loads(line)
        if "id" not in msg:
            continue
        if msg["method"] == "initialize":
            result = {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}},
                      "serverInfo": {"name": "u", "version": "1"}}
        elif msg["method"] == "tools/list":
            result = {"tools": [{"name": "records", "description": "Records.", "inputSchema": SCHEMA},
                                {"name": "plain", "inputSchema": {"type": "object", "properties": {}}}]}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
""")


def _model_visible(tool) -> dict:
    schema = _convert_mcp_schema("u", tool)
    return sanitize_tool_schemas([{"type": "function", "function": schema}])[0]["function"]


def test_bare_root_union_server_lists_every_tool_with_its_arguments(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    script = tmp_path / "server.py"
    script.write_text(_SERVER, encoding="utf-8")

    async def listing(errlog):
        params = StdioServerParameters(command=sys.executable, args=[str(script), json.dumps(_UNION)])
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await asyncio.wait_for(list_tools_repairing_schemas(session)(), 30)

    with open(tmp_path / "stderr.log", "w", encoding="utf-8") as errlog:
        tools = {tool.name: tool for tool in asyncio.run(listing(errlog)).tools}
    assert set(tools) == {"records", "plain"}
    fn = _model_visible(tools["records"])
    assert fn["parameters"]["properties"] == {
        "action": {"type": "string", "enum": ["get", "search"]}, "id": {"type": "string"}, "query": {"type": "string"}}
    assert fn["parameters"]["required"] == ["action"]
    assert "anyOf" not in fn["parameters"]
    assert fn["description"].endswith('Argument sets (provide one): action="get", id | action="search", query')


def test_root_combinator_branches_lift_into_properties_and_dispatch_maps_keys_back():
    allof = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"],
             "allOf": [{"properties": {"y": {"type": "integer"}}, "required": ["y"]}]}
    oneof = {"type": "object", "oneOf": [{"properties": {"a b": {"type": "string"}}, "required": ["a b"]},
                                         {"properties": {"c": {"type": "string"}}, "required": ["c"]}]}
    fragments = {"type": "object", "properties": {"m": {"type": "string"}},
                 "oneOf": [{"required": ["m"]}, {"required": ["n"]}]}
    out = sanitize_tool_schemas([{"type": "function", "function": {"name": n, "description": "d", "parameters": p}}
                                 for n, p in (("allof", allof), ("oneof", oneof), ("fragments", fragments))])
    params = {tool["function"]["name"]: tool["function"]["parameters"] for tool in out}
    assert params["allof"]["properties"] == {"x": {"type": "string"}, "y": {"type": "integer"}}
    assert params["allof"]["required"] == ["x", "y"]  # every allOf branch applies
    assert params["oneof"]["properties"] == {"a_b": {"type": "string"}, "c": {"type": "string"}}
    assert params["oneof"]["required"] == []  # nothing is required by every branch
    assert params["fragments"] == {"type": "object", "properties": {"m": {"type": "string"}}, "required": []}
    assert out[2]["function"]["description"] == "d"  # constraint-only branches add no hint
    assert unrename_tool_args(oneof, {"a_b": "v", "c": "w"}) == {"a b": "v", "c": "w"}
