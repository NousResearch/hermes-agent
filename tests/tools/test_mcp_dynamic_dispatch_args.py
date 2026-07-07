from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest


EXPECTED_SPAWN_ARGS = {
    "actor_name": "Cube",
    "location": [0, 0, 100],
    "rotation": {"pitch": 0, "yaw": 90, "roll": 0},
    "visible": True,
    "count": 2,
}


SPAWN_SCHEMA = {
    "type": "object",
    "properties": {
        "actor_name": {"type": "string"},
        "location": {"type": "array", "items": {"type": "number"}},
        "rotation": {
            "type": "object",
            "properties": {
                "pitch": {"type": "number"},
                "yaw": {"type": "number"},
                "roll": {"type": "number"},
            },
        },
        "visible": {"type": "boolean"},
        "count": {"type": "integer"},
        "label": {"type": "string"},
    },
    "required": ["actor_name"],
    "additionalProperties": True,
}


def _tool(name: str, schema: dict, description: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description or name,
        inputSchema=schema,
    )


def _result(payload: dict, *, is_error: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        isError=is_error,
        content=[SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))],
    )


class FakeUnrealSearchSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.actual_received: dict | None = None
        self.meta_received: dict | None = None

    async def call_tool(self, tool_name: str, arguments: dict):
        recorded = copy.deepcopy(arguments)
        self.calls.append((tool_name, recorded))
        if tool_name == "spawn_actor":
            self.actual_received = recorded
            return _result({"ok": recorded == EXPECTED_SPAWN_ARGS, "received": recorded})
        if tool_name == "list_toolsets":
            return _result({"toolsets": [{"name": "Editor", "description": "Editor tools"}]})
        if tool_name == "describe_toolset":
            return _result({
                "toolset": {
                    "name": "Editor",
                    "tools": [
                        {
                            "name": "spawn_actor",
                            "description": "Spawn an actor",
                            "inputSchema": SPAWN_SCHEMA,
                        }
                    ],
                }
            })
        if tool_name == "call_tool":
            self.meta_received = recorded
            nested = recorded.get("arguments")
            self.actual_received = copy.deepcopy(nested)
            return _result({"ok": nested == EXPECTED_SPAWN_ARGS, "received": nested})
        raise AssertionError(f"unexpected tool call: {tool_name}")


@pytest.fixture()
def fake_unreal_mcp(monkeypatch):
    import tools.mcp_tool as mcp_tool
    from tools.registry import registry

    server_name = f"argtest_{uuid4().hex[:8]}"
    session = FakeUnrealSearchSession()
    server = mcp_tool.MCPServerTask(server_name)
    server.session = session
    server._tools = [
        _tool("spawn_actor", SPAWN_SCHEMA, "Spawn an actor"),
        _tool("list_toolsets", {"type": "object", "properties": {}}, "List toolsets"),
        _tool(
            "describe_toolset",
            {
                "type": "object",
                "properties": {"toolset_name": {"type": "string"}},
                "required": ["toolset_name"],
            },
            "Describe a toolset",
        ),
        _tool(
            "call_tool",
            {
                "type": "object",
                "properties": {
                    "toolset_name": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "arguments": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "required": ["toolset_name", "tool_name", "arguments"],
            },
            "Call a tool from a toolset",
        ),
    ]

    def run_coro(coro_or_factory, timeout=30):
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        return asyncio.run(coro)

    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_coro)
    with mcp_tool._lock:
        mcp_tool._servers[server_name] = server
    asyncio.run(mcp_tool._discover_dynamic_dispatch_tools(server_name, server))
    registered = mcp_tool._register_server_tools(server_name, server, {})
    virtual_spawn_tool = next(
        name for name in registered
        if name.endswith("__Editor__spawn_actor")
    )

    try:
        yield SimpleNamespace(
            server_name=server_name,
            session=session,
            toolset=f"mcp-{server_name}",
            spawn_tool=mcp_tool.mcp_prefixed_tool_name(server_name, "spawn_actor"),
            call_tool=mcp_tool.mcp_prefixed_tool_name(server_name, "call_tool"),
            virtual_spawn_tool=virtual_spawn_tool,
        )
    finally:
        for name in registered:
            registry.deregister(name)
        with mcp_tool._lock:
            mcp_tool._servers.pop(server_name, None)
            mcp_tool._dynamic_dispatch_tools_by_server.pop(server_name, None)


def _dispatch(tool_name: str, args: dict, toolset: str) -> dict:
    import model_tools

    return json.loads(model_tools.handle_function_call(
        tool_name,
        copy.deepcopy(args),
        enabled_toolsets=[toolset],
    ))


def test_direct_mcp_tool_call_preserves_object_args(fake_unreal_mcp):
    result = _dispatch(
        fake_unreal_mcp.spawn_tool,
        EXPECTED_SPAWN_ARGS,
        fake_unreal_mcp.toolset,
    )

    assert result["result"]
    assert fake_unreal_mcp.session.actual_received == EXPECTED_SPAWN_ARGS


def test_direct_mcp_tool_call_coerces_schema_guided_json_strings(fake_unreal_mcp):
    result = _dispatch(
        fake_unreal_mcp.spawn_tool,
        {
            "actor_name": "Cube",
            "location": "[0, 0, 100]",
            "rotation": "{\"pitch\":0,\"yaw\":90,\"roll\":0}",
            "visible": "true",
            "count": "2",
        },
        fake_unreal_mcp.toolset,
    )

    assert result["result"]
    assert fake_unreal_mcp.session.actual_received == EXPECTED_SPAWN_ARGS


def test_unreal_style_call_tool_preserves_nested_object_args(fake_unreal_mcp):
    result = _dispatch(
        fake_unreal_mcp.call_tool,
        {
            "toolset_name": "Editor",
            "tool_name": "spawn_actor",
            "arguments": EXPECTED_SPAWN_ARGS,
        },
        fake_unreal_mcp.toolset,
    )

    assert result["result"]
    assert fake_unreal_mcp.session.meta_received["arguments"] == EXPECTED_SPAWN_ARGS
    assert fake_unreal_mcp.session.actual_received == EXPECTED_SPAWN_ARGS


def test_unreal_style_call_tool_parses_nested_arguments_json_string(fake_unreal_mcp):
    result = _dispatch(
        fake_unreal_mcp.call_tool,
        {
            "toolset_name": "Editor",
            "tool_name": "spawn_actor",
            "arguments": json.dumps(EXPECTED_SPAWN_ARGS),
        },
        fake_unreal_mcp.toolset,
    )

    assert result["result"]
    assert fake_unreal_mcp.session.meta_received["arguments"] == EXPECTED_SPAWN_ARGS
    assert fake_unreal_mcp.session.actual_received == EXPECTED_SPAWN_ARGS


def test_tool_search_bridge_does_not_drop_nested_mcp_arguments(fake_unreal_mcp):
    result = _dispatch(
        "tool_call",
        {
            "name": fake_unreal_mcp.call_tool,
            "arguments": {
                "toolset_name": "Editor",
                "tool_name": "spawn_actor",
                "arguments": json.dumps(EXPECTED_SPAWN_ARGS),
            },
        },
        fake_unreal_mcp.toolset,
    )

    assert result["result"]
    assert fake_unreal_mcp.session.meta_received["arguments"] == EXPECTED_SPAWN_ARGS
    assert fake_unreal_mcp.session.actual_received == EXPECTED_SPAWN_ARGS


def test_unreal_style_call_tool_coerces_nested_double_encoded_fields(fake_unreal_mcp):
    result = _dispatch(
        fake_unreal_mcp.call_tool,
        {
            "toolset_name": "Editor",
            "tool_name": "spawn_actor",
            "arguments": {
                "actor_name": "Cube",
                "location": "[0,0,100]",
                "rotation": "{\"pitch\":0,\"yaw\":90,\"roll\":0}",
                "visible": "true",
                "count": "2",
            },
        },
        fake_unreal_mcp.toolset,
    )

    assert result["result"]
    assert fake_unreal_mcp.session.actual_received == EXPECTED_SPAWN_ARGS


def test_dynamic_dispatch_preserves_string_typed_numeric_values(fake_unreal_mcp):
    _dispatch(
        fake_unreal_mcp.call_tool,
        {
            "toolset_name": "Editor",
            "tool_name": "spawn_actor",
            "arguments": {
                **EXPECTED_SPAWN_ARGS,
                "label": "001",
            },
        },
        fake_unreal_mcp.toolset,
    )

    assert fake_unreal_mcp.session.actual_received["label"] == "001"


def test_dynamic_dispatch_virtual_tool_uses_real_schema(fake_unreal_mcp):
    result = _dispatch(
        fake_unreal_mcp.virtual_spawn_tool,
        {
            "actor_name": "Cube",
            "location": "[0,0,100]",
            "rotation": "{\"pitch\":0,\"yaw\":90,\"roll\":0}",
            "visible": "true",
            "count": "2",
        },
        fake_unreal_mcp.toolset,
    )

    assert result["result"]
    assert fake_unreal_mcp.session.calls[-1][0] == "call_tool"
    assert fake_unreal_mcp.session.meta_received["arguments"] == EXPECTED_SPAWN_ARGS
    assert fake_unreal_mcp.session.actual_received == EXPECTED_SPAWN_ARGS


def test_dynamic_dispatch_preserves_json_like_strings_when_schema_says_string(fake_unreal_mcp):
    _dispatch(
        fake_unreal_mcp.call_tool,
        {
            "toolset_name": "Editor",
            "tool_name": "spawn_actor",
            "arguments": {
                **EXPECTED_SPAWN_ARGS,
                "label": "{\"literal\":true}",
            },
        },
        fake_unreal_mcp.toolset,
    )

    assert fake_unreal_mcp.session.actual_received["label"] == "{\"literal\":true}"


def test_dynamic_dispatch_preserves_unknown_additional_properties(fake_unreal_mcp):
    _dispatch(
        fake_unreal_mcp.call_tool,
        {
            "toolset_name": "Editor",
            "tool_name": "spawn_actor",
            "arguments": {
                **EXPECTED_SPAWN_ARGS,
                "metadata": {"raw": "[1,2,3]"},
            },
        },
        fake_unreal_mcp.toolset,
    )

    assert fake_unreal_mcp.session.actual_received["metadata"] == {"raw": "[1,2,3]"}


def test_dynamic_dispatch_invalid_nested_json_string_is_not_blindly_parsed(fake_unreal_mcp):
    result = _dispatch(
        fake_unreal_mcp.call_tool,
        {
            "toolset_name": "Editor",
            "tool_name": "spawn_actor",
            "arguments": "{\"actor_name\":",
        },
        fake_unreal_mcp.toolset,
    )

    assert "result" in result
    assert fake_unreal_mcp.session.actual_received == "{\"actor_name\":"
