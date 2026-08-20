"""Round-trip tests for strict-provider property-key sanitization (#63441).

Anthropic's validator forces schema property keys like ``filters[]`` to be
renamed before the model sees the tool (``filters``). The model then calls
the tool using the sanitized names, but the MCP server's contract is the
ORIGINAL inputSchema — so the dispatch path must map renamed argument keys
back before ``session.call_tool()``. These tests cover that restore step,
including the full ``_register_server_tools()`` → handler → ``call_tool``
round trip.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _register_server_tools
from tools.registry import ToolRegistry
from tools.schema_sanitizer import (
    restore_value_property_keys,
    sanitize_schema_property_keys,
)


def _schema(properties, required=None):
    schema = {"type": "object", "properties": properties}
    if required is not None:
        schema["required"] = required
    return schema


class TestRestoreValuePropertyKeys:
    def test_renamed_key_restored(self):
        schema = _schema({"filters[]": {"type": "array"}})
        restored = restore_value_property_keys(schema, {"filters": ["a"]})
        assert restored == {"filters[]": ["a"]}

    def test_canonical_keys_pass_through(self):
        schema = _schema({"filters[]": {"type": "array"}, "query": {"type": "string"}})
        args = {"filters[]": ["a"], "query": "q"}
        assert restore_value_property_keys(schema, args) == args

    def test_valid_schema_is_identity(self):
        schema = _schema({"query": {"type": "string"}})
        args = {"query": "q", "unknown_extra": 1}
        assert restore_value_property_keys(schema, args) == args

    def test_collision_suffix_restored(self):
        # `filters` and `filters[]` both exist: the second sanitizes to a
        # deterministic hash-suffixed name, which must map back.
        schema = _schema({"filters": {"type": "string"}, "filters[]": {"type": "array"}})
        sanitized = sanitize_schema_property_keys(schema)
        suffixed = [k for k in sanitized["properties"] if k != "filters"]
        assert len(suffixed) == 1
        restored = restore_value_property_keys(
            schema, {"filters": "x", suffixed[0]: ["y"]}
        )
        assert restored == {"filters": "x", "filters[]": ["y"]}

    def test_nested_object_and_array_items(self):
        schema = _schema(
            {
                "outer[]": {
                    "type": "array",
                    "items": _schema({"inner key": {"type": "string"}}),
                }
            }
        )
        restored = restore_value_property_keys(
            schema, {"outer": [{"inner_key": "v"}]}
        )
        assert restored == {"outer[]": [{"inner key": "v"}]}

    def test_nullable_union_wrapper_branch(self):
        # MCP/Pydantic optionals arrive as anyOf: [object, null]; the model
        # saw the collapsed non-null branch, so restore must descend into it.
        schema = _schema(
            {
                "opts[]": {
                    "anyOf": [
                        _schema({"page size": {"type": "integer"}}),
                        {"type": "null"},
                    ]
                }
            }
        )
        restored = restore_value_property_keys(
            schema, {"opts": {"page_size": 10}}
        )
        assert restored == {"opts[]": {"page size": 10}}

    def test_round_trip_matches_sanitized_schema(self):
        # Whatever key the sanitized schema teaches the model, restore maps
        # it back to the original for every property in one pass.
        schema = _schema(
            {
                "query": {"type": "string"},
                "filters[]": {"type": "array"},
                "user name": {"type": "string"},
            },
            required=["query"],
        )
        sanitized = sanitize_schema_property_keys(schema)
        model_args = {key: "v" for key in sanitized["properties"]}
        restored = restore_value_property_keys(schema, model_args)
        assert set(restored) == set(schema["properties"])

    def test_non_dict_inputs_unchanged(self):
        schema = _schema({"a[]": {"type": "array"}})
        assert restore_value_property_keys(schema, "text") == "text"
        assert restore_value_property_keys(None, {"a": 1}) == {"a": 1}
        assert restore_value_property_keys(schema, None) is None


class TestMcpDispatchRestoresKeys:
    """End-to-end: _register_server_tools() → registry handler → call_tool."""

    def _make_server(self, captured):
        tool = SimpleNamespace(
            name="search",
            description="Search with filters",
            inputSchema=_schema(
                {
                    "query": {"type": "string"},
                    "filters[]": {"type": "array", "items": {"type": "string"}},
                },
                required=["query", "filters[]"],
            ),
        )
        server = MCPServerTask("filtersrv")
        server._tools = [tool]
        server._rpc_lock = asyncio.Lock()

        async def fake_call_tool(name, arguments=None):
            captured["name"] = name
            captured["arguments"] = arguments
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text="ok")],
                structuredContent=None,
            )

        server.session = SimpleNamespace(call_tool=fake_call_tool)
        return server

    def test_model_visible_schema_is_sanitized_and_dispatch_restores(self):
        captured = {}
        server = self._make_server(captured)
        reg = ToolRegistry()

        with patch("tools.registry.registry", reg), \
             patch("tools.mcp_tool._get_connected_server_for_call", return_value=server), \
             patch("tools.mcp_tool._run_on_mcp_loop",
                   lambda coro_or_factory, timeout=30: asyncio.run(coro_or_factory())):
            registered = _register_server_tools("filtersrv", server, {})
            prefixed = "mcp__filtersrv__search"
            assert prefixed in registered

            # The Anthropic-visible schema renames filters[] -> filters.
            from agent.anthropic_adapter import convert_tools_to_anthropic

            entry = reg.get_entry(prefixed)
            anthropic_tools = convert_tools_to_anthropic(
                [{"type": "function", "function": entry.schema}]
            )
            visible = anthropic_tools[0]["input_schema"]["properties"]
            assert "filters" in visible
            assert "filters[]" not in visible

            # The model answers with the sanitized key; the MCP server must
            # still receive its declared `filters[]` argument.
            entry.handler({"query": "q", "filters": ["a", "b"]})
            assert captured["name"] == "search"
            assert captured["arguments"] == {"query": "q", "filters[]": ["a", "b"]}

    def test_canonical_arguments_untouched_by_dispatch(self):
        captured = {}
        server = self._make_server(captured)
        reg = ToolRegistry()

        with patch("tools.registry.registry", reg), \
             patch("tools.mcp_tool._get_connected_server_for_call", return_value=server), \
             patch("tools.mcp_tool._run_on_mcp_loop",
                   lambda coro_or_factory, timeout=30: asyncio.run(coro_or_factory())):
            _register_server_tools("filtersrv", server, {})
            entry = reg.get_entry("mcp__filtersrv__search")

            entry.handler({"query": "q", "filters[]": ["a"]})
            assert captured["arguments"] == {"query": "q", "filters[]": ["a"]}
