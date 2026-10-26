"""Provider-facing tool schemas must not leak output metadata (#137128)."""

import copy

import pytest

from tools.registry import ToolRegistry
from tools.schema_sanitizer import sanitize_tool_schemas


@pytest.mark.parametrize("parameters", [None, {"type": "object", "properties": {
    "outputSchema": {"type": "string"}}, "required": []}])
def test_registry_output_schema_is_local_only(parameters):
    registry = ToolRegistry()
    schema = {"name": "lookup", "description": "Lookup.", "strict": False,
              "outputSchema": {"type": "object", "properties": {"answer": {"type": "string"}}}}
    if parameters is not None:
        schema["parameters"] = parameters
    original = copy.deepcopy(schema)
    registry.register("lookup", "example", schema, lambda args: "{}")

    wire = sanitize_tool_schemas(registry.get_definitions({"lookup"}))

    assert "outputSchema" not in wire[0]["function"]
    assert wire[0]["function"]["strict"] is False
    assert wire[0]["function"]["description"] == original["description"]
    assert registry.get_schema("lookup") == original
    if parameters is not None:
        assert wire[0]["function"]["parameters"] == parameters
    assert sanitize_tool_schemas(wire) == wire


def test_unrelated_tool_shapes_and_empty_input_are_preserved():
    tools = [{"type": "web_search_preview"}, {"type": "function", "function": {
        "name": "plain", "description": "Plain.", "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": []}}}]
    assert sanitize_tool_schemas(tools) == tools
    assert sanitize_tool_schemas([]) == []
