"""Tests for tools/schema_sanitizer.py.

Targets the known llama.cpp ``json-schema-to-grammar`` failure modes that
cause ``HTTP 400: Unable to generate parser for this template. ...
Unrecognized schema: "object"`` errors on local inference backends.
"""

from __future__ import annotations

import copy

from tools.schema_sanitizer import sanitize_tool_schemas


def _tool(name: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


def test_object_without_properties_gets_empty_properties():
    tools = [_tool("t", {"type": "object"})]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_nested_object_without_properties_gets_empty_properties():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object", "description": "free-form"},
        },
        "required": ["name"],
    })]
    out = sanitize_tool_schemas(tools)
    args = out[0]["function"]["parameters"]["properties"]["arguments"]
    assert args["type"] == "object"
    assert args["properties"] == {}
    assert args["description"] == "free-form"


def test_bare_string_object_value_replaced_with_schema_dict():
    # Malformed: a property's schema value is the bare string "object".
    # This is the exact shape llama.cpp reports as `Unrecognized schema: "object"`.
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "payload": "object",  # <-- invalid, should be {"type": "object"}
        },
    })]
    out = sanitize_tool_schemas(tools)
    payload = out[0]["function"]["parameters"]["properties"]["payload"]
    assert isinstance(payload, dict)
    assert payload["type"] == "object"
    assert payload["properties"] == {}


def test_bare_string_primitive_value_replaced_with_schema_dict():
    tools = [_tool("t", {
        "type": "object",
        "properties": {"name": "string"},
    })]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"]["properties"]["name"] == {"type": "string"}


def test_nullable_type_array_collapsed_to_single_string():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "maybe_name": {"type": ["string", "null"]},
        },
    })]
    out = sanitize_tool_schemas(tools)
    prop = out[0]["function"]["parameters"]["properties"]["maybe_name"]
    assert prop["type"] == "string"
    assert prop.get("nullable") is True


def test_anyof_nested_objects_sanitized():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "opt": {
                "anyOf": [
                    {"type": "object"},               # bare object
                    {"type": "string"},
                ],
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    variants = out[0]["function"]["parameters"]["properties"]["opt"]["anyOf"]
    assert variants[0] == {"type": "object", "properties": {}}
    assert variants[1] == {"type": "string"}


def test_missing_parameters_gets_default_object_schema():
    tools = [{"type": "function", "function": {"name": "t"}}]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_non_dict_parameters_gets_default_object_schema():
    tools = [_tool("t", "object")]  # pathological
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_required_pruned_to_existing_properties():
    tools = [_tool("t", {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name", "missing_field"],
    })]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"]["required"] == ["name"]


def test_required_all_missing_is_dropped():
    tools = [_tool("t", {
        "type": "object",
        "properties": {},
        "required": ["x", "y"],
    })]
    out = sanitize_tool_schemas(tools)
    assert "required" not in out[0]["function"]["parameters"]


def test_well_formed_schema_unchanged():
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "offset": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
    }
    tools = [_tool("read_file", copy.deepcopy(schema))]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"] == schema


def test_additional_properties_bool_preserved():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    payload = out[0]["function"]["parameters"]["properties"]["payload"]
    assert payload["additionalProperties"] is True


def test_additional_properties_schema_sanitized():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "dict_field": {
                "type": "object",
                "additionalProperties": {"type": "object"},  # bare object schema
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    field = out[0]["function"]["parameters"]["properties"]["dict_field"]
    assert field["additionalProperties"] == {"type": "object", "properties": {}}


def test_deepcopy_does_not_mutate_input():
    original = {
        "type": "object",
        "properties": {"x": {"type": "object"}},
    }
    tools = [_tool("t", original)]
    _ = sanitize_tool_schemas(tools)
    # Original should still lack properties on the nested object
    assert "properties" not in original["properties"]["x"]


def test_items_sanitized_in_array_schema():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "bag": {
                "type": "array",
                "items": {"type": "object"},  # bare object items
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    items = out[0]["function"]["parameters"]["properties"]["bag"]["items"]
    assert items == {"type": "object", "properties": {}}


def test_empty_tools_list_returns_empty():
    assert sanitize_tool_schemas([]) == []


def test_none_tools_returns_none():
    assert sanitize_tool_schemas(None) is None
