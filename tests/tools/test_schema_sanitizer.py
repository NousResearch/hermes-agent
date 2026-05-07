"""Tests for tools/schema_sanitizer.py.

Targets the known llama.cpp ``json-schema-to-grammar`` failure modes that
cause ``HTTP 400: Unable to generate parser for this template. ...
Unrecognized schema: "object"`` errors on local inference backends.
"""

from __future__ import annotations

import copy

from tools.schema_sanitizer import sanitize_tool_schemas, strip_pattern_and_format


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


# ─────────────────────────────────────────────────────────────────────────
# strip_pattern_and_format — reactive recovery when llama.cpp rejects a
# schema with an HTTP 400 grammar-parse error. Must be opt-in (only
# invoked on recovery) and must not damage property names.
# ─────────────────────────────────────────────────────────────────────────


def test_strip_pattern_removes_schema_pattern_keyword():
    """`pattern` as a sibling of `type` → stripped."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "date": {"type": "string", "pattern": "\\d{4,4}-\\d{2,2}-\\d{2,2}"},
        },
    })]
    _, stripped = strip_pattern_and_format(tools)
    assert stripped == 1
    prop = tools[0]["function"]["parameters"]["properties"]["date"]
    assert "pattern" not in prop
    assert prop["type"] == "string"


def test_strip_format_removes_schema_format_keyword():
    """`format` as a sibling of `type` → stripped."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "ts": {"type": "string", "format": "date-time"},
        },
    })]
    _, stripped = strip_pattern_and_format(tools)
    assert stripped == 1
    assert "format" not in tools[0]["function"]["parameters"]["properties"]["ts"]


def test_strip_preserves_property_named_pattern():
    """Property literally *named* 'pattern' (search_files) must survive."""
    tools = [_tool("search_files", {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern..."},
            "limit": {"type": "integer"},
        },
        "required": ["pattern"],
    })]
    _, stripped = strip_pattern_and_format(tools)
    assert stripped == 0
    params = tools[0]["function"]["parameters"]
    # Property named "pattern" still exists with its schema intact
    assert "pattern" in params["properties"]
    assert params["properties"]["pattern"]["type"] == "string"
    assert params["required"] == ["pattern"]


def test_strip_recurses_into_anyof_variants():
    """Pattern/format inside anyOf variant schemas are also stripped."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string", "pattern": "[A-Z]+", "format": "uuid"},
                    {"type": "integer"},
                ],
            },
        },
    })]
    _, stripped = strip_pattern_and_format(tools)
    assert stripped == 2
    variants = tools[0]["function"]["parameters"]["properties"]["value"]["anyOf"]
    assert "pattern" not in variants[0]
    assert "format" not in variants[0]
    assert variants[0]["type"] == "string"


def test_strip_is_idempotent():
    """Second call on already-stripped tools is a no-op."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {"d": {"type": "string", "pattern": "\\d+"}},
    })]
    _, first = strip_pattern_and_format(tools)
    _, second = strip_pattern_and_format(tools)
    assert first == 1
    assert second == 0


def test_strip_empty_tools_returns_zero():
    tools, stripped = strip_pattern_and_format([])
    assert tools == []
    assert stripped == 0


def test_strip_none_returns_zero():
    tools, stripped = strip_pattern_and_format(None)
    assert tools is None
    assert stripped == 0


def test_top_level_allof_stripped_for_codex_backend_compat():
    """OpenAI Codex backend rejects top-level allOf/oneOf/anyOf/enum/not."""
    tools = [_tool("memory", {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "replace"]},
            "content": {"type": "string"},
        },
        "required": ["action"],
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "add"}}, "required": ["action"]},
                "then": {"required": ["content"]},
            },
        ],
    })]
    out = sanitize_tool_schemas(tools)
    params = out[0]["function"]["parameters"]
    assert "allOf" not in params
    # Properties and required survive.
    assert params["required"] == ["action"]
    assert "content" in params["properties"]


def test_top_level_oneof_anyof_enum_not_stripped():
    """All five forbidden top-level combinators are dropped."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "oneOf": [{"required": ["x"]}],
        "anyOf": [{"required": ["x"]}],
        "enum": ["bogus-top-level"],
        "not": {"required": ["y"]},
    })]
    out = sanitize_tool_schemas(tools)
    params = out[0]["function"]["parameters"]
    for key in ("oneOf", "anyOf", "enum", "not"):
        assert key not in params, f"{key} should be stripped from top level"


def test_nested_allof_preserved():
    """Combinators inside a property's schema are preserved (only top is strict)."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {"mode": {"type": "string"}},
                "allOf": [{"required": ["mode"]}],
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    nested = out[0]["function"]["parameters"]["properties"]["config"]
    assert "allOf" in nested
    assert nested["allOf"] == [{"required": ["mode"]}]
