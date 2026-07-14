"""Tests for tools/schema_sanitizer.py.

Targets the known llama.cpp ``json-schema-to-grammar`` failure modes that
cause ``HTTP 400: Unable to generate parser for this template. ...
Unrecognized schema: "object"`` errors on local inference backends.
"""

from __future__ import annotations

import copy

from tools.schema_sanitizer import (
    sanitize_tool_schemas,
    strip_pattern_and_format,
    strip_slash_enum,
)


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


def test_ref_with_default_sibling_stripped():
    """Strict backends reject ``default`` alongside ``$ref``."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "payload": {"$ref": "#/$defs/Payload", "default": None},
        },
        "$defs": {
            "Payload": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    payload = out[0]["function"]["parameters"]["properties"]["payload"]
    assert payload == {"$ref": "#/$defs/Payload"}


def test_nullable_union_collapse_does_not_leave_default_on_ref():
    """Nullable anyOf collapse must not attach ``default`` to a ``$ref`` branch."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "input": {
                "anyOf": [
                    {"$ref": "#/$defs/Payload"},
                    {"type": "null"},
                ],
                "default": None,
            },
        },
        "$defs": {
            "Payload": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    prop = out[0]["function"]["parameters"]["properties"]["input"]
    assert prop["$ref"] == "#/$defs/Payload"
    assert "default" not in prop
    assert prop.get("nullable") is True


def test_ref_description_preserved():
    """Annotation siblings that strict backends allow should survive."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "payload": {
                "$ref": "#/$defs/Payload",
                "description": "The payload",
            },
        },
        "$defs": {
            "Payload": {"type": "object", "properties": {}},
        },
    })]
    out = sanitize_tool_schemas(tools)
    payload = out[0]["function"]["parameters"]["properties"]["payload"]
    assert payload["description"] == "The payload"
    assert payload["$ref"] == "#/$defs/Payload"


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



def test_strip_responses_format_strips_format_keyword():
    """Responses-format:  keyword should be stripped."""
    from tools.schema_sanitizer import strip_pattern_and_format

    tools = [
        {
            "name": "get_event",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts": {"type": "string", "format": "date-time"},
                }
            },
            "type": "function"
        }
    ]

    result, stripped = strip_pattern_and_format(tools)
    assert stripped == 1, f"Expected 1 format stripped, got {stripped}"
    assert "format" not in result[0]["parameters"]["properties"]["ts"], "format should be stripped"
    assert result[0]["parameters"]["properties"]["ts"]["type"] == "string", "type should be preserved"


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


def test_strip_responses_format_tools():
    """strip_pattern_and_format should handle Responses-format tools (no function wrapper)."""
    from tools.schema_sanitizer import strip_pattern_and_format

    # Responses-format: {"name": "...", "parameters": {...}, "type": "function"}
    tools = [
        {
            "name": "mcp_firecrawl_search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "includeDomains": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$"
                        }
                    }
                }
            },
            "type": "function"
        }
    ]

    result, stripped = strip_pattern_and_format(tools)
    assert stripped == 1, f"Expected 1 pattern stripped, got {stripped}"
    
    # Verify pattern keyword was removed from includeDomains
    domains = result[0]["parameters"]["properties"]["includeDomains"]["items"]
    assert "pattern" not in domains, f"pattern should be stripped: {domains}"
    assert domains["type"] == "string", "type should be preserved"


def test_strip_responses_idempotent():
    """Second call on already-stripped Responses-format tools should return 0."""
    from tools.schema_sanitizer import strip_pattern_and_format

    tools = [
        {
            "name": "search_files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"}  # This is a property named pattern, NOT schema keyword
                }
            }
        }
    ]

    # Pass 1 - property named 'pattern' should NOT be stripped
    result, first = strip_pattern_and_format(tools)
    assert first == 0, f"Expected 0 stripped (property pattern preserved), got {first}"
    assert "pattern" in result[0]["parameters"]["properties"], "property named pattern should survive"
    
    # Pass 2 - idempotent
    _, second = strip_pattern_and_format(tools)
    assert second == 0, f"Expected 0 on second pass, got {second}"


def test_strip_responses_mixed_formats():
    """Mixed list of OpenAI-format and Responses-format tools should both be sanitized."""
    from tools.schema_sanitizer import strip_pattern_and_format

    tools = [
        # OpenAI-format: {"function": {"parameters": {...}}}
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "pattern": "^[a-z]+$"}
                    }
                }
            }
        },
        # Responses-format: {"name": "...", "parameters": {...}}
        {
            "name": "get_time",
            "parameters": {
                "type": "object",
                "properties": {
                    "tz": {"type": "string", "format": "date-time"}
                }
            },
            "type": "function"
        }
    ]

    result, stripped = strip_pattern_and_format(tools)
    assert stripped == 2, f"Expected 2 stripped (1 pattern + 1 format), got {stripped}"

    # OpenAI-format tool: pattern stripped from parameters
    openai_params = result[0]["function"]["parameters"]["properties"]["query"]
    assert "pattern" not in openai_params, f"pattern should be stripped: {openai_params}"

    # Responses-format tool: format stripped
    resp_params = result[1]["parameters"]["properties"]["tz"]
    assert "format" not in resp_params, f"format should be stripped: {resp_params}"

    # Verify structure preserved
    assert result[0]["function"]["parameters"]["type"] == "object"
    assert result[1]["parameters"]["type"] == "object"


# ─────────────────────────────────────────────────────────────────────────
# strip_slash_enum — reactive recovery when xAI's /v1/responses (and
# /v1/chat/completions) grammar-compiler rejects enum values containing
# a forward slash. Symptom: HTTP 400 "Invalid arguments passed to the
# model" before any token is emitted. Most commonly hit by MCP-derived
# tools whose enum lists HuggingFace IDs like "Qwen/Qwen3.5-0.8B".
# ─────────────────────────────────────────────────────────────────────────


def test_strip_slash_enum_removes_huggingface_id_enum():
    """enum containing HF-style 'owner/name' IDs → stripped."""
    tools = [_tool("train", {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "enum": ["Qwen/Qwen3.5-0.8B", "openai/gpt-oss-20b"],
            },
        },
    })]
    _, stripped = strip_slash_enum(tools)
    assert stripped == 1
    prop = tools[0]["function"]["parameters"]["properties"]["model"]
    assert "enum" not in prop
    # Type + description survive so the model still gets the prompting hint.
    assert prop["type"] == "string"


def test_strip_slash_enum_preserves_slashless_enum():
    """enum without any '/' → preserved."""
    tools = [_tool("pick", {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["fast", "slow"]},
        },
    })]
    _, stripped = strip_slash_enum(tools)
    assert stripped == 0
    assert tools[0]["function"]["parameters"]["properties"]["mode"]["enum"] == ["fast", "slow"]


def test_strip_slash_enum_partial_match_strips_whole_enum():
    """Any single value containing '/' triggers removal of the entire enum.

    Rationale: if we kept the slashless values, the model could still pick
    them, but xAI's grammar-compile failure is all-or-nothing on the enum
    keyword — keeping a mixed-content enum would still 400. Drop it whole.
    """
    tools = [_tool("pick", {
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": ["local", "hf://Qwen/Qwen3"]},
        },
    })]
    _, stripped = strip_slash_enum(tools)
    assert stripped == 1
    assert "enum" not in tools[0]["function"]["parameters"]["properties"]["target"]


def test_strip_slash_enum_responses_format():
    """Responses-format tools (no `function` wrapper) are also handled."""
    tools = [{
        "type": "function",
        "name": "mcp_prime_lab_train_model",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["Qwen/Qwen3.5-0.8B", "meta-llama/Llama-3.2-1B-Instruct"],
                },
            },
        },
    }]
    _, stripped = strip_slash_enum(tools)
    assert stripped == 1
    assert "enum" not in tools[0]["parameters"]["properties"]["model"]


def test_strip_slash_enum_recurses_into_anyof():
    """enum-with-slash inside an anyOf variant is also stripped."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string", "enum": ["owner/repo"]},
                    {"type": "null"},
                ],
            },
        },
    })]
    _, stripped = strip_slash_enum(tools)
    assert stripped == 1
    variants = tools[0]["function"]["parameters"]["properties"]["value"]["anyOf"]
    assert "enum" not in variants[0]
    assert variants[0]["type"] == "string"


def test_strip_slash_enum_is_idempotent():
    """Second call on already-stripped tools is a no-op."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {"m": {"type": "string", "enum": ["a/b"]}},
    })]
    _, first = strip_slash_enum(tools)
    _, second = strip_slash_enum(tools)
    assert first == 1
    assert second == 0


def test_strip_slash_enum_empty_returns_zero():
    tools, stripped = strip_slash_enum([])
    assert tools == []
    assert stripped == 0


def test_strip_slash_enum_none_returns_zero():
    tools, stripped = strip_slash_enum(None)
    assert tools is None
    assert stripped == 0


def test_strip_slash_enum_ignores_non_string_enum_values():
    """Integer/boolean enum values can't contain '/' — leave them alone."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "level": {"type": "integer", "enum": [1, 2, 3]},
            "flag": {"type": "boolean", "enum": [True, False]},
        },
    })]
    _, stripped = strip_slash_enum(tools)
    assert stripped == 0
    props = tools[0]["function"]["parameters"]["properties"]
    assert props["level"]["enum"] == [1, 2, 3]
    assert props["flag"]["enum"] == [True, False]


# ─────────────────────────────────────────────────────────────────────────
# _sanitize_single_tool — uncovered branches
# ─────────────────────────────────────────────────────────────────────────


def test_non_dict_tool_entry_returned_unchanged():
    """A tool entry whose 'function' value is not a dict is returned as-is."""
    # Line 63: _sanitize_single_tool returns out when fn is not a dict.
    tool = {"type": "function", "function": "bad_value"}
    out = sanitize_tool_schemas([tool])
    assert out[0]["function"] == "bad_value"


def test_tool_without_function_key_returned_unchanged():
    """A tool dict with no 'function' key passes through without modification."""
    # Line 62-63: fn = out.get("function") is None, not isinstance(fn, dict).
    tool = {"type": "function", "name": "bare_tool"}
    out = sanitize_tool_schemas([tool])
    assert "function" not in out[0]
    assert out[0]["name"] == "bare_tool"


def test_non_dict_parameters_string_gets_default_schema():
    """A string parameters value (not caught by existing test) gets replaced."""
    # Lines 68-69: not isinstance(params, dict) branch with a non-None string.
    tool = {"type": "function", "function": {"name": "t", "parameters": "bad"}}
    out = sanitize_tool_schemas([tool])
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_post_recursion_top_not_dict_gets_default():
    """If _sanitize_node returns a non-dict for the parameters, the top-level
    fixup replaces it with the minimal valid schema (line 75)."""
    # _sanitize_node on a bare-list node returns a list, which is not a dict.
    # Construct parameters as a list directly to trigger the line 74-75 branch.
    # We patch parameters via a raw dict that has a list as its "parameters" value;
    # simulate by passing a list inside a dict that _sanitize_node normalizes to a list.
    # The cleanest driver: pass parameters={"type": "array", "items": []} as the outer
    # schema — after _sanitize_node it is still a dict, so we need a different approach.
    # Use a parameters dict whose only key is a bare-string value that _sanitize_node
    # turns into a dict — but that still ends up as a dict.
    # The only way to get a non-dict top is to override the function entry directly:
    from tools import schema_sanitizer
    import copy

    original_sanitize_node = schema_sanitizer._sanitize_node

    def returning_list(node, path):
        # Force the top-level call to return a list so line 75 fires.
        if path == "t":
            return ["not", "a", "dict"]
        return original_sanitize_node(node, path)

    import unittest.mock as mock
    with mock.patch.object(schema_sanitizer, "_sanitize_node", side_effect=returning_list):
        tools = [_tool("t", {"type": "object", "properties": {}})]
        out = schema_sanitizer.sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_post_recursion_top_type_not_object_is_fixed():
    """After recursion, if top-level type is not 'object', it is coerced (line 78)."""
    # _sanitize_node preserves a 'string' type — pass a plain string schema as parameters.
    # parameters={"type": "string"} passes _sanitize_node intact (it is a dict),
    # then the top-level fixup on line 77-78 sets type='object'.
    tool = {"type": "function", "function": {"name": "t", "parameters": {"type": "string"}}}
    out = sanitize_tool_schemas([tool])
    assert out[0]["function"]["parameters"]["type"] == "object"
    assert out[0]["function"]["parameters"]["properties"] == {}


def test_post_recursion_properties_non_dict_is_replaced():
    """After recursion, a non-dict properties value at the top level is replaced (line 80)."""
    # properties: ["list"] survives _sanitize_node unchanged (it's under key "properties"
    # but the node loop copies it via the pass-through branch for unexpected types).
    # However, the post-recursion check at line 79-80 fires because isinstance(list) != dict.
    tool = {
        "type": "function",
        "function": {
            "name": "t",
            "parameters": {"type": "object", "properties": ["bad"]},
        },
    }
    out = sanitize_tool_schemas([tool])
    assert out[0]["function"]["parameters"]["properties"] == {}


# ─────────────────────────────────────────────────────────────────────────
# _strip_top_level_combinators — non-dict input path and logger.debug path
# ─────────────────────────────────────────────────────────────────────────


def test_strip_top_level_combinators_non_dict_passthrough():
    """_strip_top_level_combinators returns the value unchanged for non-dict input (line 118)."""
    from tools.schema_sanitizer import _strip_top_level_combinators
    assert _strip_top_level_combinators("not-a-dict") == "not-a-dict"
    assert _strip_top_level_combinators(None) is None
    assert _strip_top_level_combinators([1, 2]) == [1, 2]


def test_strip_top_level_combinators_logs_debug_on_strip(caplog):
    """logger.debug fires when a forbidden key is stripped (lines 122, 127)."""
    import logging
    from tools.schema_sanitizer import _strip_top_level_combinators
    with caplog.at_level(logging.DEBUG, logger="tools.schema_sanitizer"):
        result = _strip_top_level_combinators(
            {"type": "object", "properties": {}, "allOf": [{"required": ["x"]}]},
            path="my_tool",
        )
    assert "allOf" not in result
    # The debug message names the tool path and the stripped key.
    assert any("my_tool" in r.message and "allOf" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────
# strip_nullable_unions — keep_nullable_hint=False path; list input;
# metadata carry-over
# ─────────────────────────────────────────────────────────────────────────


def test_strip_nullable_unions_no_hint():
    """keep_nullable_hint=False: collapsed replacement must NOT get nullable:true (line 185)."""
    from tools.schema_sanitizer import strip_nullable_unions
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    result = strip_nullable_unions(schema, keep_nullable_hint=False)
    assert result["type"] == "string"
    assert "nullable" not in result


def test_strip_nullable_unions_list_input():
    """A top-level list is recursed into, each item processed independently."""
    from tools.schema_sanitizer import strip_nullable_unions
    items = [
        {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        {"type": "string"},
    ]
    result = strip_nullable_unions(items)
    assert isinstance(result, list)
    assert result[0]["type"] == "integer"
    assert result[1] == {"type": "string"}


def test_strip_nullable_unions_metadata_carried_over():
    """title/description/default/examples on the outer union node carry to the replacement."""
    from tools.schema_sanitizer import strip_nullable_unions
    schema = {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": "A name",
        "default": None,
        "title": "Name",
    }
    result = strip_nullable_unions(schema, keep_nullable_hint=False)
    assert result["type"] == "string"
    assert result["description"] == "A name"
    assert result["default"] is None
    assert result["title"] == "Name"


def test_strip_nullable_unions_non_scalar_passthrough():
    """A non-dict, non-list scalar is returned unchanged (line 165)."""
    from tools.schema_sanitizer import strip_nullable_unions
    assert strip_nullable_unions(42) == 42
    assert strip_nullable_unions(True) is True
    assert strip_nullable_unions(None) is None


# ─────────────────────────────────────────────────────────────────────────
# _sanitize_node — uncovered branches
# ─────────────────────────────────────────────────────────────────────────


def test_sanitize_node_bare_string_primitive_becomes_type_dict():
    """A bare-string schema 'string'/'integer'/etc. becomes {"type": X} (lines 206-212)."""
    # Drive through sanitize_tool_schemas: put a bare string as a property value.
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "count": "integer",
            "flag": "boolean",
            "ratio": "number",
            "data": "array",
        },
    })]
    out = sanitize_tool_schemas(tools)
    props = out[0]["function"]["parameters"]["properties"]
    assert props["count"] == {"type": "integer"}
    assert props["flag"] == {"type": "boolean"}
    assert props["ratio"] == {"type": "number"}
    assert props["data"] == {"type": "array"}


def test_sanitize_node_bare_string_null_becomes_type_null():
    """Bare string 'null' is a recognized type; becomes {"type": "null"} (lines 206-212)."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {"nothing": "null"},
    })]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"]["properties"]["nothing"] == {"type": "null"}


def test_sanitize_node_bare_unknown_string_becomes_empty_object():
    """An unrecognized bare string (not a JSON Schema type) becomes an empty object (lines 219-223)."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {"bad": "not-a-type"},
    })]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"]["properties"]["bad"] == {
        "type": "object",
        "properties": {},
    }


def test_sanitize_node_items_bool_preserved():
    """items: True/False is kept as-is without recursion (line 262)."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "bag": {"type": "array", "items": True},
            "nothing": {"type": "array", "items": False},
        },
    })]
    out = sanitize_tool_schemas(tools)
    props = out[0]["function"]["parameters"]["properties"]
    assert props["bag"]["items"] is True
    assert props["nothing"]["items"] is False


def test_sanitize_node_allof_list_sanitized():
    """allOf list inside a property schema is recursed into (line 266)."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "config": {
                "allOf": [
                    {"type": "object"},   # bare object — should get properties: {}
                    {"type": "string"},
                ],
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    allof = out[0]["function"]["parameters"]["properties"]["config"]["allOf"]
    assert allof[0] == {"type": "object", "properties": {}}
    assert allof[1] == {"type": "string"}


def test_sanitize_node_nested_object_without_properties_injected():
    """Nested object nodes that lack properties get properties:{} injected (line 285)."""
    # This is the _sanitize_node-level injection (separate from the top-level fixup).
    # Drive it via a deeply nested object.
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {"type": "object"},   # no properties — should get {}
                },
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    inner = out[0]["function"]["parameters"]["properties"]["outer"]["properties"]["inner"]
    assert inner == {"type": "object", "properties": {}}


def test_sanitize_node_required_partial_prune():
    """required with some valid and some missing property refs is trimmed to valid subset (line 296)."""
    # A nested object (not the top level) so the line 296 path inside _sanitize_node fires.
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "sub": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x", "ghost"],   # "ghost" does not exist in properties
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    sub = out[0]["function"]["parameters"]["properties"]["sub"]
    assert sub["required"] == ["x"]


def test_sanitize_node_non_dict_non_str_non_list_passthrough():
    """A numeric or boolean node value is returned unchanged (line 229 path)."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {"x": {"type": "string", "minimum": 0}},
    })]
    out = sanitize_tool_schemas(tools)
    # minimum is an int; the else branch at line 280 returns it unchanged.
    assert out[0]["function"]["parameters"]["properties"]["x"]["minimum"] == 0
