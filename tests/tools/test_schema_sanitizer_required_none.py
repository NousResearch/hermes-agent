"""
Regression tests for tools/schema_sanitizer.py handling of "required": None.

Issue #60752 - Some tool definitions in Hermes v0.11+ emit "required": None
in their JSON Schema parameters, which strict OpenAI-compatible backends
reject with HTTP 400
("None is not of type 'array'"). The contract is: "required" must be
absent or ["..."], never null.

The sanitizer already handles 8 known compatibility quirks (llama.cpp
constraints, nullable unions, $ref siblings, slash-enums, top-level
combinators, pattern/format reactive strip, MCP $defs). This regression
was introduced between v0.10 and v0.11, and the fix belongs in the same
sanitizer so every backend-path benefits.
"""

from __future__ import annotations

import pytest


def _wrap_function(parameters: dict, name: str = "fixture_tool") -> dict:
    """Wrap a parameters dict in the OpenAI tool format the sanitizer expects."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "fixture tool for required=None regression",
            "parameters": parameters,
        },
    }


def test_sanitizer_normalizes_required_none_to_empty_list():
    """Top-level 'required': None must be normalized to []."""
    from tools.schema_sanitizer import sanitize_tool_schemas

    tool = _wrap_function({
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": None,
    })
    [out] = sanitize_tool_schemas([tool])
    fn = out["function"]
    assert fn["parameters"].get("required") == [], (
        f"expected required=[] after sanitizing required=None; got "
        f"{fn['parameters'].get('required')!r}"
    )


def test_sanitizer_preserves_valid_required_list():
    """Regression guard: a real required list must not be touched."""
    from tools.schema_sanitizer import sanitize_tool_schemas

    tool = _wrap_function({
        "type": "object",
        "properties": {
            "x": {"type": "string"},
            "y": {"type": "string"},
        },
        "required": ["x", "y"],
    })
    [out] = sanitize_tool_schemas([tool])
    assert out["function"]["parameters"]["required"] == ["x", "y"]


def test_sanitizer_normalizes_required_none_when_properties_is_empty():
    """Required=None on a no-property schema also normalizes; HTTP 400 still
    triggers on strict backends regardless of whether properties is empty.
    """
    from tools.schema_sanitizer import sanitize_tool_schemas

    tool = _wrap_function({
        "type": "object",
        "properties": {},
        "required": None,
    })
    [out] = sanitize_tool_schemas([tool])
    assert out["function"]["parameters"].get("required") == []


def test_real_terminal_tool_schema_passes_through_clean():
    """Sanity check on a real built-in tool with required=['command'] -
    the sanitizer must not poison a working schema."""
    from tools.schema_sanitizer import sanitize_tool_schemas

    tool = _wrap_function({
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "shell command"},
            "background": {"type": "boolean", "default": False},
        },
        "required": ["command"],
    })
    [out] = sanitize_tool_schemas([tool])
    params = out["function"]["parameters"]
    assert params["required"] == ["command"]
    assert params["properties"]["command"]["type"] == "string"
    assert params["properties"]["background"]["type"] == "boolean"


def test_required_none_openai_format_example_matches_issue_report():
    """Reproduce the exact shape from the issue report:
       ``"required": None`` inside ``function.parameters``.
    """
    from tools.schema_sanitizer import sanitize_tool_schemas

    bad_api_payload = [{
        "type": "function",
        "function": {
            "name": "any_tool",
            "parameters": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": None,
            },
        },
    }]
    [out] = sanitize_tool_schemas(bad_api_payload)
    params = out["function"]["parameters"]
    # Post-sanitize, the params MUST NOT contain the bad null.
    assert "required" in params
    assert params["required"] is not None
    assert isinstance(params["required"], list)
