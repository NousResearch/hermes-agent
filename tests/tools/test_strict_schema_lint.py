"""Tests for tools/strict_schema_lint.py."""

import pytest

from tools.strict_schema_lint import (
    lint_openai_strict_schema,
    lint_tool,
    validate_tools_payload,
)


def _tool(name: str, params: dict) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": params}}


# ---------------------------------------------------------------------------
# lint_openai_strict_schema
# ---------------------------------------------------------------------------


def test_clean_object_passes():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    assert lint_openai_strict_schema(schema) == []


def test_array_without_items_flagged():
    schema = {"type": "array"}
    errors = lint_openai_strict_schema(schema)
    assert errors == [{"path": "$", "reason": "array node missing items (and no prefixItems)"}]


def test_array_with_items_passes():
    schema = {"type": "array", "items": {"type": "string"}}
    assert lint_openai_strict_schema(schema) == []


def test_array_with_prefix_items_passes():
    schema = {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
    }
    assert lint_openai_strict_schema(schema) == []


def test_nested_array_in_anyof_flagged():
    """The exact mcp_docker_create_container shape that started this fire."""
    schema = {
        "type": "object",
        "properties": {
            "ports": {
                "type": "object",
                "additionalProperties": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                        {"type": "array"},  # ← bad: no items
                    ],
                },
            },
        },
    }
    errors = lint_openai_strict_schema(schema)
    assert any(
        e["path"].endswith("anyOf[2]") and "missing items" in e["reason"]
        for e in errors
    ), errors


def test_type_array_form_flagged():
    schema = {"type": ["string", "null"]}
    errors = lint_openai_strict_schema(schema)
    assert any("type is list" in e["reason"] for e in errors)


def test_object_with_non_dict_properties_flagged():
    schema = {"type": "object", "properties": "oops"}
    errors = lint_openai_strict_schema(schema)
    assert any(e["path"] == "$.properties" for e in errors)


# ---------------------------------------------------------------------------
# lint_tool
# ---------------------------------------------------------------------------


def test_lint_tool_clean():
    tool = _tool("foo", {"type": "object", "properties": {}})
    assert lint_tool(tool) == []


def test_lint_tool_no_params_passes():
    """A tool with no params declared is valid."""
    tool = {"type": "function", "function": {"name": "ping"}}
    assert lint_tool(tool) == []


def test_lint_tool_propagates_errors():
    tool = _tool(
        "bad",
        {
            "type": "object",
            "properties": {"vals": {"type": "array"}},
        },
    )
    errors = lint_tool(tool)
    assert errors and any(e["path"].startswith("bad") for e in errors)


def test_lint_tool_non_dict_returns_error():
    assert lint_tool("not a dict")[0]["reason"].startswith("tool must be dict")


# ---------------------------------------------------------------------------
# validate_tools_payload
# ---------------------------------------------------------------------------


def test_validate_clean_payload_passes():
    tools = [_tool("a", {"type": "object", "properties": {}})]
    clean, quarantined = validate_tools_payload(tools, auto_quarantine=True)
    assert clean == tools
    assert quarantined == []


def test_validate_quarantines_bad_tool():
    bad = _tool("bad", {"type": "object", "properties": {"x": {"type": "array"}}})
    good = _tool("good", {"type": "object", "properties": {}})
    clean, quarantined = validate_tools_payload([bad, good], auto_quarantine=True)
    clean_names = [(t["function"]["name"]) for t in clean]
    assert clean_names == ["good"]
    assert len(quarantined) == 1 and quarantined[0]["tool"] == "bad"


def test_validate_no_auto_quarantine_keeps_bad_tool():
    bad = _tool("bad", {"type": "object", "properties": {"x": {"type": "array"}}})
    clean, quarantined = validate_tools_payload([bad], auto_quarantine=False)
    assert len(clean) == 1
    assert len(quarantined) == 1


def test_validate_empty_payload_returns_empty():
    clean, quarantined = validate_tools_payload([], auto_quarantine=True)
    assert clean == []
    assert quarantined == []


def test_validate_handles_none():
    clean, quarantined = validate_tools_payload(None, auto_quarantine=True)
    assert clean == []
    assert quarantined == []


def test_env_default_true_when_unset(monkeypatch):
    monkeypatch.delenv("MCP_STRICT_SCHEMA_AUTO_QUARANTINE", raising=False)
    bad = _tool("bad", {"type": "object", "properties": {"x": {"type": "array"}}})
    clean, quarantined = validate_tools_payload([bad])  # auto_quarantine resolved from env
    assert clean == []
    assert len(quarantined) == 1


def test_env_zero_disables_auto_quarantine(monkeypatch):
    monkeypatch.setenv("MCP_STRICT_SCHEMA_AUTO_QUARANTINE", "0")
    bad = _tool("bad", {"type": "object", "properties": {"x": {"type": "array"}}})
    clean, quarantined = validate_tools_payload([bad])
    assert len(clean) == 1
    assert len(quarantined) == 1
