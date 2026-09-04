"""Regression: boolean JSON Schema property schemas must not park MCP servers.

See #101669 — mcp 2.0.0's 2025-11-25 InputSchema typed every ``properties``
value as an object, so ``"properties": {"refresh": true}`` (legal JSON Schema
2020-12) failed ``ListToolsResult`` validation and Hermes disabled the whole
server. The same typing applied to OutputSchema.properties; both surfaces are
patched and covered here.

On the post-#102117 tree the compat patch lives in ``tools.mcp_tool_schema``;
the facade ``_ensure_mcp_sdk`` still invokes it on first SDK import.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")
pytest.importorskip("mcp_types")


def test_list_tools_result_accepts_boolean_property_schema():
    from tools.mcp_tool import _ensure_mcp_sdk
    from tools.mcp_tool_schema import _patch_mcp_boolean_property_schemas

    assert _ensure_mcp_sdk()
    # Idempotent if _ensure already patched; also covers calling the helper alone.
    _patch_mcp_boolean_property_schemas()

    import mcp_types.methods as mcp_methods

    raw = {
        "tools": [
            {
                "name": "example_tool",
                "description": "has a boolean property schema",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "refresh": True,
                        "q": {"type": "string"},
                    },
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "ok": True,
                        "blocked": False,
                        "detail": {"type": "string"},
                    },
                },
            },
            {
                "name": "other_tool",
                "description": "unaffected sibling",
                "inputSchema": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                },
            },
            {
                "name": "deny_all_prop",
                "inputSchema": {
                    "type": "object",
                    "properties": {"never": False},
                },
            },
        ]
    }

    # Negotiated pre-2026 sessions validate tools/list against the 2025-11-25
    # surface — this is the path that used to raise dict_type and park the server.
    mcp_methods.validate_server_result("tools/list", "2025-11-25", raw)

    from mcp.types import ListToolsResult

    result = ListToolsResult.model_validate(raw)
    assert len(result.tools) == 3
    assert result.tools[0].input_schema["properties"]["refresh"] is True
    assert result.tools[0].output_schema["properties"]["ok"] is True
    assert result.tools[0].output_schema["properties"]["blocked"] is False
    assert result.tools[0].output_schema["properties"]["detail"]["type"] == "string"
    assert result.tools[2].input_schema["properties"]["never"] is False

    # Wire alias round-trip keeps boolean outputSchema properties.
    dumped = result.model_dump(by_alias=True, mode="json")
    out_props = dumped["tools"][0]["outputSchema"]["properties"]
    assert out_props["ok"] is True
    assert out_props["blocked"] is False
    mcp_methods.validate_server_result("tools/list", "2025-11-25", dumped)


def test_normalize_mcp_input_schema_preserves_boolean_properties():
    from tools.mcp_tool_schema import _normalize_mcp_input_schema

    out = _normalize_mcp_input_schema(
        {
            "type": "object",
            "properties": {
                "refresh": True,
                "q": {"type": "string"},
            },
        }
    )
    assert out["properties"]["refresh"] is True
    assert out["properties"]["q"]["type"] == "string"


def test_boolean_property_schema_compat_patch_is_idempotent():
    from tools.mcp_tool import _ensure_mcp_sdk
    from tools.mcp_tool_schema import _patch_mcp_boolean_property_schemas

    assert _ensure_mcp_sdk()
    _patch_mcp_boolean_property_schemas()
    _patch_mcp_boolean_property_schemas()

    import mcp_types.methods as mcp_methods

    raw = {
        "tools": [
            {
                "name": "t",
                "inputSchema": {
                    "type": "object",
                    "properties": {"refresh": True},
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "ok": True,
                        "blocked": False,
                    },
                },
            }
        ]
    }
    mcp_methods.validate_server_result("tools/list", "2025-11-25", raw)

    from mcp.types import ListToolsResult

    result = ListToolsResult.model_validate(raw)
    assert result.tools[0].input_schema["properties"]["refresh"] is True
    assert result.tools[0].output_schema["properties"]["ok"] is True
    assert result.tools[0].output_schema["properties"]["blocked"] is False
