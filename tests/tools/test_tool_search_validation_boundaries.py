"""Boundary invariants for deferred tool-call schema validation."""

import json

import pytest


@pytest.mark.parametrize(
    "keyword",
    ["$defs", "definitions", "dependencies", "dependentSchemas", "patternProperties", "properties"],
)
def test_schema_map_names_do_not_change_value_normalization(keyword):
    from tools.tool_search_validation import _schema_for_local_validation

    schema = {"type": "integer", "nullable": True}
    normalized = _schema_for_local_validation({keyword: {"nullable": schema}})

    assert normalized[keyword]["nullable"] == _schema_for_local_validation(schema)


@pytest.mark.parametrize(
    ("value", "should_dispatch"),
    [
        (7, True),
        ([], False),
    ],
)
def test_schema_keyword_can_also_be_a_parameter_name(value, should_dispatch):
    """Schema normalization must distinguish keywords from names in ``properties`` maps."""
    import model_tools
    from tools.registry import registry

    name = "mcp_probe_nullable_parameter"
    toolset = "mcp-probe-nullable-parameter"
    calls = []
    registry.register(
        name=name,
        toolset=toolset,
        schema={
            "name": name,
            "description": "Accept a parameter whose wire name matches a schema keyword.",
            "parameters": {
                "type": "object",
                "properties": {"nullable": {"type": "integer"}},
                "required": ["nullable"],
                "additionalProperties": False,
            },
        },
        handler=lambda args, **_: calls.append(args) or json.dumps({"ok": True}),
    )

    result = json.loads(model_tools.handle_function_call(
        function_name="tool_call",
        function_args={"name": name, "arguments": {"nullable": value}},
        enabled_toolsets=[toolset],
    ))

    assert bool(calls) is should_dispatch
    if should_dispatch:
        assert result["ok"] is True
        assert calls == [{"nullable": value}]
    else:
        assert result["path"] == "arguments.nullable"
        assert result["constraint"] == "type"
