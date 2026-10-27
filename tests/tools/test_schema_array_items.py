"""Preserve the missing-array-items compatibility fix from PR #24158."""
import copy

import pytest

from tools.mcp_tool_schema import _normalize_mcp_input_schema
from tools.schema_sanitizer import sanitize_tool_schemas


def _wire(schema):
    return sanitize_tool_schemas([{"function": {"name": "probe", "parameters": schema}}])[0]["function"]["parameters"]


@pytest.mark.parametrize("normalize", [_wire, _normalize_mcp_input_schema])
def test_missing_items_repaired_only_in_schema_positions(normalize):
    literal = {"type": "array"}
    schema = {"type": "object", "properties": {
        "tags": {"type": "array", "default": literal},
        "nested": {"type": "array", "items": {"type": "array"}},
        "tuple": {"type": "array", "prefixItems": [{"type": "string"}]},
        "metadata": {"const": literal, "x-metadata": literal},
    }}
    before = copy.deepcopy(schema)
    result = normalize(schema)["properties"]
    assert result["tags"]["items"] == {}
    assert result["nested"]["items"]["items"] == {}
    assert "items" not in result["tuple"]
    assert result["tuple"]["prefixItems"] == [{"type": "string"}]
    assert result["tags"]["default"] == literal
    assert result["metadata"] == schema["properties"]["metadata"]
    assert schema == before


def test_generated_array_schema_forms_receive_permissive_items():
    schema = {"type": "object", "properties": {
        "bare": "array", "nullable": {"type": ["array", "null"]},
        "mixed": {"type": ["array", "string"]},
    }}
    result = _wire(schema)["properties"]
    assert result["bare"] == {"type": "array", "items": {}}
    assert result["nullable"]["items"] == {}
    assert result["mixed"]["anyOf"][0] == {"type": "array", "items": {}}
