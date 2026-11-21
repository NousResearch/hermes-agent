"""``type`` unions that allow "string" in ``tools/arg_coercion.py`` (regression for #49).

A string the fragment already accepts must reach the tool unchanged. A union member other than
"string" is tried only when the fragment's string-applicable constraints (enum, const, pattern,
minLength, maxLength) reject the string, as with ``"1"`` against ``enum: [1, 2]``.
"""

from unittest.mock import patch

import pytest

from tools.arg_coercion import coerce_tool_args


def _coerce(prop_schema: dict, value: str):
    schema = {"name": "test_tool", "description": "test",
              "parameters": {"type": "object", "properties": {"code": prop_schema}}}
    with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
        return coerce_tool_args("test_tool", {"code": value})["code"]


@pytest.mark.parametrize("value", ["00123", "1.10", "false", "42"])
@pytest.mark.parametrize("types", [["string", "integer"], ["integer", "string"],
                                   ["string", "number", "boolean"]])
def test_string_valid_for_a_union_that_allows_string_is_kept(value, types):
    """A string already satisfies a type union that includes "string" (a zip code, a version,
    an id with leading zeros): coercion has nothing unambiguous to repair, so the model's value
    must reach the tool unchanged — not become 123 / 1.1 / False."""
    assert _coerce({"type": types}, value) == value


@pytest.mark.parametrize("prop_schema,value,expected", [
    ({"type": ["string", "integer"], "enum": [1, 2]}, "1", 1),
    ({"type": ["string", "integer"], "const": 2}, "2", 2),
    ({"type": ["integer", "string"], "pattern": "^[a-z]+$"}, "42", 42),
    ({"type": ["string", "boolean"], "maxLength": 3}, "false", False),
    ({"type": ["string", "integer"], "enum": ["00123", 7]}, "00123", "00123"),
    ({"type": ["string", "integer"], "pattern": "^[0-9]{5}$"}, "00123", "00123"),
], ids=["enum-rejects", "const-rejects", "pattern-rejects", "maxLength-rejects", "enum-accepts",
        "pattern-accepts"])
def test_string_kept_only_while_the_fragments_string_constraints_accept_it(prop_schema, value, expected):
    """The "string" member keeps a value only when the whole fragment accepts it as a string:
    a string an enum/const/pattern/length rejects still gets the other union members' repair."""
    result = _coerce(prop_schema, value)
    assert (result, type(result)) == (expected, type(expected))
