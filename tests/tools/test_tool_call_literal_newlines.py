"""Literal newlines in deferred tool argument values survive both bridge parse sites.

Regression cases supplied by labyrinth-analytics for #119640 and #122730.
"""

import json

import pytest

from tools.tool_search_validation import normalize_tool_call_entries


@pytest.mark.parametrize("parse_site", ["outer_envelope", "entry_arguments"])
def test_normalize_preserves_literal_newlines_in_string_values(parse_site):
    name = "mcp_example_add_doc"
    arguments = {"title": "Notes", "content": "line one\nline two\nline three"}
    if parse_site == "outer_envelope":
        raw = json.dumps([{"name": name, "arguments": arguments}]).replace("\\n", "\n")
        payload = {"calls": raw}
    else:
        raw = json.dumps(arguments).replace("\\n", "\n")
        payload = {"calls": [{"name": name, "arguments": raw}]}

    # Unlike whitespace between tokens, these newlines make strict JSON invalid.
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)

    entries, error = normalize_tool_call_entries(payload)

    assert error is None, error
    assert entries == [{"name": name, "arguments": arguments}]
