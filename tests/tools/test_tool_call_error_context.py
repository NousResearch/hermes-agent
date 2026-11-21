"""Diagnostic context for malformed stringified deferred calls (issue #155976).

The carrier PR #122730 already repairs unambiguous JSON errors; unrepairable
ones must stay fail-closed but show the offending region, not a long prefix.
"""

import json

import pytest

from tools.tool_search_validation import normalize_tool_call_entries


@pytest.mark.parametrize("site", ["calls", "arguments"])
def test_unrepairable_call_reports_near_fault_without_echoing_secret(site):
    arguments = {
        "message": "x" * 350,
        "api_key": "sk-test-1234567890abcdef",
        "priority": "high",
    }
    if site == "calls":
        valid = json.dumps([{"name": "mcp_example_add_doc", "arguments": arguments}])
        wrap = lambda raw: {"calls": raw}
    else:
        valid = json.dumps(arguments)
        wrap = lambda raw: {"calls": [{"name": "mcp_example_add_doc", "arguments": raw}]}

    # A single misplaced byte well beyond the carrier's first-200-char echo.
    broken = valid.replace('", "priority"', '"!, "priority"')
    assert broken != valid
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)

    entries, error = normalize_tool_call_entries(wrap(broken))
    assert entries == []
    assert error is not None and "unrepairable" in error
    assert '"priority"' in error  # the error-local context survives long text
    assert "sk-test-1234567890abcdef" not in error
