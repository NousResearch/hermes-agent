"""Model-emitted JSON strings for the ``tool_call`` envelope.

Some models stringify the ``calls`` envelope — or a single entry's ``arguments`` —
with Python literal syntax (single quotes) instead of JSON. The module already
tolerates the stringified envelope (#114484); it must also accept the literal
variant, and when it cannot parse at all it must echo the offending text so the
model can correct itself instead of looping on a bare column number.
"""

from tools.tool_search_validation import normalize_tool_call_entries


def test_calls_envelope_as_python_literal_string_is_accepted():
    entries, error = normalize_tool_call_entries({
        "calls": "[{'name': 'mcp__clawlink__clawlink_execute', "
                 "'arguments': {'integration_id': 'bitbucket', 'input': {'id': 1}}}]"})

    assert error is None
    assert entries[0]["name"] == "mcp__clawlink__clawlink_execute"
    assert entries[0]["arguments"] == {"integration_id": "bitbucket", "input": {"id": 1}}


def test_entry_arguments_as_python_literal_string_is_accepted():
    entries, error = normalize_tool_call_entries({
        "calls": [{
            "name": "mcp__clawlink__clawlink_execute",
            "arguments": "{'integration_id': 'bitbucket', 'input': {'id': 1}}",
        }]})

    assert error is None
    assert entries[0]["arguments"] == {"integration_id": "bitbucket", "input": {"id": 1}}


def test_unparsable_calls_string_echoes_the_offending_text():
    bad = '[{"name": "x", "arguments": {"a": 1} "b": 2}]'

    entries, error = normalize_tool_call_entries({"calls": bad})

    assert entries == []
    assert error is not None
    assert "not valid JSON" in error
    assert '"b": 2' in error


def test_unparsable_arguments_string_echoes_the_offending_text():
    bad = "{'integration_id': 'bitbucket' 'action_id': 'x'}"

    entries, error = normalize_tool_call_entries({
        "calls": [{"name": "mcp__clawlink__clawlink_execute", "arguments": bad}]})

    assert entries == []
    assert error is not None
    assert "not valid JSON" in error
    assert "'action_id'" in error
