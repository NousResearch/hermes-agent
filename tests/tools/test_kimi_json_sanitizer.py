from tools.kimi_json_sanitizer import repair_tool_call_arguments, sanitize_kimi_json


def test_repairs_missing_closing_brace():
    repaired, error = repair_tool_call_arguments('{"path": "README.md"')

    assert error is None
    assert repaired == {"path": "README.md"}


def test_repairs_truncated_trailing_member():
    repaired, error = repair_tool_call_arguments('{"command":"pwd","timeout"', tool_name="terminal")

    assert error is None
    assert repaired == {"command": "pwd"}


def test_regex_fallback_recovers_partial_pairs():
    repaired, error = repair_tool_call_arguments('{"command":"pwd","timeout":180,"background":tru')

    assert error is None
    assert repaired == {"command": "pwd", "timeout": 180}


def test_unrepairable_payload_returns_error():
    repaired, error = sanitize_kimi_json('{"command":')

    assert repaired is None
    assert error
