import json

from tools.delegate_tool import _looks_like_error_output


def test_error_json_with_appended_loop_warning_still_counts_as_error():
    payload = json.dumps({"error": "child failed"})
    combined = payload + "\n\n⚠️ Loop warning: child hit the iteration limit"

    assert _looks_like_error_output(combined) is True


def test_failed_status_json_with_appended_loop_warning_still_counts_as_error():
    payload = json.dumps({"status": "failed", "message": "child failed"})
    combined = payload + "\n\nLoop warning: repeated tool call detected"

    assert _looks_like_error_output(combined) is True
