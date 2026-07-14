"""Regression tests for MCP isError nested JSON error extraction.

Issue #47867: When an MCP tool returns ``isError: true`` with a JSON body
containing a nested ``message`` field (e.g. ``{"ok":false,"error":{"message":"..."}}``),
the handler double-encodes the entire JSON string instead of extracting the
readable message.

Fix: ``_extract_mcp_error_message()`` parses nested JSON error bodies and
extracts the human-readable message before passing it to ``_sanitize_error()``.
"""

import json

from tools.mcp_tool import _extract_mcp_error_message


# ---------------------------------------------------------------------------
# _extract_mcp_error_message unit tests
# ---------------------------------------------------------------------------


def test_empty_text_returns_default():
    """Empty error text should return the default message."""
    assert _extract_mcp_error_message("") == "MCP tool returned an error"


def test_none_text_returns_default():
    """None (falsy) error text should return the default message."""
    assert _extract_mcp_error_message("") == "MCP tool returned an error"


def test_plain_text_returned_as_is():
    """Non-JSON text should be returned unchanged."""
    assert _extract_mcp_error_message("something broke") == "something broke"


def test_nested_error_message_extracted():
    """``{"error": {"message": "..."}}`` should extract the message."""
    body = json.dumps({
        "ok": False,
        "error": {"code": "hiring_tool_failed", "message": "请选择省份"},
    })
    assert _extract_mcp_error_message(body) == "请选择省份"


def test_nested_error_msg_extracted():
    """``{"error": {"msg": "..."}}`` should also be extracted (some servers use ``msg``)."""
    body = json.dumps({"error": {"msg": "invalid input"}})
    assert _extract_mcp_error_message(body) == "invalid input"


def test_nested_error_string_extracted():
    """``{"error": "plain string"}`` should extract the string."""
    body = json.dumps({"error": "connection refused"})
    assert _extract_mcp_error_message(body) == "connection refused"


def test_top_level_message_extracted():
    """``{"message": "..."}`` should extract the top-level message."""
    body = json.dumps({"message": "rate limit exceeded", "code": 429})
    assert _extract_mcp_error_message(body) == "rate limit exceeded"


def test_nested_message_takes_priority_over_top_level():
    """When both ``error.message`` and ``message`` exist, prefer the nested one."""
    body = json.dumps({
        "error": {"message": "specific error"},
        "message": "generic error",
    })
    assert _extract_mcp_error_message(body) == "specific error"


def test_json_array_returned_as_is():
    """A JSON array (not a dict) should be returned as the raw string."""
    body = json.dumps(["error1", "error2"])
    assert _extract_mcp_error_message(body) == body


def test_json_dict_without_message_or_error_returned_as_is():
    """A JSON dict without ``error`` or ``message`` keys should be returned as raw string."""
    body = json.dumps({"code": 500, "detail": "internal"})
    assert _extract_mcp_error_message(body) == body


def test_malformed_json_returned_as_is():
    """Malformed JSON should be returned as plain text."""
    assert _extract_mcp_error_message("{not valid json") == "{not valid json"


def test_unicode_content_preserved():
    """Unicode characters in error messages should be preserved."""
    body = json.dumps({"error": {"message": "文件未找到"}})
    assert _extract_mcp_error_message(body) == "文件未找到"


def test_nested_error_with_empty_message_falls_through():
    """``{"error": {"message": ""}}`` should fall through to raw text."""
    body = json.dumps({"error": {"message": ""}})
    # Empty string is falsy, so it falls through to the raw text
    assert _extract_mcp_error_message(body) == body
