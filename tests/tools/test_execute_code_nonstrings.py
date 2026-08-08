"""execute_code must reject non-string code without AttributeError."""

from __future__ import annotations

import json

from tools.code_execution_tool import execute_code


def test_non_string_code_returns_error_not_attribute_error():
    for bad in (42, ["print(1)"], {"code": "x"}, None, ""):
        result = json.loads(execute_code(code=bad))
        assert "error" in result, bad
        assert "No code provided" in result["error"]


def test_blank_code_returns_error():
    result = json.loads(execute_code(code="   "))
    assert "error" in result
    assert "No code provided" in result["error"]
