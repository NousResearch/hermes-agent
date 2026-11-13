"""Regression tests for invalid/None terminal command handling."""

import json

from tools.terminal_tool import terminal_tool




def test_terminal_tool_none_command_returns_clean_error():
    result = json.loads(terminal_tool(None))  # type: ignore[arg-type]

    assert result["exit_code"] == -1
    assert result["status"] == "error"
    assert result["error"]
