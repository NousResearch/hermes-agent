"""Regression tests for invalid/None terminal command handling."""

import json

from tools.terminal_tool import _transform_sudo_command, terminal_tool


def test_transform_sudo_command_none_returns_cleanly():
    transformed, sudo_stdin = _transform_sudo_command(None)

    assert transformed is None
    assert sudo_stdin is None


def test_terminal_tool_none_command_returns_clean_error():
    result = json.loads(terminal_tool(None))  # type: ignore[arg-type]

    assert result["exit_code"] == -1
    assert result["status"] == "error"
    assert "expected string" in result["error"].lower()
    assert "nonetype" in result["error"].lower()
