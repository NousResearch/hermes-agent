"""Regression tests for invalid/None terminal command handling."""

import json
from unittest.mock import patch

from tools.terminal_tool import (
    _macos_chrome_keychain_guard,
    _transform_sudo_command,
    terminal_tool,
)


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


def test_terminal_tool_blocks_macos_chrome_without_mock_keychain_flags():
    command = (
        "'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' "
        "--headless=new --screenshot=out.png http://127.0.0.1:8785/"
    )

    with patch("tools.terminal_tool.platform.system", return_value="Darwin"):
        result = json.loads(terminal_tool(command))

    assert result["exit_code"] == -1
    assert result["status"] == "error"
    assert "blocked macos chrome launch" in result["error"].lower()
    assert "--use-mock-keychain" in result["error"]


def test_terminal_tool_allows_macos_chrome_with_mock_keychain_flags():
    command = (
        "'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' "
        "--headless=new --password-store=basic --use-mock-keychain --version"
    )

    with patch("tools.terminal_tool.platform.system", return_value="Darwin"):
        assert _macos_chrome_keychain_guard(command) is None
