"""Regression tests for invalid/None terminal command handling."""

import json
from unittest.mock import patch

from tools.terminal_tool import (
    _handle_terminal,
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


# ---------------------------------------------------------------------------
# _handle_terminal argument normalization + missing-command validation
#
# The production failure path is args={} (empty/absent tool arguments
# normalized upstream to "{}"), which must produce a structured, actionable
# error WITHOUT ever calling terminal_tool with command=None.
# ---------------------------------------------------------------------------


def _assert_missing_command_error(raw_result):
    result = json.loads(raw_result)
    assert result["status"] == "error"
    assert result["exit_code"] == -1
    assert "command" in result["error"].lower()
    return result


def test_handle_terminal_empty_dict_returns_structured_error():
    """args={} (the reported production path) → structured error, no dispatch."""
    with patch("tools.terminal_tool.terminal_tool") as mock_tt:
        _assert_missing_command_error(_handle_terminal({}))
    mock_tt.assert_not_called()


def test_handle_terminal_command_none_returns_structured_error():
    with patch("tools.terminal_tool.terminal_tool") as mock_tt:
        _assert_missing_command_error(_handle_terminal({"command": None}))
    mock_tt.assert_not_called()


def test_handle_terminal_command_empty_string_returns_structured_error():
    with patch("tools.terminal_tool.terminal_tool") as mock_tt:
        _assert_missing_command_error(_handle_terminal({"command": ""}))
        _assert_missing_command_error(_handle_terminal({"command": "   "}))
    mock_tt.assert_not_called()


def test_handle_terminal_none_args_returns_structured_error():
    """Defensive: args=None normalizes to {} → missing-command error."""
    with patch("tools.terminal_tool.terminal_tool") as mock_tt:
        _assert_missing_command_error(_handle_terminal(None))
    mock_tt.assert_not_called()


def test_handle_terminal_invalid_type_args_returns_structured_error():
    with patch("tools.terminal_tool.terminal_tool") as mock_tt:
        result = json.loads(_handle_terminal(12345))  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "expected dict or string" in result["error"].lower()
    mock_tt.assert_not_called()


def test_handle_terminal_valid_dict_command_dispatches():
    with patch("tools.terminal_tool.terminal_tool", return_value='{"ok":true}') as mock_tt:
        _handle_terminal({"command": "echo ok"})
    _, kwargs = mock_tt.call_args
    assert kwargs["command"] == "echo ok"


def test_handle_terminal_bare_string_command_dispatches():
    """args is a bare string → wrapped into {"command": str} and dispatched."""
    with patch("tools.terminal_tool.terminal_tool", return_value='{"ok":true}') as mock_tt:
        _handle_terminal("echo ok")
    _, kwargs = mock_tt.call_args
    assert kwargs["command"] == "echo ok"


def test_handle_terminal_task_id_only_from_dispatch_kwargs():
    """task_id comes from the trusted dispatch kwarg, NOT from model args."""
    with patch("tools.terminal_tool.terminal_tool", return_value='{"ok":true}') as mock_tt:
        _handle_terminal({"command": "echo ok"}, task_id="trusted-task")
    _, kwargs = mock_tt.call_args
    assert kwargs["task_id"] == "trusted-task"


def test_handle_terminal_ignores_model_supplied_task_id():
    """A model-supplied task_id in args must NOT influence isolation."""
    with patch("tools.terminal_tool.terminal_tool", return_value='{"ok":true}') as mock_tt:
        _handle_terminal({"command": "echo ok", "task_id": "evil"})
    _, kwargs = mock_tt.call_args
    assert kwargs["task_id"] is None

    with patch("tools.terminal_tool.terminal_tool", return_value='{"ok":true}') as mock_tt:
        _handle_terminal(
            {"command": "echo ok", "task_id": "evil"}, task_id="trusted-task"
        )
    _, kwargs = mock_tt.call_args
    assert kwargs["task_id"] == "trusted-task"
