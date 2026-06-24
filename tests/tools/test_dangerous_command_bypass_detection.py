"""Regression tests for shell-encoded dangerous-command bypasses."""

import pytest

import tools.approval as approval_mod
from tools.approval import (
    check_all_command_guards,
    check_dangerous_command,
    detect_dangerous_command,
    detect_hardline_command,
    disable_session_yolo,
    reset_current_session_key,
    set_current_session_key,
)


@pytest.fixture
def clean_session(monkeypatch):
    """Reset approval state around each integration test."""
    monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    token = set_current_session_key("bypass_detection_test")
    try:
        disable_session_yolo("bypass_detection_test")
        yield
    finally:
        disable_session_yolo("bypass_detection_test")
        reset_current_session_key(token)


@pytest.mark.parametrize(
    "command",
    [
        r"r\m -rf /home/victim",
        r"c\hmod 777 /",
        r"ch\own -R root /home",
        r"mk\fs /dev/sda",
        "r''m -rf /home/victim",
        "c''hmod 777 /",
        "ch''own -R root /",
        "$(echo rm) -rf /home/victim",
        "$(echo chmod) 777 /",
        "$(echo chown) -R root /",
        "`echo rm` -rf /",
        "`echo chmod` 777 /etc/passwd",
        "${0/x/r}m -rf /home/victim",
        "${VAR//x/r}m -rf /",
        "${1}hmod 777 /",
        "${1}own -R root /",
    ],
)
def test_shell_encoded_dangerous_commands_are_detected(command):
    is_dangerous, pattern_key, description = detect_dangerous_command(command)

    assert is_dangerous is True
    assert pattern_key
    assert description


@pytest.mark.parametrize(
    "command",
    [
        "cd ${WORKSPACE}backend",
        "cp ${CONFIG}backup.yaml ./backup.yaml",
        "echo ${VERSION}beta",
        "echo 'hello world'",
        'echo "test string"',
        "grep 'search term' file.txt",
    ],
)
def test_common_shell_usage_is_not_flagged(command):
    is_dangerous, pattern_key, description = detect_dangerous_command(command)

    assert is_dangerous is False
    assert pattern_key is None
    assert description is None


@pytest.mark.parametrize(
    "command",
    [
        r"r\m -rf /",
        "r''m -rf /",
        "$(echo rm) -rf /",
        "`echo rm` -rf /",
        "${0/x/r}m -rf /",
    ],
)
def test_shell_encoded_root_delete_is_hardline(command):
    is_hardline, description = detect_hardline_command(command)

    assert is_hardline is True
    assert description


@pytest.mark.parametrize(
    "command",
    [
        "$(echo rm) -rf /",
        "`echo rm` -rf /",
        "${0/x/r}m -rf /",
    ],
)
def test_yolo_cannot_bypass_shell_constructed_hardline(
    clean_session,
    monkeypatch,
    command,
):
    monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", True)

    dangerous_result = check_dangerous_command(command, "local")
    assert dangerous_result["approved"] is False
    assert dangerous_result.get("hardline") is True

    all_guards_result = check_all_command_guards(command, "local")
    assert all_guards_result["approved"] is False
    assert all_guards_result.get("hardline") is True
