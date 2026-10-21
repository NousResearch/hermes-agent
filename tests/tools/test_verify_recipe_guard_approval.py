import pytest
from tools.approval_detection import detect_dangerous_command

@pytest.mark.parametrize("command", [
    "hermes verify /tmp/project --phase test",
    "/home/hermes/.local/bin/hermes verify /tmp/project --phase test --json",
])
def test_hermes_verify_requires_terminal_approval(command):
    detected, _pattern_key, description = detect_dangerous_command(command)
    assert detected, f"hermes verify was not flagged: {command}"
    assert "verification recipe" in description
