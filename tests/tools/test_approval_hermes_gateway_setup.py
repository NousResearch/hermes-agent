"""Messaging setup requires approval, including quoted words and global flags."""

import pytest

from tools.approval_detection import detect_dangerous_command


@pytest.mark.parametrize("command", [
    "hermes gateway setup telegram --quick", "hermes -p work gateway setup",
    'hermes "gateway" setup telegram --quick', "hermes gateway 'setup' telegram",
    # -p/--profile is accepted anywhere before `--`, and a quoted flag word reaches argparse unquoted.
    "hermes gateway -p work setup telegram --quick", 'hermes "-p" work gateway setup telegram --quick',
    "hermes gateway --profile=work setup",
])
def test_setup_requires_approval(command):
    dangerous, description, _ = detect_dangerous_command(command)
    assert dangerous is True
    assert description == "configure a messaging platform (changes who can reach the agent)"


@pytest.mark.parametrize("command", [
    "hermes gateway status", "hermes gateway list", "hermes gateway setup-help", "hermes gateway setup_extra",
    "hermes gateway -p work status",
])
def test_other_gateway_commands_are_unchanged(command):
    assert detect_dangerous_command(command)[0] is False
