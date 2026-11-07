"""Messaging setup requires approval, including quoted words and global flags."""

import pytest

from tools.approval_detection import detect_dangerous_command


@pytest.mark.parametrize("command", [
    "hermes gateway setup telegram --quick", "hermes -p work gateway setup",
    'hermes "gateway" setup telegram --quick', "hermes gateway 'setup' telegram",
    # -p/--profile is accepted anywhere before `--`, and a quoted flag word reaches argparse unquoted.
    "hermes gateway -p work setup telegram --quick", 'hermes "-p" work gateway setup telegram --quick',
    "hermes gateway --profile=work setup",
    # A quoted value may hold spaces (the profile scanner strips them).
    'hermes gateway -p " work " setup telegram --quick', 'hermes -p " work " gateway setup',
    "hermes gateway '--profile=work' setup",
    # argparse's `--` may sit between the path words, and quotes may sit inside the profile flag's name.
    "hermes gateway -- setup telegram", 'hermes gateway -"p" work setup telegram --quick',
    'hermes gateway --"profile"=work setup', 'hermes gateway "--profile"=work setup',
])
def test_setup_requires_approval(command):
    dangerous, description, _ = detect_dangerous_command(command)
    assert dangerous is True
    assert description == "configure a messaging platform (changes who can reach the agent)"


@pytest.mark.parametrize("command", [
    "hermes gateway status", "hermes gateway list", "hermes gateway setup-help", "hermes gateway setup_extra",
    "hermes gateway -p work status",
    # After the group only the profile flag takes a value, so the next word is the verb.
    "hermes gateway --profile=work status setup", "hermes gateway -- status",
])
def test_other_gateway_commands_are_unchanged(command):
    assert detect_dangerous_command(command)[0] is False


@pytest.mark.parametrize("other, key", [
    ("git reset --hard", "git reset --hard (destroys uncommitted changes)"),
    # Found only in a later command variant (subshell, quoted command word) or by a structural check.
    ("(npm uninstall foo)", "package manager uninstall"),
    ('"git" reset --hard', "git reset --hard (destroys uncommitted changes)"),
    ("python3 -c 'import os'", "script execution via -e/-c flag"),
    ("bash -c 'echo hi'", "shell command via -c/-lc flag"),
])
def test_any_other_finding_keeps_its_key_in_a_compound_command(other, key):
    # Approval keys on the first finding only, and this rule is checked after everything else, so approving it for
    # the session never covers another finding in the same command.
    for command in ("hermes gateway setup telegram --quick; " + other, other + "; hermes gateway setup telegram --quick"):
        assert detect_dangerous_command(command)[:2] == (True, key), command
