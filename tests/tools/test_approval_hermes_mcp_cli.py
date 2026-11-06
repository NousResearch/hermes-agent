"""MCP installation and removal use the terminal approval boundary."""

import pytest

from tools.approval_detection import detect_dangerous_command


@pytest.mark.parametrize("command", [
    "hermes mcp add foo --yes --command npx",
    "hermes -p work mcp install linear",
    "hermes mcp remove foo",
    "hermes mcp rm foo",
    'hermes "mcp" add foo --yes --url https://example.com/mcp',
    "hermes mcp 'install' linear",
    'hermes mcp "remove" foo',
    # -p/--profile works anywhere before `--`, argparse's `--` may sit between the path words, and quoted
    # flag words and values reach argparse unquoted.
    "hermes mcp -p work add foo --yes --command npx", "hermes mcp --profile=work install linear",
    "hermes mcp -- remove foo", 'hermes -p " work " mcp rm foo', 'hermes mcp "-p" work install linear',
    'hermes mcp "--profile"=work add foo --yes --command npx',
])
def test_mcp_mutations_require_approval(command):
    description = "add/remove an MCP server (changes the tools and programs every session runs)"
    assert detect_dangerous_command(command) == (True, description, description)


@pytest.mark.parametrize("command", [
    "hermes mcp list",
    "hermes mcp test foo",
    "hermes mcp configure foo",
    "hermes mcp --profile=work list add", "hermes mcp -- list", "hermes mcp add-ons",
])
def test_other_mcp_commands_do_not_require_approval(command):
    assert detect_dangerous_command(command) == (False, None, None)


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
    for command in ("hermes mcp add foo --yes --command npx; " + other, other + "; hermes mcp add foo --yes --command npx"):
        assert detect_dangerous_command(command)[:2] == (True, key), command
