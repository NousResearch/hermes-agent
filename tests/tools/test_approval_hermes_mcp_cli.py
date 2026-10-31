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
])
def test_mcp_mutations_require_approval(command):
    description = "add/remove an MCP server (changes the tools and programs every session runs)"
    assert detect_dangerous_command(command) == (True, description, description)


@pytest.mark.parametrize("command", [
    "hermes mcp list",
    "hermes mcp test foo",
    "hermes mcp configure foo",
])
def test_other_mcp_commands_do_not_require_approval(command):
    assert detect_dangerous_command(command) == (False, None, None)
