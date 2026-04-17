"""Terminal + sandbox.type coordination (logging hook surface)."""

from unittest.mock import patch

from tools.terminal_tool import sandbox_type_from_cli


def test_sandbox_type_from_cli_defaults_local():
    with patch("cli.CLI_CONFIG", {}, create=True):
        assert sandbox_type_from_cli() == "local"


def test_sandbox_type_from_cli_reads_config():
    fake = {"sandbox": {"type": "docker"}}
    with patch("cli.CLI_CONFIG", fake, create=True):
        assert sandbox_type_from_cli() == "docker"
