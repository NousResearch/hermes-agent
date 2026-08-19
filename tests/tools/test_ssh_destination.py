"""Tests for OpenSSH destination formatting (CWE-88 option smuggling)."""

import pytest

from tools.environments.ssh_destination import (
    build_ssh_probe_command,
    format_ssh_destination,
)


class TestFormatSshDestination:
    def test_host_only(self):
        assert format_ssh_destination("example.com") == "example.com"

    def test_user_and_host(self):
        assert format_ssh_destination("example.com", "alice") == "alice@example.com"

    def test_blank_user_treated_as_host_only(self):
        assert format_ssh_destination("example.com", "  ") == "example.com"

    def test_rejects_empty_host(self):
        with pytest.raises(ValueError, match="empty"):
            format_ssh_destination("")

    def test_rejects_leading_dash_host(self):
        with pytest.raises(ValueError, match="must not start with"):
            format_ssh_destination("-oProxyCommand=evil")

    def test_rejects_leading_dash_user(self):
        with pytest.raises(ValueError, match="must not start with"):
            format_ssh_destination("example.com", "-oProxyCommand=evil")


class TestBuildSshProbeCommand:
    def test_includes_double_dash_and_echo(self):
        cmd = build_ssh_probe_command("example.com", "alice", port="2222", key="~/.ssh/id")
        assert cmd[0] == "ssh"
        dash = cmd.index("--")
        assert cmd[dash + 1] == "alice@example.com"
        assert cmd[dash + 2] == "echo ok"
        assert "-p" in cmd and "2222" in cmd


class TestValidateSshIdentity:
    def test_returns_destination_when_host_set(self):
        from tools.environments.ssh_destination import validate_ssh_identity

        assert validate_ssh_identity("h", "u") == "u@h"

    def test_blank_host_ok_with_safe_user(self):
        from tools.environments.ssh_destination import validate_ssh_identity

        assert validate_ssh_identity("", "alice") is None

    def test_blank_host_rejects_leading_dash_user(self):
        from tools.environments.ssh_destination import validate_ssh_identity

        with pytest.raises(ValueError, match="must not start with"):
            validate_ssh_identity("", "-oProxyCommand=evil")
