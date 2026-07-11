"""Tests for tools.mcp_command_guard.validate_stdio_command (tasks-69t.4 C2).

Handoff from the 69t.9 audit: an MCP stdio command must be validated against
a fixed allowlist (npx/uvx/python/python3/node/docker/deno) before it is
handed to StdioServerParameters. These tests cover the allowed set, the
rejected set, symlink/realpath canonicalization, Windows suffixes, and the
per-call-site ``extra_allowed`` widening used by cua_backend.py.
"""

import os
from unittest.mock import patch

import pytest

from tools.mcp_command_guard import (
    ALLOWED_STDIO_COMMANDS,
    DisallowedMcpCommandError,
    is_enabled,
    validate_stdio_command,
)


class TestAllowedCommands:
    @pytest.mark.parametrize("cmd", sorted(ALLOWED_STDIO_COMMANDS))
    def test_bare_allowed_command_passes(self, cmd):
        validate_stdio_command(cmd, server_name="srv")

    @pytest.mark.parametrize("cmd", sorted(ALLOWED_STDIO_COMMANDS))
    def test_absolute_path_allowed_command_passes(self, cmd):
        validate_stdio_command(f"/usr/local/bin/{cmd}", server_name="srv")

    @pytest.mark.parametrize("cmd", sorted(ALLOWED_STDIO_COMMANDS))
    def test_uppercase_and_mixed_case_passes(self, cmd):
        # Basename comparison is case-insensitive.
        validate_stdio_command(cmd.upper(), server_name="srv")

    @pytest.mark.parametrize(
        "suffix", [".exe", ".cmd", ".bat", ".ps1", ".EXE"],
    )
    def test_windows_suffix_stripped(self, suffix):
        validate_stdio_command(f"C:\\tools\\node{suffix}", server_name="srv")


class TestRejectedCommands:
    @pytest.mark.parametrize(
        "cmd",
        [
            "bash", "sh", "zsh", "curl", "wget", "rm", "eval",
            "/bin/bash", "/usr/bin/env", "perl", "ruby", "osascript",
            "powershell", "cmd", "cmd.exe",
        ],
    )
    def test_disallowed_command_rejected(self, cmd):
        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(cmd, server_name="srv")

    def test_empty_command_rejected(self):
        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command("", server_name="srv")

    def test_whitespace_only_command_rejected(self):
        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command("   ", server_name="srv")

    def test_none_command_rejected(self):
        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(None, server_name="srv")  # type: ignore[arg-type]

    def test_error_message_includes_server_name(self):
        with pytest.raises(DisallowedMcpCommandError, match="evil-srv"):
            validate_stdio_command("bash", server_name="evil-srv")

    def test_logs_security_error(self, caplog):
        import logging
        with caplog.at_level(logging.ERROR, logger="tools.mcp_command_guard"):
            with pytest.raises(DisallowedMcpCommandError):
                validate_stdio_command("bash", server_name="srv")
        assert any("SECURITY" in r.message for r in caplog.records)


class TestNameBasedCheck:
    """The allowlist check is name-based (basename of the given path), not a
    realpath/provenance check — see the module docstring for why: real npx
    installs are routinely symlinks/shims to a differently-named target
    (e.g. Homebrew's npx -> npm/bin/npx-cli.js), so resolving through
    os.path.realpath before the check rejects legitimate installs."""

    def test_symlink_literally_named_allowed_command_passes(self, tmp_path):
        """A path literally named 'npx' passes on its name, regardless of
        what it resolves to — matching real-world npx shim behavior."""
        real_bad = tmp_path / "definitely-not-npx"
        real_bad.write_text("#!/bin/sh\necho pwned\n")
        real_bad.chmod(0o755)
        fake_npx = tmp_path / "npx"
        fake_npx.symlink_to(real_bad)

        validate_stdio_command(str(fake_npx), server_name="srv")

    def test_differently_named_target_is_rejected_by_name(self, tmp_path):
        """A path NOT named after an allowlisted command is rejected even
        when it points at a real, otherwise-legitimate script."""
        real_python = tmp_path / "python3"
        real_python.write_text("#!/bin/sh\n")
        real_python.chmod(0o755)
        link = tmp_path / "my-custom-launcher"
        link.symlink_to(real_python)

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(str(link), server_name="srv")

    def test_dotdot_traversal_basename_still_checked(self, tmp_path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        traversal_path = str(sub / ".." / ".." / "bash")

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(traversal_path, server_name="srv")

    def test_dotdot_traversal_to_allowed_name_passes_by_name(self, tmp_path):
        """Documents the current name-only scope: a traversal path ending
        in an allowlisted basename passes this check. Directory-containment
        (jail) enforcement is a separate concern — see
        docs/security/mcp-stdio-sandbox.md — this module only restricts
        WHICH INTERPRETER may be spawned, not WHERE it may read/write."""
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        traversal_path = str(sub / ".." / ".." / "npx")

        validate_stdio_command(traversal_path, server_name="srv")


class TestExtraAllowed:
    def test_extra_allowed_widens_for_specific_call_site(self):
        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command("cua-driver", server_name="cua-driver")
        validate_stdio_command(
            "cua-driver", server_name="cua-driver",
            extra_allowed=frozenset({"cua-driver"}),
        )

    def test_extra_allowed_does_not_widen_globally(self):
        """Passing extra_allowed at one call site must not mutate the
        shared ALLOWED_STDIO_COMMANDS set used elsewhere."""
        validate_stdio_command(
            "cua-driver", extra_allowed=frozenset({"cua-driver"}),
        )
        assert "cua-driver" not in ALLOWED_STDIO_COMMANDS
        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command("cua-driver")


class TestIsEnabled:
    """The allowlist is opt-in (default off) — see the module docstring for
    why: some operators run MCP servers launched via a custom binary or
    wrapper script outside the fixed interpreter set, and this check would
    otherwise refuse to start those servers on upgrade."""

    def test_defaults_to_disabled_when_config_unreadable(self):
        with patch("hermes_cli.config.load_config", side_effect=Exception("boom")):
            assert is_enabled() is False

    def test_defaults_to_disabled_with_empty_config(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert is_enabled() is False

    def test_enabled_via_config(self):
        cfg = {"security": {"mcp_stdio_command_allowlist_enabled": True}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert is_enabled() is True

    def test_disabled_via_config_explicitly(self):
        cfg = {"security": {"mcp_stdio_command_allowlist_enabled": False}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert is_enabled() is False

    def test_env_var_overrides_config_on(self, monkeypatch):
        monkeypatch.setenv("HERMES_MCP_STDIO_COMMAND_ALLOWLIST_ENABLED", "true")
        cfg = {"security": {"mcp_stdio_command_allowlist_enabled": False}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert is_enabled() is True

    def test_env_var_overrides_config_off(self, monkeypatch):
        monkeypatch.setenv("HERMES_MCP_STDIO_COMMAND_ALLOWLIST_ENABLED", "false")
        cfg = {"security": {"mcp_stdio_command_allowlist_enabled": True}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert is_enabled() is False
