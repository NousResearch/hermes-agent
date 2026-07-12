"""Tests for tools.mcp_command_guard.validate_stdio_command (tasks-69t.4 C2).

Handoff from the 69t.9 audit: an MCP stdio command must be validated against
a fixed allowlist (npx/uvx/python/python3/node/docker/deno) before it is
handed to StdioServerParameters. These tests cover the allowed set, the
rejected set, symlink/realpath canonicalization, Windows suffixes, absolute
path provenance (teknium1 review on PR #62808), and the per-call-site
``extra_allowed`` widening used by cua_backend.py.
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
    """The allowlist check starts with a name check (basename of the given
    path) — see the module docstring for why: real npx installs are
    routinely symlinks/shims to a differently-named target (e.g. Homebrew's
    npx -> npm/bin/npx-cli.js), so requiring the resolved TARGET file to
    itself be literally named 'npx' rejects legitimate installs. But a
    resolved ABSOLUTE path additionally needs trusted provenance (see
    TestProvenanceCheck below) — basename alone is bypassable via an
    attacker-controlled PATH (teknium1 review on PR #62808)."""

    def test_bare_name_passes_without_provenance(self):
        """A bare command (no path separator) hasn't been resolved to a
        specific file yet, so provenance doesn't apply — it either gets
        resolved before spawn (and re-checked then) or fails at spawn with
        ENOENT regardless of this guard."""
        validate_stdio_command("npx", server_name="srv")

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


class TestProvenanceCheck:
    """Regression coverage for teknium1's review on PR #62808: basename-only
    checking let an attacker-controlled MCP server ``env.PATH`` resolve a
    bare command like ``npx`` to an arbitrary binary (e.g. ``/attacker/npx``)
    before the guard ever saw it — ``tools.mcp_tool._resolve_stdio_command``
    resolves bare commands through the server's own configured PATH. A
    resolved absolute path must now additionally have trusted provenance:
    live under one of Hermes' own fixed install dirs, or resolve to the
    exact same file the AMBIENT (non-server-controlled) PATH would find."""

    def test_symlink_named_allowed_command_in_untrusted_dir_is_rejected(
        self, tmp_path,
    ):
        """The literal attack from the review: a binary literally named
        'npx', living somewhere that is neither a Hermes-trusted install
        dir nor reachable via the ambient PATH (i.e. an attacker-controlled
        directory a malicious server config's env.PATH could point at), is
        rejected even though its basename matches — including through a
        symlink, where the REAL (realpath-resolved) location is what's
        actually checked, not just the symlink's own directory."""
        attacker_dir = tmp_path / "attacker"
        attacker_dir.mkdir()
        real_bad = attacker_dir / "definitely-not-npx"
        real_bad.write_text("#!/bin/sh\necho pwned\n")
        real_bad.chmod(0o755)
        fake_npx = attacker_dir / "npx"
        fake_npx.symlink_to(real_bad)

        with patch("shutil.which", return_value=None):
            with pytest.raises(DisallowedMcpCommandError):
                validate_stdio_command(str(fake_npx), server_name="srv")

    def test_dotdot_traversal_to_allowed_name_in_untrusted_dir_is_rejected(
        self, tmp_path,
    ):
        """A traversal path ending in an allowlisted basename no longer
        passes purely on name — it resolves into an untrusted directory,
        so provenance now rejects it too."""
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        traversal_path = str(sub / ".." / ".." / "npx")

        with patch("shutil.which", return_value=None):
            with pytest.raises(DisallowedMcpCommandError):
                validate_stdio_command(traversal_path, server_name="srv")

    def test_path_under_trusted_install_dir_passes(self):
        """Hermes' own fixed install locations (mirroring the fallback
        candidates in tools.mcp_tool._resolve_stdio_command) are trusted
        without needing anything on disk or on PATH."""
        validate_stdio_command("/usr/local/bin/npx", server_name="srv")

    def test_ambient_path_resolution_grants_trust(self, tmp_path, monkeypatch):
        """A legitimate nonstandard install (e.g. asdf/nvm shim dir) is
        trusted when it's on the AMBIENT PATH — this process's own
        inherited PATH — even though it's outside the fixed trusted dirs,
        because the operator's own PATH would launch this exact binary
        regardless of any MCP server config."""
        custom_bin = tmp_path / "custombin"
        custom_bin.mkdir()
        real_npx = custom_bin / "npx"
        real_npx.write_text("#!/bin/sh\n")
        real_npx.chmod(0o755)

        monkeypatch.setenv("PATH", str(custom_bin))

        validate_stdio_command(str(real_npx), server_name="srv")

    def test_ambient_path_match_does_not_launder_a_different_file(
        self, tmp_path, monkeypatch,
    ):
        """The ambient-PATH branch requires the SAME file, not just the same
        basename somewhere on PATH — a distinct attacker binary named 'npx'
        must still be rejected even when a legitimate 'npx' also happens to
        be on the ambient PATH."""
        legit_bin = tmp_path / "legitbin"
        legit_bin.mkdir()
        (legit_bin / "npx").write_text("#!/bin/sh\n")
        (legit_bin / "npx").chmod(0o755)

        attacker_dir = tmp_path / "attacker"
        attacker_dir.mkdir()
        fake_npx = attacker_dir / "npx"
        fake_npx.write_text("#!/bin/sh\necho pwned\n")
        fake_npx.chmod(0o755)

        monkeypatch.setenv("PATH", str(legit_bin))

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(str(fake_npx), server_name="srv")

    @pytest.mark.parametrize("form", ["./npx", "subdir/npx"])
    def test_relative_path_form_attacker_named_rejected(
        self, form, tmp_path, monkeypatch,
    ):
        """A RELATIVE path form (contains a separator but isn't absolute) is
        still a specific file, resolved against CWD — it must not skip
        provenance. Keying provenance on os.path.isabs alone would let
        './npx' / 'subdir/npx' pass on basename only (grok adversarial
        review follow-up to teknium1's ask)."""
        cwd = tmp_path / "cwd"
        (cwd / "subdir").mkdir(parents=True)
        # Drop an attacker 'npx' at whichever relative location `form` names.
        target = cwd / form
        target.write_text("#!/bin/sh\necho pwned\n")
        target.chmod(0o755)

        monkeypatch.chdir(cwd)
        # No trust source: ambient PATH empty, HERMES_HOME empty.
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(form, server_name="srv")

    def test_relative_path_form_into_trusted_dir_passes(
        self, tmp_path, monkeypatch,
    ):
        """A legitimate relative form that (once made absolute against CWD)
        resolves INTO a trusted install dir still passes — guards against
        over-rejecting real relative invocations."""
        hermes_home = tmp_path / "hermes-home"
        node_bin = hermes_home / "node" / "bin"
        node_bin.mkdir(parents=True)
        real_npx = node_bin / "npx"
        real_npx.write_text("#!/bin/sh\n")
        real_npx.chmod(0o755)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        # CWD is HERMES_HOME/node, so "bin/npx" absolutizes into the trusted
        # node bindir.
        monkeypatch.chdir(hermes_home / "node")

        validate_stdio_command("bin/npx", server_name="srv")


class TestResolveThenValidateIntegration:
    """Integration across the real seam grok flagged as untested: a bare
    command is resolved through the *server's* configured PATH by
    tools.mcp_tool._resolve_stdio_command BEFORE validate_stdio_command
    runs. This exercises both together so the guard is proven to catch an
    evil server PATH, not just a pre-resolved absolute path in isolation."""

    def test_evil_server_path_resolution_is_rejected(self, tmp_path, monkeypatch):
        from tools.mcp_tool import _resolve_stdio_command

        attacker_dir = tmp_path / "attacker"
        attacker_dir.mkdir()
        evil = attacker_dir / "npx"
        evil.write_text("#!/bin/sh\necho pwned\n")
        evil.chmod(0o755)

        # Neutralize every trust source: ambient PATH has no npx, and
        # HERMES_HOME points somewhere empty so the attacker path matches
        # no trusted install dir.
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

        # The server config's own env.PATH wins the resolution — exactly the
        # attacker-controlled-PATH scenario from teknium1's review.
        resolved_cmd, _ = _resolve_stdio_command(
            "npx", {"PATH": str(attacker_dir)}
        )
        assert resolved_cmd == str(evil)

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(resolved_cmd, server_name="srv")


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

    def test_no_env_var_override(self, monkeypatch):
        """Non-secret behavioral config lives in config.yaml only per
        AGENTS.md — there is deliberately no
        HERMES_MCP_STDIO_COMMAND_ALLOWLIST_ENABLED env var (teknium1 review
        on PR #62808). Setting it must have zero effect in either
        direction; config.yaml stays authoritative."""
        monkeypatch.setenv("HERMES_MCP_STDIO_COMMAND_ALLOWLIST_ENABLED", "true")
        cfg = {"security": {"mcp_stdio_command_allowlist_enabled": False}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert is_enabled() is False

        monkeypatch.setenv("HERMES_MCP_STDIO_COMMAND_ALLOWLIST_ENABLED", "false")
        cfg = {"security": {"mcp_stdio_command_allowlist_enabled": True}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert is_enabled() is True
