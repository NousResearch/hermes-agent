"""Tests for tools.mcp_command_guard.validate_stdio_command (tasks-69t.4 C2).

Handoff from the 69t.9 audit: an MCP stdio command must be validated against
a fixed allowlist (npx/uvx/python/python3/node/docker/deno) before it is
handed to StdioServerParameters. These tests cover the allowed set, the
rejected set, symlink/realpath canonicalization, Windows suffixes, absolute
path provenance (teknium1 review on PR #62808), the per-call-site
``extra_allowed`` widening used by cua_backend.py and its own provenance
requirement via ``extra_trusted_paths`` (egilewski review on PR #62808),
the canonicalized return value callers must spawn instead of the original
argument (egilewski review), the ``is_enabled()`` fail-open warning, and the
documented ``node_modules/.bin`` compatibility tradeoff (both Enough1122
review on PR #62808).
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

    def test_unresolved_bare_command_with_relative_server_path_is_rejected(
        self, tmp_path, monkeypatch,
    ):
        """BLOCKER 2 (round-1 panel, PR #62808): a server env.PATH with a
        RELATIVE entry can leave a command bare — tools.mcp_tool
        ._resolve_stdio_command only has fallback candidates for
        npx/npm/node, so e.g. "python" stays unresolved when shutil.which
        can't find it in that PATH. A bare command previously skipped
        provenance outright, so it would reach StdioServerParameters and
        be resolved by the OS exec call against whatever the SPAWNING
        process's cwd is at THAT moment — not anything checked here.
        Reproduced end to end: no attacker file exists yet when
        _resolve_stdio_command/validate_stdio_command run (command stays
        bare, "python" not "resolved/python"), but the guard now rejects
        it anyway because the PATH it will hand to the subprocess has a
        relative entry."""
        from tools.mcp_tool import _resolve_stdio_command

        monkeypatch.chdir(tmp_path)
        # No attacker binary on disk anywhere -- which() legitimately fails.
        resolved_cmd, safe_env = _resolve_stdio_command(
            "python", {"PATH": "reldir"}
        )
        assert resolved_cmd == "python"  # stayed bare, no fallback for python

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(
                resolved_cmd, server_name="srv",
                resolved_path=safe_env.get("PATH"),
            )

    def test_bare_command_resolves_via_ambient_path_and_passes(
        self, tmp_path, monkeypatch,
    ):
        """The common, legitimate case: resolved_path finds a real binary
        that also has provenance (here, via the ambient PATH matching the
        same directory) -- passes, and returns the resolved realpath, not
        the original bare string."""
        bindir = tmp_path / "bindir"
        bindir.mkdir()
        real_python = bindir / "python"
        real_python.write_text("#!/bin/sh\n")
        real_python.chmod(0o755)

        monkeypatch.setenv("PATH", str(bindir))

        result = validate_stdio_command(
            "python", server_name="srv", resolved_path=str(bindir),
        )
        assert result == os.path.realpath(str(real_python))

    def test_bare_command_without_resolved_path_passes(self):
        """Callers that don't supply resolved_path (the pre-existing,
        still-supported call shape) see unchanged behavior."""
        validate_stdio_command("python", server_name="srv")

    def test_bare_command_with_unresolvable_path_shapes_is_rejected(
        self, tmp_path,
    ):
        """Round-3/round-6 panel history: a bare command must be rejected
        whenever it does not resolve against the given PATH -- covering
        the shapes a narrower, PATH-entry-pattern-matching version of this
        check used to special-case one at a time (a relative entry, then
        an empty one too) plus the shape that check still missed: an
        ABSOLUTE PATH entry that simply has no matching file in it yet at
        validation time (round-6 panel BLOCKER B; reproduced end to end,
        with an attacker actually populating the directory, below). None
        of these directories contain a 'python' binary, so
        shutil.which legitimately finds nothing for any of them."""
        empty_abs = tmp_path / "empty-abs"
        empty_abs.mkdir()
        shapes = [
            "reldir",                 # relative entry (round-3 BLOCKER 2)
            "",                       # whole-string empty (round-3 BLOCKER)
            f":{empty_abs}",          # leading empty entry
            f"{empty_abs}:",          # trailing empty entry
            str(empty_abs),           # absolute but untrusted+empty (BLOCKER B)
        ]
        for path_value in shapes:
            with pytest.raises(DisallowedMcpCommandError):
                validate_stdio_command(
                    "python", server_name="srv", resolved_path=path_value,
                )

    def test_absolute_untrusted_path_dir_bypass_is_closed_end_to_end(
        self, tmp_path, monkeypatch,
    ):
        """Round-6 panel BLOCKER B, reproduced end to end: an ABSOLUTE PATH
        entry is not automatically trusted just because it's absolute.
        Reproduced independently by direct execution during development
        (guard acceptance, then a real subprocess.run() actually launching
        the planted binary) before this fix -- not re-asserted here, only
        the guard's own verdict is. Neutralize every trust source, confirm
        the guard rejects the bare command while the directory is empty
        (this alone was the whole bypass under the old PATH-shape check),
        then plant the binary the way an attacker able to write into this
        directory would, and confirm the guard still rejects it once
        populated -- because resolving a bare name now always falls
        through to the same provenance check as a path-form command, and
        this directory is neither a Hermes trusted install dir nor on the
        ambient PATH."""
        monkeypatch.setenv("PATH", str(tmp_path / "empty-ambient"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-hermes-home"))

        evil_dir = tmp_path / "attacker-writable"
        evil_dir.mkdir()

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(
                "python", server_name="srv", resolved_path=str(evil_dir),
            )

        evil = evil_dir / "python"
        evil.write_text("#!/bin/sh\necho PWNED_VIA_ABSOLUTE_UNTRUSTED_PATH_DIR\n")
        evil.chmod(0o755)

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(
                "python", server_name="srv", resolved_path=str(evil_dir),
            )


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


class TestExtraAllowedPathFormProvenance:
    """Regression coverage for egilewski's review on PR #62808: an
    ``extra_allowed`` name given in PATH FORM used to be exempted from
    provenance entirely, so any absolute path whose basename matched an
    ``extra_allowed`` entry was accepted — e.g. a cua-driver manifest
    (untrusted subprocess output, not a call-site literal) naming an
    arbitrary ``/attacker/cua-driver``. ``extra_trusted_paths`` now binds a
    PATH-FORM ``extra_allowed`` command to an explicit, caller-vouched-for
    identity instead of exempting it outright."""

    def test_path_form_without_extra_trusted_paths_is_rejected(self, tmp_path):
        """The bypass egilewski reported: no extra_trusted_paths means a
        path-form extra_allowed command has no legitimate identity to
        match, so (unlike before this fix) it is rejected rather than
        silently exempted."""
        fake = tmp_path / "cua-driver"
        fake.write_text("#!/bin/sh\necho pwned\n")
        fake.chmod(0o755)

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(
                str(fake), server_name="cua-driver",
                extra_allowed=frozenset({"cua-driver"}),
            )

    def test_path_form_matching_extra_trusted_path_passes(self, tmp_path):
        """The intended, legitimate case: the manifest reports the SAME
        binary the call site already resolved and trusts."""
        real = tmp_path / "cua-driver"
        real.write_text("#!/bin/sh\n")
        real.chmod(0o755)

        result = validate_stdio_command(
            str(real), server_name="cua-driver",
            extra_allowed=frozenset({"cua-driver"}),
            extra_trusted_paths=frozenset({str(real)}),
        )
        assert result == os.path.realpath(str(real))

    def test_path_form_different_binary_same_basename_is_rejected(
        self, tmp_path,
    ):
        """The precise exploit shape: an attacker-controlled binary that
        merely shares a basename with the trusted one must still be
        rejected, even when SOME extra_trusted_paths is supplied — it must
        match, not just exist."""
        trusted = tmp_path / "trusted" / "cua-driver"
        trusted.parent.mkdir()
        trusted.write_text("#!/bin/sh\n")
        trusted.chmod(0o755)

        attacker = tmp_path / "attacker" / "cua-driver"
        attacker.parent.mkdir()
        attacker.write_text("#!/bin/sh\necho pwned\n")
        attacker.chmod(0o755)

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(
                str(attacker), server_name="cua-driver",
                extra_allowed=frozenset({"cua-driver"}),
                extra_trusted_paths=frozenset({str(trusted)}),
            )

    def test_extra_trusted_paths_matches_via_realpath(self, tmp_path):
        """A symlink to the trusted target still matches — comparison is by
        realpath, consistent with the base interpreter set's provenance
        check. The RETURNED value is the resolved real target, not the
        symlink path, so a caller spawning it can't be affected by the
        symlink being repointed after this check runs (BLOCKER 1, PR
        #62808 round-1 panel: returning the symlink path here would have
        left exactly that TOCTOU window open)."""
        real = tmp_path / "real-cua-driver"
        real.write_text("#!/bin/sh\n")
        real.chmod(0o755)
        link = tmp_path / "cua-driver"
        link.symlink_to(real)

        result = validate_stdio_command(
            str(link), server_name="cua-driver",
            extra_allowed=frozenset({"cua-driver"}),
            extra_trusted_paths=frozenset({str(real)}),
        )
        assert result == os.path.realpath(str(real))
        assert result != str(link)

    def test_bare_extra_allowed_name_unaffected(self):
        """A bare (separator-free) extra_allowed name stays exempt from
        provenance, same as before — only PATH-FORM commands are affected
        by this fix."""
        validate_stdio_command(
            "cua-driver", server_name="cua-driver",
            extra_allowed=frozenset({"cua-driver"}),
        )

    def test_bare_extra_allowed_name_with_resolved_path_rejects_unresolvable(
        self, tmp_path, monkeypatch,
    ):
        """Regression for round-6 panel BLOCKER A: this is the exact kwarg
        combination tools/computer_use/cua_backend.py's _lifecycle_coro
        passes (extra_allowed + extra_trusted_paths + resolved_path
        together) — resolved_path was silently omitted at that call site
        until this fix, leaving the bare-name PATH check inert there even
        though the parameter existed and worked at the mcp_tool.py call
        site. Confirms resolved_path is honored through this exact
        combination: a bare extra_allowed name that doesn't resolve
        against the given PATH must still be rejected, not passed through
        on the theory that extra_trusted_paths alone vouches for it."""
        monkeypatch.setenv("PATH", str(tmp_path / "empty-ambient"))
        empty_path_dir = tmp_path / "empty-path-dir"
        empty_path_dir.mkdir()

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(
                "cua-driver", server_name="cua-driver",
                extra_allowed=frozenset({"cua-driver"}),
                extra_trusted_paths=frozenset(
                    {str(tmp_path / "trusted-dir" / "cua-driver")}
                ),
                resolved_path=str(empty_path_dir),
            )

    def test_bare_extra_allowed_name_with_resolved_path_passes_when_trusted(
        self, tmp_path,
    ):
        """Same call-site kwarg combination as above, now resolving to the
        exact binary extra_trusted_paths vouches for — proves resolved_path
        is correctly threaded all the way through to a successful
        resolution + provenance match, not just to a rejection path."""
        driver_dir = tmp_path / "driverdir"
        driver_dir.mkdir()
        driver = driver_dir / "cua-driver"
        driver.write_text("#!/bin/sh\n")
        driver.chmod(0o755)

        result = validate_stdio_command(
            "cua-driver", server_name="cua-driver",
            extra_allowed=frozenset({"cua-driver"}),
            extra_trusted_paths=frozenset({str(driver)}),
            resolved_path=str(driver_dir),
        )
        assert result == os.path.realpath(str(driver))


class TestValidatedCommandIsCanonical:
    """Regression coverage for egilewski's review on PR #62808: the
    returned command must be the exact string that was provenance-checked,
    so a caller who spawns the RETURNED value (not the original argument)
    can't be tricked by a relative command resolving differently at spawn
    time than it did at validation time (e.g. a chdir() elsewhere in the
    process between validation and spawn)."""

    def test_bare_name_returned_unchanged(self):
        assert validate_stdio_command("npx", server_name="srv") == "npx"

    def test_absolute_path_returned_unchanged(self):
        result = validate_stdio_command("/usr/local/bin/npx", server_name="srv")
        assert result == os.path.realpath("/usr/local/bin/npx")

    def test_relative_path_form_returned_as_absolute(self, tmp_path, monkeypatch):
        """A relative command that passes provenance is returned as the
        ABSOLUTE form it was actually checked against, not the original
        relative string — so a caller using the return value can no longer
        be affected by a later cwd change."""
        hermes_home = tmp_path / "hermes-home"
        node_bin = hermes_home / "node" / "bin"
        node_bin.mkdir(parents=True)
        real_npx = node_bin / "npx"
        real_npx.write_text("#!/bin/sh\n")
        real_npx.chmod(0o755)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.chdir(hermes_home / "node")

        result = validate_stdio_command("bin/npx", server_name="srv")
        assert result == os.path.realpath(str(real_npx))
        assert os.path.isabs(result)


class TestNodeModulesBinCompat:
    """Documents (Enough1122 review on PR #62808) that a project-local,
    pinned toolchain launcher under node_modules/.bin is a legitimate npx
    install but is rejected by the fixed trusted-dir provenance policy when
    referenced by its own absolute path — a known compatibility tradeoff,
    not a bug. See the config_defaults.py help text for
    security.mcp_stdio_command_allowlist_enabled."""

    def test_node_modules_bin_npx_is_rejected(self, tmp_path, monkeypatch):
        project_npx = tmp_path / "myproject" / "node_modules" / ".bin" / "npx"
        project_npx.parent.mkdir(parents=True)
        project_npx.write_text("#!/bin/sh\n")
        project_npx.chmod(0o755)

        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

        with pytest.raises(DisallowedMcpCommandError):
            validate_stdio_command(str(project_npx), server_name="srv")


class TestIsEnabled:
    """The allowlist is opt-in (default off) — see the module docstring for
    why: some operators run MCP servers launched via a custom binary or
    wrapper script outside the fixed interpreter set, and this check would
    otherwise refuse to start those servers on upgrade."""

    def test_defaults_to_disabled_when_config_unreadable(self):
        with patch("hermes_cli.config.load_config", side_effect=Exception("boom")):
            assert is_enabled() is False

    def test_warns_when_config_unreadable(self, caplog):
        """Enough1122 review on PR #62808: is_enabled() fails open (stays
        disabled) on a config read error, which is correct — but must not
        do so silently, or an operator who believes the allowlist is on has
        no way to notice a misconfiguration turned it off."""
        import logging
        with caplog.at_level(logging.WARNING, logger="tools.mcp_command_guard"):
            with patch(
                "hermes_cli.config.load_config",
                side_effect=RuntimeError("bad config.yaml"),
            ):
                assert is_enabled() is False
        assert any(
            "mcp_stdio_command_allowlist_enabled" in r.message for r in caplog.records
        )

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
