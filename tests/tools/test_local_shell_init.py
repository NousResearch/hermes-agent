"""Tests for terminal.shell_init_files / terminal.auto_source_bashrc.

A bash ``-l -c`` invocation does NOT source ``~/.bashrc``, so tools that
register themselves there (nvm, asdf, pyenv) stay invisible to the
environment snapshot built by ``LocalEnvironment.init_session``.  These
tests verify the config-driven prelude that fixes that.
"""

import os
from unittest.mock import patch

import pytest

from tools.environments.local import (
    LocalEnvironment,
    _prepend_shell_init,
    _resolve_shell_init_files,
)


class TestResolveShellInitFiles:
    @pytest.mark.platforms("linux")
    def test_auto_sources_bashrc_when_present(self, tmp_path, monkeypatch):
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text('export MARKER=seen\n')
        monkeypatch.setenv("HOME", str(tmp_path))

        # Default config: auto_source_bashrc on, no explicit list.
        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([], True),
        ):
            resolved = _resolve_shell_init_files()

        assert resolved == [str(bashrc)]

    @pytest.mark.platforms("linux")
    def test_auto_sources_profile_when_present(self, tmp_path, monkeypatch):
        """~/.profile is where ``n`` / ``nvm`` installers typically write
        their PATH export on Debian/Ubuntu, and it has no interactivity
        guard so a non-interactive source actually runs it.
        """
        profile = tmp_path / ".profile"
        profile.write_text('export PATH="$HOME/n/bin:$PATH"\n')
        monkeypatch.setenv("HOME", str(tmp_path))

        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([], True),
        ):
            resolved = _resolve_shell_init_files()

        assert resolved == [str(profile)]


    @pytest.mark.platforms("linux")
    def test_auto_sources_profile_before_bashrc(self, tmp_path, monkeypatch):
        """Both files present: profile runs first so PATH exports in
        profile take effect even if bashrc short-circuits on the
        non-interactive ``case $- in *i*) ;; *) return;; esac`` guard.
        """
        profile = tmp_path / ".profile"
        profile.write_text('export FROM_PROFILE=1\n')
        bash_profile = tmp_path / ".bash_profile"
        bash_profile.write_text('export FROM_BASH_PROFILE=1\n')
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text('export FROM_BASHRC=1\n')
        monkeypatch.setenv("HOME", str(tmp_path))

        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([], True),
        ):
            resolved = _resolve_shell_init_files()

        assert resolved == [str(profile), str(bash_profile), str(bashrc)]

    def test_skips_bashrc_when_missing(self, tmp_path, monkeypatch):
        # No rc files written.
        monkeypatch.setenv("HOME", str(tmp_path))

        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([], True),
        ):
            resolved = _resolve_shell_init_files()

        assert resolved == []


    def test_missing_explicit_files_are_skipped_silently(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([str(tmp_path / "does-not-exist.sh")], False),
        ):
            resolved = _resolve_shell_init_files()

        assert resolved == []


class TestPrependShellInit:
    def test_empty_list_returns_command_unchanged(self):
        assert _prepend_shell_init("echo hi", []) == "echo hi"

    def test_prepends_guarded_source_lines(self):
        wrapped = _prepend_shell_init("echo hi", ["/tmp/a.sh", "/tmp/b.sh"])
        assert "echo hi" in wrapped
        # Each file is sourced through a guarded [ -r … ] && . '…' || true
        # pattern so a missing/broken rc can't abort the bootstrap.
        assert "/tmp/a.sh" in wrapped
        assert "/tmp/b.sh" in wrapped
        assert "|| true" in wrapped
        assert "set +e" in wrapped

    def test_escapes_single_quotes(self):
        wrapped = _prepend_shell_init("echo hi", ["/tmp/o'malley.sh"])
        # The path must survive as the shell receives it; embedded single
        # quote is escaped as '\'' rather than breaking the outer quoting.
        assert "o'\\''malley" in wrapped


@pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.path.isfile("/bin/bash"),
    reason="Requires bash; CI sandbox may strip it.",
)
class TestSnapshotEndToEnd:
    """Spin up a real LocalEnvironment and confirm the snapshot sources
    extra init files."""

    def test_exported_env_changes_persist_between_commands(self, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
        try:
            first = env.execute(
                'export HERMES_STICKY_ENV_PROBE="sticky"; '
                'export PATH="/tmp/hermes-session-bin:$PATH"; '
                'echo "first=$HERMES_STICKY_ENV_PROBE"'
            )
            second = env.execute(
                'echo "second=$HERMES_STICKY_ENV_PROBE"; echo "PATH=$PATH"'
            )
        finally:
            env.cleanup()

        assert first["returncode"] == 0
        assert second["returncode"] == 0
        assert "first=sticky" in first.get("output", "")
        output = second.get("output", "")
        assert "second=sticky" in output
        assert "/tmp/hermes-session-bin" in output


    def test_snapshot_picks_up_init_file_exports(self, tmp_path, monkeypatch):
        init_file = tmp_path / "custom-init.sh"
        init_file.write_text(
            'export HERMES_SHELL_INIT_PROBE="probe-ok"\n'
            'export PATH="/opt/shell-init-probe/bin:$PATH"\n'
        )

        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([str(init_file)], False),
        ):
            env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
            try:
                result = env.execute(
                    'echo "PROBE=$HERMES_SHELL_INIT_PROBE"; echo "PATH=$PATH"'
                )
            finally:
                env.cleanup()

        output = result.get("output", "")
        assert "PROBE=probe-ok" in output
        assert "/opt/shell-init-probe/bin" in output

    def test_profile_path_export_survives_bashrc_interactive_guard(
        self, tmp_path, monkeypatch
    ):
        """Reproduces the Debian/Ubuntu + ``n``/``nvm`` case.

        Setup:
          - ``~/.bashrc`` starts with ``case $- in *i*) ;; *) return;; esac``
            (the default on Debian/Ubuntu) and would happily export a PATH
            entry below that guard — but never gets there because a
            non-interactive source short-circuits.
          - ``~/.profile`` exports ``$HOME/fake-n/bin`` onto PATH, no guard.

        Expectation: auto-sourced rc list picks up ``~/.profile`` before
        ``~/.bashrc``, so the snapshot ends up with ``fake-n/bin`` on PATH
        even though the bashrc export is silently skipped.
        """
        fake_n_bin = tmp_path / "fake-n" / "bin"
        fake_n_bin.mkdir(parents=True)

        profile = tmp_path / ".profile"
        profile.write_text(
            f'export PATH="{fake_n_bin}:$PATH"\n'
            'export FROM_PROFILE=profile-ok\n'
        )
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text(
            'case $- in\n'
            '    *i*) ;;\n'
            '      *) return;;\n'
            'esac\n'
            'export FROM_BASHRC=bashrc-should-not-appear\n'
        )

        monkeypatch.setenv("HOME", str(tmp_path))

        with patch(
            "tools.environments.local._read_terminal_shell_init_config",
            return_value=([], True),
        ):
            env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
            try:
                result = env.execute(
                    'echo "PATH=$PATH"; '
                    'echo "FROM_PROFILE=$FROM_PROFILE"; '
                    'echo "FROM_BASHRC=$FROM_BASHRC"'
                )
            finally:
                env.cleanup()

        output = result.get("output", "")
        assert "FROM_PROFILE=profile-ok" in output
        assert str(fake_n_bin) in output
        # bashrc short-circuited on the interactive guard — its export never ran
        assert "FROM_BASHRC=bashrc-should-not-appear" not in output


class TestSnapshotExcludesLaunchVaultNamesNotInScope:
    """Reproduce the rc-file credential leak the credential-hygiene gate raised.

    A terminal whose login shell sources ``~/.zshrc`` must NOT carry launch-vault
    names re-exported from ``~/.hermes/.env`` when the routed profile's scope
    doesn't supply them. The Python-side ``build_subprocess_env`` scope scrub
    cannot see those re-exports (they happen inside the bash login subprocess
    reading the file on disk). The fix lives in the snapshot's ``export -p``
    dump, where ``_export_dump_excluding_session_vars`` ``unset``s names BEFORE
    the dump — the helper-under-test computes exactly that set.
    """

    def test_helper_returns_only_residue_minus_scope(self):
        """Launch-vault names absent from scope appear in the exclusion set;
        scope-bound names do NOT (those stay for the profile that legitimately
        holds them)."""
        from tools.environments.base import _compute_launch_vault_exclusions_for_snapshot

        excluded = _compute_launch_vault_exclusions_for_snapshot(
            launch_env_keys=["GITHUB_TOKEN", "FIGMA_API_KEY", "API_KEY_21ST",
                              "MINIMAX_API_KEY", "OTHER_NON_SECRET"],
            scope_keys=["MINIMAX_API_KEY"],
        )

        # Residue dropped (not in the routed profile's scope).
        assert "GITHUB_TOKEN" in excluded
        assert "FIGMA_API_KEY" in excluded
        assert "API_KEY_21ST" in excluded
        assert "OTHER_NON_SECRET" in excluded
        # Scope-bound name kept -- the routed profile is allowed to see it.
        assert "MINIMAX_API_KEY" not in excluded

    def test_helper_is_case_insensitive(self):
        """The helper folds name cases so the bash ``unset`` call sees the
        case the rc file used to ``export`` (uppercase, by convention)."""
        from tools.environments.base import _compute_launch_vault_exclusions_for_snapshot

        excluded = _compute_launch_vault_exclusions_for_snapshot(
            launch_env_keys=["github_token"],   # lowercase inside the launch .env
            scope_keys=["GITHUB_TOKEN"],       # uppercase in scope
        )
        assert excluded == ()

    def test_launch_profile_target_returns_empty_exclusion(self):
        """When the active ``HERMES_HOME`` IS the launch home, the method must
        return ``()`` -- the launch profile is entitled to everything in its
        own ``.env``, so excluding anything from it would be a regression."""
        import pathlib as _pathlib

        from tools.environments.local import LocalEnvironment

        launch = LocalEnvironment(cwd="/tmp", timeout=5)
        try:
            with patch(
                "hermes_constants.get_hermes_home_override",
                return_value=None,
            ), patch(
                "hermes_constants.get_routing_process_hermes_home",
                return_value=_pathlib.Path("/tmp"),
            ):
                excluded = launch._snapshot_excluded_launch_vault_names()
        finally:
            launch.cleanup()
        assert excluded == ()

    def test_routed_worker_zshrc_export_excluded_from_snapshot(
        self, tmp_path, monkeypatch,
    ):
        """Full-path: fake LAUNCH ``~/.hermes/.env`` + routed HOME + ``~/.zshrc``
        that exports ``GITHUB_TOKEN`` from the launch .env, then boot
        ``LocalEnvironment`` and verify the exclusion set includes it."""
        import pathlib as _pathlib

        # LAUNCH home: ~/.hermes with .env defining all three rc-leak names.
        launch_home = tmp_path / "launch"
        launch_home.mkdir()
        (launch_home / ".env").write_text(
            "GITHUB_TOKEN=launch-gh\n"
            "FIGMA_API_KEY=launch-figma\n"
            "API_KEY_21ST=launch-21st\n"
            "MINIMAX_API_KEY=launch-minimax\n"
        )
        # Routed HOME: profile dir whose .env only ships the model key.
        routed_home = tmp_path / "routed"
        routed_home.mkdir()
        (routed_home / ".env").write_text("MINIMAX_API_KEY=routed-minimax\n")
        # Fake ~/.zshrc re-exporting GITHUB_TOKEN from the launch .env --
        # the real leak vector.
        (tmp_path / ".zshrc").write_text(
            "export GITHUB_TOKEN=$(grep '^GITHUB_TOKEN=' "
            '"$HOME/.hermes/.env" | head -1 | cut -d= -f2-)\n'
        )
        # HERMES_HOME points at the launch home -- .zshrc's $HOME/.hermes is
        # the LAUNCH vault, not the routed override.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(launch_home))

        from tools.environments.local import LocalEnvironment

        env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
        try:
            with patch(
                "tools.environments.local._read_terminal_shell_init_config",
                return_value=([str(tmp_path / ".zshrc")], False),
            ), patch(
                "hermes_constants.get_hermes_home_override",
                return_value=str(routed_home),
            ), patch(
                "hermes_constants.get_routing_process_hermes_home",
                return_value=launch_home,
            ), patch(
                "agent.secret_scope.build_profile_secret_scope",
                return_value={"MINIMAX_API_KEY": "routed-minimax"},
            ):
                excluded = env._snapshot_excluded_launch_vault_names()
            # Routed scope = {MINIMAX_API_KEY}; launch .env ⊃ the three rc-leak
            # names. Expect the latter three in the exclusion set.
            assert "GITHUB_TOKEN" in excluded
            assert "FIGMA_API_KEY" in excluded
            assert "API_KEY_21ST" in excluded
            assert "MINIMAX_API_KEY" not in excluded
        finally:
            env.cleanup()
