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
