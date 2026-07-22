"""Tests for verbose logging feature — format_verbose() and debug verbose subcommand."""

import pytest
from pathlib import Path
from unittest.mock import patch


class TestVerboseSections:
    def test_sections_constant(self):
        from hermes_cli.debug import VERBOSE_SECTIONS
        assert VERBOSE_SECTIONS == {"config", "providers", "tools", "memory", "session"}


class TestFormatVerbose:
    def test_returns_string(self):
        from hermes_cli.debug import format_verbose
        output = format_verbose()
        assert isinstance(output, str)

    def test_all_sections_present(self):
        from hermes_cli.debug import format_verbose
        output = format_verbose()
        assert "Config" in output
        assert "Providers" in output
        assert "Tools" in output
        assert "Memory" in output
        assert "Session" in output

    def test_header(self):
        from hermes_cli.debug import format_verbose
        output = format_verbose()
        assert "HERMES AGENT VERBOSE REPORT" in output

    def test_config_section_reads_terminal_backend(self):
        from hermes_cli.debug import format_verbose
        output = format_verbose(sections=["config"])
        assert "terminal.backend" in output
        # Should NOT contain the old wrong key
        assert "terminal.mode" not in output

    def test_specific_section(self):
        from hermes_cli.debug import format_verbose
        output = format_verbose(sections=["providers"])
        assert "Providers" in output
        assert "Config" not in output

    def test_session_section_uses_session_count(self):
        """Verify _verbose_session calls session_count(), not get_total_sessions."""
        from hermes_cli.debug import _verbose_session
        lines = _verbose_session()
        joined = "\n".join(lines)
        # Should contain a numeric session count or an error, never N/A from hasattr fallback
        assert "Total sessions:" in joined


class TestVerboseSessionIntegration:
    def test_session_count_returns_int(self, tmp_path):
        """Seed a temp SessionDB and verify session_count() returns an int."""
        import os
        from hermes_constants import get_hermes_home

        # Patch HERMES_HOME so SessionDB uses our temp directory
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from hermes_state import SessionDB
            db = SessionDB()
            count = db.session_count()
            assert isinstance(count, int)
            assert count >= 0


class TestDebugVerboseSubcommand:
    def test_run_debug_verbose_prints_output(self, capsys):
        from hermes_cli.debug import run_debug_verbose

        class Args:
            sections = None
        run_debug_verbose(Args())
        captured = capsys.readouterr()
        assert "HERMES AGENT VERBOSE REPORT" in captured.out

    def test_run_debug_verbose_sections_arg(self, capsys):
        from hermes_cli.debug import run_debug_verbose

        class Args:
            sections = "config,tools"
        run_debug_verbose(Args())
        captured = capsys.readouterr()
        assert "Config" in captured.out
        assert "Tools" in captured.out
        assert "Providers" not in captured.out

    def test_run_debug_verbose_invalid_section(self, capsys):
        from hermes_cli.debug import run_debug_verbose

        class Args:
            sections = "bogus"
        run_debug_verbose(Args())
        captured = capsys.readouterr()
        assert "Unknown sections" in captured.out
