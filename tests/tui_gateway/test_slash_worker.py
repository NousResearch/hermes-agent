"""Tests for tui_gateway/slash_worker.py — targeting ≥70% statement coverage.

The module exposes two functions:
- _run(cli, command): executes a slash command via HermesCLI, capturing stdout
- main(): argparse entry point that reads JSON from stdin, dispatches _run, writes JSON to stdout
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

from tui_gateway import slash_worker


# ─── _run ─────────────────────────────────────────────────────────────────────


class TestRun:
    """Tests for _run() command execution."""

    def _make_mock_cli(self):
        cli = MagicMock()
        cli.console = MagicMock()
        cli.process_command = MagicMock()
        return cli

    def test_empty_string_returns_empty(self):
        cli = self._make_mock_cli()
        assert slash_worker._run(cli, "") == ""

    def test_whitespace_only_returns_empty(self):
        cli = self._make_mock_cli()
        assert slash_worker._run(cli, "   ") == ""

    def test_prepends_slash_to_bare_command(self):
        cli = self._make_mock_cli()
        slash_worker._run(cli, "help")
        cli.process_command.assert_called_once_with("/help")

    def test_preserves_existing_slash(self):
        cli = self._make_mock_cli()
        slash_worker._run(cli, "/status")
        cli.process_command.assert_called_once_with("/status")

    def test_captures_stdout_from_process_command(self):
        cli = self._make_mock_cli()

        def fake_process(cmd):
            print("hello from cli")

        cli.process_command = fake_process
        result = slash_worker._run(cli, "/echo")
        assert "hello from cli" in result

    def test_strips_trailing_whitespace(self):
        cli = self._make_mock_cli()

        def fake_process(cmd):
            print("output\n\n")

        cli.process_command = fake_process
        result = slash_worker._run(cli, "/test")
        assert result == "output"

    def test_restores_original_cprint(self):
        import cli as cli_mod

        cli = self._make_mock_cli()
        original = getattr(cli_mod, "_cprint", None)
        slash_worker._run(cli, "/noop")
        assert getattr(cli_mod, "_cprint", None) is original

    def test_restores_cprint_even_on_error(self):
        """cprint must be restored even if process_command raises."""
        import cli as cli_mod

        cli = self._make_mock_cli()
        cli.process_command.side_effect = RuntimeError("boom")
        original = getattr(cli_mod, "_cprint", None)

        try:
            slash_worker._run(cli, "/fail")
        except RuntimeError:
            pass

        assert getattr(cli_mod, "_cprint", None) is original


# ─── main() ───────────────────────────────────────────────────────────────────


class TestMain:
    """Tests for the main() entry point."""

    def test_processes_valid_json_command(self, monkeypatch):
        """main() reads a JSON command from stdin and writes a JSON response."""
        fake_stdin = io.StringIO(json.dumps({"id": "r1", "command": "/help"}) + "\n")
        fake_stdout = io.StringIO()

        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr("sys.stdout", fake_stdout)
        monkeypatch.setattr("sys.argv", ["slash_worker", "--session-key", "test-session", "--model", ""])

        mock_cli = MagicMock()
        mock_cli.console = MagicMock()
        mock_cli.process_command = MagicMock()
        monkeypatch.setattr("tui_gateway.slash_worker.HermesCLI", lambda **kwargs: mock_cli)

        slash_worker.main()

        output = fake_stdout.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["id"] == "r1"
        assert parsed["ok"] is True

    def test_handles_invalid_json(self, monkeypatch):
        """main() writes an error response when stdin contains invalid JSON."""
        fake_stdin = io.StringIO("not json\n")
        fake_stdout = io.StringIO()

        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr("sys.stdout", fake_stdout)
        monkeypatch.setattr("sys.argv", ["slash_worker", "--session-key", "test", "--model", ""])

        mock_cli = MagicMock()
        monkeypatch.setattr("tui_gateway.slash_worker.HermesCLI", lambda **kwargs: mock_cli)

        slash_worker.main()

        output = fake_stdout.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["ok"] is False
        assert "error" in parsed

    def test_skips_blank_lines(self, monkeypatch):
        """main() skips blank lines in stdin without producing output."""
        fake_stdin = io.StringIO("\n\n\n")
        fake_stdout = io.StringIO()

        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr("sys.stdout", fake_stdout)
        monkeypatch.setattr("sys.argv", ["slash_worker", "--session-key", "test", "--model", ""])

        mock_cli = MagicMock()
        monkeypatch.setattr("tui_gateway.slash_worker.HermesCLI", lambda **kwargs: mock_cli)

        slash_worker.main()

        # No output — blank lines were skipped
        assert fake_stdout.getvalue().strip() == ""

    def test_sets_env_vars(self, monkeypatch):
        """main() sets HERMES_SESSION_KEY and HERMES_INTERACTIVE env vars."""
        import os

        fake_stdin = io.StringIO("")
        fake_stdout = io.StringIO()

        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr("sys.stdout", fake_stdout)
        monkeypatch.setattr("sys.argv", ["slash_worker", "--session-key", "my-session", "--model", "gpt-4"])

        mock_cli = MagicMock()
        monkeypatch.setattr("tui_gateway.slash_worker.HermesCLI", lambda **kwargs: mock_cli)

        slash_worker.main()

        assert os.environ.get("HERMES_SESSION_KEY") == "my-session"
        assert os.environ.get("HERMES_INTERACTIVE") == "1"
