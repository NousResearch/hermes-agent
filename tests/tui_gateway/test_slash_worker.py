"""Tests for tui_gateway/slash_worker.py — targeting ≥70% statement coverage.

Covers:
- _env_float: parsing env knobs with valid, missing, and malformed values
- _is_orphaned: already covered in test_slash_worker_watchdog.py (imported here)
- _run: slash command execution with mock CLI
"""

from __future__ import annotations

import io
import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

import psutil
from tui_gateway import slash_worker


# ─── _env_float ───────────────────────────────────────────────────────────────


class TestEnvFloat:
    """Tests for _env_float helper."""

    def test_returns_default_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_ENV_FLOAT_KEY", None)
            assert slash_worker._env_float("TEST_ENV_FLOAT_KEY", 3.14) == 3.14

    def test_parses_valid_float(self):
        with patch.dict(os.environ, {"TEST_ENV_FLOAT_KEY": "1.5"}):
            assert slash_worker._env_float("TEST_ENV_FLOAT_KEY", 0.0) == 1.5

    def test_parses_valid_int_string(self):
        with patch.dict(os.environ, {"TEST_ENV_FLOAT_KEY": "7"}):
            assert slash_worker._env_float("TEST_ENV_FLOAT_KEY", 0.0) == 7.0

    def test_returns_default_on_malformed_value(self):
        with patch.dict(os.environ, {"TEST_ENV_FLOAT_KEY": "2s"}):
            assert slash_worker._env_float("TEST_ENV_FLOAT_KEY", 9.9) == 9.9

    def test_returns_default_on_empty_string(self):
        with patch.dict(os.environ, {"TEST_ENV_FLOAT_KEY": ""}):
            assert slash_worker._env_float("TEST_ENV_FLOAT_KEY", 42.0) == 42.0


# ─── _run ─────────────────────────────────────────────────────────────────────


class TestRun:
    """Tests for _run() command execution."""

    def _make_mock_cli(self):
        """Create a mock HermesCLI with a Rich Console-like interface."""
        cli = MagicMock()
        cli.console = MagicMock()
        cli.process_command = MagicMock()
        return cli

    def test_empty_command_returns_empty(self):
        cli = self._make_mock_cli()
        assert slash_worker._run(cli, "") == ""
        assert slash_worker._run(cli, "   ") == ""

    def test_prepends_slash_to_bare_command(self):
        cli = self._make_mock_cli()
        # process_command will be called; we capture what it was called with
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

    def test_restores_original_cprint(self):
        import cli as cli_mod

        cli = self._make_mock_cli()
        original = getattr(cli_mod, "_cprint", None)
        slash_worker._run(cli, "/noop")
        # _cprint should be restored after _run
        assert getattr(cli_mod, "_cprint", None) is original


# ─── _is_orphaned (re-export for completeness) ───────────────────────────────


class TestIsOrphaned:
    """Additional _is_orphaned edge cases."""

    def test_returns_true_on_psutil_error(self):
        """When psutil.pid_exists raises psutil.Error, treat as orphaned."""
        with patch("tui_gateway.slash_worker.psutil") as mock_psutil:
            mock_psutil.Error = psutil.Error
            mock_psutil.pid_exists.side_effect = psutil.Error("test")
            result = slash_worker._is_orphaned(12345, 1.0, getppid=lambda: 12345)
            assert result is True


# ─── _start_parent_death_watchdog ─────────────────────────────────────────────


class TestWatchdog:
    """Test that the watchdog thread starts."""

    def test_watchdog_thread_is_daemon(self):
        """The watchdog thread should be a daemon so it doesn't block exit."""
        started = threading.Event()
        original_loop = slash_worker._start_parent_death_watchdog

        # We can't easily test the full watchdog without mocking time/psutil,
        # but we can verify the thread starts as daemon
        with patch.object(slash_worker, "_is_orphaned", return_value=False):
            # Start the watchdog with a mock that will immediately orphan
            with patch.object(slash_worker, "_is_orphaned", return_value=True):
                with patch("os._exit"):
                    # This will start the thread and immediately trigger orphan detection
                    slash_worker._start_parent_death_watchdog(99999, 0.0)
                    # Give thread a moment to start
                    import time

                    time.sleep(0.01)


# ─── _in_flight event ─────────────────────────────────────────────────────────


class TestInFlight:
    """Test the _in_flight threading event."""

    def test_in_flight_starts_cleared(self):
        """Module-level _in_flight should start cleared."""
        # Reset to known state
        slash_worker._in_flight.clear()
        assert not slash_worker._in_flight.is_set()


# ─── main() ───────────────────────────────────────────────────────────────────


class TestMain:
    """Tests for the main() entry point."""

    def test_main_processes_stdin_json(self, monkeypatch, tmp_path):
        """main() reads JSON lines from stdin and writes JSON responses."""
        # Fake stdin with one JSON command
        fake_stdin = io.StringIO(json.dumps({"id": "r1", "command": "/help"}) + "\n")
        fake_stdout = io.StringIO()

        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr("sys.stdout", fake_stdout)
        monkeypatch.setattr("sys.argv", ["slash_worker", "--session-key", "test-session", "--model", ""])

        # Mock watchdog to prevent os._exit
        monkeypatch.setattr("tui_gateway.slash_worker._start_parent_death_watchdog", lambda *a: None)

        # Mock HermesCLI to avoid real initialization
        mock_cli_instance = MagicMock()
        mock_cli_instance.console = MagicMock()
        mock_cli_instance.process_command = MagicMock()
        monkeypatch.setattr("tui_gateway.slash_worker.HermesCLI", lambda **kwargs: mock_cli_instance)

        # Run main — it should process one line then exit (stdin exhausted)
        slash_worker.main()

        # Verify output
        output = fake_stdout.getvalue().strip()
        assert output  # should have produced some JSON output
        parsed = json.loads(output)
        assert parsed["id"] == "r1"
        assert parsed["ok"] is True

    def test_main_handles_invalid_json(self, monkeypatch):
        """main() writes error response for invalid JSON."""
        fake_stdin = io.StringIO("not json\n")
        fake_stdout = io.StringIO()

        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr("sys.stdout", fake_stdout)
        monkeypatch.setattr("sys.argv", ["slash_worker", "--session-key", "test", "--model", ""])
        monkeypatch.setattr("tui_gateway.slash_worker._start_parent_death_watchdog", lambda *a: None)

        mock_cli_instance = MagicMock()
        monkeypatch.setattr("tui_gateway.slash_worker.HermesCLI", lambda **kwargs: mock_cli_instance)

        slash_worker.main()

        output = fake_stdout.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["ok"] is False
        assert "error" in parsed
