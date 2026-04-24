"""Tests for browser.chrome_args config and AGENT_BROWSER_ARGS env injection.

Covers the config -> env var wiring that lets users pass --no-sandbox and
other Chrome launch flags when running in containers/VMs.
"""

import os
from unittest.mock import patch

import pytest

import tools.browser_tool as browser_tool


class TestGetChromeArgs:
    """Tests for the _get_chrome_args() config reader."""

    def setup_method(self):
        # Clear the module-level cache before each test
        browser_tool._cached_chrome_args = None
        browser_tool._chrome_args_resolved = False

    def teardown_method(self):
        browser_tool._cached_chrome_args = None
        browser_tool._chrome_args_resolved = False

    def test_returns_empty_when_unset(self):
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert browser_tool._get_chrome_args() == ""

    def test_reads_from_config(self):
        cfg = {"browser": {"chrome_args": "--no-sandbox"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert browser_tool._get_chrome_args() == "--no-sandbox"

    def test_strips_whitespace(self):
        cfg = {"browser": {"chrome_args": "  --no-sandbox  "}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert browser_tool._get_chrome_args() == "--no-sandbox"

    def test_caches_result(self):
        cfg = {"browser": {"chrome_args": "--no-sandbox"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg) as mock_read:
            browser_tool._get_chrome_args()
            browser_tool._get_chrome_args()
            mock_read.assert_called_once()

    def test_merges_env_var_before_config(self):
        """AGENT_BROWSER_ARGS env var is prepended; config value is appended."""
        cfg = {"browser": {"chrome_args": "--disable-gpu"}}
        with patch.dict(os.environ, {"AGENT_BROWSER_ARGS": "--no-sandbox"}):
            with patch("hermes_cli.config.read_raw_config", return_value=cfg):
                result = browser_tool._get_chrome_args()
                assert "--no-sandbox" in result
                assert "--disable-gpu" in result
                # env var comes first, then config, comma-joined
                assert result == "--no-sandbox,--disable-gpu"

    def test_env_var_alone_when_config_empty(self):
        with patch.dict(os.environ, {"AGENT_BROWSER_ARGS": "--no-sandbox"}):
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert browser_tool._get_chrome_args() == "--no-sandbox"

    def test_normalizes_spaces_to_commas(self):
        """Shell-style space-separated flags are converted to commas."""
        cfg = {"browser": {"chrome_args": "--no-sandbox --disable-gpu"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert browser_tool._get_chrome_args() == "--no-sandbox,--disable-gpu"

    def test_does_not_normalize_when_commas_already_present(self):
        """If user already used commas, don't touch spaces inside values."""
        cfg = {"browser": {"chrome_args": "--foo,--bar=b a z"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            # commas present → spaces preserved
            assert browser_tool._get_chrome_args() == "--foo,--bar=b a z"

    def test_handles_config_read_failure(self):
        with patch(
            "hermes_cli.config.read_raw_config", side_effect=Exception("disk error")
        ):
            assert browser_tool._get_chrome_args() == ""


class TestRunBrowserCommandEnvInjection:
    """Tests that _run_browser_command injects AGENT_BROWSER_ARGS."""

    def test_sets_agent_browser_args_when_chrome_args_configured(self, monkeypatch, tmp_path):
        monkeypatch.setattr(browser_tool, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            lambda task_id: {"session_name": "sess_1", "cdp_url": None},
        )
        monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(browser_tool, "_write_owner_pid", lambda *a: None)
        monkeypatch.setattr(browser_tool, "_get_chrome_args", lambda: "--no-sandbox")

        captured_env = {}

        def fake_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            # Return a mock process that exits immediately
            class MockProc:
                returncode = 0
                def wait(self, timeout=None):
                    pass
            return MockProc()

        monkeypatch.setattr(browser_tool.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(browser_tool.os, "open", lambda *a, **k: 0)
        monkeypatch.setattr(browser_tool.os, "close", lambda fd: None)

        browser_tool._run_browser_command("task-1", "open", ["https://example.com"])

        assert captured_env.get("AGENT_BROWSER_ARGS") == "--no-sandbox"

    def test_preserves_existing_env_var_when_no_config(self, monkeypatch, tmp_path):
        monkeypatch.setattr(browser_tool, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            lambda task_id: {"session_name": "sess_2", "cdp_url": None},
        )
        monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(browser_tool, "_write_owner_pid", lambda *a: None)
        # Do NOT mock _get_chrome_args — let it read the real env var

        captured_env = {}

        def fake_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            class MockProc:
                returncode = 0
                def wait(self, timeout=None):
                    pass
            return MockProc()

        monkeypatch.setattr(browser_tool.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(browser_tool.os, "open", lambda *a, **k: 0)
        monkeypatch.setattr(browser_tool.os, "close", lambda fd: None)

        # Pre-set env var in os.environ (simulating user export)
        monkeypatch.setenv("AGENT_BROWSER_ARGS", "--disable-gpu")
        # Clear cache so _get_chrome_args picks up the env var
        browser_tool._cached_chrome_args = None
        browser_tool._chrome_args_resolved = False

        browser_tool._run_browser_command("task-2", "snapshot", [])

        assert captured_env.get("AGENT_BROWSER_ARGS") == "--disable-gpu"


class TestCleanupClearsChromeArgsCache:
    """Tests that cleanup_all_browsers resets the chrome_args cache."""

    def test_cleanup_resets_chrome_args_cache(self, monkeypatch):
        # Seed the cache
        browser_tool._cached_chrome_args = "--no-sandbox"
        browser_tool._chrome_args_resolved = True

        monkeypatch.setattr(browser_tool, "_active_sessions", {})
        monkeypatch.setattr(browser_tool, "_stop_cdp_supervisor", lambda t: None)

        browser_tool.cleanup_all_browsers()

        assert browser_tool._cached_chrome_args is None
        assert browser_tool._chrome_args_resolved is False
