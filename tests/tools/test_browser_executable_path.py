"""Tests for browser.executable_path config propagation to agent-browser (#66111).

``browser.executable_path`` in config.yaml must be read and propagated as
``AGENT_BROWSER_EXECUTABLE_PATH`` to the agent-browser subprocess env, and
must make ``_chromium_installed()`` report True so the browser tool isn't
disabled.  Before the fix only the env var worked; the config key was a dead
value.
"""

import sys

import pytest

import tools.browser_tool as bt


@pytest.fixture(autouse=True)
def _reset_chromium_cache():
    """Reset cached chromium-installation state between tests."""
    bt._cached_chromium_installed = None
    yield
    bt._cached_chromium_installed = None


def _no_system_chrome(monkeypatch):
    """Neutralize system-Chrome / Playwright-cache discovery so tests are
    deterministic regardless of the host's installed browsers."""
    monkeypatch.setattr(bt.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(bt, "_chromium_search_roots", lambda: [])


# ---------------------------------------------------------------------------
# _build_browser_env — config propagation
# ---------------------------------------------------------------------------
class TestBuildBrowserEnvExecutablePath:
    def test_env_includes_config_executable_path(self, monkeypatch):
        """Config-set ``browser.executable_path`` reaches the subprocess env."""
        fake_binary = sys.executable  # guaranteed to exist
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"executable_path": fake_binary}},
        )
        env = bt._build_browser_env()
        assert env.get("AGENT_BROWSER_EXECUTABLE_PATH") == fake_binary

    def test_env_var_takes_priority_over_config(self, monkeypatch):
        """Explicit env var wins over config value."""
        monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", "/env/chrome")
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"executable_path": "/config/chrome"}},
        )
        env = bt._build_browser_env()
        assert env["AGENT_BROWSER_EXECUTABLE_PATH"] == "/env/chrome"

    def test_no_executable_path_when_config_unset(self, monkeypatch):
        """No ``AGENT_BROWSER_EXECUTABLE_PATH`` when config has empty value."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"executable_path": ""}},
        )
        env = bt._build_browser_env()
        assert "AGENT_BROWSER_EXECUTABLE_PATH" not in env


# ---------------------------------------------------------------------------
# _chromium_installed — config-aware capability check
# ---------------------------------------------------------------------------
class TestChromiumInstalledConfigPath:
    def test_returns_true_for_config_path(self, monkeypatch, tmp_path):
        """Config-set path to a real binary makes the check pass."""
        fake_binary = tmp_path / "chromium"
        fake_binary.touch(mode=0o755)
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        _no_system_chrome(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"executable_path": str(fake_binary)}},
        )
        assert bt._chromium_installed() is True

    def test_returns_false_for_nonexistent_config_path(self, monkeypatch):
        """Config-set path to a missing binary does not false-positive."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        _no_system_chrome(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"executable_path": "/nonexistent/chrome"}},
        )
        assert bt._chromium_installed() is False
