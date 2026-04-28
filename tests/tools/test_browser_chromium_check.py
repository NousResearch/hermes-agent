"""Tests for Chromium-presence detection in browser_tool.

Regression guard for the "browser tool advertised but Chromium missing"
class of bug — where ``agent-browser`` CLI is discoverable but no
Chromium build is on disk, causing every browser_* tool call to hang
for the full command timeout before surfacing a useless error.
"""

import os
from pathlib import Path

import pytest

from tools import browser_tool as bt


@pytest.fixture(autouse=True)
def _reset_chromium_cache():
    bt._cached_chromium_installed = None
    yield
    bt._cached_chromium_installed = None


class TestChromiumSearchRoots:
    def test_respects_playwright_browsers_path_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        roots = bt._chromium_search_roots()
        assert str(tmp_path) == roots[0]

    def test_ignores_playwright_browsers_path_zero(self, monkeypatch):
        # Playwright treats "0" as "skip browser download" — not a real path.
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
        roots = bt._chromium_search_roots()
        assert "0" not in roots

    def test_always_includes_default_ms_playwright_cache(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        roots = bt._chromium_search_roots()
        home = os.path.expanduser("~")
        assert any(r == os.path.join(home, ".cache", "ms-playwright") for r in roots)


class TestChromiumInstalled:
    def test_true_when_chromium_dir_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()
        assert bt._chromium_installed() is True

    def test_true_when_headless_shell_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium_headless_shell-1208").mkdir()
        assert bt._chromium_installed() is True

    def test_false_when_dir_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))
        assert bt._chromium_installed() is False

    def test_false_when_only_unrelated_browsers(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))
        (tmp_path / "firefox-1234").mkdir()
        (tmp_path / "webkit-5678").mkdir()
        assert bt._chromium_installed() is False

    def test_false_when_path_not_a_dir(self, monkeypatch, tmp_path):
        # User points PLAYWRIGHT_BROWSERS_PATH at a file by mistake.
        bogus = tmp_path / "nope"
        bogus.write_text("")
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(bogus))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))
        assert bt._chromium_installed() is False

    def test_result_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()
        assert bt._chromium_installed() is True
        # Delete after first call — cached True should still return True.
        (tmp_path / "chromium-1208").rmdir()
        assert bt._chromium_installed() is True


class TestCheckBrowserRequirementsChromium:
    def test_local_mode_missing_chromium_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        assert bt.check_browser_requirements() is False

    def test_local_mode_with_chromium_returns_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()

        assert bt.check_browser_requirements() is True

    def test_cloud_mode_does_not_require_local_chromium(self, monkeypatch, tmp_path):
        """Cloud browsers (Browserbase etc.) host their own Chromium."""
        class FakeProvider:
            def is_configured(self):
                return True
            def provider_name(self):
                return "browserbase"

        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: FakeProvider())
        # Point chromium search at an empty dir — should not matter for cloud.
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        assert bt.check_browser_requirements() is True

    def test_camofox_mode_does_not_require_chromium(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: True)
        # Even with no chromium on disk, camofox drives its own backend.
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        assert bt.check_browser_requirements() is True


class TestRunBrowserCommandChromiumGuard:
    """Verify _run_browser_command fails fast (no timeout hang) when
    Chromium is missing in local mode.
    """

    def test_local_mode_missing_chromium_returns_error_immediately(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt, "_is_local_mode", lambda: True)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        # If we ever reached subprocess.Popen the test would hang — the
        # fast-fail guard prevents that.
        def _fail_popen(*args, **kwargs):
            raise AssertionError("Should have failed before spawning subprocess")

        monkeypatch.setattr("subprocess.Popen", _fail_popen)

        result = bt._run_browser_command("task-1", "navigate", ["https://example.com"])
        assert result["success"] is False
        assert "Chromium" in result["error"]

    def test_docker_hint_mentions_image_pull(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt, "_is_local_mode", lambda: True)
        monkeypatch.setattr(bt, "_running_in_docker", lambda: True)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        result = bt._run_browser_command("task-1", "navigate", ["https://example.com"])
        assert result["success"] is False
        assert "docker pull" in result["error"].lower()

    def test_non_docker_hint_mentions_agent_browser_install(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt, "_is_local_mode", lambda: True)
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        result = bt._run_browser_command("task-1", "navigate", ["https://example.com"])
        assert result["success"] is False
        assert "agent-browser install" in result["error"]
