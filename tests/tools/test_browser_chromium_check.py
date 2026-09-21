"""Tests for Chromium-presence detection in browser_tool.

Regression guard for the "browser tool advertised but Chromium missing"
class of bug — where ``agent-browser`` CLI is discoverable but no
Chromium build is on disk, causing every browser_* tool call to hang
for the full command timeout before surfacing a useless error.
"""

import os
import shutil

import pytest

from tools import browser_tool as bt
from tools import browser_tool_install as bt_install
from tools import browser_tool_cloud as bt_cloud


@pytest.fixture(autouse=True)
def _reset_chromium_cache():
    bt._cached_chromium_installed = None
    yield
    bt._cached_chromium_installed = None


class TestChromiumSearchRoots:
    def test_respects_playwright_browsers_path_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        roots = bt_install._chromium_search_roots()
        assert str(tmp_path) == roots[0]


    def test_always_includes_default_ms_playwright_cache(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        roots = bt_install._chromium_search_roots()
        home = os.path.expanduser("~")
        assert any(r == os.path.join(home, ".cache", "ms-playwright") for r in roots)


    def test_includes_agent_browser_managed_cache(self, monkeypatch, tmp_path):
        """``agent-browser install`` — the command the missing-Chromium error prints — puts Chrome
        in ``$AGENT_BROWSER_HOME/browsers``, never in Playwright's cache."""
        monkeypatch.setenv("AGENT_BROWSER_HOME", str(tmp_path))
        assert os.path.join(str(tmp_path), "browsers") in bt_install._chromium_search_roots()


class TestChromiumInstalled:
    def test_true_when_chrome_lives_in_agent_browser_managed_cache(self, monkeypatch, tmp_path):
        """Regression: the probe must see the browser ``agent-browser`` itself would launch.

        Reported state: ``agent-browser open <url>`` succeeds, no system Chrome and no Playwright
        cache on the box, yet every browser tool call answered "Chromium browser is missing. Install
        it with: npx agent-browser install" — running that command just re-creates the directory the
        probe refused to read, so the hint could never satisfy the gate.
        """
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        monkeypatch.setattr(shutil, "which", lambda name, path=None: None)

        assert bt_install._chromium_installed() is False
        (tmp_path / ".agent-browser" / "browsers" / "chrome-153.0.8010.52").mkdir(parents=True)
        bt._cached_chromium_installed = None
        assert bt_install._chromium_installed() is True


    def test_true_when_plain_chromium_on_path(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, path=None: "/usr/bin/chromium" if name == "chromium" else None,
        )

        assert bt_install._chromium_installed() is True


    def test_result_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()
        assert bt_install._chromium_installed() is True
        # Delete after first call — cached True should still return True.
        (tmp_path / "chromium-1208").rmdir()
        assert bt_install._chromium_installed() is True


class TestCheckBrowserRequirementsChromium:

    def test_local_mode_with_chromium_returns_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda **_kw: "/usr/local/bin/agent-browser")
        monkeypatch.setattr("tools.browser_tool_install._requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()

        assert bt_install.check_browser_requirements() is True


    def test_camofox_mode_does_not_require_chromium(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: True)
        # Even with no chromium on disk, camofox drives its own backend.
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        assert bt_install.check_browser_requirements() is True


class TestRunBrowserCommandChromiumGuard:
    """Verify _run_browser_command fails fast (no timeout hang) when
    Chromium is missing in local mode.
    """


