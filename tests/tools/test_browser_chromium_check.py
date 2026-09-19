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


class TestChromiumInstalled:
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


class TestAgentBrowserBrowsersRoot:
    """agent-browser 0.26+ installs Chrome for Testing into ``~/.agent-browser/browsers/chrome-<ver>/``
    — a layout the Playwright-only scan roots never saw, so the gate reported "missing" while
    agent-browser itself launched that build fine (PR #30161)."""

    def test_includes_agent_browser_browsers_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path))
        roots = bt_install._chromium_search_roots()
        assert str(tmp_path / ".agent-browser" / "browsers") in roots


class TestChromeForTestingBuild:
    def _make_cfT(self, tmp_path, rel="chrome", with_binary=True, executable=True):
        """agent-browser's Chrome-for-Testing cache: browsers/chrome-<version>/<layout>."""
        browsers = tmp_path / "browsers"
        chrome_dir = browsers / "chrome-148.0.7778.56"
        exe = chrome_dir / rel
        exe.parent.mkdir(parents=True)
        if with_binary:
            exe.touch()
            if executable:
                exe.chmod(0o755)
        return browsers

    def test_true_when_agent_browser_chrome_dir_present(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        browsers = self._make_cfT(tmp_path)
        assert bt_install._has_chromium_build(str(browsers)) is True

    def test_true_for_nested_chrome_linux64_layout(self, monkeypatch, tmp_path):
        """CfT CDN's linux64 archive unpacks to chrome-<ver>/chrome-linux64/chrome — agent-browser
        0.37.1's own discovery resolves this layout (verified live), so the gate must too."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        browsers = self._make_cfT(tmp_path, rel="chrome-linux64/chrome")
        assert bt_install._has_chromium_build(str(browsers)) is True

    def test_chrome_dir_without_executable_returns_false(self, monkeypatch, tmp_path):
        """An interrupted install can leave a version dir with no binary — not a usable browser."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        browsers = self._make_cfT(tmp_path, with_binary=False)
        assert bt_install._has_chromium_build(str(browsers)) is False

    def test_non_executable_binary_returns_false(self, monkeypatch, tmp_path):
        """agent-browser's probe 'finds' a non-executable chrome but launch fails with EACCES —
        the gate must not advertise tools that then hang/fail (verified live vs 0.37.1 doctor)."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        browsers = self._make_cfT(tmp_path, executable=False)
        assert bt_install._has_chromium_build(str(browsers)) is False

    def test_ignores_non_chrome_dirs_in_agent_browser(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        browsers = tmp_path / "browsers"
        (browsers / "not-a-browser").mkdir(parents=True)
        assert bt_install._has_chromium_build(str(browsers)) is False

    def test_playwright_chromium_prefix_still_accepted_without_probe(self, tmp_path):
        """chromium-* dirs keep the plain directory-name check (Playwright layouts always carry the binary)."""
        root = tmp_path / "chromium-1208"
        root.mkdir()
        assert bt_install._has_chromium_build(str(tmp_path)) is True


class TestRunBrowserCommandChromiumGuard:
    """Verify _run_browser_command fails fast (no timeout hang) when
    Chromium is missing in local mode.
    """


