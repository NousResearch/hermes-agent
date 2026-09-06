"""Tests for Chromium-presence detection in browser_tool.

Regression guard for the "browser tool advertised but Chromium missing"
class of bug — where ``agent-browser`` CLI is discoverable but no
Chromium build is on disk, causing every browser_* tool call to hang
for the full command timeout before surfacing a useless error.
"""

import os
import shutil
import sys

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


class TestManagedChromiumExecutable:
    def test_finds_newest_agent_browser_download(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [])
        older = tmp_path / ".agent-browser" / "browsers" / "chrome-99.0.1.2" / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "Google Chrome for Testing"
        newest = tmp_path / ".agent-browser" / "browsers" / "chrome-152.0.7977.82" / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "Google Chrome for Testing"
        for executable in (older, newest):
            executable.parent.mkdir(parents=True)
            executable.touch()
            executable.chmod(0o755)
        monkeypatch.setattr(bt_install.sys, "platform", "darwin")

        assert bt_install._managed_chromium_executable() == str(newest)

    def test_system_chrome_alone_is_not_managed(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [])
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, path=None: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            if name == "google-chrome"
            else None,
        )

        assert bt_install._managed_chromium_executable() is None

    def test_local_browser_subprocess_is_pinned_to_managed_download(self, monkeypatch):
        managed = "/managed/Chrome for Testing"
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install, "_managed_chromium_executable", lambda: managed)
        monkeypatch.setattr(
            "tools.environments.local.hermes_subprocess_env",
            lambda **_kwargs: {"PATH": "/usr/bin"},
        )

        from tools import browser_tool_session as bt_session

        env = bt_session._agent_browser_command_env("/tmp/socket", managed_chromium=True)
        assert env["AGENT_BROWSER_EXECUTABLE_PATH"] == managed

    def test_non_browser_subprocess_is_not_pinned(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install, "_managed_chromium_executable", lambda: "/managed/Chrome for Testing")
        monkeypatch.setattr("tools.environments.local.hermes_subprocess_env", lambda **_kwargs: {"PATH": "/usr/bin"})
        from tools import browser_tool_session as bt_session

        assert "AGENT_BROWSER_EXECUTABLE_PATH" not in bt_session._agent_browser_command_env("/tmp/socket")


class TestChromiumInstalled:
    def test_true_when_agent_browser_managed_download_exists(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install, "_managed_chromium_executable", lambda: "/managed/Chrome for Testing")
        monkeypatch.setattr(shutil, "which", lambda _name, path=None: None)

        assert bt_install._chromium_installed() is True

    def test_false_when_only_system_chromium_is_on_path(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install, "_managed_chromium_executable", lambda: None)
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, path=None: "/usr/bin/chromium" if name == "chromium" else None,
        )

        assert bt_install._chromium_installed() is False


    def test_result_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        release = tmp_path / "chromium-1208"
        if sys.platform == "darwin":
            executable = release / "chrome-mac-arm64" / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "Google Chrome for Testing"
        elif sys.platform == "win32":
            executable = release / "chrome-win64" / "chrome.exe"
        else:
            executable = release / "chrome-linux64" / "chrome"
        executable.parent.mkdir(parents=True)
        executable.touch(mode=0o755)
        assert bt_install._chromium_installed() is True
        # Delete after first call — cached True should still return True.
        executable.unlink()
        assert bt_install._chromium_installed() is True


class TestCheckBrowserRequirementsChromium:

    def test_local_mode_with_chromium_returns_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda **_kw: "/usr/local/bin/agent-browser")
        monkeypatch.setattr("tools.browser_tool_install._requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        release = tmp_path / "chromium-1208"
        if sys.platform == "darwin":
            executable = release / "chrome-mac-arm64" / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "Google Chrome for Testing"
        elif sys.platform == "win32":
            executable = release / "chrome-win64" / "chrome.exe"
        else:
            executable = release / "chrome-linux64" / "chrome"
        executable.parent.mkdir(parents=True)
        executable.touch(mode=0o755)

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


