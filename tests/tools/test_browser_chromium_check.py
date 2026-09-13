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
def _isolated_browser_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))



class TestChromiumInstalled:
    def test_shell_only_cache_does_not_satisfy_full_browser(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium_headless_shell-1234").mkdir()
        assert bt_install._chromium_installed() is False

    def test_system_chromium_on_path_alone_is_not_enough(self, monkeypatch, tmp_path):
        """Pinned-store-only (gap plan D3): a system Chromium in PATH does
        NOT satisfy the check — only AGENT_BROWSER_EXECUTABLE_PATH or the
        pinned browser store do."""
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, path=None: "/usr/bin/chromium" if name == "chromium" else None,
        )

        assert bt_install._chromium_installed() is False


    def test_true_when_agent_browser_executable_path_set(self, monkeypatch, tmp_path):
        exe = tmp_path / ("chrome.exe" if os.name == "nt" else "chrome")
        exe.write_bytes(b"")
        monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(exe))
        assert bt_install._chromium_installed() is True


    def test_removed_override_is_no_longer_ready(self, monkeypatch, tmp_path):
        browser = tmp_path / "browser"
        browser.touch()
        monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(browser))
        assert bt_install._chromium_installed() is True
        browser.unlink()
        assert bt_install._chromium_installed() is False


class TestCheckBrowserRequirementsChromium:

    def test_local_mode_with_chromium_returns_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda **_kw: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
        browser = tmp_path / "browser"
        browser.touch()
        monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(browser))

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


