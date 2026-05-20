"""Tests for Chromium-presence detection in browser_tool.

Regression guard for the "browser tool advertised but Chromium missing"
class of bug — where ``agent-browser`` CLI is discoverable but no
Chromium build is on disk, causing every browser_* tool call to hang
for the full command timeout before surfacing a useless error.
"""

import os
import subprocess
from pathlib import Path

import pytest

from tools import browser_tool as bt


@pytest.fixture(autouse=True)
def _reset_chromium_cache():
    bt._cached_chromium_installed = None
    cache_clear = getattr(bt._browser_sandbox_bypass_reason, "cache_clear", None)
    if cache_clear:
        cache_clear()
    yield
    bt._cached_chromium_installed = None
    cache_clear = getattr(bt._browser_sandbox_bypass_reason, "cache_clear", None)
    if cache_clear:
        cache_clear()


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
    def test_true_when_plain_chromium_on_path(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(
            bt.shutil,
            "which",
            lambda name: "/usr/bin/chromium" if name == "chromium" else None,
        )

        assert bt._chromium_installed() is True


class TestBrowserSandboxBypass:
    def test_root_requires_sandbox_bypass(self, monkeypatch):
        monkeypatch.setattr(bt.os, "name", "posix")
        monkeypatch.setattr(bt.sys, "platform", "linux")
        monkeypatch.setattr(bt.os, "geteuid", lambda: 0, raising=False)

        assert bt._browser_sandbox_bypass_reason() == "running as root"

    def test_apparmor_userns_restriction_requires_sandbox_bypass(self, monkeypatch):
        monkeypatch.setattr(bt.os, "name", "posix")
        monkeypatch.setattr(bt.sys, "platform", "linux")
        monkeypatch.setattr(bt.os, "geteuid", lambda: 1000, raising=False)
        monkeypatch.setattr(bt, "_apparmor_restricts_unprivileged_userns", lambda: True)
        monkeypatch.setattr(bt, "_unprivileged_userns_probe_fails", lambda: False)

        assert (
            bt._browser_sandbox_bypass_reason()
            == "AppArmor user namespace restrictions detected"
        )

    def test_failed_unshare_probe_requires_sandbox_bypass(self, monkeypatch):
        monkeypatch.setattr(bt.os, "name", "posix")
        monkeypatch.setattr(bt.sys, "platform", "linux")
        monkeypatch.setattr(bt.os, "geteuid", lambda: 1000, raising=False)
        monkeypatch.setattr(bt, "_apparmor_restricts_unprivileged_userns", lambda: False)
        monkeypatch.setattr(bt, "_unprivileged_userns_probe_fails", lambda: True)

        assert (
            bt._browser_sandbox_bypass_reason()
            == "unprivileged user namespace probe failed"
        )

    def test_unshare_probe_failure_detects_blocked_userns(self, monkeypatch):
        monkeypatch.setattr(bt.os, "name", "posix")
        monkeypatch.setattr(bt.sys, "platform", "linux")
        monkeypatch.setattr(bt.shutil, "which", lambda name: "/usr/bin/unshare")

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 1)

        monkeypatch.setattr(bt.subprocess, "run", fake_run)

        assert bt._unprivileged_userns_probe_fails() is True

    def test_unshare_probe_skips_non_linux(self, monkeypatch):
        monkeypatch.setattr(bt.os, "name", "nt")
        monkeypatch.setattr(bt.sys, "platform", "win32")
        monkeypatch.setattr(bt.shutil, "which", lambda name: "/usr/bin/unshare")

        assert bt._unprivileged_userns_probe_fails() is False

    def test_injects_browser_args_when_sandbox_bypass_needed(self, monkeypatch):
        browser_env = {}
        monkeypatch.setattr(
            bt,
            "_browser_sandbox_bypass_reason",
            lambda: "unprivileged user namespace probe failed",
        )

        reason = bt._inject_browser_sandbox_args_if_needed(browser_env)

        assert reason == "unprivileged user namespace probe failed"
        assert browser_env["AGENT_BROWSER_ARGS"] == "--no-sandbox,--disable-dev-shm-usage"

    @pytest.mark.parametrize(
        "browser_env",
        [
            {"AGENT_BROWSER_ARGS": "--custom-flag"},
            {"AGENT_BROWSER_CHROME_FLAGS": "--custom-legacy-flag"},
        ],
    )
    def test_preserves_user_browser_args(self, monkeypatch, browser_env):
        monkeypatch.setattr(bt, "_browser_sandbox_bypass_reason", lambda: "running as root")
        original = dict(browser_env)

        reason = bt._inject_browser_sandbox_args_if_needed(browser_env)

        assert reason is None
        assert browser_env == original

    def test_true_when_chromium_dir_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()
        assert bt._chromium_installed() is True

    def test_true_when_headless_shell_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium_headless_shell-1208").mkdir()
        assert bt._chromium_installed() is True




    def test_result_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()
        assert bt._chromium_installed() is True
        # Delete after first call — cached True should still return True.
        (tmp_path / "chromium-1208").rmdir()
        assert bt._chromium_installed() is True


class TestCheckBrowserRequirementsChromium:

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


