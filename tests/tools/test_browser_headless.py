"""Tests for headless-Chrome provisioning for the Browser Use backend.

Regression guard for the "browser_exec fails on a headless VPS with
'chrome-not-running'" bug — the browser-use harness's local mode needs a GUI
Chrome, so on Linux with no display Hermes must provision a detached headless
Chrome and point the harness at it via BU_CDP_URL.
"""

import pytest

from tools import browser_headless as bh


class TestIsHeadlessLinux:
    def test_true_on_linux_without_display(self, monkeypatch):
        monkeypatch.setattr(bh.sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert bh.is_headless_linux() is True

    def test_false_with_display(self, monkeypatch):
        monkeypatch.setattr(bh.sys, "platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        assert bh.is_headless_linux() is False

    def test_false_on_non_linux(self, monkeypatch):
        monkeypatch.setattr(bh.sys, "platform", "darwin")
        monkeypatch.delenv("DISPLAY", raising=False)
        assert bh.is_headless_linux() is False


class TestFindChromiumBinary:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        fake = tmp_path / "chrome"
        fake.write_text("")
        monkeypatch.setenv("BH_CHROME_PATH", str(fake))
        assert bh._find_chromium_binary() == str(fake)

    def test_system_chrome_in_path(self, monkeypatch):
        for key in ("AGENT_BROWSER_EXECUTABLE_PATH", "BH_CHROME_PATH", "CHROME_PATH"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            bh.shutil,
            "which",
            lambda name: "/usr/bin/google-chrome" if name == "google-chrome" else None,
        )
        assert bh._find_chromium_binary() == "/usr/bin/google-chrome"

    def test_playwright_cache_fallback(self, monkeypatch, tmp_path):
        for key in ("AGENT_BROWSER_EXECUTABLE_PATH", "BH_CHROME_PATH", "CHROME_PATH"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(bh.shutil, "which", lambda name: None)
        build = tmp_path / "chromium-1208" / "chrome-linux64"
        build.mkdir(parents=True)
        binary = build / "chrome"
        binary.write_text("")
        # _find_chromium_binary imports _chromium_search_roots lazily from
        # browser_tool, so patch it at the source.
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert bh._find_chromium_binary() == str(binary)


class TestEnsureHeadlessChrome:
    def test_noop_when_not_headless(self, monkeypatch):
        monkeypatch.setattr(bh, "is_headless_linux", lambda: False)
        env = {}
        assert bh.ensure_headless_chrome(env) is None
        assert "BU_CDP_URL" not in env

    def test_noop_when_cdp_url_already_set(self, monkeypatch):
        monkeypatch.setattr(bh, "is_headless_linux", lambda: True)
        env = {"BU_CDP_URL": "http://127.0.0.1:9222"}
        assert bh.ensure_headless_chrome(env) is None
        assert env["BU_CDP_URL"] == "http://127.0.0.1:9222"

    def test_noop_when_cloud_autospawn_set(self, monkeypatch):
        monkeypatch.setattr(bh, "is_headless_linux", lambda: True)
        env = {"BU_AUTOSPAWN": "1"}
        assert bh.ensure_headless_chrome(env) is None
        assert "BU_CDP_URL" not in env

    def test_reuses_running_chrome_without_relaunch(self, monkeypatch):
        monkeypatch.setattr(bh, "is_headless_linux", lambda: True)
        monkeypatch.setattr(bh, "_cdp_ready", lambda: True)
        launched = []
        monkeypatch.setattr(bh, "_launch", lambda binary: launched.append(binary) or True)
        env = {}
        assert bh.ensure_headless_chrome(env) is None
        assert env["BU_CDP_URL"] == bh._CDP_URL
        assert launched == []  # nothing re-launched

    def test_launches_and_sets_env_when_cdp_comes_up(self, monkeypatch):
        monkeypatch.setattr(bh, "is_headless_linux", lambda: True)
        monkeypatch.setattr(bh, "_find_chromium_binary", lambda: "/usr/bin/chromium")
        states = {"ready": False}

        def launch_then_ready(binary):
            states["ready"] = True
            return True

        monkeypatch.setattr(bh, "_launch", launch_then_ready)
        monkeypatch.setattr(bh, "_cdp_ready", lambda: states["ready"])

        env = {}
        assert bh.ensure_headless_chrome(env) is None
        assert env["BU_CDP_URL"] == bh._CDP_URL

    def test_error_when_no_chromium_and_no_autoinstall(self, monkeypatch):
        monkeypatch.setattr(bh, "is_headless_linux", lambda: True)
        monkeypatch.setattr(bh, "_cdp_ready", lambda: False)
        monkeypatch.setattr(bh, "_find_chromium_binary", lambda: None)

        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_chromium_installed", lambda: False)
        monkeypatch.setattr(bt, "_maybe_autoinstall_chromium", lambda: False)

        err = bh.ensure_headless_chrome({})
        assert err is not None
        assert "Chrome/Chromium" in err

    def test_error_when_launch_fails(self, monkeypatch):
        monkeypatch.setattr(bh, "is_headless_linux", lambda: True)
        monkeypatch.setattr(bh, "_cdp_ready", lambda: False)
        monkeypatch.setattr(bh, "_find_chromium_binary", lambda: "/usr/bin/chromium")
        monkeypatch.setattr(bh, "_launch", lambda binary: False)

        err = bh.ensure_headless_chrome({})
        assert err is not None
        assert "failed to launch" in err
