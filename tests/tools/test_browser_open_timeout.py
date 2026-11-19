"""Tests for the Chromium first-launch timeout floor and the sandbox-bypass host probe."""

import pytest

from tools import browser_launcher, browser_session_cloud


@pytest.fixture(autouse=True)
def _fresh_cache():
    browser_session_cloud.reset_caches()
    yield
    browser_session_cloud.reset_caches()


class TestOpenCommandTimeout:
    def test_first_open_uses_longer_floor(self, monkeypatch):
        monkeypatch.setattr(browser_session_cloud, "command_timeout", lambda: 30)
        assert browser_launcher._launch_timeout(None) == browser_launcher._MIN_FIRST_LAUNCH_TIMEOUT_S

    def test_respects_config_above_floor(self, monkeypatch):
        monkeypatch.setattr(browser_session_cloud, "command_timeout", lambda: 180)
        assert browser_launcher._launch_timeout(None) == 180


class TestSandboxBypass:

    def test_apparmor_userns_triggers_bypass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(browser_launcher, "running_in_docker", lambda: False)
        # Non-root, so only the AppArmor sysctl can answer True.
        monkeypatch.setattr(browser_launcher.os, "geteuid", lambda: 1000, raising=False)
        sysctl = tmp_path / "apparmor_restrict_unprivileged_userns"
        sysctl.write_text("1\n", encoding="utf-8")

        import builtins

        real_open = builtins.open

        def _open(path, *args, **kwargs):
            if "apparmor_restrict_unprivileged_userns" in str(path):
                return real_open(sysctl, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _open)
        assert browser_launcher.needs_chromium_sandbox_bypass() is True
