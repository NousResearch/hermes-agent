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
from tools import browser_tool_session as bt_session
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


    def test_true_when_macos_user_app_bundle_present(self, monkeypatch):
        # macOS app bundles are not on PATH, so which() must not be what saves us.
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.sys, "platform", "darwin")
        monkeypatch.setattr(bt_install.shutil, "which", lambda _name: None)
        monkeypatch.setenv("HOME", "/Users/alice")

        user_chrome = os.path.join(
            "/Users/alice", "Applications", "Google Chrome.app", "Contents", "MacOS", "Google Chrome",
        )
        monkeypatch.setattr(bt_install.os.path, "isfile", lambda path: path == user_chrome)
        monkeypatch.setattr(bt_install.os.path, "isdir", lambda _path: False)

        assert bt_install._detect_system_chromium_executable() == user_chrome
        assert bt_install._chromium_installed() is True

    def test_system_app_bundle_wins_over_user_bundle(self, monkeypatch):
        """Ordering contract: /Applications is probed before ~/Applications."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.sys, "platform", "darwin")
        monkeypatch.setattr(bt_install.shutil, "which", lambda _name: None)
        monkeypatch.setenv("HOME", "/Users/alice")

        system_chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        user_chrome = os.path.join(
            "/Users/alice", "Applications", "Google Chrome.app", "Contents", "MacOS", "Google Chrome",
        )
        monkeypatch.setattr(bt_install.os.path, "isfile", lambda path: path in {system_chrome, user_chrome})

        assert bt_install._detect_system_chromium_executable() == system_chrome

    def test_command_env_sets_macos_app_bundle_executable(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.sys, "platform", "darwin")
        monkeypatch.setattr(bt_install.shutil, "which", lambda _name: None)
        monkeypatch.setenv("HOME", "/Users/alice")

        user_chrome = os.path.join(
            "/Users/alice", "Applications", "Google Chrome.app", "Contents", "MacOS", "Google Chrome",
        )
        monkeypatch.setattr(bt_install.os.path, "isfile", lambda path: path == user_chrome)
        monkeypatch.setattr(bt_install.os.path, "isdir", lambda _path: False)

        env = bt_session._agent_browser_command_env("/tmp/hermes-browser-socket")

        assert env["AGENT_BROWSER_SOCKET_DIR"] == "/tmp/hermes-browser-socket"
        assert env["AGENT_BROWSER_EXECUTABLE_PATH"] == user_chrome

    def test_command_env_excludes_unrelated_credentials(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-agent-browser")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_PROFILE", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/tmp/browser-credentials")
        monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/tmp/browser-token")
        monkeypatch.setenv("BROWSERBASE_API_KEY", "allowed-browser-key")

        env = bt_session._agent_browser_command_env("/tmp/hermes-browser-socket")

        assert "ANTHROPIC_API_KEY" not in env
        assert not {
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
            "AWS_SHARED_CREDENTIALS_FILE", "AWS_WEB_IDENTITY_TOKEN_FILE",
        } & env.keys()
        # The documented browser-backend passthrough must still survive the scrub.
        assert env["BROWSERBASE_API_KEY"] == "allowed-browser-key"

    def test_popen_receives_the_scrubbed_env_verbatim(self, monkeypatch, tmp_path):
        """Launch-level guard: the spawn helper must pass our env through, never os.environ."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-agent-browser")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "browser-must-not-inherit")

        launched_envs = []

        class FakeProcess:
            returncode = 0

            def wait(self, timeout=None):
                return 0

        def fake_popen(_argv, *, stdout, stderr, stdin, env, **_kwargs):
            launched_envs.append(env)
            return FakeProcess()

        monkeypatch.setattr(bt_session.subprocess, "Popen", fake_popen)

        env = bt_session._agent_browser_command_env(str(tmp_path))
        bt_session._popen_agent_browser(["agent-browser", "snapshot"], env, str(tmp_path), "tag")

        assert launched_envs
        assert all("ANTHROPIC_API_KEY" not in e for e in launched_envs)
        assert all(
            not {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"} & e.keys() for e in launched_envs
        )

    def test_command_env_skips_browser_executable_for_cloud(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(
            bt_install, "_detect_system_chromium_executable", lambda: "/Applications/Chrome")

        env = bt_session._agent_browser_command_env(
            "/tmp/hermes-browser-socket", include_browser_executable=False)

        assert "AGENT_BROWSER_EXECUTABLE_PATH" not in env

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
