"""Tests for Chromium-presence detection in browser_tool.

Regression guard for the "browser tool advertised but Chromium missing"
class of bug — where ``agent-browser`` CLI is discoverable but no
Chromium build is on disk, causing every browser_* tool call to hang
for the full command timeout before surfacing a useless error.
"""

import os

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
    def test_true_when_plain_chromium_on_path(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(
            bt.shutil,
            "which",
            lambda name: "/usr/bin/chromium" if name == "chromium" else None,
        )

        assert bt._chromium_installed() is True

    def test_true_when_chromium_dir_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()
        assert bt._chromium_installed() is True

    def test_true_when_headless_shell_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium_headless_shell-1208").mkdir()
        assert bt._chromium_installed() is True




    def test_true_when_macos_user_app_bundle_present(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt.sys, "platform", "darwin")
        monkeypatch.setattr(bt.shutil, "which", lambda _name: None)
        monkeypatch.setenv("HOME", "/Users/alice")

        user_chrome = os.path.join(
            "/Users/alice",
            "Applications",
            "Google Chrome.app",
            "Contents",
            "MacOS",
            "Google Chrome",
        )
        monkeypatch.setattr(bt.os.path, "isfile", lambda path: path == user_chrome)
        monkeypatch.setattr(bt.os.path, "isdir", lambda _path: False)

        assert bt._detect_system_chromium_executable() == user_chrome
        assert bt._chromium_installed() is True

    def test_browser_subprocess_env_sets_macos_app_bundle_executable(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt.sys, "platform", "darwin")
        monkeypatch.setattr(bt.shutil, "which", lambda _name: None)
        monkeypatch.setenv("HOME", "/Users/alice")

        user_chrome = os.path.join(
            "/Users/alice",
            "Applications",
            "Google Chrome.app",
            "Contents",
            "MacOS",
            "Google Chrome",
        )
        monkeypatch.setattr(bt.os.path, "isfile", lambda path: path == user_chrome)
        monkeypatch.setattr(bt.os.path, "isdir", lambda _path: False)

        env = bt._browser_subprocess_env("/tmp/hermes-browser-socket")

        assert env["AGENT_BROWSER_SOCKET_DIR"] == "/tmp/hermes-browser-socket"
        assert env["AGENT_BROWSER_EXECUTABLE_PATH"] == user_chrome

    def test_browser_subprocess_env_excludes_unrelated_credentials(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-agent-browser")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_PROFILE", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/tmp/browser-credentials")
        monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/tmp/browser-token")
        monkeypatch.setenv("BROWSERBASE_API_KEY", "allowed-browser-key")

        env = bt._browser_subprocess_env("/tmp/hermes-browser-socket")

        assert "ANTHROPIC_API_KEY" not in env
        assert not {
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
            "AWS_SHARED_CREDENTIALS_FILE", "AWS_WEB_IDENTITY_TOKEN_FILE",
        } & env.keys()
        assert env["BROWSERBASE_API_KEY"] == "allowed-browser-key"

    def test_browser_popen_paths_exclude_unrelated_credentials(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-agent-browser")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "browser-must-not-inherit")
        monkeypatch.setenv("AWS_PROFILE", "browser-must-not-inherit")
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _command: False)
        monkeypatch.setattr(bt, "_chromium_installed", lambda: True)

        launched_envs = []

        class FakeProcess:
            returncode = 0

            def wait(self, timeout=None):
                return 0

        def fake_popen(_command, *, stdout, env, **_kwargs):
            launched_envs.append(env)
            os.write(stdout, b'{"success": true, "data": {"result": "https://example.test"}}\n')
            return FakeProcess()

        monkeypatch.setattr(bt.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(bt, "_is_local_mode", lambda: False)
        monkeypatch.setattr(
            bt,
            "_get_session_info",
            lambda _task_id: {"session_name": "cloud-session", "cdp_url": "ws://example.test"},
        )

        assert bt._run_browser_command("task", "snapshot", timeout=1)["success"] is True

        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *_args, **_kwargs: {"success": True, "data": {"result": "https://example.test"}},
        )
        assert bt._run_chrome_fallback_command("task", "snapshot", [], timeout=1)["success"] is True

        assert launched_envs
        assert all("ANTHROPIC_API_KEY" not in env for env in launched_envs)
        assert all(
            not {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"} & env.keys()
            for env in launched_envs
        )

    def test_browser_subprocess_env_skips_browser_executable_for_cloud(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt, "_detect_system_chromium_executable", lambda: "/Applications/Chrome")

        env = bt._browser_subprocess_env(
            "/tmp/hermes-browser-socket",
            include_browser_executable=False,
        )

        assert "AGENT_BROWSER_EXECUTABLE_PATH" not in env

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
        monkeypatch.setattr(bt, "_find_agent_browser", lambda **_kw: "/usr/local/bin/agent-browser")
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
        monkeypatch.setattr(bt, "_find_agent_browser", lambda **_kw: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: FakeProvider())
        # Point chromium search at an empty dir — should not matter for cloud.
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        assert bt.check_browser_requirements() is True

    def test_startup_check_uses_lightweight_agent_browser_lookup(self, monkeypatch, tmp_path):
        seen = []

        def fake_find_agent_browser(**kwargs):
            seen.append(kwargs)
            return "/usr/local/bin/agent-browser"

        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_find_agent_browser", fake_find_agent_browser)
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()

        assert bt.check_browser_requirements() is True
        assert seen == [{"validate": False}]

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
