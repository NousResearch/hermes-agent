"""Tests for macOS Homebrew PATH discovery in browser_tool.py."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

from tools.browser_tool import (
    _discover_homebrew_node_dirs,
    _find_agent_browser,
    _run_browser_command,
    _SANE_PATH,
    check_browser_requirements,
)
import tools.browser_tool as _bt


@pytest.fixture(autouse=True)
def _clear_browser_caches():
    """Clear lru_cache and manual caches between tests."""
    _discover_homebrew_node_dirs.cache_clear()
    _bt._cached_agent_browser = None
    _bt._agent_browser_resolved = False
    yield
    _discover_homebrew_node_dirs.cache_clear()
    _bt._cached_agent_browser = None
    _bt._agent_browser_resolved = False


class TestSanePath:
    """Verify _SANE_PATH includes fallback directories used by browser_tool."""

    def test_includes_termux_bin(self):
        assert "/data/data/com.termux/files/usr/bin" in _SANE_PATH.split(os.pathsep)


    def test_includes_standard_dirs(self):
        path_parts = _SANE_PATH.split(os.pathsep)
        assert "/usr/local/bin" in path_parts
        assert "/usr/bin" in path_parts
        assert "/bin" in path_parts


class TestDiscoverHomebrewNodeDirs:
    """Tests for _discover_homebrew_node_dirs()."""

    def test_returns_empty_when_no_homebrew(self):
        """Non-macOS systems without /opt/homebrew/opt should return empty."""
        with patch("os.path.isdir", return_value=False):
            assert _discover_homebrew_node_dirs() == ()


    def test_excludes_plain_node(self):
        """'node' (unversioned) should be excluded — covered by /opt/homebrew/bin."""
        with patch("os.path.isdir", return_value=True), \
             patch("os.listdir", return_value=["node"]):
            result = _discover_homebrew_node_dirs()
        assert result == ()

    def test_handles_oserror_gracefully(self):
        """Should return empty list if listdir raises OSError."""
        with patch("os.path.isdir", return_value=True), \
             patch("os.listdir", side_effect=OSError("Permission denied")):
            assert _discover_homebrew_node_dirs() == ()


class TestFindAgentBrowser:
    """Tests for _find_agent_browser() Homebrew path search."""

    def test_finds_in_current_path(self):
        """Should return result from shutil.which if available on current PATH."""
        with patch("shutil.which", return_value="/usr/local/bin/agent-browser"), \
             patch("tools.browser_tool.agent_browser_runnable", return_value=True):
            assert _find_agent_browser() == "/usr/local/bin/agent-browser"


    def test_raises_when_not_found(self):
        """Should raise FileNotFoundError when nothing works."""
        original_path_exists = Path.exists

        def mock_path_exists(self):
            if "node_modules" in str(self) and "agent-browser" in str(self):
                return False
            return original_path_exists(self)

        with patch("shutil.which", return_value=None), \
             patch("os.path.isdir", return_value=False), \
             patch.object(Path, "exists", mock_path_exists), \
             patch(
                 "tools.browser_tool._discover_homebrew_node_dirs",
                 return_value=[],
             ):
            with pytest.raises(FileNotFoundError, match="agent-browser CLI not found"):
                _find_agent_browser()


class TestBrowserRequirements:
    def test_cdp_override_does_not_require_agent_browser_cli(self, monkeypatch):
        monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222/devtools/browser/test")
        monkeypatch.setattr("tools.browser_tool._is_camofox_mode", lambda: False)
        monkeypatch.setattr("tools.browser_tool._find_agent_browser", lambda: (_ for _ in ()).throw(FileNotFoundError("not found")))

        assert check_browser_requirements() is True

    def test_termux_requires_real_agent_browser_install_not_npx_fallback(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.setattr("tools.browser_tool._is_camofox_mode", lambda: False)
        monkeypatch.setattr("tools.browser_tool._get_cloud_provider", lambda: None)
        monkeypatch.setattr("tools.browser_tool._find_agent_browser", lambda **_kw: "npx agent-browser")

        assert check_browser_requirements() is False


class TestRunBrowserCommandTermuxFallback:
    def test_termux_local_mode_rejects_bare_npx_fallback(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.setattr("tools.browser_tool._find_agent_browser", lambda **_kw: "npx agent-browser")
        monkeypatch.setattr("tools.browser_tool._get_cloud_provider", lambda: None)

        result = _run_browser_command("task-1", "navigate", ["https://example.com"])

        assert result["success"] is False
        assert "bare npx fallback" in result["error"]
        assert "agent-browser install" in result["error"]


class TestRunBrowserCommandPathConstruction:
    """Verify _run_browser_command() includes Homebrew node dirs in subprocess PATH."""

    def test_subprocess_preserves_executable_path_with_spaces(self, tmp_path):
        """A local agent-browser path containing spaces must stay one argv entry."""
        captured_cmd = None

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        def capture_popen(cmd, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            return mock_proc

        fake_session = {
            "session_name": "test-session",
            "session_id": "test-id",
            "cdp_url": None,
        }
        fake_json = json.dumps({"success": True})
        browser_path = "/Users/test/Library/Application Support/hermes/node_modules/.bin/agent-browser"
        hermes_home = str(tmp_path / "hermes-home")

        with patch("tools.browser_tool._find_agent_browser", return_value=browser_path), \
 patch("tools.browser_tool._chromium_installed", return_value=True), \
             patch("tools.browser_tool._get_session_info", return_value=fake_session), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._discover_homebrew_node_dirs", return_value=[]), \
             patch("hermes_constants.Path.home", return_value=tmp_path), \
             patch("subprocess.Popen", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.dict(
                 os.environ,
                 {
                     "PATH": "/usr/bin:/bin",
                     "HOME": "/home/test",
                     "HERMES_HOME": hermes_home,
                 },
                 clear=True,
             ):
            with patch("builtins.open", mock_open(read_data=fake_json)):
                _run_browser_command("test-task", "navigate", ["https://example.com"])

        assert captured_cmd is not None
        assert captured_cmd[0] == browser_path
        assert captured_cmd[1:5] == [
            "--session",
            "test-session",
            "--json",
            "navigate",
        ]


    def test_subprocess_path_includes_termux_fallback_dirs(self, tmp_path):
        """Termux fallback dirs should survive browser PATH rebuilding."""
        captured_env = {}

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        def capture_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        fake_session = {
            "session_name": "test-session",
            "session_id": "test-id",
            "cdp_url": None,
        }

        fake_json = json.dumps({"success": True})
        real_isdir = os.path.isdir

        def selective_isdir(path):
            if path in {
                "/data/data/com.termux/files/usr/bin",
                "/data/data/com.termux/files/usr/sbin",
            }:
                return True
            if path.startswith(str(tmp_path)):
                return True
            return real_isdir(path)

        with patch("tools.browser_tool._find_agent_browser", return_value="/usr/local/bin/agent-browser"), \
 patch("tools.browser_tool._chromium_installed", return_value=True), \
             patch("tools.browser_tool._get_session_info", return_value=fake_session), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._discover_homebrew_node_dirs", return_value=[]), \
             patch("os.path.isdir", side_effect=selective_isdir), \
             patch("subprocess.Popen", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.dict(os.environ, {"PATH": "/usr/bin:/bin", "HOME": "/home/test"}, clear=True):
            with patch("builtins.open", mock_open(read_data=fake_json)):
                _run_browser_command("test-task", "navigate", ["https://example.com"])

        result_path = captured_env.get("PATH", "")
        assert "/data/data/com.termux/files/usr/bin" in result_path
        assert "/data/data/com.termux/files/usr/sbin" in result_path
