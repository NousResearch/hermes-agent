"""Tests for fnm-managed Node discovery in browser_tool.py."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

import tools.browser_tool as _bt
from tools.browser_tool import (
    _discover_fnm_node_dirs,
    _find_agent_browser,
    _run_browser_command,
    cleanup_all_browsers,
)


@pytest.fixture(autouse=True)
def _clear_browser_caches():
    _discover_fnm_node_dirs.cache_clear()
    _bt._cached_agent_browser = None
    _bt._agent_browser_resolved = False
    yield
    _discover_fnm_node_dirs.cache_clear()
    _bt._cached_agent_browser = None
    _bt._agent_browser_resolved = False


def _installation(root: Path, version: str) -> Path:
    path = root / "node-versions" / version / "installation"
    path.mkdir(parents=True)
    return path


def test_discovers_configured_fnm_versions_newest_first(tmp_path, monkeypatch):
    fnm_root = tmp_path / "fnm"
    node20 = _installation(fnm_root, "v20.19.0")
    node24 = _installation(fnm_root, "v24.12.0")
    monkeypatch.setenv("FNM_DIR", str(fnm_root))
    monkeypatch.delenv("APPDATA", raising=False)

    assert _discover_fnm_node_dirs() == (str(node24), str(node20))


def test_discovers_windows_appdata_fnm_root(tmp_path, monkeypatch):
    appdata = tmp_path / "Roaming"
    installation = _installation(appdata / "fnm", "v24.12.0")
    monkeypatch.delenv("FNM_DIR", raising=False)
    monkeypatch.setenv("APPDATA", str(appdata))

    assert _discover_fnm_node_dirs() == (str(installation),)


def test_find_agent_browser_uses_windows_appdata_fnm_installation(tmp_path, monkeypatch):
    appdata = tmp_path / "Roaming"
    installation = _installation(appdata / "fnm", "v24.12.0")
    expected = str(installation / "agent-browser.cmd")
    monkeypatch.delenv("FNM_DIR", raising=False)
    monkeypatch.setenv("APPDATA", str(appdata))

    def fake_which(command, path=None):
        if command == "agent-browser" and path and str(installation) in path:
            return expected
        return None

    with patch("shutil.which", side_effect=fake_which), patch(
        "tools.browser_tool.agent_browser_runnable", return_value=True
    ):
        assert _find_agent_browser() == expected


def test_cleanup_all_browsers_refreshes_cached_fnm_dirs(tmp_path, monkeypatch):
    fnm_root = tmp_path / "fnm"
    monkeypatch.setenv("FNM_DIR", str(fnm_root))
    monkeypatch.delenv("APPDATA", raising=False)

    assert _discover_fnm_node_dirs() == ()
    installation = _installation(fnm_root, "v24.12.0")
    assert _discover_fnm_node_dirs() == ()

    cleanup_all_browsers()

    assert _discover_fnm_node_dirs() == (str(installation),)


def test_run_browser_command_includes_fnm_installation_in_path(tmp_path):
    installation = _installation(tmp_path / "fnm", "v24.12.0")
    captured_env = {}
    mock_proc = MagicMock(returncode=0)
    mock_proc.wait.return_value = 0

    def capture_popen(_cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return mock_proc

    fake_session = {
        "session_name": "test-session",
        "session_id": "test-id",
        "cdp_url": None,
    }
    fake_json = json.dumps({"success": True})

    with patch(
        "tools.browser_tool._find_agent_browser",
        return_value=str(installation / "agent-browser.cmd"),
    ), patch("tools.browser_tool._chromium_installed", return_value=True), patch(
        "tools.browser_tool._get_session_info", return_value=fake_session
    ), patch(
        "tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)
    ), patch(
        "tools.browser_tool._discover_fnm_node_dirs", return_value=[str(installation)]
    ), patch(
        "tools.browser_tool._discover_homebrew_node_dirs", return_value=[]
    ), patch(
        "subprocess.Popen", side_effect=capture_popen
    ), patch(
        "os.open", return_value=99
    ), patch(
        "os.close"
    ), patch(
        "tools.interrupt.is_interrupted", return_value=False
    ), patch.dict(
        os.environ,
        {
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "HERMES_HOME": str(tmp_path / "hermes-home"),
        },
        clear=True,
    ), patch(
        "builtins.open", mock_open(read_data=fake_json)
    ):
        result = _run_browser_command("test-task", "navigate", ["https://example.com"])

    assert result["success"] is True
    assert str(installation) in captured_env["PATH"].split(os.pathsep)


def test_missing_or_unreadable_fnm_root_is_ignored(monkeypatch):
    monkeypatch.setenv("FNM_DIR", os.path.join("missing", "fnm"))
    monkeypatch.delenv("APPDATA", raising=False)

    with patch("os.listdir", side_effect=OSError("permission denied")):
        assert _discover_fnm_node_dirs() == ()
