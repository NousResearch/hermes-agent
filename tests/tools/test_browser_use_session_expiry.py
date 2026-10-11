"""Regression coverage for provider-authoritative cloud browser expiry."""

import hashlib
from unittest.mock import Mock

from tools import browser_tool
from plugins.browser.browser_use import provider as browser_use_provider
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_session as bt_session
from tools import browser_tool_cloud as bt_cloud


def _isolate_browser_state(monkeypatch):
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_session_last_activity", {})
    monkeypatch.setattr("tools.browser_tool_lifecycle._start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr("tools.browser_tool_cdp._ensure_cdp_supervisor", lambda task_id: None)


def test_browser_use_preserves_provider_timeout(monkeypatch):
    provider = browser_use_provider.BrowserUseBrowserProvider()
    response = Mock(
        ok=True,
        headers={},
    )
    response.json.return_value = {
        "id": "browser-session-1",
        "cdpUrl": "ws://browser-use.example/devtools/browser/1",
        "timeoutAt": "2030-01-01T00:05:00Z",
    }

    monkeypatch.setattr(
        provider,
        "_get_config",
        lambda: {
            "api_key": "test-key",
            "base_url": "https://api.browser-use.example/api/v3",
            "managed_mode": False,
        },
    )
    monkeypatch.setattr(browser_use_provider.requests, "post", Mock(return_value=response))

    session = provider.create_session("task-1")

    assert session["expires_at"] == "2030-01-01T00:05:00Z"


def test_uuid_browser_session_uses_compact_socket_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(bt_session._bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt_session.os, "makedirs", Mock())
    monkeypatch.setattr(bt_session._lifecycle, "_write_owner_pid", Mock())
    session_name = "hermes_12345678-1234-5678-1234-567812345678_ab12cd34"

    socket_dir = bt_session._prepare_session_socket_dir(session_name)

    assert len(socket_dir.rsplit("/", 1)[-1]) == len("agent-browser-") + 16
    assert socket_dir.endswith("agent-browser-" + hashlib.sha256(
        session_name.encode("utf-8")).hexdigest()[:16])


def test_orphan_reaper_recovers_session_name_from_compact_socket_dir(monkeypatch, tmp_path):
    session_name = "hermes_12345678-1234-5678-1234-567812345678_ab12cd34"
    socket_dir = tmp_path / ("agent-browser-" + hashlib.sha256(
        session_name.encode("utf-8")).hexdigest()[:16])
    socket_dir.mkdir()
    (socket_dir / f"{session_name}.owner_pid").write_text("123", encoding="utf-8")

    monkeypatch.setattr(bt_lifecycle._bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt_lifecycle._bt, "_REAL_PROFILE_SESSION", "real-profile")
    monkeypatch.setattr(bt_lifecycle._bt, "_active_sessions", {})
    monkeypatch.setattr(bt_lifecycle, "_best_effort", lambda label, fn: None)
    reap = Mock(return_value=False)
    monkeypatch.setattr(bt_lifecycle, "_reap_socket_dir", reap)

    bt_lifecycle._reap_orphaned_browser_sessions()

    reap.assert_called_once_with(str(socket_dir), session_name, {"real-profile"})


def test_orphan_reaper_recovers_session_name_from_pid_marker(monkeypatch, tmp_path):
    session_name = "hermes_12345678-1234-5678-1234-567812345678_ab12cd34"
    socket_dir = tmp_path / ("agent-browser-" + hashlib.sha256(
        session_name.encode("utf-8")).hexdigest()[:16])
    socket_dir.mkdir()
    (socket_dir / f"{session_name}.pid").write_text("123", encoding="utf-8")

    monkeypatch.setattr(bt_lifecycle._bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt_lifecycle._bt, "_REAL_PROFILE_SESSION", "real-profile")
    monkeypatch.setattr(bt_lifecycle._bt, "_active_sessions", {})
    monkeypatch.setattr(bt_lifecycle, "_best_effort", lambda label, fn: None)
    reap = Mock(return_value=False)
    monkeypatch.setattr(bt_lifecycle, "_reap_socket_dir", reap)

    bt_lifecycle._reap_orphaned_browser_sessions()

    reap.assert_called_once_with(str(socket_dir), session_name, {"real-profile"})


def test_release_session_resources_uses_compact_socket_dir(monkeypatch, tmp_path):
    session_name = "hermes_12345678-1234-5678-1234-567812345678_ab12cd34"
    socket_dir = tmp_path / ("agent-browser-" + hashlib.sha256(
        session_name.encode("utf-8")).hexdigest()[:16])
    socket_dir.mkdir()

    monkeypatch.setattr(bt_lifecycle._bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt_lifecycle, "_forget_session_tracking", Mock())
    kill = Mock(return_value=True)
    monkeypatch.setattr(bt_lifecycle, "_kill_verified_daemon", kill)

    bt_lifecycle._release_session_resources("task-1", {"session_name": session_name})

    kill.assert_called_once_with(str(socket_dir), session_name)
    assert not socket_dir.exists()


def test_live_cloud_session_is_reused(monkeypatch):
    _isolate_browser_state(monkeypatch)
    existing = {
        "session_name": "existing",
        "bb_session_id": "browser-session-1",
        "cdp_url": "ws://browser-use.example/devtools/browser/1",
        "expires_at": "2999-01-01T00:05:00Z",
    }
    browser_tool._active_sessions["task-1"] = existing
    provider = Mock()
    monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)

    session = bt_session._get_session_info("task-1")

    assert session is existing
    provider.create_session.assert_not_called()


def test_expired_cloud_session_is_replaced_without_reusing_dead_cdp(monkeypatch):
    _isolate_browser_state(monkeypatch)
    browser_tool._active_sessions["task-1"] = {
        "session_name": "expired",
        "bb_session_id": "browser-session-old",
        "cdp_url": "ws://browser-use.example/devtools/browser/old",
        "expires_at": "2020-01-01T00:05:00Z",
    }
    browser_tool._session_last_activity["task-1"] = 1.0

    provider = Mock()
    provider.create_session.return_value = {
        "session_name": "replacement",
        "bb_session_id": "browser-session-new",
        "cdp_url": "ws://browser-use.example/devtools/browser/new",
        "expires_at": "2999-01-01T00:05:00Z",
    }
    monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_tool_cdp._stop_cdp_supervisor", Mock())
    monkeypatch.setattr(browser_tool, "_maybe_stop_recording", Mock())
    monkeypatch.setattr(bt_session, "_run_browser_command", Mock())
    monkeypatch.setattr(browser_tool.os.path, "exists", lambda path: False)

    session = bt_session._get_session_info("task-1")

    assert session["bb_session_id"] == "browser-session-new"
    assert browser_tool._active_sessions["task-1"] is session
    assert "task-1" in browser_tool._session_last_activity
    provider.close_session.assert_called_once_with("browser-session-old")
    provider.create_session.assert_called_once_with("task-1")
    bt_session._run_browser_command.assert_not_called()
