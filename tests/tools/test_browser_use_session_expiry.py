"""Regression coverage for provider-authoritative cloud browser expiry."""

from unittest.mock import Mock

import tools.browser_tool as browser_tool
from plugins.browser.browser_use import provider as browser_use_provider
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
