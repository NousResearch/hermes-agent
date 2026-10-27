"""Regression coverage for provider-authoritative cloud browser expiry."""

from unittest.mock import Mock

from plugins.browser.browser_use import provider as browser_use_provider
from tools import browser_session as bs
from tools import browser_session_cloud as bs_cloud


def _isolate_browser_state(monkeypatch):
    monkeypatch.setattr(bs, "_active_sessions", {})
    monkeypatch.setattr(bs, "_session_last_activity", {})
    monkeypatch.setattr(bs, "_session_owner_homes", {})
    monkeypatch.setattr("tools.browser_session_lifecycle.start_cleanup_thread", lambda: None)
    monkeypatch.setattr("tools.browser_session_cdp.ensure_cdp_supervisor", lambda task_id: None)


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
    bs._active_sessions["task-1"] = existing
    provider = Mock()
    monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: provider)

    session = bs.get_session_info("task-1")

    assert session is existing
    provider.create_session.assert_not_called()


def test_expired_cloud_session_is_replaced_without_reusing_dead_cdp(monkeypatch):
    _isolate_browser_state(monkeypatch)
    bs._active_sessions["task-1"] = {
        "session_name": "expired",
        "bb_session_id": "browser-session-old",
        "cdp_url": "ws://browser-use.example/devtools/browser/old",
        "expires_at": "2020-01-01T00:05:00Z",
    }
    bs._session_last_activity["task-1"] = 1.0

    events = []
    provider = Mock()
    provider.close_session.side_effect = lambda sid: events.append(("close", sid))
    provider.create_session.side_effect = lambda task_id: events.append(("create", task_id)) or {
        "session_name": "replacement",
        "bb_session_id": "browser-session-new",
        "cdp_url": "ws://browser-use.example/devtools/browser/new",
        "expires_at": "2999-01-01T00:05:00Z",
    }
    monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: provider)
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_session_cdp.stop_cdp_supervisor", lambda key: events.append(("stop", key)))

    session = bs.get_session_info("task-1")

    assert session["bb_session_id"] == "browser-session-new"
    assert session["cdp_url"] == "ws://browser-use.example/devtools/browser/new"
    assert bs._active_sessions["task-1"] is session
    assert "task-1" in bs._session_last_activity
    # The supervisor WebSocket on the dead endpoint closes first, then the provider session, and only
    # then is the replacement created: nothing keeps driving the expired CDP URL.
    assert events == [("stop", "task-1"), ("close", "browser-session-old"), ("create", "task-1")]
