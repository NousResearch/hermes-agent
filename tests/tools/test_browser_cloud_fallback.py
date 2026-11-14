"""Tests for cloud browser provider runtime fallback to local Chromium.

Covers the fallback logic in ``browser_session.get_session_info()`` when a cloud provider
is configured but fails at runtime (issue #10883).
"""
from types import SimpleNamespace
from unittest.mock import Mock

from tools.bot_desktop import placement
from tools import browser_launcher
from tools import browser_session as bs
from tools import browser_session_cloud as bs_cloud

_LOCAL_CDP = "ws://127.0.0.1:9333/devtools/browser/local"


def _reset_session_state(monkeypatch):
    """Clear caches so each test starts fresh; the local browser is a stub launcher browser."""
    monkeypatch.setattr(bs, "_active_sessions", {})
    monkeypatch.setattr("tools.browser_session_lifecycle.start_cleanup_thread", lambda: None)
    monkeypatch.setattr("tools.browser_session_lifecycle.update_session_activity", lambda t: None)
    monkeypatch.setattr("tools.browser_session_cdp.ensure_cdp_supervisor", lambda task_id: None)
    monkeypatch.setattr("tools.bot_desktop.runtime.tool_placement", lambda: placement.GATEWAY)  # host browser
    monkeypatch.setattr(browser_launcher, "launch_chromium",
                        lambda key, **kw: (SimpleNamespace(cdp_url=_LOCAL_CDP, pid=4242), None))


class TestCloudProviderRuntimeFallback:
    """Tests for get_session_info cloud -> local fallback."""

    def test_cloud_failure_falls_back_to_local(self, monkeypatch):
        """When cloud provider.create_session raises, fall back to local."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.side_effect = RuntimeError("401 Unauthorized")
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: None)

        session = bs.get_session_info("task-1")

        assert session["fallback_from_cloud"] is True
        assert "401 Unauthorized" in session["fallback_reason"]
        assert session["fallback_provider"] == "Mock"
        assert session["features"]["local"] is True
        assert session["cdp_url"] == _LOCAL_CDP


    def test_no_provider_uses_local_directly(self, monkeypatch):
        """When no cloud provider is configured, local mode is used with no fallback markers."""
        _reset_session_state(monkeypatch)

        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: None)
        monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: None)

        session = bs.get_session_info("task-4")

        assert session["features"]["local"] is True
        assert "fallback_from_cloud" not in session


    def test_cloud_returns_invalid_session_triggers_fallback(self, monkeypatch):
        """Cloud provider returning None or empty dict triggers fallback."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.return_value = None
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: None)

        session = bs.get_session_info("task-7")

        assert session["fallback_from_cloud"] is True
        assert "invalid session" in session["fallback_reason"]
