"""Tests for Browserbase-specific warnings being suppressed in local browser mode.

When cloud_provider is local (bb_session_id is None), navigate results should
not contain Browserbase-specific upsell text in stealth_warning or
bot_detection_warning (issue #54197).
"""
import json
from unittest.mock import Mock

import pytest

import tools.browser_tool as browser_tool


def _mock_navigate_deps(monkeypatch, session_info, nav_result):
    """Mock all dependencies for browser_navigate to reach the warning logic."""
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda *a, **kw: session_info)
    call_count = [0]
    def mock_run_cmd(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return nav_result
        # Auto-snapshot call
        return {"success": True, "data": {"snapshot": "", "refs": {}}}
    monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)
    monkeypatch.setattr(browser_tool, "_normalize_url_for_request", lambda url: url)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_command_timeout", lambda: 30)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
    monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task_id, url: task_id)
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    monkeypatch.setattr(browser_tool, "_maybe_start_recording", lambda key: None)
    monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_tool, "_update_session_activity", lambda t: None)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)


def _make_local_session_info():
    """Session info for a local Chromium session (no Browserbase)."""
    return {
        "session_name": "h_local_test",
        "bb_session_id": None,
        "cdp_url": None,
        "features": {"local": True},
        "_first_nav": True,
    }


def _make_cloud_session_info(proxies=True):
    """Session info for a Browserbase cloud session."""
    return {
        "session_name": "hermes_test_abc",
        "bb_session_id": "bb_test_123",
        "cdp_url": "wss://connect.browserbase.com/test",
        "features": {
            "basic_stealth": True,
            "proxies": proxies,
            "advanced_stealth": False,
            "keep_alive": True,
            "custom_timeout": False,
        },
        "_first_nav": True,
    }


def _make_nav_result(title="Welcome", url="https://example.com"):
    """Successful browser navigation result."""
    return {
        "success": True,
        "data": {"title": title, "url": url},
    }


class TestLocalModeWarnings:
    """Verify that local browser sessions don't show Browserbase warnings."""

    def test_local_session_no_stealth_warning(self, monkeypatch):
        """Local session should not include stealth_warning about Browserbase."""
        session_info = _make_local_session_info()
        _mock_navigate_deps(monkeypatch, session_info, _make_nav_result())

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True
        assert "stealth_warning" not in result

    def test_local_session_no_browserbase_in_bot_detection(self, monkeypatch):
        """Local session bot_detection_warning should not mention Browserbase."""
        session_info = _make_local_session_info()
        _mock_navigate_deps(monkeypatch, session_info, _make_nav_result(title="Access Denied"))

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True
        assert "bot_detection_warning" in result
        assert "BROWSERBASE" not in result["bot_detection_warning"]
        assert "Browserbase" not in result["bot_detection_warning"]

    def test_local_session_still_reports_stealth_features(self, monkeypatch):
        """Local session should still include stealth_features for visibility."""
        session_info = _make_local_session_info()
        _mock_navigate_deps(monkeypatch, session_info, _make_nav_result())

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True
        assert "stealth_features" in result
        assert "local" in result["stealth_features"]


class TestCloudModeWarnings:
    """Verify that Browserbase cloud sessions still show relevant warnings."""

    def test_cloud_session_without_proxies_shows_stealth_warning(self, monkeypatch):
        """Cloud session without proxies should show stealth_warning."""
        session_info = _make_cloud_session_info(proxies=False)
        _mock_navigate_deps(monkeypatch, session_info, _make_nav_result())

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True
        assert "stealth_warning" in result
        assert "Browserbase" in result["stealth_warning"]

    def test_cloud_session_with_proxies_no_stealth_warning(self, monkeypatch):
        """Cloud session with proxies should NOT show stealth_warning."""
        session_info = _make_cloud_session_info(proxies=True)
        _mock_navigate_deps(monkeypatch, session_info, _make_nav_result())

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True
        assert "stealth_warning" not in result

    def test_cloud_session_bot_detection_mentions_browserbase(self, monkeypatch):
        """Cloud session bot_detection_warning should mention Browserbase options."""
        session_info = _make_cloud_session_info(proxies=True)
        _mock_navigate_deps(monkeypatch, session_info, _make_nav_result(title="Access Denied"))

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True
        assert "bot_detection_warning" in result
        assert "BROWSERBASE_ADVANCED_STEALTH" in result["bot_detection_warning"]
