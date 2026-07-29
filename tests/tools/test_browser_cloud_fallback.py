"""Tests for cloud browser provider runtime fallback to local Chromium.

Covers the fallback logic in _get_session_info() when a cloud provider
is configured but fails at runtime (issue #10883).
"""
import logging
from unittest.mock import Mock

import pytest

import tools.browser_tool as browser_tool


def _reset_session_state(monkeypatch):
    """Clear caches so each test starts fresh."""
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_cached_cloud_provider", None)
    monkeypatch.setattr(browser_tool, "_cloud_provider_resolved", False)
    monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_tool, "_update_session_activity", lambda t: None)


class TestCloudProviderRuntimeFallback:
    """Tests for _get_session_info cloud → local fallback."""

    def test_cloud_failure_falls_back_to_local(self, monkeypatch):
        """When cloud provider.create_session raises, fall back to local."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.side_effect = RuntimeError("401 Unauthorized")
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)

        session = browser_tool._get_session_info("task-1")

        assert session["fallback_from_cloud"] is True
        assert "401 Unauthorized" in session["fallback_reason"]
        assert session["fallback_provider"] == "Mock"
        assert session["features"]["local"] is True
        assert session["cdp_url"] is None


    def test_no_provider_uses_local_directly(self, monkeypatch):
        """When no cloud provider is configured, local mode is used with no fallback markers."""
        _reset_session_state(monkeypatch)

        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)

        session = browser_tool._get_session_info("task-4")

        assert session["features"]["local"] is True
        assert "fallback_from_cloud" not in session


    def test_cloud_returns_invalid_session_triggers_fallback(self, monkeypatch):
        """Cloud provider returning None or empty dict triggers fallback."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.return_value = None
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)

        session = browser_tool._get_session_info("task-7")

        assert session["fallback_from_cloud"] is True
        assert "invalid session" in session["fallback_reason"]
