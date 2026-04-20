"""Tests for cloud browser provider runtime fallback to local Chromium.

Covers the fallback logic in _get_session_info() when a cloud provider
is configured but fails at runtime (issue #10883).
"""
import logging
from unittest.mock import Mock, patch

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

    def test_cloud_success_no_fallback(self, monkeypatch):
        """When cloud succeeds, no fallback markers are present."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "cloud-sess",
            "bb_session_id": "bb_123",
            "cdp_url": None,
            "features": {"browser_use": True},
        }
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)

        session = browser_tool._get_session_info("task-2")

        assert session["session_name"] == "cloud-sess"
        assert "fallback_from_cloud" not in session
        assert "fallback_reason" not in session

    def test_cloud_and_local_both_fail(self, monkeypatch):
        """When both cloud and local fail, raise RuntimeError with both contexts."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.side_effect = RuntimeError("cloud boom")
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)
        monkeypatch.setattr(
            browser_tool, "_create_local_session",
            Mock(side_effect=OSError("no chromium")),
        )

        with pytest.raises(RuntimeError, match="cloud boom.*local.*no chromium"):
            browser_tool._get_session_info("task-3")

    def test_no_provider_uses_local_directly(self, monkeypatch):
        """When no cloud provider is configured, local mode is used with no fallback markers."""
        _reset_session_state(monkeypatch)

        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)

        session = browser_tool._get_session_info("task-4")

        assert session["features"]["local"] is True
        assert "fallback_from_cloud" not in session

    def test_cdp_override_bypasses_provider(self, monkeypatch):
        """CDP override takes priority — cloud provider is never consulted."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://host:9222/devtools/browser/abc")

        session = browser_tool._get_session_info("task-5")

        provider.create_session.assert_not_called()
        assert session["cdp_url"] == "ws://host:9222/devtools/browser/abc"

    def test_fallback_logs_warning_with_provider_name(self, monkeypatch, caplog):
        """Fallback emits a warning log with the provider class name and error."""
        _reset_session_state(monkeypatch)

        BrowserUseProviderFake = type("BrowserUseProvider", (), {
            "create_session": Mock(side_effect=ConnectionError("timeout")),
        })
        provider = BrowserUseProviderFake()
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)

        with caplog.at_level(logging.WARNING, logger="tools.browser_tool"):
            session = browser_tool._get_session_info("task-6")

        assert session["fallback_from_cloud"] is True
        assert any("BrowserUseProvider" in r.message and "timeout" in r.message
                    for r in caplog.records)

    def test_cloud_failure_does_not_poison_next_task(self, monkeypatch):
        """A fallback for one task_id doesn't affect a new task_id when cloud recovers."""
        _reset_session_state(monkeypatch)

        call_count = 0

        def create_session_flaky(task_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            return {
                "session_name": "cloud-ok",
                "bb_session_id": "bb_999",
                "cdp_url": None,
                "features": {"browser_use": True},
            }

        provider = Mock()
        provider.create_session.side_effect = create_session_flaky
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)

        # First call fails → fallback
        s1 = browser_tool._get_session_info("task-a")
        assert s1["fallback_from_cloud"] is True

        # Second call (different task) → cloud succeeds
        s2 = browser_tool._get_session_info("task-b")
        assert "fallback_from_cloud" not in s2
        assert s2["session_name"] == "cloud-ok"

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
