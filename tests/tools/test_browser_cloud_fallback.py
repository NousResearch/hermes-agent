"""Tests that cloud browser providers fail closed instead of local fallback."""
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


class TestCloudProviderFailClosed:
    """Tests for _get_session_info cloud session creation failures."""

    def test_cloud_failure_does_not_fall_back_to_local(self, monkeypatch):
        """When provider.create_session raises, do not create a local browser."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.side_effect = RuntimeError("401 Unauthorized")
        create_local = Mock(wraps=browser_tool._create_local_session)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)
        monkeypatch.setattr(browser_tool, "_create_local_session", create_local)

        with pytest.raises(RuntimeError, match="refusing to fall back to local Chromium"):
            browser_tool._get_session_info("task-1")

        create_local.assert_not_called()
        assert "task-1" not in browser_tool._active_sessions

    def test_cloud_success_no_fallback(self, monkeypatch):
        """When cloud succeeds, no fallback markers are present."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "cloud-sess",
            "bb_session_id": "bb_123",
            "cdp_url": "ws://cloud.example/devtools/browser/123",
            "features": {"browser_use": True},
        }
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda task_id: None)

        session = browser_tool._get_session_info("task-2")

        assert session["session_name"] == "cloud-sess"
        assert session["cdp_url"] == "ws://cloud.example/devtools/browser/123"
        assert "fallback_from_cloud" not in session
        assert "fallback_reason" not in session

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
        monkeypatch.setattr(
            browser_tool,
            "_get_cdp_override",
            lambda: "ws://host:9222/devtools/browser/abc",
        )
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda task_id: None)

        session = browser_tool._get_session_info("task-5")

        provider.create_session.assert_not_called()
        assert session["cdp_url"] == "ws://host:9222/devtools/browser/abc"

    def test_cloud_failure_does_not_poison_next_task(self, monkeypatch):
        """A failure for one task_id doesn't affect a new task_id when cloud recovers."""
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
                "cdp_url": "ws://cloud.example/devtools/browser/999",
                "features": {"browser_use": True},
            }

        provider = Mock()
        provider.create_session.side_effect = create_session_flaky
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda task_id: None)

        with pytest.raises(RuntimeError, match="refusing to fall back to local Chromium"):
            browser_tool._get_session_info("task-a")

        s2 = browser_tool._get_session_info("task-b")
        assert "fallback_from_cloud" not in s2
        assert s2["session_name"] == "cloud-ok"

    @pytest.mark.parametrize(
        "session_metadata",
        [None, {}, {"session_name": "cloud-sess", "cdp_url": None}, {"cdp_url": "   "}],
    )
    def test_cloud_returns_invalid_session_fails_closed(self, monkeypatch, session_metadata):
        """Invalid cloud metadata must not silently select local --session mode."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.return_value = session_metadata
        create_local = Mock(wraps=browser_tool._create_local_session)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)
        monkeypatch.setattr(browser_tool, "_create_local_session", create_local)

        with pytest.raises(RuntimeError):
            browser_tool._get_session_info("task-7")

        create_local.assert_not_called()
        assert "task-7" not in browser_tool._active_sessions
