"""Tests for the browser.min_available_mb cold-start low-memory guard."""

import json
from unittest.mock import ANY, MagicMock, patch

import tools.browser_tool as browser_tool
from tools.browser_tool import (
    _available_memory_mb,
    _cold_start_lowmem_error,
    _get_min_available_mb,
)


class TestAvailableMemoryParsing:
    def test_parses_memavailable_from_meminfo(self, tmp_path):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(
            "MemTotal:        1024000 kB\n"
            "MemFree:          102400 kB\n"
            "MemAvailable:     204800 kB\n",
            encoding="utf-8",
        )
        assert _available_memory_mb(str(meminfo)) == 200

    def test_returns_none_when_memavailable_missing(self, tmp_path):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal: 1024000 kB\n", encoding="utf-8")
        assert _available_memory_mb(str(meminfo)) is None

    def test_returns_none_when_unreadable(self, tmp_path):
        assert _available_memory_mb(str(tmp_path / "does-not-exist")) is None


class TestMinAvailableConfig:
    def test_defaults_to_disabled(self):
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert _get_min_available_mb() == 0

    def test_reads_threshold_from_config(self):
        cfg = {"browser": {"min_available_mb": 350}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_min_available_mb() == 350

    def test_negative_values_clamp_to_disabled(self):
        cfg = {"browser": {"min_available_mb": -5}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_min_available_mb() == 0

    def test_config_error_fails_open(self):
        with patch("hermes_cli.config.read_raw_config", side_effect=RuntimeError("boom")):
            assert _get_min_available_mb() == 0


class TestColdStartGuard:
    def test_disabled_guard_allows_cold_start(self):
        with patch("tools.browser_tool._get_min_available_mb", return_value=0), \
             patch("tools.browser_tool._available_memory_mb", return_value=10):
            assert _cold_start_lowmem_error(
                {"cdp_url": None}, session_was_active=False
            ) is None

    def test_cold_start_refused_below_threshold(self):
        with patch("tools.browser_tool._get_min_available_mb", return_value=350), \
             patch("tools.browser_tool._available_memory_mb", return_value=120):
            error = _cold_start_lowmem_error(
                {"cdp_url": None}, session_was_active=False
            )
        assert error is not None
        assert "web_extract" in error
        assert "min_available_mb" in error
        assert "120MB" in error
        assert "350MB" in error

    def test_cold_start_allowed_with_enough_memory(self):
        with patch("tools.browser_tool._get_min_available_mb", return_value=350), \
             patch("tools.browser_tool._available_memory_mb", return_value=800):
            assert _cold_start_lowmem_error(
                {"cdp_url": None}, session_was_active=False
            ) is None

    def test_session_reuse_never_blocked(self):
        with patch("tools.browser_tool._get_min_available_mb", return_value=350), \
             patch("tools.browser_tool._available_memory_mb", return_value=10):
            assert _cold_start_lowmem_error(
                {"cdp_url": None}, session_was_active=True
            ) is None

    def test_unreadable_memory_fails_open(self):
        with patch("tools.browser_tool._get_min_available_mb", return_value=350), \
             patch("tools.browser_tool._available_memory_mb", return_value=None):
            assert _cold_start_lowmem_error(
                {"cdp_url": None}, session_was_active=False
            ) is None


class TestBrowserNavigateWiring:
    @staticmethod
    def _patch_navigation_dependencies(monkeypatch):
        runner = MagicMock(
            return_value={
                "success": True,
                "data": {"title": "Example", "url": "https://example.com"},
            }
        )
        memory_probe = MagicMock(return_value=120)
        monkeypatch.setattr(browser_tool, "_active_sessions", {})
        monkeypatch.setattr(browser_tool, "_session_last_activity", {})
        monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
        monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(browser_tool, "_update_session_activity", lambda task_id: None)
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda task_id: None)
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_auto_local_for_private_urls", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
        monkeypatch.setattr(browser_tool, "_maybe_start_recording", lambda task_id: None)
        monkeypatch.setattr(browser_tool, "_run_browser_command", runner)
        monkeypatch.setattr(browser_tool, "_get_min_available_mb", lambda: 350)
        monkeypatch.setattr(browser_tool, "_available_memory_mb", memory_probe)
        return runner, memory_probe

    def test_local_cold_start_blocked_at_low_memory(self, monkeypatch):
        runner, memory_probe = self._patch_navigation_dependencies(monkeypatch)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)

        result = browser_tool.browser_navigate(
            "https://example.com", task_id="local-task"
        )

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "web_extract" in parsed["error"]
        memory_probe.assert_called_once_with()
        runner.assert_not_called()
        assert "local-task" not in browser_tool._active_sessions

    def test_cloud_session_not_blocked_at_low_memory(self, monkeypatch):
        runner, memory_probe = self._patch_navigation_dependencies(monkeypatch)
        provider = MagicMock()
        provider.create_session.return_value = {
            "session_name": "cloud-session",
            "bb_session_id": "cloud-123",
            "cdp_url": "wss://cloud.example/devtools/browser/abc",
            "features": {"cloud": True},
        }
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_url_is_private", lambda url: False)

        result = browser_tool.browser_navigate(
            "https://example.com", task_id="cloud-task"
        )

        assert json.loads(result)["success"] is True
        provider.create_session.assert_called_once_with("cloud-task")
        memory_probe.assert_not_called()
        runner.assert_any_call(
            "cloud-task", "open", ["https://example.com"], timeout=ANY
        )

    def test_explicit_cdp_override_not_blocked_at_low_memory(self, monkeypatch):
        runner, memory_probe = self._patch_navigation_dependencies(monkeypatch)
        provider = MagicMock()
        cdp_url = "ws://cdp.example/devtools/browser/abc"
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: cdp_url)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)

        result = browser_tool.browser_navigate(
            "https://example.com", task_id="cdp-task"
        )

        assert json.loads(result)["success"] is True
        provider.create_session.assert_not_called()
        assert browser_tool._active_sessions["cdp-task"]["cdp_url"] == cdp_url
        memory_probe.assert_not_called()
        runner.assert_any_call(
            "cdp-task", "open", ["https://example.com"], timeout=ANY
        )

    def test_hybrid_local_sidecar_blocked_at_low_memory(self, monkeypatch):
        runner, memory_probe = self._patch_navigation_dependencies(monkeypatch)
        provider = MagicMock()
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_url_is_private", lambda url: True)
        browser_tool._active_sessions["hybrid-task"] = {
            "session_name": "cloud-session",
            "bb_session_id": "cloud-123",
            "cdp_url": "wss://cloud.example/devtools/browser/abc",
            "features": {"cloud": True},
        }

        result = browser_tool.browser_navigate(
            "http://localhost:3000", task_id="hybrid-task"
        )

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "web_extract" in parsed["error"]
        provider.create_session.assert_not_called()
        memory_probe.assert_called_once_with()
        runner.assert_not_called()
        assert "hybrid-task" in browser_tool._active_sessions
        assert "hybrid-task::local" not in browser_tool._active_sessions

    def test_existing_local_session_reuse_allowed_at_low_memory(self, monkeypatch):
        runner, memory_probe = self._patch_navigation_dependencies(monkeypatch)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
        browser_tool._active_sessions["reuse-task"] = {
            "session_name": "local-session",
            "bb_session_id": None,
            "cdp_url": None,
            "features": {"local": True},
            "_first_nav": False,
        }

        result = browser_tool.browser_navigate(
            "https://example.com", task_id="reuse-task"
        )

        assert json.loads(result)["success"] is True
        memory_probe.assert_not_called()
        runner.assert_any_call(
            "reuse-task", "open", ["https://example.com"], timeout=ANY
        )
