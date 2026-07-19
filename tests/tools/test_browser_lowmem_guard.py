"""Tests for the browser.min_available_mb cold-start low-memory guard."""

import json
from unittest.mock import MagicMock, patch

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
             patch("tools.browser_tool._available_memory_mb", return_value=10), \
             patch.dict("tools.browser_tool._active_sessions", {}, clear=True):
            assert _cold_start_lowmem_error("default", "default") is None

    def test_cold_start_refused_below_threshold(self):
        with patch("tools.browser_tool._get_min_available_mb", return_value=350), \
             patch("tools.browser_tool._available_memory_mb", return_value=120), \
             patch.dict("tools.browser_tool._active_sessions", {}, clear=True):
            error = _cold_start_lowmem_error("default", "default")
        assert error is not None
        assert "web_extract" in error
        assert "min_available_mb" in error
        assert "120MB" in error
        assert "350MB" in error

    def test_cold_start_allowed_with_enough_memory(self):
        with patch("tools.browser_tool._get_min_available_mb", return_value=350), \
             patch("tools.browser_tool._available_memory_mb", return_value=800), \
             patch.dict("tools.browser_tool._active_sessions", {}, clear=True):
            assert _cold_start_lowmem_error("default", "default") is None

    def test_session_reuse_never_blocked(self):
        with patch("tools.browser_tool._get_min_available_mb", return_value=350), \
             patch("tools.browser_tool._available_memory_mb", return_value=10), \
             patch.dict("tools.browser_tool._active_sessions", {"default": object()}, clear=True):
            assert _cold_start_lowmem_error("default", "default") is None

    def test_unreadable_memory_fails_open(self):
        with patch("tools.browser_tool._get_min_available_mb", return_value=350), \
             patch("tools.browser_tool._available_memory_mb", return_value=None), \
             patch.dict("tools.browser_tool._active_sessions", {}, clear=True):
            assert _cold_start_lowmem_error("default", "default") is None


class TestBrowserNavigateWiring:
    def test_navigate_refuses_cold_start_before_creating_session(self):
        from tools.browser_tool import browser_navigate

        session_info = MagicMock(side_effect=AssertionError("session must not be created"))
        with patch("tools.browser_tool._get_min_available_mb", return_value=350), \
             patch("tools.browser_tool._available_memory_mb", return_value=120), \
             patch.dict("tools.browser_tool._active_sessions", {}, clear=True), \
             patch("tools.browser_tool._is_local_backend", return_value=True), \
             patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool.check_website_access", return_value=None), \
             patch("tools.browser_tool._get_session_info", session_info):
            result = browser_navigate("https://example.com")

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "web_extract" in parsed["error"]
        session_info.assert_not_called()
