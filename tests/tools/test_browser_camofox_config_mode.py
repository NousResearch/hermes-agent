"""Tests for selecting Camofox via Hermes browser config."""

from unittest.mock import patch

import pytest

from tools import browser_camofox as cf
from tools import browser_tool as bt


@pytest.fixture(autouse=True)
def _reset_camofox_state(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    cf._vnc_url = None
    cf._vnc_url_checked = False
    with cf._sessions_lock:
        cf._sessions.clear()
    yield
    cf._vnc_url = None
    cf._vnc_url_checked = False
    with cf._sessions_lock:
        cf._sessions.clear()


class TestCamofoxConfigMode:
    def test_browser_mode_camofox_uses_default_local_url(self):
        with patch("tools.browser_camofox.load_config", return_value={"browser": {"mode": "camofox"}}):
            assert cf.get_camofox_url() == "http://localhost:9377"
            assert cf.is_camofox_mode() is True

    def test_browser_cloud_provider_camofox_uses_default_local_url(self):
        with patch("tools.browser_camofox.load_config", return_value={"browser": {"cloud_provider": "camofox"}}):
            assert cf.get_camofox_url() == "http://localhost:9377"
            assert cf.is_camofox_mode() is True

    def test_env_url_overrides_config_default(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://camofox.internal:9999/")
        with patch("tools.browser_camofox.load_config", return_value={"browser": {"mode": "camofox"}}):
            assert cf.get_camofox_url() == "http://camofox.internal:9999"

    def test_cdp_override_still_takes_priority(self, monkeypatch):
        monkeypatch.setenv("BROWSER_CDP_URL", "ws://localhost:9222/devtools/browser/abc")
        with patch("tools.browser_camofox.load_config", return_value={"browser": {"mode": "camofox"}}):
            assert cf.is_camofox_mode() is False

    def test_browser_tool_does_not_construct_cloud_provider_for_camofox_config(self):
        with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"cloud_provider": "camofox"}}), \
             patch("tools.browser_tool.BrowserUseProvider") as browser_use_provider, \
             patch("tools.browser_tool.BrowserbaseProvider") as browserbase_provider:
            bt._cached_cloud_provider = None
            bt._cloud_provider_resolved = False

            assert bt._get_cloud_provider() is None
            browser_use_provider.assert_not_called()
            browserbase_provider.assert_not_called()
