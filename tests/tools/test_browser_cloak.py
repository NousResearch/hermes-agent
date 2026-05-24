"""Tests for the native CloakBrowser backend integration."""

import json
from unittest.mock import patch


def test_cloakbrowser_mode_env_enabled_and_cdp_override_takes_precedence(monkeypatch):
    from tools.browser_cloak import is_cloakbrowser_mode

    monkeypatch.setenv("CLOAKBROWSER_ENABLED", "true")
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    assert is_cloakbrowser_mode() is True

    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    assert is_cloakbrowser_mode() is False


def test_browser_navigate_routes_to_cloakbrowser_before_camofox(monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_is_cloakbrowser_mode", lambda: True)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: True)

    with patch("tools.browser_cloak.cloakbrowser_navigate", return_value=json.dumps({"success": True, "backend": "cloakbrowser"})) as cloak_nav, \
         patch("tools.browser_camofox.camofox_navigate", return_value=json.dumps({"success": True, "backend": "camofox"})) as camo_nav:
        result = json.loads(bt.browser_navigate("https://example.com", task_id="pytest-cloak"))

    assert result["success"] is True
    assert result["backend"] == "cloakbrowser"
    cloak_nav.assert_called_once_with("https://example.com", "pytest-cloak")
    camo_nav.assert_not_called()


def test_check_browser_requirements_uses_cloakbrowser_availability(monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_is_cloakbrowser_mode", lambda: True)

    monkeypatch.setattr("tools.browser_cloak.check_cloakbrowser_available", lambda: True)
    assert bt.check_browser_requirements() is True

    monkeypatch.setattr("tools.browser_cloak.check_cloakbrowser_available", lambda: False)
    assert bt.check_browser_requirements() is False
