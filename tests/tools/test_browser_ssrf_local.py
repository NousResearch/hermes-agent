"""Tests that browser_navigate SSRF checks respect the allow_private_urls setting.

When ``browser.allow_private_urls`` is ``False`` (default), private/internal
addresses are blocked.  When set to ``True``, they are allowed — useful for
local development, LAN access, and Hermes self-testing.
"""

import json

import pytest

from tools import browser_tool


def _make_browser_result(url="https://example.com"):
    """Return a mock successful browser command result."""
    return {"success": True, "data": {"title": "OK", "url": url}}


# ---------------------------------------------------------------------------
# Pre-navigation SSRF check
# ---------------------------------------------------------------------------


class TestPreNavigationSsrf:
    PRIVATE_URL = "http://127.0.0.1:8080/dashboard"

    @pytest.fixture()
    def _common_patches(self, monkeypatch):
        """Shared patches for pre-navigation tests that pass the SSRF check."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            lambda task_id: {
                "session_name": f"s_{task_id}",
                "bb_session_id": None,
                "cdp_url": None,
                "features": {"local": True},
                "_first_nav": False,
            },
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(),
        )

    def test_blocks_private_url_by_default(self, monkeypatch, _common_patches):
        """SSRF protection is on when allow_private_urls is not set (False)."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_blocks_private_url_when_setting_false(self, monkeypatch, _common_patches):
        """SSRF protection is on when allow_private_urls is explicitly False."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is False

    def test_allows_private_url_when_setting_true(self, monkeypatch, _common_patches):
        """Private URLs are allowed when allow_private_urls is True."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        # _is_safe_url would block this, but the setting overrides it
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True

    def test_allows_public_url_regardless_of_setting(self, monkeypatch, _common_patches):
        """Public URLs always pass regardless of the allow_private_urls setting."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True


# ---------------------------------------------------------------------------
# Post-redirect SSRF check
# ---------------------------------------------------------------------------


class TestPostRedirectSsrf:
    PUBLIC_URL = "https://example.com/redirect"
    PRIVATE_FINAL_URL = "http://192.168.1.1/internal"

    @pytest.fixture()
    def _common_patches(self, monkeypatch):
        """Shared patches for redirect tests."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            lambda task_id: {
                "session_name": f"s_{task_id}",
                "bb_session_id": None,
                "cdp_url": None,
                "features": {"local": True},
                "_first_nav": False,
            },
        )

    def test_blocks_redirect_to_private_by_default(self, monkeypatch, _common_patches):
        """Redirects to private addresses are blocked when setting is False."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(
            browser_tool, "_is_safe_url", lambda url: "192.168" not in url,
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=self.PRIVATE_FINAL_URL),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is False
        assert "redirect landed on a private/internal address" in result["error"]

    def test_allows_redirect_to_private_when_setting_true(self, monkeypatch, _common_patches):
        """Redirects to private addresses are allowed when setting is True."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(
            browser_tool, "_is_safe_url", lambda url: "192.168" not in url,
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=self.PRIVATE_FINAL_URL),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == self.PRIVATE_FINAL_URL

    def test_allows_redirect_to_public_regardless_of_setting(self, monkeypatch, _common_patches):
        """Redirects to public addresses always pass."""
        final = "https://example.com/final"
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=final),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == final
