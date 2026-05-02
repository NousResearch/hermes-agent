"""Tests that browser_navigate SSRF checks respect local-backend mode and
the allow_private_urls setting.

Local backends (Camofox, headless Chromium without a cloud provider) skip
SSRF checks entirely — the agent already has full local-network access via
the terminal tool.

Cloud backends (Browserbase, BrowserUse) enforce SSRF by default.  Users
can opt out for cloud mode via ``browser.allow_private_urls: true``.
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

    # -- Cloud mode: SSRF active -----------------------------------------------

    def test_cloud_blocks_private_url_by_default(self, monkeypatch, _common_patches):
        """SSRF protection blocks private URLs in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_cloud_allows_private_url_when_setting_true(self, monkeypatch, _common_patches):
        """Private URLs pass in cloud mode when allow_private_urls is True."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True

    def test_cloud_allows_public_url(self, monkeypatch, _common_patches):
        """Public URLs always pass in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True

    # -- Local mode: SSRF skipped ----------------------------------------------

    def test_local_allows_private_url(self, monkeypatch, _common_patches):
        """Local backends skip SSRF — private URLs are always allowed."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True

    def test_local_allows_public_url(self, monkeypatch, _common_patches):
        """Local backends pass public URLs too (sanity check)."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True


# ---------------------------------------------------------------------------
# _is_local_backend() unit tests
# ---------------------------------------------------------------------------


class TestIsLocalBackend:
    def test_camofox_is_local(self, monkeypatch):
        """Camofox mode counts as a local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: "anything")

        assert browser_tool._is_local_backend() is True

    def test_no_cloud_provider_is_local(self, monkeypatch):
        """No cloud provider configured → local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)

        assert browser_tool._is_local_backend() is True

    def test_cloud_provider_is_not_local(self, monkeypatch):
        """Cloud provider configured and not Camofox → NOT local."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: "bb")

        assert browser_tool._is_local_backend() is False


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

    # -- Cloud mode: redirect SSRF active --------------------------------------

    def test_cloud_blocks_redirect_to_private(self, monkeypatch, _common_patches):
        """Redirects to private addresses are blocked in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
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

    def test_cloud_allows_redirect_to_private_when_setting_true(self, monkeypatch, _common_patches):
        """Redirects to private addresses pass in cloud mode with allow_private_urls."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
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

    # -- Local mode: redirect SSRF skipped -------------------------------------

    def test_local_allows_redirect_to_private(self, monkeypatch, _common_patches):
        """Redirects to private addresses pass in local mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
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

        assert result["success"] is True
        assert result["url"] == self.PRIVATE_FINAL_URL

    def test_cloud_allows_redirect_to_public(self, monkeypatch, _common_patches):
        """Redirects to public addresses always pass (cloud mode)."""
        final = "https://example.com/final"
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
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


class TestAllowPrivateUrlsConfig:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        browser_tool._allow_private_urls_resolved = False
        browser_tool._cached_allow_private_urls = None
        yield
        browser_tool._allow_private_urls_resolved = False
        browser_tool._cached_allow_private_urls = None

    def test_browser_config_string_false_stays_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"allow_private_urls": "false"}},
        )

        assert browser_tool._allow_private_urls() is False
