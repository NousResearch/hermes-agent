"""Regression tests for browser_navigate first-navigation warning scoping.

Before this fix, every session whose feature map lacked ``proxies=True``
received a Browserbase-specific residential-proxy warning. The warning
should only apply to Browserbase-backed sessions that are actually running
without proxies.
"""
import json
from unittest.mock import patch

import pytest

import tools.browser_tool as bt


def _navigate_with_patched_session(
    session_info,
    task_id="warn-test",
    title="Example",
):
    """Run browser_navigate with a fake _get_session_info + fake command results.

    Patches backend-availability checks so the test does not need a real
    browser and avoids private-URL / SSRF guards.
    """
    session_info = dict(session_info, _first_nav=True)
    with (
        patch("tools.browser_tool._is_local_backend", return_value=True),
        patch("tools.browser_tool._get_cloud_provider", return_value=None),
        patch("tools.browser_tool._get_session_info", return_value=session_info),
        patch(
            "tools.browser_tool._run_browser_command",
            side_effect=[
                {
                    "success": True,
                    "data": {"title": title, "url": "https://example.com/"},
                },
                {
                    "success": True,
                    "data": {
                        "snapshot": "- heading \"Example\" [ref=e1]",
                        "refs": {"e1": {}},
                    },
                },
            ],
        ),
    ):
        response = json.loads(bt.browser_navigate("https://example.com", task_id=task_id))
    bt._last_active_session_key.pop(task_id, None)
    return response


class TestBrowserbaseProxyWarning:
    """Browserbase sessions without proxies show the warning."""

    def test_browserbase_without_proxies_shows_warning(self):
        response = _navigate_with_patched_session({
            "session_name": "bb-test",
            "browser_backend": "browserbase",
            "features": {"basic_stealth": True, "proxies": False},
        })

        assert response["success"] is True
        assert "stealth_warning" in response
        assert "Browserbase" in response["stealth_warning"]
        assert response["stealth_features"] == ["basic_stealth"]

    def test_browserbase_blocked_title_shows_backend_specific_advanced_stealth_guidance(
        self,
    ):
        response = _navigate_with_patched_session({
            "session_name": "bb-block-test",
            "browser_backend": "browserbase",
            "features": {"basic_stealth": True, "proxies": True},
        }, title="Access Denied")

        assert response["success"] is True
        assert "bot_detection_warning" in response
        assert "stealth_warning" not in response
        assert "BROWSERBASE_ADVANCED_STEALTH=true" in response["bot_detection_warning"]
        assert "Scale plan" in response["bot_detection_warning"]

    def test_browserbase_with_proxies_no_warning(self):
        response = _navigate_with_patched_session({
            "session_name": "bb-test",
            "browser_backend": "browserbase",
            "features": {"basic_stealth": True, "proxies": True},
        })

        assert response["success"] is True
        assert "stealth_warning" not in response
        assert "proxies" in response["stealth_features"]


class TestNonBrowserbaseNoProxyWarning:
    """Non-Browserbase backends must not receive the Browserbase warning."""

    def test_cdp_override_no_browserbase_warning(self):
        response = _navigate_with_patched_session({
            "session_name": "cdp-test",
            "browser_backend": "cdp_override",
            "features": {"cdp_override": True},
        })

        assert response["success"] is True
        assert "stealth_warning" not in response
        assert "cdp_override" in response["stealth_features"]

    def test_local_chrome_no_browserbase_warning(self):
        response = _navigate_with_patched_session({
            "session_name": "local-test",
            "browser_backend": "local",
            "features": {"local": True},
        })

        assert response["success"] is True
        assert "stealth_warning" not in response
        assert "local" in response["stealth_features"]

    def test_browser_use_no_browserbase_warning(self):
        response = _navigate_with_patched_session({
            "session_name": "bu-test",
            "browser_backend": "browser-use",
            "features": {"browser_use": True},
        })

        assert response["success"] is True
        assert "stealth_warning" not in response

    def test_firecrawl_no_browserbase_warning(self):
        response = _navigate_with_patched_session({
            "session_name": "fc-test",
            "browser_backend": "firecrawl",
            "features": {"firecrawl": True},
        })

        assert response["success"] is True
        assert "stealth_warning" not in response

    @pytest.mark.parametrize(
        ("session_info", "blocked_title"),
        [
            (
                {
                    "session_name": "cdp-test",
                    "browser_backend": "cdp_override",
                    "features": {"cdp_override": True},
                },
                "Access denied",
            ),
            (
                {
                    "session_name": "local-test",
                    "browser_backend": "local",
                    "features": {"local": True},
                },
                "Just a moment",
            ),
            (
                {
                    "session_name": "bu-test",
                    "browser_backend": "browser-use",
                    "features": {"browser_use": True},
                },
                "Attention Required",
            ),
            (
                {
                    "session_name": "fc-test",
                    "browser_backend": "firecrawl",
                    "features": {"firecrawl": True},
                },
                "Blocked",
            ),
            (
                {
                    "session_name": "legacy-test",
                    "features": {"basic_stealth": True, "proxies": False},
                },
                "Bot detected",
            ),
        ],
    )
    def test_non_browserbase_blocked_title_keeps_generic_warning(
        self,
        session_info,
        blocked_title,
    ):
        response = _navigate_with_patched_session(session_info, title=blocked_title)

        assert response["success"] is True
        assert "bot_detection_warning" in response
        assert "3) Some sites have very aggressive bot detection" in response[
            "bot_detection_warning"
        ]
        assert "4)" not in response["bot_detection_warning"]
        assert "BROWSERBASE_ADVANCED_STEALTH" not in response["bot_detection_warning"]
        assert "Browserbase" not in response["bot_detection_warning"]
        assert "Scale plan" not in response["bot_detection_warning"]

    def test_legacy_session_without_backend_no_warning(self):
        """Backward-compat: old sessions that lack browser_backend must not warn."""
        response = _navigate_with_patched_session({
            "session_name": "legacy-test",
            "features": {"basic_stealth": True, "proxies": False},
        })

        assert response["success"] is True
        assert "stealth_warning" not in response
