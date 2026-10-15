"""Real-profile provenance in ``browser_navigate`` responses (#133415).

``features["real_profile"]`` records launch *intent*. The response may only claim
``used_real_profile: true`` when the endpoint that actually served the navigation is
proven to run on the recorded profile-copy dir (its port equals the copy dir's
``DevToolsActivePort``). A stale daemon that fell back to a throwaway temp profile must
surface as an explicit downgrade (``used_real_profile: false`` + warning), never as a
silent success, and the same stale record must not advertise ``real_profile`` through
``stealth_features``. The Browserbase proxy upsell must stay off local sessions — they
never touch Browserbase.
"""

import json
from unittest.mock import patch

import pytest

import tools.browser_tool as browser_tool


def _write_devtools_port(copy_dir, port: str) -> str:
    copy_dir.mkdir(parents=True, exist_ok=True)
    (copy_dir / "DevToolsActivePort").write_text(
        f"{port}\n/devtools/browser/test-browser-id\n", encoding="utf-8"
    )
    return str(copy_dir)


@pytest.fixture(autouse=True)
def _isolated_module_state():
    """Keep the facade caches navigate writes to out of the shared module state."""
    saved_cache = browser_tool._real_profile_cdp_cache
    saved_last = browser_tool._last_active_session_key
    browser_tool._real_profile_cdp_cache = {}
    browser_tool._last_active_session_key = {}
    yield
    browser_tool._real_profile_cdp_cache = saved_cache
    browser_tool._last_active_session_key = saved_last


@pytest.fixture
def navigate():
    """``navigate(session)`` → parsed ``browser_navigate`` response over a stubbed backend."""

    def run(session: dict) -> dict:
        with (
            patch("tools.browser_tool._navigation_session_key", return_value="default"),
            patch("tools.browser_tool_session._get_session_info", return_value=session),
            patch(
                "tools.browser_tool_session._run_browser_command",
                return_value={
                    "success": True,
                    "data": {"title": "Example Domain", "url": "https://example.com/"},
                },
            ),
            patch("tools.browser_tool_cloud._is_local_backend", return_value=True),
            patch("tools.browser_tool._maybe_start_recording", lambda key: None),
        ):
            return json.loads(browser_tool.browser_navigate("https://example.com/"))

    return run


def _rp_session(cdp_url=None, features=None) -> dict:
    return {
        "session_name": "rp_test0000000000",
        "bb_session_id": None,
        "cdp_url": cdp_url,
        "features": features
        if features is not None
        else {"local": True, "real_profile": True},
        "_first_nav": True,
    }


class TestRealProfileProvenance:
    def test_endpoint_on_copy_dir_reports_true(self, navigate, tmp_path):
        browser_tool._real_profile_cdp_cache["copy_dir"] = _write_devtools_port(
            tmp_path / "copy", "53333"
        )
        resp = navigate(_rp_session(cdp_url="http://127.0.0.1:53333"))
        assert resp["used_real_profile"] is True
        assert "real_profile_warning" not in resp
        assert "real_profile" in resp["stealth_features"]

    def test_resolved_websocket_url_still_verifies(self, navigate, tmp_path):
        """``_resolve_cdp_override`` hands navigate a ws:// URL — the port match must hold."""
        browser_tool._real_profile_cdp_cache["copy_dir"] = _write_devtools_port(
            tmp_path / "copy", "53333"
        )
        resp = navigate(
            _rp_session(cdp_url="ws://127.0.0.1:53333/devtools/browser/test-browser-id")
        )
        assert resp["used_real_profile"] is True
        assert "real_profile_warning" not in resp

    def test_throwaway_endpoint_fails_closed(self, navigate, tmp_path):
        """The #133415 state: the session claims real_profile, a throwaway serves the page."""
        browser_tool._real_profile_cdp_cache["copy_dir"] = _write_devtools_port(
            tmp_path / "copy", "53333"
        )
        resp = navigate(
            _rp_session(cdp_url="http://127.0.0.1:59999")
        )  # throwaway's port
        assert resp["success"] is True
        assert resp["used_real_profile"] is False
        assert "NOT served by the real-profile browser" in resp["real_profile_warning"]
        assert "real_profile" not in resp.get("stealth_features", [])

    def test_unknown_copy_dir_fails_closed(self, navigate):
        """No recorded copy dir (cache miss) must not inherit the intent flag."""
        resp = navigate(_rp_session(cdp_url="http://127.0.0.1:53333"))
        assert resp["used_real_profile"] is False
        assert "real_profile_warning" in resp

    def test_port_file_gone_fails_closed(self, navigate, tmp_path):
        (tmp_path / "copy").mkdir()
        browser_tool._real_profile_cdp_cache["copy_dir"] = str(tmp_path / "copy")
        resp = navigate(_rp_session(cdp_url="http://127.0.0.1:53333"))
        assert resp["used_real_profile"] is False
        assert "real_profile_warning" in resp


class TestProxyUpsellGate:
    def test_plain_local_session_gets_no_browserbase_upsell(self, navigate):
        resp = navigate(_rp_session(features={"local": True}))
        assert "stealth_warning" not in resp
        assert "used_real_profile" not in resp

    def test_verified_real_profile_session_gets_no_upsell(self, navigate, tmp_path):
        browser_tool._real_profile_cdp_cache["copy_dir"] = _write_devtools_port(
            tmp_path / "copy", "53333"
        )
        resp = navigate(_rp_session(cdp_url="http://127.0.0.1:53333"))
        assert resp["used_real_profile"] is True
        assert "stealth_warning" not in resp

    def test_cloud_session_without_proxies_keeps_upsell(self, navigate):
        resp = navigate(_rp_session(features={"cloud": True, "proxies": False}))
        assert "residential proxies" in resp["stealth_warning"]
        assert "proxies" not in resp["stealth_features"]
