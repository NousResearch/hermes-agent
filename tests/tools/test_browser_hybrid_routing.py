"""Tests for hybrid browser-backend routing (LAN/localhost auto-local).

When a cloud browser provider (Browserbase / Browser-Use / Firecrawl) is
configured globally, ``browser.auto_local_for_private_urls`` (default True)
causes ``browser_navigate`` to transparently spawn a local Chromium sidecar
for URLs whose host resolves to a private/loopback/LAN address, while
public URLs continue to hit the cloud session in the same conversation.

These tests cover the routing decision layer — session_key selection,
sidecar detection, last-active-session tracking, and the config toggle.
The downstream session creation is covered by test_browser_cloud_fallback.py.
"""
from unittest.mock import Mock

import pytest

import tools.browser_tool as browser_tool


@pytest.fixture(autouse=True)
def _reset_routing_state(monkeypatch):
    """Clear module-level caches so each test starts clean."""
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    monkeypatch.setattr(browser_tool, "_cached_cloud_provider", None)
    monkeypatch.setattr(browser_tool, "_cloud_provider_resolved", False)
    monkeypatch.setattr(browser_tool, "_auto_local_for_private_urls_resolved", False)
    monkeypatch.setattr(browser_tool, "_cached_auto_local_for_private_urls", True)
    monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_tool, "_update_session_activity", lambda t: None)
    # Default: no CDP override, no Camofox
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)


class TestNavigationSessionKey:
    """Tests for _navigation_session_key URL-based routing decisions."""

    def test_public_url_uses_bare_task_id(self, monkeypatch):
        """Public URL with cloud provider configured → bare task_id (cloud)."""
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "https://github.com/x/y")
        assert key == "default"

    def test_localhost_routes_to_local_sidecar(self, monkeypatch):
        """``localhost`` URL → ``::local`` suffix when cloud configured + flag on."""
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "http://localhost:3000/")
        assert key == "default::local"

    def test_loopback_ipv4_routes_to_local_sidecar(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "http://127.0.0.1:8080/")
        assert key == "default::local"

    def test_rfc1918_lan_routes_to_local_sidecar(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "http://192.168.1.50:8000/")
        assert key == "default::local"

    def test_ipv6_loopback_routes_to_local_sidecar(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "http://[::1]:3000/")
        assert key == "default::local"

    def test_public_ip_literal_uses_bare_task_id(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "https://8.8.8.8/")
        assert key == "default"

    def test_mdns_local_hostname_routes_to_sidecar(self, monkeypatch):
        """``*.local`` mDNS / ``*.lan`` / ``*.internal`` hostnames route to sidecar."""
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        for host in ("raspberrypi.local", "printer.lan", "db.internal"):
            key = browser_tool._navigation_session_key("default", f"http://{host}/")
            assert key == "default::local", f"host {host!r} did not route to sidecar"

    def test_no_cloud_provider_stays_on_bare_task_id(self, monkeypatch):
        """When cloud provider is not configured, no hybrid routing happens."""
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
        key = browser_tool._navigation_session_key("default", "http://localhost:3000/")
        assert key == "default"

    def test_camofox_mode_stays_on_bare_task_id(self, monkeypatch):
        """Camofox is already local — no hybrid routing needed."""
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
        key = browser_tool._navigation_session_key("default", "http://localhost:3000/")
        assert key == "default"

    def test_cdp_override_stays_on_bare_task_id(self, monkeypatch):
        """A user-supplied CDP endpoint owns the whole session — no hybrid."""
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://localhost:9222")
        key = browser_tool._navigation_session_key("default", "http://localhost:3000/")
        assert key == "default"

    def test_feature_flag_off_disables_hybrid_routing(self, monkeypatch):
        """``auto_local_for_private_urls: false`` keeps private URLs on cloud."""
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        monkeypatch.setattr(browser_tool, "_auto_local_for_private_urls", lambda: False)
        key = browser_tool._navigation_session_key("default", "http://localhost:3000/")
        assert key == "default"

    def test_none_task_id_defaults(self, monkeypatch):
        """``None`` task_id resolves to 'default'."""
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key(None, "http://localhost:3000/")
        assert key == "default::local"


class TestSessionKeyHelpers:
    def test_is_local_sidecar_key(self):
        assert browser_tool._is_local_sidecar_key("default::local")
        assert browser_tool._is_local_sidecar_key("my_task::local")
        assert not browser_tool._is_local_sidecar_key("default")
        assert not browser_tool._is_local_sidecar_key("my_task")

    def test_last_session_key_falls_back_to_task_id(self, monkeypatch):
        """Without a recorded last-active key, returns the bare task_id."""
        monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
        assert browser_tool._last_session_key("default") == "default"
        assert browser_tool._last_session_key("task-42") == "task-42"
        assert browser_tool._last_session_key(None) == "default"

    def test_last_session_key_returns_recorded_key(self, monkeypatch):
        monkeypatch.setattr(
            browser_tool,
            "_last_active_session_key",
            {"default": "default::local", "task-42": "task-42"},
        )
        assert browser_tool._last_session_key("default") == "default::local"
        assert browser_tool._last_session_key("task-42") == "task-42"
        # Unknown task_id still falls back
        assert browser_tool._last_session_key("other") == "other"


class TestHybridRoutingSessionCreation:
    """_get_session_info must force a local session when the key carries ``::local``."""

    def test_local_sidecar_key_skips_cloud_provider(self, monkeypatch):
        """A ``::local``-suffixed key creates a local session even when cloud is set."""
        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "should_not_be_used",
            "bb_session_id": "bb_xxx",
            "cdp_url": "wss://fake.browserbase.com/ws",
        }
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda t: None)

        session = browser_tool._get_session_info("default::local")

        assert provider.create_session.call_count == 0
        assert session["bb_session_id"] is None
        assert session["cdp_url"] is None
        assert session["features"]["local"] is True

    def test_bare_task_id_with_cloud_provider_uses_cloud(self, monkeypatch):
        """A bare task_id with cloud provider configured hits the cloud path."""
        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "cloud-sess",
            "bb_session_id": "bb_123",
            "cdp_url": "wss://real.browserbase.com/ws",
        }
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda t: None)
        monkeypatch.setattr(browser_tool, "_resolve_cdp_override", lambda u: u)

        session = browser_tool._get_session_info("default")

        assert provider.create_session.call_count == 1
        assert session["bb_session_id"] == "bb_123"


class TestCleanupHybridSessions:
    """cleanup_browser(bare_task_id) must reap both cloud + local sidecar sessions."""

    def test_cleanup_reaps_both_primary_and_sidecar(self, monkeypatch):
        """Given a bare task_id with both sessions alive, both get cleaned."""
        reaped = []

        def _fake_cleanup_one(key):
            reaped.append(key)

        monkeypatch.setattr(browser_tool, "_cleanup_single_browser_session", _fake_cleanup_one)
        monkeypatch.setattr(
            browser_tool,
            "_active_sessions",
            {
                "default": {"session_name": "cloud_sess"},
                "default::local": {"session_name": "local_sess"},
            },
        )
        monkeypatch.setattr(
            browser_tool, "_last_active_session_key", {"default": "default::local"}
        )

        browser_tool.cleanup_browser("default")

        assert set(reaped) == {"default", "default::local"}
        # last-active pointer dropped
        assert "default" not in browser_tool._last_active_session_key

    def test_cleanup_reaps_only_primary_when_no_sidecar(self, monkeypatch):
        """When no sidecar exists, only the primary is reaped."""
        reaped = []

        def _fake_cleanup_one(key):
            reaped.append(key)

        monkeypatch.setattr(browser_tool, "_cleanup_single_browser_session", _fake_cleanup_one)
        monkeypatch.setattr(
            browser_tool,
            "_active_sessions",
            {"default": {"session_name": "cloud_sess"}},
        )

        browser_tool.cleanup_browser("default")

        assert reaped == ["default"]

    def test_cleanup_sidecar_directly_keeps_primary(self, monkeypatch):
        """Calling cleanup with a ``::local`` key reaps only the sidecar."""
        reaped = []

        def _fake_cleanup_one(key):
            reaped.append(key)

        monkeypatch.setattr(browser_tool, "_cleanup_single_browser_session", _fake_cleanup_one)
        monkeypatch.setattr(
            browser_tool,
            "_active_sessions",
            {
                "default": {"session_name": "cloud_sess"},
                "default::local": {"session_name": "local_sess"},
            },
        )
        monkeypatch.setattr(
            browser_tool, "_last_active_session_key", {"default": "default::local"}
        )

        browser_tool.cleanup_browser("default::local")

        assert reaped == ["default::local"]
        # Last-active pointer NOT dropped (primary task is still alive)
        assert browser_tool._last_active_session_key.get("default") == "default::local"
