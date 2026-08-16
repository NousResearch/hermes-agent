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
    monkeypatch.setattr(browser_tool, "_session_last_activity", {})
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    monkeypatch.setattr(browser_tool, "_retired_browser_tasks", set())
    monkeypatch.setattr(browser_tool, "_browser_task_states", {})
    monkeypatch.setattr(browser_tool, "_browser_task_generations", {})
    monkeypatch.setattr(browser_tool, "_browser_task_cleanup_reasons", {})
    monkeypatch.setattr(browser_tool, "_pending_provider_cleanups", {})
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


    def test_rfc1918_lan_routes_to_local_sidecar(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "http://192.168.1.50:8000/")
        assert key == "default::local"


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


    def test_last_session_key_drops_mismatched_owner_metadata(self, monkeypatch):
        """Explicit ownership metadata prevents retargeting to another task's session."""
        last_active = {"default": "other-task::local"}
        monkeypatch.setattr(browser_tool, "_last_active_session_key", last_active)
        monkeypatch.setattr(
            browser_tool,
            "_active_sessions",
            {
                "other-task::local": {
                    "session_name": "local_sess",
                    "session_key": "other-task::local",
                    "owner_task_id": "other-task",
                }
            },
        )

        assert browser_tool._last_session_key("default") == "default"
        assert last_active == {}


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
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda *_a, **_kw: None)

        session = browser_tool._get_session_info("default::local")

        assert provider.create_session.call_count == 0
        assert session["bb_session_id"] is None
        assert session["cdp_url"] is None
        assert session["features"]["local"] is True
        assert session["session_key"] == "default::local"
        assert session["owner_task_id"] == "default"

    def test_bare_task_id_with_cloud_provider_uses_cloud(self, monkeypatch):
        """A bare task_id with cloud provider configured hits the cloud path."""
        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "cloud-sess",
            "bb_session_id": "bb_123",
            "cdp_url": "wss://real.browserbase.com/ws",
        }
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda *_a, **_kw: None)
        monkeypatch.setattr(browser_tool, "_resolve_cdp_override", lambda u: u)

        session = browser_tool._get_session_info("default")

        assert provider.create_session.call_count == 1
        assert session["bb_session_id"] == "bb_123"
        assert session["session_key"] == "default"
        assert session["owner_task_id"] == "default"


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
        # The cleaned sidecar must not remain the recorded owner; otherwise a
        # later click/snapshot could resurrect it instead of using the primary.
        assert "default" not in browser_tool._last_active_session_key
