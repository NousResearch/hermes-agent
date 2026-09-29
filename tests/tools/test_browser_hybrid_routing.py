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
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_session as bt_session
from tools import browser_tool_cloud as bt_cloud
from tools.browser_task_identity import BrowserTaskKey, browser_task_key


@pytest.fixture(autouse=True)
def _reset_routing_state(monkeypatch):
    """Clear module-level caches so each test starts clean."""
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    monkeypatch.setattr(browser_tool, "_cached_cloud_provider", None)
    monkeypatch.setattr(browser_tool, "_cloud_provider_resolved", False)
    monkeypatch.setattr(browser_tool, "_auto_local_for_private_urls_resolved", False)
    monkeypatch.setattr(browser_tool, "_cached_auto_local_for_private_urls", True)
    monkeypatch.setattr("tools.browser_tool_lifecycle._start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr("tools.browser_tool_lifecycle._update_session_activity", lambda t: None)
    # Default: no CDP override, no Camofox
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: None)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)


class TestNavigationSessionKey:
    """Tests for _navigation_session_key URL-based routing decisions."""

    def test_public_url_uses_bare_task_id(self, monkeypatch):
        """Public URL with cloud provider configured → bare task_id (cloud)."""
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "https://github.com/x/y")
        assert key == browser_task_key("default")
        assert isinstance(key, BrowserTaskKey)
        assert key.local is False

    def test_localhost_routes_to_local_sidecar(self, monkeypatch):
        """``localhost`` URL → ``::local`` suffix when cloud configured + flag on."""
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "http://localhost:3000/")
        assert key == browser_task_key("default").with_local(True)
        assert key.local is True


    def test_rfc1918_lan_routes_to_local_sidecar(self, monkeypatch):
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key("default", "http://192.168.1.50:8000/")
        assert key == browser_task_key("default").with_local(True)


    def test_none_task_id_defaults(self, monkeypatch):
        """``None`` task_id resolves to 'default'."""
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: Mock())
        key = browser_tool._navigation_session_key(None, "http://localhost:3000/")
        assert key == browser_task_key().with_local(True)

    def test_raw_local_suffix_is_an_ordinary_task_id(self):
        """Only typed internal keys can designate a sidecar."""
        raw_suffix = browser_task_key("default::local")

        assert raw_suffix.owner_task_id == "default::local"
        assert raw_suffix.local is False
        assert browser_tool._is_local_sidecar_key(raw_suffix) is False


class TestSessionKeyHelpers:


    def test_last_session_key_drops_mismatched_owner_metadata(self, monkeypatch):
        """Explicit ownership metadata prevents retargeting to another task's session."""
        owner = browser_task_key("default")
        other_sidecar = browser_task_key("other-task").with_local(True)
        last_active = {owner: other_sidecar}
        monkeypatch.setattr(browser_tool, "_last_active_session_key", last_active)
        monkeypatch.setattr(
            browser_tool,
            "_active_sessions",
            {
                other_sidecar: {
                    "session_name": "local_sess",
                    "session_key": other_sidecar,
                    "owner_task_id": browser_task_key("other-task"),
                }
            },
        )

        assert browser_tool._last_session_key(owner) == owner
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
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_tool_cdp._ensure_cdp_supervisor", lambda t: None)

        sidecar = browser_task_key("default").with_local(True)
        session = bt_session._get_session_info(sidecar)

        assert provider.create_session.call_count == 0
        assert session["bb_session_id"] is None
        assert session["cdp_url"] is None
        assert session["features"]["local"] is True
        assert session["session_key"] == sidecar
        assert session["owner_task_id"] == sidecar.with_local(False)

    def test_bare_task_id_with_cloud_provider_uses_cloud(self, monkeypatch):
        """A bare task_id with cloud provider configured hits the cloud path."""
        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "cloud-sess",
            "bb_session_id": "bb_123",
            "cdp_url": "wss://real.browserbase.com/ws",
        }
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_tool_cdp._ensure_cdp_supervisor", lambda t: None)
        monkeypatch.setattr("tools.browser_tool_cdp._resolve_cdp_override", lambda u: u)

        task_key = browser_task_key("default")
        session = bt_session._get_session_info(task_key)

        assert provider.create_session.call_count == 1
        assert session["bb_session_id"] == "bb_123"
        assert session["session_key"] == task_key
        assert session["owner_task_id"] == task_key


class TestCleanupHybridSessions:
    """cleanup_browser(bare_task_id) must reap both cloud + local sidecar sessions."""

    def test_cleanup_reaps_both_primary_and_sidecar(self, monkeypatch):
        """Given a bare task_id with both sessions alive, both get cleaned."""
        reaped = []

        def _fake_cleanup_one(key):
            reaped.append(key)

        task_key = browser_task_key("default")
        sidecar_key = task_key.with_local(True)
        monkeypatch.setattr(bt_lifecycle, "_cleanup_single_browser_session", _fake_cleanup_one)
        monkeypatch.setattr(
            browser_tool,
            "_active_sessions",
            {
                task_key: {"session_name": "cloud_sess"},
                sidecar_key: {"session_name": "local_sess"},
            },
        )
        monkeypatch.setattr(
            browser_tool, "_last_active_session_key", {task_key: sidecar_key}
        )

        bt_lifecycle.cleanup_browser(task_key)

        assert set(reaped) == {task_key, sidecar_key}
        # last-active pointer dropped
        assert task_key not in browser_tool._last_active_session_key


    def test_cleanup_sidecar_directly_keeps_primary(self, monkeypatch):
        """Calling cleanup with a ``::local`` key reaps only the sidecar."""
        reaped = []

        def _fake_cleanup_one(key):
            reaped.append(key)

        task_key = browser_task_key("default")
        sidecar_key = task_key.with_local(True)
        monkeypatch.setattr(bt_lifecycle, "_cleanup_single_browser_session", _fake_cleanup_one)
        monkeypatch.setattr(
            browser_tool,
            "_active_sessions",
            {
                task_key: {"session_name": "cloud_sess"},
                sidecar_key: {"session_name": "local_sess"},
            },
        )
        monkeypatch.setattr(
            browser_tool, "_last_active_session_key", {task_key: sidecar_key}
        )

        bt_lifecycle.cleanup_browser(sidecar_key)

        assert reaped == [sidecar_key]
        # The cleaned sidecar must not remain the recorded owner; otherwise a
        # later click/snapshot could resurrect it instead of using the primary.
        assert task_key not in browser_tool._last_active_session_key
