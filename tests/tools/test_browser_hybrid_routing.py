"""Tests for hybrid browser-backend routing (LAN/localhost auto-local).

When a cloud browser provider (Browserbase / Browser-Use / Firecrawl) is
configured globally, ``browser.auto_local_for_private_urls`` (default True)
routes a ``browser_exec`` call whose code carries a URL whose host resolves to a
private/loopback/LAN address to a local Chromium sidecar (the ``::local`` session
key), while public URLs continue to hit the cloud session in the same conversation.

These tests cover the routing decision layer — session_key selection, sidecar
detection, task ownership, and sidecar-aware cleanup. The downstream session
creation is covered by test_browser_cloud_fallback.py.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.bot_desktop import placement
from tools import browser_launcher
from tools import browser_session as bs
from tools import browser_session_cloud as bs_cloud
from tools import browser_session_lifecycle as lifecycle
from tools import browser_use_cli as bu

_LOCAL_CDP = "ws://127.0.0.1:9444/devtools/browser/sidecar"


@pytest.fixture(autouse=True)
def _reset_routing_state(monkeypatch):
    """Clear module-level caches so each test starts clean."""
    monkeypatch.setattr(bs, "_active_sessions", {})
    monkeypatch.setattr("tools.browser_session_lifecycle.start_cleanup_thread", lambda: None)
    monkeypatch.setattr("tools.browser_session_lifecycle.update_session_activity", lambda t: None)
    monkeypatch.setattr(bs_cloud, "auto_local_for_private_urls", lambda: True)
    # Default: no CDP override
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: None)
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override_raw", lambda: "")
    monkeypatch.setattr("tools.bot_desktop.runtime.tool_placement", lambda: placement.GATEWAY)  # host browser
    monkeypatch.setattr(browser_launcher, "launch_chromium",
                        lambda key, **kw: (SimpleNamespace(cdp_url=_LOCAL_CDP, pid=4242), None))


def _exec_session_key(task_id, url: str) -> str:
    """The session key browser_exec routes a call carrying ``url`` to."""
    verdict = bu._code_url_verdict(f'new_tab("{url}")')
    return bu._backend_cache_key(task_id, sidecar=verdict.local_sidecar)


class TestNavigationSessionKey:
    """URL-based routing decisions for a browser_exec call."""

    def test_public_url_uses_bare_task_id(self, monkeypatch):
        """Public URL with cloud provider configured → bare task_id (cloud)."""
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: Mock())
        key = _exec_session_key("default", "https://github.com/x/y")
        assert key == "default"

    def test_localhost_routes_to_local_sidecar(self, monkeypatch):
        """``localhost`` URL → ``::local`` suffix when cloud configured + flag on."""
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: Mock())
        key = _exec_session_key("default", "http://localhost:3000/")
        assert key == "default::local"


    def test_rfc1918_lan_routes_to_local_sidecar(self, monkeypatch):
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: Mock())
        key = _exec_session_key("default", "http://192.168.1.50:8000/")
        assert key == "default::local"


    def test_none_task_id_defaults(self, monkeypatch):
        """``None`` task_id resolves to browser_exec's default key."""
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: Mock())
        key = _exec_session_key(None, "http://localhost:3000/")
        assert key == "browser-exec-default::local"


class TestSessionKeyHelpers:


    def test_task_lookup_ignores_another_tasks_sidecar(self, monkeypatch):
        """Explicit ownership metadata prevents retargeting to another task's session."""
        monkeypatch.setattr(
            bs,
            "_active_sessions",
            {
                "other-task::local": {
                    "session_name": "local_sess",
                    "session_key": "other-task::local",
                    "owner_task_id": "other-task",
                }
            },
        )

        assert bs.session_for_task("default") == {}


class TestHybridRoutingSessionCreation:
    """get_session_info must force a local session when the key carries ``::local``."""

    def test_local_sidecar_key_skips_cloud_provider(self, monkeypatch):
        """A ``::local``-suffixed key creates a local session even when cloud is set."""
        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "should_not_be_used",
            "bb_session_id": "bb_xxx",
            "cdp_url": "wss://fake.browserbase.com/ws",
        }
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_session_cdp.ensure_cdp_supervisor", lambda t: None)

        session = bs.get_session_info("default::local")

        assert provider.create_session.call_count == 0
        assert session["bb_session_id"] is None
        assert session["cdp_url"] == _LOCAL_CDP
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
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_session_cdp.ensure_cdp_supervisor", lambda t: None)
        monkeypatch.setattr("tools.browser_session_cdp.resolve_cdp_override", lambda u: u)

        session = bs.get_session_info("default")

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

        monkeypatch.setattr(lifecycle, "cleanup_single_session", _fake_cleanup_one)
        monkeypatch.setattr(
            bs,
            "_active_sessions",
            {
                "default": {"session_name": "cloud_sess"},
                "default::local": {"session_name": "local_sess"},
            },
        )

        lifecycle.cleanup_browser("default")

        assert set(reaped) == {"default", "default::local"}


    def test_cleanup_sidecar_directly_keeps_primary(self, monkeypatch):
        """Calling cleanup with a ``::local`` key reaps only the sidecar."""
        reaped = []

        def _fake_cleanup_one(key):
            reaped.append(key)

        monkeypatch.setattr(lifecycle, "cleanup_single_session", _fake_cleanup_one)
        monkeypatch.setattr(
            bs,
            "_active_sessions",
            {
                "default": {"session_name": "cloud_sess"},
                "default::local": {"session_name": "local_sess"},
            },
        )

        lifecycle.cleanup_browser("default::local")

        assert reaped == ["default::local"]
