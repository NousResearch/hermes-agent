"""Tripwire tests: the SSRF guard MUST be active whenever the Camofox-labelled backend is remote.

Regression provenance: ``_is_local_backend()`` returned ``True`` for tenant-key gateways
(``CAMOFOX_URL``/``SEKRETO_URL`` pointing at a non-loopback host), silently disabling the
SSRF guard on the hosted-browser lane. ``_is_camofox_mode()`` selects by product family,
not topology; these tests pin the invariant that topology decides.
"""

import pytest

from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_eval_policy as bt_eval_policy


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("SEKRETO_URL", raising=False)
    monkeypatch.delenv("SEKRETO_API_KEY", raising=False)
    monkeypatch.delenv("CAMOFOX_API_KEY", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)


def test_guard_active_on_remote_gateway(monkeypatch):
    """CAMOFOX_URL pointing at a non-loopback host = remote browser = guard ON."""
    monkeypatch.setenv("CAMOFOX_URL", "https://api.gateway-hosted-browser.example")
    monkeypatch.setenv("CAMOFOX_API_KEY", "k" * 8)
    assert bt_cloud._is_local_backend() is False
    assert bt_eval_policy._eval_ssrf_guard_active("test") is True


def test_guard_off_on_loopback_sidecar(monkeypatch):
    """Loopback Camofox sidecar = genuinely local = guard stays off (no behaviour change)."""
    monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
    monkeypatch.setenv("CAMOFOX_API_KEY", "k" * 8)
    assert bt_cloud._is_local_backend() is True
    assert bt_eval_policy._eval_ssrf_guard_active("test") in (False, None) or not bt_eval_policy._eval_ssrf_guard_active("test")


def test_guard_active_on_sekreto_lane(monkeypatch):
    """SEKRETO_URL-only profile (no CAMOFOX vars): gateway detect must still fire."""
    monkeypatch.setenv("SEKRETO_URL", "https://api.sekreto-gw.example")
    monkeypatch.setenv("SEKRETO_API_KEY", "s" * 8)
    assert bt_cloud._is_local_backend() is False
    assert bt_eval_policy._eval_ssrf_guard_active("test") is True