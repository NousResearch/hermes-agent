"""``free_tier.status`` / ``free_tier.ack_notice`` (pull) and the free-tier branch of ``billing.state``,
driven through the registered RPC handlers against a seeded free-tier identity."""

from __future__ import annotations

import base64
import json
import time

import pytest

import tui_gateway.server as srv
from hermes_cli import anon_auth
from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store


def _jwt(**claims) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": "nas_user:1", "client_id": "nas-anonymous", "account_tier": "anonymous",
               "scope": "inference:invoke tool:invoke", "exp": int(time.time()) + 900, **claims}
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.sig"


def _call(method: str, params: dict | None = None) -> dict:
    return srv._methods[method](1, params or {})["result"]


@pytest.fixture
def guest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
    monkeypatch.delenv("HERMES_FORCE_GUEST", raising=False)
    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("providers", {})["nous"] = {
            "auth_method": anon_auth.ANON_AUTH_METHOD, "account_tier": "anonymous", "anon_token": "anon_0001",
            "client_id": "nas-anonymous", "access_token": _jwt(), "expires_at": "2999-01-01T00:00:00+00:00",
            "inference_base_url": "https://welcome-api.nousresearch.com/v1"}
        store["active_provider"] = "nous"
        _save_auth_store(store)


def _set_guest_off(monkeypatch):
    from hermes_cli import config as cfg_mod
    monkeypatch.setattr(anon_auth, "guest_enabled", lambda: False)
    return cfg_mod


def test_status_is_pull_from_local_state_and_ack_persists_on_the_identity(guest, monkeypatch):
    status = _call("free_tier.status")
    assert status == {"has_guest": True, "enabled": True, "available": True, "notice_pending": True,
                      "model": "nous/welcome", "label": anon_auth.FREE_TIER_LABEL}

    assert _call("free_tier.ack_notice") == {"acked": True}
    assert _call("free_tier.status")["notice_pending"] is False
    assert _load_auth_store()["providers"]["nous"][anon_auth.GUEST_NOTICE_FLAG] is True

    # nous.guest: false -> the identity exists but carries nothing; no notice either.
    _set_guest_off(monkeypatch)
    status = _call("free_tier.status")
    assert status["has_guest"] is True and status["enabled"] is False
    assert status["available"] is False and status["notice_pending"] is False


def test_billing_state_answers_the_free_tier_locally(guest, monkeypatch):
    import agent.billing_view as bv
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: pytest.fail("free tier must not call the portal"))
    res = _call("billing.state")
    assert res["ok"] is True and res["logged_in"] is False
    assert res["free_tier"] is True and res["free_tier_model"] == "nous/welcome"
    assert res["usage"] == {"available": False}

    _set_guest_off(monkeypatch)
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: bv.BillingState(logged_in=False))
    res = _call("billing.state")
    assert res["free_tier"] is False and res["free_tier_model"] is None


def test_status_without_an_identity_starts_the_background_setup_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
    calls = []
    monkeypatch.setattr(anon_auth, "ensure_portal_identity", lambda **kw: calls.append(kw) or None)
    status = _call("free_tier.status")
    assert status["has_guest"] is False and status["available"] is False
    assert calls == [{"blocking": False}]
