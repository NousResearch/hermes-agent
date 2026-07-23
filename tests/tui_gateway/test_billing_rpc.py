"""Tests for the Phase 2b billing JSON-RPC methods (tui_gateway/server.py).

Verifies the structured envelope contract the Ink side branches on:
- billing.state serializes BillingState (Decimals → strings) + fails open.
- billing.charge / charge_status / auto_reload return typed error envelopes
  (result.ok=false, result.error=<code>) instead of JSON-RPC errors.
- billing.charge mints + echoes an idempotency_key for retry reuse.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import tui_gateway.server as srv
import hermes_cli.nous_billing as nb
import agent.billing_view as bv
from agent.billing_view import BillingState, CardInfo, MonthlyCap


def _call(method: str, params: dict) -> dict:
    """Invoke a registered RPC method and return its result dict."""
    envelope = srv._methods[method](1, params)
    return envelope["result"]


# ---------------------------------------------------------------------------
# billing.state
# ---------------------------------------------------------------------------


def test_billing_state_serializes_decimals_as_strings(monkeypatch):
    state = BillingState(
        logged_in=True,
        org_name="Acme",
        role="OWNER",
        balance_usd=Decimal("142.5"),
        cli_billing_enabled=True,
        charge_presets=(Decimal("100"), Decimal("250")),
        min_usd=Decimal("10"),
        max_usd=Decimal("10000"),
        card=CardInfo(brand="visa", last4="4242"),
        monthly_cap=MonthlyCap(
            limit_usd=Decimal("1000"), spent_this_month_usd=Decimal("180"), is_default_ceiling=True
        ),
        portal_url="https://portal/billing?topup=open",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    res = _call("billing.state", {})
    assert res["ok"] is True and res["logged_in"] is True
    # Money on the wire is STRING, not float/number.
    assert res["balance_usd"] == "142.5"
    assert res["balance_display"] == "$142.50"
    assert res["charge_presets"] == ["100", "250"]
    assert res["card"]["masked"] == "visa ····4242"
    assert res["monthly_cap"]["is_default_ceiling"] is True
    assert res["is_admin"] is True and res["can_charge"] is True


def test_billing_state_fail_open(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("portal down")

    monkeypatch.setattr(bv, "build_billing_state", _boom)
    res = _call("billing.state", {})
    assert res["ok"] is True and res["logged_in"] is False


# ---------------------------------------------------------------------------
# billing.charge — typed error envelopes
# ---------------------------------------------------------------------------


def test_billing_charge_success_echoes_charge_id(monkeypatch):
    monkeypatch.setattr(nb, "post_charge", lambda **kw: {"chargeId": "ch_123"})
    res = _call("billing.charge", {"amount_usd": "100", "idempotency_key": "key-1"})
    assert res["ok"] is True
    assert res["charge_id"] == "ch_123"
    assert res["idempotency_key"] == "key-1"


def test_billing_charge_mints_key_when_absent(monkeypatch):
    seen = {}

    def _post(**kw):
        seen["key"] = kw["idempotency_key"]
        return {"chargeId": "ch_x"}

    monkeypatch.setattr(nb, "post_charge", _post)
    res = _call("billing.charge", {"amount_usd": "50"})
    assert res["ok"] is True
    assert res["idempotency_key"] == seen["key"]  # minted key echoed back
    assert len(res["idempotency_key"]) == 36


def test_billing_charge_insufficient_scope_envelope(monkeypatch):
    def _post(**kw):
        raise nb.BillingScopeRequired("need scope", status=403, error="insufficient_scope")

    monkeypatch.setattr(nb, "post_charge", _post)
    res = _call("billing.charge", {"amount_usd": "100", "idempotency_key": "k"})
    assert res["ok"] is False
    assert res["error"] == "insufficient_scope"
    assert res["idempotency_key"] == "k"  # preserved for reuse post-stepup


def test_billing_charge_no_payment_method_envelope(monkeypatch):
    def _post(**kw):
        raise nb.BillingError(
            "no reusable card", status=403, error="no_payment_method",
            portal_url="/billing?topup=open",
        )

    monkeypatch.setattr(nb, "post_charge", _post)
    res = _call("billing.charge", {"amount_usd": "100", "idempotency_key": "k"})
    assert res["ok"] is False
    assert res["error"] == "no_payment_method"
    assert res["portal_url"] == "/billing?topup=open"


def test_billing_charge_rate_limited_envelope(monkeypatch):
    def _post(**kw):
        raise nb.BillingRateLimited("slow down", status=429, error="rate_limited", retry_after=60)

    monkeypatch.setattr(nb, "post_charge", _post)
    res = _call("billing.charge", {"amount_usd": "100", "idempotency_key": "k"})
    assert res["error"] == "rate_limited"
    assert res["retry_after"] == 60


# ---------------------------------------------------------------------------
# billing.charge_status — the poll
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server_resp,expected",
    [
        ({"status": "pending"}, {"status": "pending"}),
        (
            {"status": "settled", "amountUsd": "50", "settledAt": "2026-06-13T00:00:00Z"},
            {"status": "settled", "amount_usd": "50"},
        ),
        ({"status": "failed", "reason": "card_declined"}, {"status": "failed", "reason": "card_declined"}),
    ],
)
def test_billing_charge_status_maps_fields(monkeypatch, server_resp, expected):
    monkeypatch.setattr(nb, "get_charge_status", lambda cid, **kw: server_resp)
    res = _call("billing.charge_status", {"charge_id": "ch_1"})
    assert res["ok"] is True
    for k, v in expected.items():
        assert res[k] == v


def test_billing_charge_status_requires_id():
    res = _call("billing.charge_status", {})
    assert res["ok"] is False and res["error"] == "invalid_charge_id"


# ---------------------------------------------------------------------------
# billing.auto_reload
# ---------------------------------------------------------------------------


def test_billing_auto_reload_success(monkeypatch):
    seen = {}
    monkeypatch.setattr(nb, "patch_auto_top_up", lambda **kw: seen.update(kw) or {"ok": True})
    res = _call("billing.auto_reload", {"enabled": True, "threshold": 20, "top_up_amount": 100})
    assert res["ok"] is True
    assert seen == {"enabled": True, "threshold": 20, "top_up_amount": 100}


def test_billing_auto_reload_validation_error_envelope(monkeypatch):
    def _patch(**kw):
        raise nb.BillingError("bad", status=400, error="validation_failed")

    monkeypatch.setattr(nb, "patch_auto_top_up", _patch)
    res = _call("billing.auto_reload", {"enabled": True, "threshold": 20, "top_up_amount": 100})
    assert res["ok"] is False and res["error"] == "validation_failed"


def test_billing_auto_reload_requires_fields():
    res = _call("billing.auto_reload", {"enabled": True})
    assert res["ok"] is False and res["error"] == "invalid_request"


# ---------------------------------------------------------------------------
# billing.step_up
# ---------------------------------------------------------------------------


def test_billing_step_up_granted(monkeypatch):
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "step_up_nous_billing_scope", lambda **kw: True)
    res = _call("billing.step_up", {})
    assert res["ok"] is True and res["granted"] is True


def test_billing_step_up_downscoped(monkeypatch):
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "step_up_nous_billing_scope", lambda **kw: False)
    res = _call("billing.step_up", {})
    assert res["ok"] is True and res["granted"] is False
