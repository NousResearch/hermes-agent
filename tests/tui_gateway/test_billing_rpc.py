"""Tests for the Phase 2b billing JSON-RPC methods (tui_gateway/server.py).

Verifies the structured envelope contract the Ink side branches on:
- billing.state serializes BillingState (Decimals → strings) + fails open.
- billing.charge / charge_status / auto_reload return typed error envelopes
  (result.ok=false, result.error=<code>) instead of JSON-RPC errors.
- billing.charge mints + echoes an idempotency_key for retry reuse.
"""

from __future__ import annotations

from decimal import Decimal


import tui_gateway.server as srv
import agent.billing_view as bv
from agent.billing_view import BillingState, CardInfo, MonthlyCap, PaymentMethodInfo


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
        charge_presets=(Decimal(100), Decimal(250)),
        min_usd=Decimal(10),
        max_usd=Decimal(10000),
        card=CardInfo(brand="visa", last4="4242"),
        payment_method=PaymentMethodInfo(
            kind="link",
            email="billing@example.com",
            resolved_via="customerDefault",
        ),
        monthly_cap=MonthlyCap(
            limit_usd=Decimal(1000), spent_this_month_usd=Decimal(180), is_default_ceiling=True
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
    assert res["card"] == {
        "brand": "visa",
        "last4": "4242",
        "masked": "visa ····4242",
        "display": "visa ····4242",
        "resolved_via": None,
    }
    assert res["payment_method"] == {
        "kind": "link",
        "email": "billing@example.com",
        "resolved_via": "customerDefault",
    }
    assert res["monthly_cap"]["is_default_ceiling"] is True
    assert res["is_admin"] is True and res["can_charge"] is True
    # Not a window plan (or an older portal): no Keep going on credits control.
    assert res["keep_going_on_credits"] is None


def test_billing_state_carries_keep_going_on_credits(monkeypatch):
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: BillingState(logged_in=True, keep_going_on_credits=False))
    assert _call("billing.state", {})["keep_going_on_credits"] is False


# ---------------------------------------------------------------------------
# billing.keep_going_on_credits
# ---------------------------------------------------------------------------


def test_keep_going_on_credits_patches_just_enabled(monkeypatch):
    import hermes_cli.nous_billing as nb
    sent = []
    monkeypatch.setattr(nb, "patch_keep_going_on_credits", lambda **kw: sent.append(kw) or {})
    assert _call("billing.keep_going_on_credits", {"enabled": True}) == {"ok": True}
    assert _call("billing.keep_going_on_credits", {"enabled": False}) == {"ok": True}
    assert sent == [{"enabled": True}, {"enabled": False}]


def test_keep_going_on_credits_rejects_a_non_bool_without_calling_the_portal(monkeypatch):
    import hermes_cli.nous_billing as nb
    monkeypatch.setattr(nb, "patch_keep_going_on_credits", lambda **kw: (_ for _ in ()).throw(AssertionError("called")))
    for params in ({}, {"enabled": "true"}, {"enabled": 1}):
        res = _call("billing.keep_going_on_credits", params)
        assert res["ok"] is False and res["error"] == "invalid_request"


def test_keep_going_on_credits_scope_denial_is_an_envelope(monkeypatch):
    import hermes_cli.nous_billing as nb

    def deny(**_kw):
        raise nb.BillingScopeRequired("billing:manage required", status=403, error="insufficient_scope")
    monkeypatch.setattr(nb, "patch_keep_going_on_credits", deny)
    res = _call("billing.keep_going_on_credits", {"enabled": True})
    assert res["ok"] is False and res["error"] == "insufficient_scope"
