"""Billing / usage / subscription serializers for the TUI RPC surface.

STRUCTURED envelopes (result.ok / result.error) rather than JSON-RPC errors, so
rpc() always resolves and the client branches on the typed billing code.
Data-building lives in agent/billing_view.py + hermes_cli/nous_billing.py.
Bodies are rebound onto server.py's globals at install time (method_ctx.bind_module),
so tests may still monkeypatch e.g. ``server._usage_payload``.
"""

from __future__ import annotations

from typing import Optional

from .method_ctx import bind_module


def _wire_str(value):
    """Decimal/number → wire string (None passes through)."""
    return None if value is None else str(value)


def _serialize_billing_error(exc) -> dict:
    """Map a BillingError into the result.error envelope the TUI branches on."""
    from hermes_cli.nous_billing import (
        BillingRemoteSpendingRevoked, BillingScopeRequired, BillingSessionRevoked, BillingTransient)
    typed = {BillingRemoteSpendingRevoked: "remote_spending_revoked",
             BillingSessionRevoked: "session_revoked", BillingScopeRequired: "insufficient_scope"}
    kind = next((k for cls, k in typed.items() if isinstance(exc, cls)), None)
    if kind is None:
        error = getattr(exc, "error", None)
        fallback = "rate_limited" if isinstance(exc, BillingTransient) else "error"
        kind = str(error) if error else fallback
    return {
        "ok": False, "error": kind, "message": str(exc),
        "portal_url": getattr(exc, "portal_url", None),
        "retry_after": getattr(exc, "retry_after", None),
        "payload": getattr(exc, "payload", {}) or {},
        # Remote-Spending contract extras: the TUI renders actor-aware copy + recovery from these.
        "actor": getattr(exc, "actor", None), "code": getattr(exc, "code", None),
        "recovery": getattr(exc, "recovery", None)}


def _serialize_payment_method(pm) -> dict | None:
    # Each kind sends only its own fields. Emitting every key with nulls would contradict
    # the shared type — a client checking `'brand' in pm` would read every Link method as a card.
    if pm is None:
        return None
    if pm.kind == "card":
        return {
            "kind": "card", "brand": pm.brand, "last4": pm.last4, "wallet": pm.wallet,
            "resolved_via": pm.resolved_via,
        }
    if pm.kind == "link":
        return {"kind": "link", "email": pm.email, "resolved_via": pm.resolved_via}
    return {"kind": "unknown", "raw_kind": pm.raw_kind, "resolved_via": pm.resolved_via}


def _serialize_auto_reload(ar, format_money) -> dict | None:
    if ar is None:
        return None
    card_out = None
    if ar.card is not None:
        card_out = {"kind": ar.card.kind}
        if ar.card.kind == "distinct":
            card_out.update(payment_method_id=ar.card.payment_method_id, brand=ar.card.brand,
                            last4=ar.card.last4)
    return {
        "enabled": ar.enabled, "threshold_usd": _wire_str(ar.threshold_usd),
        "threshold_display": format_money(ar.threshold_usd),
        "reload_to_usd": _wire_str(ar.reload_to_usd),
        "reload_to_display": format_money(ar.reload_to_usd), "card": card_out}


def _serialize_billing_state(state) -> dict:
    """Serialize a BillingState for the wire (Decimals → strings, money-safe)."""
    from agent.billing_view import format_money

    card = mc = None
    if state.card is not None:
        card = {
            "brand": state.card.brand, "last4": state.card.last4, "masked": state.card.masked,
            # None/False on older NAS payloads; resolved_via = rung for rung-gated surfaces.
            "display": state.card.display, "resolved_via": state.card.resolved_via}
    if state.monthly_cap is not None:
        m = state.monthly_cap
        mc = {"limit_usd": _wire_str(m.limit_usd), "limit_display": format_money(m.limit_usd),
              "spent_this_month_usd": _wire_str(m.spent_this_month_usd),
              "spent_display": format_money(m.spent_this_month_usd),
              "is_default_ceiling": m.is_default_ceiling}
    return {
        "ok": True, "logged_in": state.logged_in, "org_name": state.org_name,
        "org_slug": state.org_slug, "role": state.role, "is_admin": state.is_admin,
        "can_change_plan": state.can_change_plan, "can_charge": state.can_charge,
        "balance_usd": _wire_str(state.balance_usd),
        "balance_display": format_money(state.balance_usd),
        "cli_billing_enabled": state.cli_billing_enabled,
        "charge_presets": [_wire_str(p) for p in state.charge_presets],
        "charge_presets_display": [format_money(p) for p in state.charge_presets],
        "min_usd": _wire_str(state.min_usd), "max_usd": _wire_str(state.max_usd),
        "card": card, "payment_method": _serialize_payment_method(state.payment_method),
        "monthly_cap": mc, "auto_reload": _serialize_auto_reload(state.auto_reload, format_money),
        "portal_url": state.portal_url, "error": state.error,
        # Shared two-bar dollar usage model so /topup matches /usage and /subscription; fail-open.
        "usage": _usage_payload(state)}


def _usage_payload(state) -> dict:
    """Shared usage model for the /topup + /subscription bars: only when logged in, fail-open."""
    if not getattr(state, "logged_in", False):
        return {"available": False}
    try:
        from agent.billing_usage import build_usage_model
        return _serialize_usage_model(build_usage_model())
    except Exception:
        return {"available": False}


def _serialize_usage_bar(bar) -> Optional[dict]:
    """Serialize a UsageBar (dollar magnitudes → display strings + fractions)."""
    if bar is None:
        return None
    from agent.billing_usage import _fmt_usd
    return {
        "kind": bar.kind, "remaining_display": _fmt_usd(bar.remaining_usd),
        "total_display": _fmt_usd(bar.total_usd), "spent_display": _fmt_usd(bar.spent_usd),
        "pct_used": bar.pct_used, "fill_fraction": bar.fill_fraction}


def _serialize_usage_model(model) -> dict:
    """Serialize a UsageModel for the wire — the shared two-bar dollar view (fail-open)."""
    from agent.billing_usage import _fmt_usd, format_renews
    if model is None or not getattr(model, "available", False):
        return {"ok": True, "available": False}

    def _usd(value):
        return None if value is None else _fmt_usd(value)
    return {
        "ok": True, "available": True, "status": model.status, "plan_name": model.plan_name,
        "renews_at": model.renews_at,
        "renews_display": getattr(model, "renews_display", None) or format_renews(model.renews_at),
        "subscription_remaining_display": _usd(model.subscription_remaining_usd),
        "topup_remaining_display": _usd(model.topup_remaining_usd),
        "total_spendable_display": _usd(model.total_spendable_usd),
        "has_topup": model.has_topup, "plan_bar": _serialize_usage_bar(model.plan_bar),
        "topup_bar": _serialize_usage_bar(model.topup_bar)}


def _serialize_subscription_state(state) -> dict:
    """Serialize a SubscriptionState for the wire (Decimals → strings)."""
    from agent.billing_usage import format_renews
    from agent.billing_view import format_money

    current = None
    if state.current is not None:
        c = state.current
        current = {
            "tier_id": c.tier_id, "tier_name": c.tier_name,
            "monthly_credits": _wire_str(c.monthly_credits),
            "credits_remaining": _wire_str(c.credits_remaining), "cycle_ends_at": c.cycle_ends_at,
            "pending_downgrade_tier_name": c.pending_downgrade_tier_name,
            "pending_downgrade_at": c.pending_downgrade_at,
            "pending_downgrade_display": format_renews(c.pending_downgrade_at),
            "cancel_at_period_end": c.cancel_at_period_end,
            "cancellation_effective_at": c.cancellation_effective_at,
            "cancellation_effective_display": format_renews(c.cancellation_effective_at)}
    # Selectable catalog for the in-terminal tier picker; price pre-formatted ($X / $X.YY).
    tiers = [
        {"tier_id": t.tier_id, "name": t.name, "tier_order": t.tier_order,
         "dollars_per_month_display": format_money(t.dollars_per_month),
         "monthly_credits": _wire_str(t.monthly_credits), "is_current": t.is_current,
         "is_enabled": t.is_enabled}
        for t in state.tiers]
    return {
        "ok": True, "logged_in": state.logged_in, "is_admin": state.is_admin,
        "can_change_plan": state.can_change_plan, "org_name": state.org_name,
        "org_id": state.org_id,
        "role": state.role, "context": state.context, "current": current, "tiers": tiers,
        "portal_url": state.portal_url, "error": state.error,
        # Shared two-bar usage model (account-info is the only source with top-up dollars);
        # fail-open → {available:false}; lazy when logged out.
        "usage": _usage_payload(state)}


def _serialize_subscription_preview(p) -> dict:
    """Serialize a SubscriptionChangePreview for the wire (Decimal → string)."""
    return {
        "ok": True, "effect": p.effect, "reason": p.reason,
        "current_tier_id": p.current_tier_id, "current_tier_name": p.current_tier_name,
        "target_tier_id": p.target_tier_id, "target_tier_name": p.target_tier_name,
        "monthly_credits_delta": _wire_str(p.monthly_credits_delta),
        "amount_due_now_cents": p.amount_due_now_cents, "effective_at": p.effective_at}


def register(server) -> None:
    """Publish this module's serializers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
