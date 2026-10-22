"""Billing / subscription session handlers (``methods_session`` split).

Moved verbatim from ``tui_gateway/methods_session.py`` (file-line ratchet): the bodies close over
server.py globals through ``method_ctx.bind_module`` exactly as before, and registration runs
from the parent's ``register()`` via ``HandlerRegistry``. Serializers live in ``billing_view``.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


# ── billing / subscription ───────────────────────────────────────────
# All fail-open: a logged-out / unreachable portal yields an ``ok`` envelope with a typed
# ``error`` (not a JSON-RPC error) so the TUI maps it to copy. ``billing:manage`` routes
# return error=insufficient_scope on 403, which drives the ``billing.step_up`` device flow.
def _billing_view(name: str, module: str, builder: str, serializer: str, fallback: dict) -> None:
    """Read-only view RPC (no scope required): ``serializer(module.builder())``, ``fallback`` on any error.
    The view module stays a lazy import (startup budget); the serializer is a server global."""
    @method(name)
    def _(rid, params: dict) -> dict:
        try:
            from importlib import import_module
            return _ok(rid, globals()[serializer](getattr(import_module(module), builder)()))
        except Exception:
            return _ok(rid, dict(fallback))


@method("billing.state")
def _(rid, params: dict) -> dict:
    """Read-only billing view (no scope required); fail-open. The Nous free tier has no account to
    bill, so its state is answered locally (``free_tier_account`` set, ``logged_in`` false) without a portal
    round-trip that could only fail."""
    try:
        from agent.billing_view import BillingState, build_billing_state
        from hermes_cli.anon_auth import has_free_tier_account
        if has_free_tier_account():
            return _ok(rid, _serialize_billing_state(BillingState(logged_in=False), free_tier_account=True))
        return _ok(rid, _serialize_billing_state(build_billing_state()))
    except Exception:  # health: allow BLE001 -- moved unchanged from methods_session (main keeps the same handler): billing.state is fail-open by contract
        return _ok(rid, {"ok": True, "logged_in": False, "free_tier_account": False, "error": "could not load billing state"})


_billing_view("usage.bars", "agent.billing_usage", "build_usage_model", "_serialize_usage_model",  # two-bar $ view
              {"ok": True, "available": False})
_billing_view("subscription.state", "agent.subscription_view", "build_subscription_state",
              "_serialize_subscription_state",
              {"ok": True, "logged_in": False, "error": "could not load subscription state"})


@method("subscription.preview")
def _(rid, params: dict) -> dict:
    """POST /api/billing/subscription/preview → chargeless effect quote. billing:manage."""
    from agent.subscription_view import subscription_change_preview_from_payload
    from hermes_cli.nous_billing import post_subscription_preview
    if not (tier_id := params.get("subscription_type_id")):
        return _billing_invalid(rid, "subscription_type_id is required")
    return _billing_call(rid, lambda: _serialize_subscription_preview(
        subscription_change_preview_from_payload(post_subscription_preview(subscription_type_id=tier_id))))


def _billing_route(name: str, call, *, invalid=None, message: str = "", error: str = "invalid_request",
                   idempotent: bool = False):
    """Portal write route on ``hermes_cli.nous_billing`` (lazy; tests patch its functions): ``invalid(params)``
    → ``_billing_invalid(message, error)``; ``call(nb, params, key)`` performs the request. ``idempotent``
    mints ``idempotency_key`` if absent and echoes it (also on error) so the TUI retries the SAME operation."""
    @method(name)
    def _(rid, params: dict) -> dict:
        import hermes_cli.nous_billing as nb
        if invalid is not None and invalid(params):
            return _billing_invalid(rid, message, error=error)
        key = extra = None
        if idempotent:
            from agent.billing_view import new_idempotency_key
            key = params.get("idempotency_key") or new_idempotency_key()
            extra = {"idempotency_key": key}
        return _billing_call(rid, lambda: call(nb, params, key) | (extra or {}), extra=extra)


# PUT pending-change: schedule a downgrade / same-price change OR a period-end cancellation.
_billing_route("subscription.change", lambda nb, p, _k: _billing_pending_change(nb.put_subscription_pending_change(
    subscription_type_id=p.get("subscription_type_id"), cancel=bool(p.get("cancel")))),
    invalid=lambda p: not p.get("cancel") and not p.get("subscription_type_id"),
    message="subscription_type_id or cancel is required")
# DELETE pending-change: clear a scheduled downgrade / cancellation (re-enables recurring spend).
_billing_route("subscription.resume",
               lambda nb, p, _k: _billing_pending_change(nb.delete_subscription_pending_change()))
# The money route (prorate + charge + flip plan). SCA / decline → status requires_action / payment_failed +
# recovery_url.
_billing_route("subscription.upgrade", lambda nb, p, key: _billing_pick(
    nb.post_subscription_upgrade(subscription_type_id=p.get("subscription_type_id"), idempotency_key=key),
    status="status", target_tier_name="targetTierName", recovery_url="recoveryUrl", reason="reason"),
    invalid=lambda p: not p.get("subscription_type_id"), message="subscription_type_id is required", idempotent=True)
# POST /api/billing/charge → {ok, charge_id, idempotency_key}.
_billing_route("billing.charge", lambda nb, p, key: _billing_pick(
    nb.post_charge(amount_usd=p.get("amount_usd"), idempotency_key=key), charge_id="chargeId"),
    invalid=lambda p: p.get("amount_usd") is None, message="amount_usd is required", idempotent=True)
# GET /api/billing/charge/{id} — a single status read; the caller drives the poll cadence.
_billing_route("billing.charge_status", lambda nb, p, _k: _billing_pick(
    nb.get_charge_status(p.get("charge_id")), status="status", amount_usd="amountUsd", settled_at="settledAt",
    reason="reason"), invalid=lambda p: not p.get("charge_id"), message="charge_id is required",
    error="invalid_charge_id")


def _auto_reload(nb, p: dict, _key) -> dict:
    """PATCH /api/billing/auto-top-up. params: {enabled, threshold, top_up_amount}."""
    nb.patch_auto_top_up(enabled=bool(p.get("enabled")), threshold=p.get("threshold"),
                         top_up_amount=p.get("top_up_amount"))
    return {"ok": True}


_billing_route("billing.auto_reload", _auto_reload, message="threshold and top_up_amount are required",
               invalid=lambda p: p.get("threshold") is None or p.get("top_up_amount") is None)


@method("billing.step_up")
def _(rid, params: dict) -> dict:
    """billing:manage step-up device flow → {ok, granted} (false when the server downscopes). Pooled (blocks
    for minutes); URL/code reach the TUI via ``billing.step_up.verification`` (stdout is the RPC pipe) and the
    browser opens TUI-side, never via the gateway's headless webbrowser.open."""
    sid = params.get("session_id") or ""

    def call():
        from hermes_cli.auth import step_up_nous_billing_scope
        granted = step_up_nous_billing_scope(
            open_browser=False,
            on_verification=lambda url, code: _emit(
                "billing.step_up.verification", sid, {"verification_url": url, "user_code": code}))
        return {"ok": True, "granted": bool(granted)}
    return _billing_call(rid, call, extra={"granted": False})


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
