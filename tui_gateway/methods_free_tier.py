"""Nous free-tier JSON-RPC handlers. ``free_tier.status`` answers from the
auth store with zero network and zero side effects; ``free_tier.provision`` retries a welcome-trial claim
that failed (desktop-only entry); ``free_tier.ack_notice``
persists the one-time notice flag on the free-tier identity itself, so it dies with that identity.
``free_tier.challenge_result`` records a presentation hint for the pending attempt; the
backend still exchanges with the portal to determine whether the credential is cleared.
Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

import logging

from .method_ctx import HandlerRegistry, bind_module

logger = logging.getLogger(__name__)
_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


@method("free_tier.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """``{has_guest, enabled, available, notice_pending, model, label}`` for the focused profile.
    ``available``: a guest that carries inference here (the welcome trial, on the desktop's local
    backend). ``enabled``: available, or the trial can still be claimed on this machine. Whether
    inference actually runs on it is a ROUTE question answered by ``setup.runtime_check.free_tier``.
    ``notice_pending`` is true until ``free_tier.ack_notice`` ran for this identity.
    ``nudge_due_in``: seconds until the sign-in offer after a finished task is due (install-wide,
    ``hermes_cli.free_tier_offer``); absent when none is pending. ``connector_nudge_due_in``: the
    same for a guest without the trial that has connected an app.

    A pure read: an identity is never created as a side effect of polling this (NS-845 Q1.2)."""
    try:
        from hermes_cli import anon_auth
        has_guest = anon_auth.has_guest()
        available = anon_auth.free_inference_active()
        enabled = available or _trial_claimable()
        payload = {
            "has_guest": has_guest, "enabled": enabled, "available": available,
            "notice_pending": bool(available and anon_auth.guest_notice_pending()),
            "model": anon_auth.GUEST_MODEL, "label": anon_auth.FREE_TIER_LABEL}
        from hermes_cli import free_tier_offer
        for key, kind in (("nudge_due_in", free_tier_offer.TRIAL), ("connector_nudge_due_in", free_tier_offer.CONNECTOR)):
            if (due_in := free_tier_offer.offer_due_in(kind)) is not None:
                payload[key] = due_in
        if enabled and not has_guest:
            # Why there is no identity, when the last claim failed:
            # ``{error, error_code, retryable, retry_after}`` (the mint memo's verdict).
            payload.update(anon_auth.last_mint_failure() or {})
        # A browser challenge the account service is waiting on (``anon_challenge``): the same
        # payload the ``free_tier.challenge`` event carried, for a client that connected after it.
        from hermes_cli import anon_challenge
        challenge = anon_challenge.pending_challenge()
        if challenge:
            payload["challenge"] = challenge
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5090, str(e))


def _trial_claimable() -> bool:
    from hermes_cli.welcome_trial import claim_blocker
    return claim_blocker() is None


@method("free_tier.provision")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """The user's retry of a welcome-trial claim that failed: claims it when nothing blocks it here
    (blocking, short timeout; the one attempt allowed inside the mint memo's cooldown). ``{has_guest,
    enabled}``, plus ``{error, error_code, retryable, retry_after}`` when the portal refused."""
    try:
        from hermes_cli import anon_auth, free_tier_bootstrap, welcome_trial
        from hermes_cli.auth_constants import AuthError
        if _trial_claimable():
            try:
                welcome_trial.claim_desktop_free_tier(force=True)
            except AuthError as exc:   # memoised by the mint primitive before it re-raised
                logger.info("welcome trial claim failed: %s", exc)
            if not params.get("profile"):
                # The launch profile: ``setup.status`` and the ``setup.ready`` listeners move with it.
                free_tier_bootstrap.refresh_record()
        has_guest = anon_auth.has_guest()
        enabled = anon_auth.free_inference_active() or _trial_claimable()
        payload = {"has_guest": has_guest, "enabled": enabled}
        if enabled and not has_guest:
            payload.update(anon_auth.last_mint_failure() or {})
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5092, str(e))


@method("free_tier.ack_notice")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Mark the availability notice shown on the free-tier identity. ``acked`` is false when there is
    no free-tier identity to mark (nothing to show again either)."""
    try:
        from hermes_cli import anon_auth
        return _ok(rid, {"acked": bool(anon_auth.mark_guest_notice_shown())})
    except Exception as e:
        return _err(rid, 5091, str(e))


@method("free_tier.claim_nudge")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Claim the due sign-in offer of ``kind`` (``trial`` or ``connector``). Claim before showing,
    so two windows never both show it: ``claimed`` is true for exactly one caller each time an offer
    comes due."""
    try:
        from hermes_cli import free_tier_offer
        return _ok(rid, {"claimed": bool(free_tier_offer.claim_offer(params.get("kind") or free_tier_offer.TRIAL))})
    except Exception as e:  # the RPC boundary: report it, never crash the dispatcher
        logger.warning("free_tier.claim_nudge failed", exc_info=True)
        return _err(rid, 5093, str(e))


@method("free_tier.challenge_result")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from hermes_cli.anon_challenge import record_host_outcome
    return _ok(rid, {"accepted": record_host_outcome(params["url"], params.get("attempt", 0), params["outcome"])})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
