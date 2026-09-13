"""Nous free-tier JSON-RPC handlers: a renderer reads the profile's local auth state (pull); nothing
is pushed except the boot bootstrap's one ``setup.ready`` event. ``free_tier.status`` answers from the
auth store with zero network and zero side effects; ``free_tier.provision`` is the explicit retry when
the boot bootstrap could not create the identity (desktop-only entry); ``free_tier.ack_notice``
persists the one-time notice flag on the free-tier identity itself, so it dies with that identity.
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
    ``available`` = an identity exists AND the tier is on: the free tier (connectors, and the model
    when nothing else carries inference) is there for this install. Whether inference actually runs
    on it is a ROUTE question answered by ``setup.runtime_check.free_tier``, never by this flag.
    ``notice_pending`` is true until ``free_tier.ack_notice`` ran for this identity.

    A pure read. The identity is created by the boot bootstrap (``free_tier_bootstrap``), never as
    a side effect of a client polling this method (NS-845 Q1.2)."""
    try:
        from hermes_cli import anon_auth
        has_guest = anon_auth.has_guest()
        enabled = anon_auth.guest_enabled()
        return _ok(rid, {
            "has_guest": has_guest, "enabled": enabled, "available": has_guest and enabled,
            "notice_pending": bool(has_guest and enabled and anon_auth.guest_notice_pending()),
            "model": anon_auth.GUEST_MODEL, "label": anon_auth.FREE_TIER_LABEL})
    except Exception as e:
        return _err(rid, 5090, str(e))


@method("free_tier.provision")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Explicit retry of the free-tier set-up for the focused profile: adopt the shared store's
    identity, else mint one (blocking, short timeout). The boot bootstrap normally did this already;
    the desktop calls this when the record says the identity is missing (portal down at boot, gate
    turned on later) and the user asks again. ``{has_guest, enabled}``; ``error`` when the portal
    refused."""
    try:
        from hermes_cli import anon_auth
        enabled = anon_auth.guest_enabled()
        error = None
        if enabled and not anon_auth.has_guest():
            try:
                anon_auth.ensure_portal_identity(explicit=True)
            except Exception as exc:
                logger.info("free tier provisioning failed: %s", exc)
                error = str(exc)
        payload = {"has_guest": anon_auth.has_guest(), "enabled": enabled}
        if error:
            payload["error"] = error
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


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
