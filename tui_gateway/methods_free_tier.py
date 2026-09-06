"""Nous free-tier JSON-RPC handlers: a renderer reads the profile's local auth state (pull); nothing
is pushed. ``free_tier.status`` answers from the auth store with zero network; ``free_tier.ack_notice``
persists the one-time notice flag on the free-tier identity itself, so it dies with that identity.
Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


@method("free_tier.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """``{has_guest, enabled, carries_inference, notice_pending, model, label}`` for the focused
    profile. ``carries_inference`` is the one flag surfaces key on (identity present AND ``nous.guest``
    on); ``notice_pending`` is true until ``free_tier.ack_notice`` ran for this identity."""
    try:
        from hermes_cli import anon_auth
        has_guest = anon_auth.has_guest()
        enabled = anon_auth.guest_enabled()
        return _ok(rid, {
            "has_guest": has_guest, "enabled": enabled, "carries_inference": has_guest and enabled,
            "notice_pending": bool(has_guest and enabled and anon_auth.guest_notice_pending()),
            "model": anon_auth.GUEST_MODEL, "label": anon_auth.FREE_TIER_LABEL})
    except Exception as e:
        return _err(rid, 5090, str(e))


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
