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
        from hermes_cli import anon_auth, free_tier_usage
        has_guest = anon_auth.has_guest()
        enabled = anon_auth.guest_enabled()
        state = free_tier_usage.status()
        continuation_required = False
        if params.get("session_id"):
            session, err = _sess_nowait(params, rid)
            if err:
                return err
            with _session_profile_runtime_scope(session):
                state = free_tier_usage.status()
                continuation_required = _free_tier_session_refusal(session) is not None
        elif state["capped"] and (params.get("provider") or params.get("model")):
            from types import SimpleNamespace
            from agent.free_tier import refusal
            from hermes_cli.runtime_provider import resolve_runtime_provider
            # Only the capped new-chat recovery probe resolves credentials; ordinary
            # identity polling remains local and never provisions an identity.
            runtime = resolve_runtime_provider(requested=params.get("provider") or None,
                                               target_model=params.get("model") or None)
            continuation_required = refusal(SimpleNamespace(**runtime)) is not None
        return _ok(rid, {
            "has_guest": has_guest, "enabled": enabled, "available": has_guest and enabled,
            "notice_pending": bool(has_guest and enabled and anon_auth.guest_notice_pending()),
            "model": anon_auth.GUEST_MODEL, "label": anon_auth.FREE_TIER_LABEL,
            **state, "continuation_required": continuation_required})
    except Exception:
        return _err(rid, 5090, "Could not check free-tier continuation. Please retry.")


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


def _free_tier_session_refusal(session):
    from agent.free_tier import refusal
    with _hermes_home_scope(_session_home(session)):
        if session.get("_compute_host_active") or session.get("agent") is None:
            from types import SimpleNamespace
            from hermes_cli import free_tier_usage
            if not free_tier_usage.status()["capped"]:
                return None
            runtime = _metadata_mirror(session).get("runtime")
            if not isinstance(runtime, dict):
                # A cold lazy session has no host mirror yet. Resolve the same
                # selected route as construction, never construct a parent agent.
                with _session_profile_runtime_scope(session):
                    _, runtime = _resolve_agent_model_runtime(session.get("model_override"), None)
            return refusal(SimpleNamespace(**runtime), session.get("history"))
        return refusal(session["agent"], session.get("history"))


def _sync_free_tier_notice(sid, session):
    """Idle-only hydration/recovery; never display the gate in an active tool batch."""
    if session.get("running"):
        return
    blocked = _free_tier_session_refusal(session)
    if blocked:
        _emit("notification.show", sid, {
            "text": blocked["final_response"], "level": "warn", "kind": "sticky",
            "key": "free_tier.limit", "id": "free_tier.limit", "ttl_ms": None})
    else:
        _emit("notification.clear", sid, {"key": "free_tier.limit"})


def _settle_free_tier_sessions():
    from agent.free_tier import rehome_after_sign_in
    from hermes_cli.anon_auth import route_is_welcome_host
    for sid, session in list(_sessions.items()):
        agent = session.get("agent")
        if agent is None or session.get("running"):
            continue  # common admission rehomes active turns at the next boundary
        with _hermes_home_scope(_session_home(session)):
            if route_is_welcome_host(getattr(agent, "base_url", None)):
                if rehome_after_sign_in(agent):
                    session.pop("model_override", None)
                    _persist_live_session_runtime(session)
            _sync_free_tier_notice(sid, session)


def _start_free_tier_login(sid, session):
    """Render the canonical flow live, not into slash_worker's already-returned StringIO."""
    from hermes_cli import anon_auth
    with _sessions_lock:
        prior = session.get("_free_tier_login_thread")
        if prior is not None and prior.is_alive():
            return anon_auth.UPGRADE_WAITING
        home = _session_home(session)
        transport = current_transport() or session.get("transport")

        def run_login():
            token = bind_transport(transport)
            try:
                for state in anon_auth.run_sign_in(
                    timeout_seconds=8.0, scope=lambda: _hermes_home_scope(home),
                    cancelled=lambda: bool(session.get("_closing"))):
                    if isinstance(state, anon_auth.Waiting):
                        continue
                    text = (f"{state.link}\n{state.code}\n{state.copy_with_wait}"
                            if isinstance(state, anon_auth.Code) else state.copy)
                    if isinstance(state, anon_auth.Completed):
                        _settle_free_tier_sessions()
                    _emit("notification.show", sid, {"text": text, "level": "info", "kind": "sticky",
                          "key": "free_tier.login", "id": "free_tier.login", "ttl_ms": None})
                    if state.terminal:
                        break
            except Exception:
                logger.exception("Live sign-in failed")
                _emit("notification.show", sid, {"text": anon_auth.UPGRADE_NOT_COMPLETED,
                      "level": "warn", "kind": "sticky", "key": "free_tier.login", "id": "free_tier.login"})
            finally:
                reset_transport(token)

        thread = threading.Thread(target=run_login, daemon=True, name="tui-sign-in")
        session["_free_tier_login_thread"] = thread
        thread.start()
    return anon_auth.LOGIN_STARTING


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
