"""Browser controller registration and result routing for the dashboard.

The controller extension registers over the authenticated ``/api/ws`` gateway. Everything
binds to the SERVER-MINTED identity (``WSTransport.auth_identity``, stamped from the single-use
ticket); a client-supplied ``principal_id`` is ignored and replaced by a digest of it. Broker
frames are re-enveloped as Gateway ``event`` frames; ``result`` resolves a command only on the
owning transport for the exact attached scope (the broker's exact-scope ``complete`` is the
backstop). Capabilities come from the broker's explicit allowlist (no raw CDP/eval/uploads).
Bodies are rebound onto server.py's globals (bind_module publishes this module's helpers too).
"""

from __future__ import annotations

import hashlib
import logging

from hermes_cli.dashboard_auth.ws_tickets import (
    INTERNAL_PROVIDER as _INTERNAL_PROVIDER, INTERNAL_USER_ID as _INTERNAL_USER_ID)

from .method_ctx import HandlerRegistry, bind_module

logger = logging.getLogger(__name__)

_registry = HandlerRegistry()
method = _registry.method

# Transport family stamped into every scope attached here; the broker treats it as an
# identity field, so an API transport can never address a dashboard controller.
_CLOUD_TRANSPORT_FAMILY = "cloud-ticket-ws"
_ERR_FORBIDDEN = 4403  # identity / session / flag denials
_IDENTITY_REQUIRED = "authenticated controller identity required"
_NOT_OWNED = "controller is not owned by this transport"
_NO_CONTROLLER = "no controller registered for this session"


def _is_authenticated_identity(identity: object) -> bool:
    """True for a server-minted, non-internal ``{user_id, provider}`` identity."""
    if not isinstance(identity, dict):
        return False
    user_id, provider = identity.get("user_id"), identity.get("provider")
    if not isinstance(user_id, str) or not user_id.strip():
        return False
    if not isinstance(provider, str) or not provider.strip():
        return False
    return not (user_id == _INTERNAL_USER_ID and provider == _INTERNAL_PROVIDER)


def _principal_digest(identity: dict) -> str:
    """Server-derived principal id: stable per user, unspoofable without the minted identity."""
    raw = f"{identity.get('provider')}\x00{identity.get('user_id')}"
    return f"principal:dashboard:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"


def _broker_event_writer(transport: object, session_id: str):
    """Broker send callback: re-envelope ``{method, params}`` as a Gateway ``event`` frame
    (``type`` = method, ``payload`` = params, plus the owning ``session_id``)."""

    def send(frame: dict) -> None:
        try:
            accepted = transport.write({
                "jsonrpc": "2.0", "method": "event",
                "params": {
                    "type": frame.get("method"), "session_id": session_id,
                    "payload": frame.get("params"),
                }})
        except Exception:
            logger.exception(
                "browser controller event write failed session=%s frame=%s",
                session_id, frame.get("method"),
            )
            raise
        if accepted is False:
            raise ConnectionError("browser controller event write failed")

    return send


def _controller_method(
    name: str, *, identity_message: str = _IDENTITY_REQUIRED, lookup_scope: bool = True,
    missing_scope_message: str = _NO_CONTROLLER, precheck=None):
    """Register a handler behind the shared fail-closed (4403) controller gates.

    Order: ``precheck(rid, params)`` (may return an error envelope) → caller holds a
    server-authenticated, non-internal identity → the named session exists and its
    ``transport`` is exactly the caller → when ``lookup_scope``, a scope is attached for
    this session/principal/family and the caller owns it. Then
    ``fn(rid, params, transport, identity, session_id, broker, scope, session)`` runs.
    """

    def dec(fn):
        def handler(rid, params: dict) -> dict:
            from gateway import browser_control_broker

            if precheck is not None:
                denied = precheck(rid, params)
                if denied is not None:
                    return denied
            transport = current_transport()
            identity = getattr(transport, "auth_identity", None)
            if not _is_authenticated_identity(identity):
                return _err(rid, _ERR_FORBIDDEN, identity_message)
            session_id = str(params.get("session_id") or "")
            with _sessions_lock:
                session = _sessions.get(session_id)
                if session is None or session.get("transport") is not transport:
                    return _err(rid, _ERR_FORBIDDEN, "session is not owned by this transport")
            broker = browser_control_broker.get_browser_control_broker()
            scope = None
            if lookup_scope:
                scope = broker.scope_for_session(
                    session_id=session_id, principal_id=_principal_digest(identity),
                    transport_family=_CLOUD_TRANSPORT_FAMILY)
                if scope is None:
                    return _err(rid, _ERR_FORBIDDEN, missing_scope_message)
                # Defense in depth: the broker's exact-scope ops already reject foreign
                # scopes; the owner check makes the same-transport rule explicit here too.
                if not broker.is_owner(scope, transport):
                    return _err(rid, _ERR_FORBIDDEN, _NOT_OWNED)
            return fn(rid, params, transport, identity, session_id, broker, scope, session)

        handler.__doc__ = fn.__doc__
        return method(name)(handler)

    return dec


def _register_precheck(rid, params: dict):
    from gateway import browser_control_broker

    if not browser_control_broker.browser_control_enabled():
        return _err(rid, _ERR_FORBIDDEN, "browser.extension_control.enabled is not set")
    broker_mod = browser_control_broker
    if not broker_mod.browser_control_protocol_supported(params.get("protocol_version")):
        expected = broker_mod.BROWSER_CONTROL_PROTOCOL_VERSION
        return _err(
            rid, _ERR_FORBIDDEN,
            f"unsupported browser-control protocol version; expected {expected}",
        )
    return None


@_controller_method(
    "browser.controller.register",
    identity_message="browser.controller.register requires an authenticated non-internal identity",
    lookup_scope=False, precheck=_register_precheck)
def _(rid, params: dict, transport, identity, session_id, broker, _scope, session) -> dict:
    """Attach this connection as the browser controller for one session; fails closed (4403) unless
    the flag is on, the protocol version is supported, the gates pass and a capability survives."""
    from gateway import browser_control_broker

    controller_id = str(params.get("controller_id") or "").strip()
    browser_profile_id = str(params.get("browser_profile_id") or "").strip()
    profile_id = str(session.get("profile") or "").strip()
    if not controller_id or not browser_profile_id or not profile_id:
        return _err(
            rid, _ERR_FORBIDDEN,
            "controller_id, browser_profile_id, and server session profile are required",
        )
    capabilities = browser_control_broker.filter_browser_control_capabilities(
        params.get("capabilities")
    )
    if not capabilities:
        return _err(rid, _ERR_FORBIDDEN, "no permitted controller capabilities requested")
    scope = browser_control_broker.ControllerScope(
        principal_id=_principal_digest(identity), profile_id=profile_id, session_id=session_id,
        controller_id=controller_id, browser_profile_id=browser_profile_id,
        transport_family=_CLOUD_TRANSPORT_FAMILY, capabilities=capabilities)
    broker.attach(scope, _broker_event_writer(transport, session_id), owner=transport)
    return _ok(rid, {
        "scope": {
            "principal_id": scope.principal_id, "profile_id": scope.profile_id,
            "session_id": scope.session_id, "controller_id": scope.controller_id,
            "browser_profile_id": scope.browser_profile_id,
            "transport_family": scope.transport_family,
            "capabilities": sorted(scope.capabilities)}})


@_controller_method("browser.controller.result")
def _(rid, params: dict, _transport, _identity, _session_id, broker, scope, _session) -> dict:
    """Deliver one command result to the broker; ``accepted`` is False for unknown / resolved /
    cancelled command ids (the broker's idempotent answer, surfaced verbatim)."""
    command_id = str(params.get("command_id") or "")
    if not command_id:
        return _err(rid, _ERR_FORBIDDEN, "command_id required")
    ok = params.get("ok") is True
    accepted = broker.complete(
        command_id, scope=scope, ok=ok, result=params.get("result") if ok else params.get("error"))
    return _ok(rid, {"accepted": accepted})


@_controller_method("browser.controller.heartbeat")
def _(rid, params: dict, *_gate) -> dict:
    """Acknowledge a heartbeat only for this transport's attached controller."""
    return _ok(rid, {"ok": True})


@_controller_method("browser.controller.detach", missing_scope_message=_NOT_OWNED)
def _(rid, params: dict, transport, _identity, _session_id, broker, scope, _session) -> dict:
    """Hard-detach only the controller owned by this authenticated transport."""
    broker.detach(scope, owner=transport, notify_controller=False)
    return _ok(rid, {"detached": True})


def register(server) -> None:
    """Publish helpers/constants onto ``server`` and install handlers (rebound to its globals)."""
    bind_module(globals(), server, skip=("_",))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'BROWSER_CONTROL_PROTOCOL_VERSION': ('gateway.browser_control_broker', 'BROWSER_CONTROL_PROTOCOL_VERSION'),
    'browser_control_protocol_supported': ('gateway.browser_control_broker', 'browser_control_protocol_supported'),
    'filter_browser_control_capabilities': ('gateway.browser_control_broker', 'filter_browser_control_capabilities'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
