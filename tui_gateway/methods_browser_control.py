"""Browser controller registration and result routing for the dashboard.

The dashboard's browser controller (the extension that physically drives a
browser) registers itself over the authenticated ``/api/ws`` JSON-RPC
gateway. Everything here is bound to the **server-minted identity** that the
dashboard auth layer stamped onto the WS connection: ``hermes_cli.web_server``
consumes the single-use ticket and records ``ws._hermes_auth_identity``, the
WS transport carries it as ``WSTransport.auth_identity``, and the client can
never name its own principal (a spoofed ``principal_id`` param is ignored and
replaced by a server-derived digest of the authenticated identity).

Registration attaches the shared transport-neutral broker
(:mod:`gateway.browser_control_broker`) with the calling transport as owner;
broker command/cancel frames are wrapped as standard Gateway ``event`` frames
(``type`` = broker method name, ``payload`` = broker params, plus the owning
``session_id``) so the dashboard consumes the same envelope as every other
gateway event. ``browser.controller.result`` resolves a pending command only
when the request arrives on the same transport that owns the session, and
only for the exact attached scope — the broker's exact-scope ``complete`` is
the last line of defense against cross-tenant completion.

Both dashboard and local API transports use the broker's shared, explicit
capability allowlist. Raw CDP, script evaluation, console access, uploads, and
other privileged surfaces are not controller capabilities.

Note on handler globals: ``HandlerRegistry.install`` (method_ctx.py) rebinds
each handler's ``__globals__`` onto server.py's namespace, so handler bodies
may only reference names server.py defines/imports (``_ok``, ``_err``,
``_sessions``, ``_sessions_lock``, ``current_transport``, ``logger``, ...).
This module's own helpers and constants are therefore captured through
keyword-default arguments, which ``install`` preserves.
"""

from __future__ import annotations

import hashlib
import logging

from gateway.browser_control_broker import (
    BROWSER_CONTROL_PROTOCOL_VERSION,
    browser_control_protocol_supported,
    filter_browser_control_capabilities,
)
from hermes_cli.dashboard_auth.ws_tickets import (
    INTERNAL_PROVIDER as _INTERNAL_PROVIDER,
    INTERNAL_USER_ID as _INTERNAL_USER_ID,
)

from .method_ctx import HandlerRegistry

logger = logging.getLogger(__name__)

_registry = HandlerRegistry()
method = _registry.method

#: Transport family stamped into every scope attached from this gateway. The
#: broker's exact-match contract treats it as an identity field, so an API
#: transport can never address a dashboard controller (and vice versa).
_CLOUD_TRANSPORT_FAMILY = "cloud-ticket-ws"

#: JSON-RPC error code for identity / session / flag denials (forbidden).
_ERR_FORBIDDEN = 4403


def _is_authenticated_identity(identity: object) -> bool:
    """True for a server-minted, non-internal ``{user_id, provider}`` identity."""
    if not isinstance(identity, dict):
        return False
    user_id = identity.get("user_id")
    provider = identity.get("provider")
    if not isinstance(user_id, str) or not user_id.strip():
        return False
    if not isinstance(provider, str) or not provider.strip():
        return False
    if user_id == _INTERNAL_USER_ID and provider == _INTERNAL_PROVIDER:
        return False
    return True


def _principal_digest(identity: dict) -> str:
    """Server-derived principal id: a digest of the server-minted identity.

    The client-supplied ``principal_id`` RPC param is never trusted; the
    digest is deterministic (stable across reconnects for the same user) but
    unspoofable by a peer that does not hold the authenticated identity.
    """
    raw = f"{identity.get('provider')}\x00{identity.get('user_id')}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"principal:dashboard:{digest[:32]}"


def _broker_event_writer(transport: object, session_id: str):
    """Wrap broker command/cancel frames as standard Gateway event frames.

    The broker's send callback receives transport-neutral envelopes like
    ``{"method": "browser.controller.command", "params": {...}}``; the
    dashboard speaks Gateway events, so we re-envelope them:
    ``{"jsonrpc": "2.0", "method": "event", "params": {"type": <method>,
    "session_id": <owner>, "payload": <params>}}``.
    """

    def send(frame: dict) -> None:
        try:
            accepted = transport.write(
                {
                    "jsonrpc": "2.0",
                    "method": "event",
                    "params": {
                        "type": frame.get("method"),
                        "session_id": session_id,
                        "payload": frame.get("params"),
                    },
                }
            )
        except Exception:
            logger.exception(
                "browser controller event write failed session=%s frame=%s",
                session_id,
                frame.get("method"),
            )
            raise
        if accepted is False:
            raise ConnectionError("browser controller event write failed")

    return send


@method("browser.controller.register")
def _(
    rid,
    params: dict,
    _family=_CLOUD_TRANSPORT_FAMILY,
    _protocol_version=BROWSER_CONTROL_PROTOCOL_VERSION,
    _protocol_supported=browser_control_protocol_supported,
    _filter_capabilities=filter_browser_control_capabilities,
    _forbidden=_ERR_FORBIDDEN,
    _identity_ok=_is_authenticated_identity,
    _digest=_principal_digest,
    _event_writer=_broker_event_writer,
) -> dict:
    """Attach this connection as the browser controller for one session.

    Fails closed (4403) unless *every* gate passes:

    * the ``browser.extension_control.enabled`` feature flag is on;
    * the calling transport holds a server-authenticated, non-internal
      identity (``WSTransport.auth_identity`` — never the RPC params);
    * the named session exists in the live session registry and its
      ``transport`` is exactly the calling transport;
    * at least one requested capability survives the shared allowlist.

    The returned ``scope`` names a server-derived ``principal_id``, the
    ``cloud-ticket-ws`` transport family, and the filtered capability set.
    """
    from gateway import browser_control_broker

    if not browser_control_broker.browser_control_enabled():
        return _err(
            rid,
            _forbidden,
            "browser.extension_control.enabled is not set",
        )

    if not _protocol_supported(params.get("protocol_version")):
        return _err(
            rid,
            _forbidden,
            f"unsupported browser-control protocol version; expected {_protocol_version}",
        )

    transport = current_transport()
    identity = getattr(transport, "auth_identity", None)
    if not _identity_ok(identity):
        return _err(
            rid,
            _forbidden,
            "browser.controller.register requires an authenticated "
            "non-internal identity",
        )

    session_id = str(params.get("session_id") or "")
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None or session.get("transport") is not transport:
            return _err(
                rid,
                _forbidden,
                "session is not owned by this transport",
            )

    controller_id = str(params.get("controller_id") or "").strip()
    browser_profile_id = str(params.get("browser_profile_id") or "").strip()
    profile_id = str(session.get("profile") or "").strip()
    if not controller_id or not browser_profile_id or not profile_id:
        return _err(
            rid,
            _forbidden,
            "controller_id, browser_profile_id, and server session profile are required",
        )

    capabilities = _filter_capabilities(params.get("capabilities"))
    if not capabilities:
        return _err(
            rid,
            _forbidden,
            "no permitted controller capabilities requested",
        )

    scope = browser_control_broker.ControllerScope(
        principal_id=_digest(identity),
        profile_id=profile_id,
        session_id=session_id,
        controller_id=controller_id,
        browser_profile_id=browser_profile_id,
        transport_family=_family,
        capabilities=capabilities,
    )

    broker = browser_control_broker.get_browser_control_broker()
    broker.attach(
        scope,
        _event_writer(transport, session_id),
        owner=transport,
    )

    return _ok(
        rid,
        {
            "scope": {
                "principal_id": scope.principal_id,
                "profile_id": scope.profile_id,
                "session_id": scope.session_id,
                "controller_id": scope.controller_id,
                "browser_profile_id": scope.browser_profile_id,
                "transport_family": scope.transport_family,
                "capabilities": sorted(scope.capabilities),
            }
        },
    )


@method("browser.controller.result")
def _(
    rid,
    params: dict,
    _family=_CLOUD_TRANSPORT_FAMILY,
    _forbidden=_ERR_FORBIDDEN,
    _identity_ok=_is_authenticated_identity,
    _digest=_principal_digest,
) -> dict:
    """Deliver one controller command result back to the broker.

    Only the transport that owns the session may resolve its commands, and
    only against the exact scope attached for that session (the broker's
    exact-scope ``complete`` rejects any other scope). ``accepted`` is
    ``False`` for unknown / already-resolved / cancelled command ids — the
    broker's idempotent answer, surfaced verbatim.
    """
    from gateway import browser_control_broker

    transport = current_transport()
    identity = getattr(transport, "auth_identity", None)
    if not _identity_ok(identity):
        return _err(rid, _forbidden, "authenticated controller identity required")
    session_id = str(params.get("session_id") or "")
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None or session.get("transport") is not transport:
            return _err(
                rid,
                _forbidden,
                "session is not owned by this transport",
            )

    command_id = str(params.get("command_id") or "")
    if not command_id:
        return _err(rid, _forbidden, "command_id required")

    broker = browser_control_broker.get_browser_control_broker()
    scope = broker.scope_for_session(
        session_id=session_id,
        principal_id=_digest(identity),
        transport_family=_family,
    )
    if scope is None:
        return _err(
            rid,
            _forbidden,
            "no controller registered for this session",
        )
    # Defense in depth: the exact-scope complete below already rejects any
    # foreign scope, but the owner check makes the "same transport" rule
    # explicit at this layer too.
    if not broker.is_owner(scope, transport):
        return _err(
            rid,
            _forbidden,
            "controller is not owned by this transport",
        )

    ok = params.get("ok") is True
    accepted = broker.complete(
        command_id,
        scope=scope,
        ok=ok,
        result=params.get("result") if ok else params.get("error"),
    )
    return _ok(rid, {"accepted": accepted})


@method("browser.controller.heartbeat")
def _(
    rid,
    params: dict,
    _family=_CLOUD_TRANSPORT_FAMILY,
    _forbidden=_ERR_FORBIDDEN,
    _identity_ok=_is_authenticated_identity,
    _digest=_principal_digest,
) -> dict:
    """Acknowledge a heartbeat only for this transport's attached controller."""
    from gateway import browser_control_broker

    transport = current_transport()
    identity = getattr(transport, "auth_identity", None)
    if not _identity_ok(identity):
        return _err(rid, _forbidden, "authenticated controller identity required")
    session_id = str(params.get("session_id") or "")
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None or session.get("transport") is not transport:
            return _err(rid, _forbidden, "session is not owned by this transport")

    broker = browser_control_broker.get_browser_control_broker()
    scope = broker.scope_for_session(
        session_id=session_id,
        principal_id=_digest(identity),
        transport_family=_family,
    )
    if scope is None:
        return _err(rid, _forbidden, "no controller registered for this session")
    if not broker.is_owner(scope, transport):
        return _err(rid, _forbidden, "controller is not owned by this transport")
    return _ok(rid, {"ok": True})


@method("browser.controller.detach")
def _(
    rid,
    params: dict,
    _family=_CLOUD_TRANSPORT_FAMILY,
    _forbidden=_ERR_FORBIDDEN,
    _identity_ok=_is_authenticated_identity,
    _digest=_principal_digest,
) -> dict:
    """Hard-detach only the controller owned by this authenticated transport."""
    from gateway import browser_control_broker

    transport = current_transport()
    identity = getattr(transport, "auth_identity", None)
    if not _identity_ok(identity):
        return _err(rid, _forbidden, "authenticated controller identity required")
    session_id = str(params.get("session_id") or "")
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None or session.get("transport") is not transport:
            return _err(rid, _forbidden, "session is not owned by this transport")

    broker = browser_control_broker.get_browser_control_broker()
    scope = broker.scope_for_session(
        session_id=session_id,
        principal_id=_digest(identity),
        transport_family=_family,
    )
    if scope is None or not broker.is_owner(scope, transport):
        return _err(rid, _forbidden, "controller is not owned by this transport")
    broker.detach(scope, owner=transport, notify_controller=False)
    return _ok(rid, {"detached": True})


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
