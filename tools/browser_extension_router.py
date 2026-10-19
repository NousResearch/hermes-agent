"""Registry-level browser extension router.

Agent-side half of browser-extension-control: decides, per ``browser_*`` handler
invocation, whether an attached extension controller (via
:mod:`gateway.browser_control_broker`) or the legacy backend executes it.

Routing contract (see ``tests/tools/test_browser_extension_router.py``):
- Feature off ⇒ legacy, broker never touched, ``fallback()`` called exactly once.
- No server-bound identity ⇒ legacy.
- Bound identity ⇒ authoritative extension lane; missing/ambiguous scope,
  disconnect, or capability mismatch fail closed (never jump to another browser).
- Selected controller ⇒ authoritative; its errors propagate, legacy never retried.
- ``args`` is never mutated.

:func:`routed_browser_handler` resolves the flag and broker lazily on every call
so importing this module never pulls in the gateway and config changes apply
without restart.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteContext:
    """What a guard may use about the call in flight on the extension lane.

    ``key`` names the lane (session, principal and transport family), so a guard can remember something about
    the browser between calls. ``send(action, arguments)`` sends the same controller one more command and
    returns its result (it raises what the broker raises)."""

    key: tuple
    send: Callable[[str, dict[str, Any]], Any]


#: ``(action, args, route) -> (args to dispatch, refusal result or None)``; see :func:`route_browser_tool`.
ArgsGuard = Callable[[str, dict[str, Any], RouteContext], tuple[dict[str, Any], Optional[str]]]
#: ``(action, args sent, controller result, route) -> result to give the model``; see :func:`route_browser_tool`.
ResultGuard = Callable[[str, dict[str, Any], Any, RouteContext], Any]


def _bound_identity() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(session_id, principal_id, transport_family) from the session context."""
    from gateway.session_context import get_session_env

    return tuple(  # type: ignore[return-value]
        get_session_env(key, "") or None
        for key in ("HERMES_SESSION_ID", "HERMES_BROWSER_CONTROL_PRINCIPAL",
                    "HERMES_BROWSER_CONTROL_TRANSPORT_FAMILY")
    )


#: Tells the model what to say when a session that can bind a controller has none attached, which is the one
#: state it cannot fix itself (the tool surface is the same whether or not a controller is attached).
TURN_ON_HERMES_CONTROL = "Ask the person to turn on Hermes Control in the browser extension and attach a tab, then retry."


def _controller_unavailable(message: str) -> Exception:
    from gateway.browser_control_broker import ControllerUnavailable

    return ControllerUnavailable(message)


def extension_lane_configured() -> bool:
    """Whether this request belongs to a session that can bind a browser extension controller: the feature flag
    (``browser.extension_control.enabled``) is on and the request carries the server-bound principal and
    transport. It does not ask what is attached right now, so the answer is the same on every request of a
    session. The tool surface follows it: a surface that changed whenever the person turned Hermes Control on
    or off, or the extension reconnected, would rewrite the tool block mid-conversation and lose the provider's
    prompt cache."""
    try:
        from gateway.browser_control_broker import browser_control_enabled

        return browser_control_enabled() and all(_bound_identity())
    except Exception:
        logger.debug("browser extension lane check failed", exc_info=True)
        return False


def extension_controller_available(action: str) -> bool:
    """Whether this request owns one exact controller capable of ``action``.
    Runs during tool-schema assembly inside the request's session context;
    consults the process-local broker directly and fails closed on any gap."""
    try:
        from gateway.browser_control_broker import browser_control_enabled, get_browser_control_broker

        if not browser_control_enabled():
            return False
        session_id, principal_id, transport_family = identity = _bound_identity()
        if not all(identity):
            return False
        broker = get_browser_control_broker()
        scope = broker.scope_for_session(session_id=session_id, principal_id=principal_id, transport_family=transport_family)
        return scope is not None and broker.select(scope, action) is not None
    except Exception:
        logger.debug("browser extension availability check failed for %s", action, exc_info=True)
        return False


def route_browser_tool(
    action: str, args: dict[str, Any], *, fallback: Callable[[], Any], broker: Any, enabled: bool,
    session_id: Optional[str] = None, task_id: Optional[str] = None, principal_id: Optional[str] = None,
    transport_family: Optional[str] = None, tool_call_id: Optional[str] = "", guard: Optional[ArgsGuard] = None,
    result_guard: Optional[ResultGuard] = None,
) -> Any:
    """Route one browser action through the extension-control broker.

    ``broker`` exposes ``scope_for_session(**identity)``, ``select(scope, cap)``
    and ``dispatch(scope, *, action, arguments, tool_call_id)``. ``fallback`` is
    called exactly once when the feature is off or no server-bound identity
    exists; once a controller is selected its result/exception is final.
    ``guard`` runs only on that extension path, right before dispatch: it returns
    the arguments to send and, when the call must not reach the browser, a result
    string to answer with instead (the legacy backend runs its own checks).
    ``result_guard`` also runs only on that path, on what the controller sent back, and returns the result the
    model gets (the page a navigation landed on is only known once the controller reports it). Both receive a
    :class:`RouteContext`.
    """
    if not enabled or not str(principal_id or "").strip() or not str(transport_family or "").strip():
        return fallback()

    identity = dict(session_id=session_id, task_id=task_id, principal_id=principal_id, transport_family=transport_family)
    scope = broker.scope_for_session(**identity)
    if scope is None:
        # A stamped identity only becomes authoritative once a controller has
        # registered for the lane; unregistered lanes keep the legacy backend,
        # registered-but-offline lanes fail closed.
        lane_bound = getattr(broker, "lane_registered", None)
        if callable(lane_bound) and not lane_bound(**identity):
            return fallback()
        raise _controller_unavailable(f"bound browser controller unavailable for {action}. {TURN_ON_HERMES_CONTROL}")

    if broker.select(scope, action) is None:
        raise _controller_unavailable(
            f"bound browser controller cannot execute {action}. {TURN_ON_HERMES_CONTROL} If it is on and connected, "
            "the browser extension does not offer this action (an older extension, or a browser such as Firefox "
            "that cannot do it): use another browser action, or update the extension and reconnect.")

    # Controller is authoritative: never retry the legacy backend. Registry
    # handlers must return a string; keep string results byte-identical and
    # serialize decoded JSON values at this boundary.
    route = RouteContext(
        key=(session_id, principal_id, transport_family),
        send=lambda command, arguments: broker.dispatch(
            scope, action=command, arguments=arguments, tool_call_id=tool_call_id))
    if guard is not None:
        args, refusal = guard(action, args, route)
        if refusal is not None:
            return refusal
    result = broker.dispatch(scope, action=action, arguments=args, tool_call_id=tool_call_id)
    if result_guard is not None:
        result = result_guard(action, args, result, route)
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def current_tool_call_id() -> str:
    """Active tool_call_id bound by the agent executor, or ``""`` when none."""
    try:
        from tools.approval_context import _approval_tool_call_id

        return _approval_tool_call_id.get() or ""
    except Exception:
        return ""


def routed_browser_handler(
    action: str, args: dict[str, Any], *, fallback: Callable[[], Any], task_id: Optional[str] = None,
    session_id: Optional[str] = None, principal_id: Optional[str] = None,
    transport_family: Optional[str] = None, tool_call_id: Optional[str] = None, guard: Optional[ArgsGuard] = None,
    result_guard: Optional[ResultGuard] = None,
) -> Any:
    """Lazy registry-handler route wrapper for ``browser_*`` tools.
    Feature off (or gateway unimportable) ⇒ the legacy handler runs unchanged."""
    try:
        from gateway.browser_control_broker import browser_control_enabled, get_browser_control_broker
    except Exception as exc:  # pragma: no cover - defensive, gateway always present
        logger.debug("browser extension router unavailable (%s); using legacy backend", exc)
        return fallback()
    if not browser_control_enabled():
        return fallback()

    try:
        env_session, env_principal, env_transport = _bound_identity()
    except Exception:
        env_session = env_principal = env_transport = None

    return route_browser_tool(
        action, args, fallback=fallback, broker=get_browser_control_broker(), enabled=True,
        session_id=session_id or env_session, task_id=task_id, principal_id=principal_id or env_principal,
        transport_family=transport_family or env_transport,
        tool_call_id=current_tool_call_id() if tool_call_id is None else tool_call_id, guard=guard,
        result_guard=result_guard,
    )
