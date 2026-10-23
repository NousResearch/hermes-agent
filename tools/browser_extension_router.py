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
import threading
from collections import OrderedDict
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


#: Session lanes the extension has registered a controller for, oldest first (bounded; see :func:`_lane_pinned`).
_PINNED_LANES: OrderedDict[tuple[str, str, str], None] = OrderedDict()
_PINNED_LANES_LOCK = threading.Lock()
_MAX_PINNED_LANES = 4096


def _lane_key(session_id: Optional[str], principal_id: Optional[str], transport_family: Optional[str]) -> tuple:
    return tuple(str(v or "").strip() for v in (session_id, principal_id, transport_family))


def _lane_pinned(key: tuple, broker: Any) -> bool:
    """Whether the browser extension has registered a controller for this session lane in this process. Once it
    has, the answer stays yes for the rest of the session, attached or not, so the tool block does not change
    when the person pauses Hermes Control or the extension reconnects."""
    if not all(key):
        return False
    with _PINNED_LANES_LOCK:
        if key in _PINNED_LANES:
            _PINNED_LANES.move_to_end(key)
            return True
    lane_registered = getattr(broker, "lane_registered", None)
    session_id, principal_id, transport_family = key
    if not callable(lane_registered):  # a broker that cannot say is treated as registered: fail closed
        return True
    if not lane_registered(session_id=session_id, principal_id=principal_id, transport_family=transport_family):
        return False
    with _PINNED_LANES_LOCK:
        _PINNED_LANES[key] = None
        _PINNED_LANES.move_to_end(key)
        while len(_PINNED_LANES) > _MAX_PINNED_LANES:
            _PINNED_LANES.popitem(last=False)
    return True


def reset_pinned_lanes() -> None:
    """Forget every pinned session lane (tests; a new process starts empty)."""
    with _PINNED_LANES_LOCK:
        _PINNED_LANES.clear()


def extension_lane_configured() -> bool:
    """Whether this request belongs to a browser extension session: the feature flag
    (``browser.extension_control.enabled``) is on, the request carries the server-bound principal and transport,
    and the extension has registered a controller for this session (the side panel registers it before the
    session's first turn when Hermes Control is on). Other API clients of the same profile, such as a chat UI
    that never attaches a browser, keep the stock surface. Once yes, the answer stays yes for the session
    whatever is attached later, because a surface that changed whenever the person paused Hermes Control or the
    extension reconnected would rewrite the tool block mid-conversation and lose the provider's prompt cache."""
    try:
        from gateway.browser_control_broker import browser_control_enabled, get_browser_control_broker

        return browser_control_enabled() and _lane_pinned(_lane_key(*_bound_identity()), get_browser_control_broker())
    except Exception:
        logger.debug("browser extension lane check failed", exc_info=True)
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
        # registered-but-offline lanes fail closed, and so does a lane that was
        # registered earlier in the session (its tools describe the extension).
        if not _lane_pinned(_lane_key(session_id or task_id, principal_id, transport_family), broker):
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
