"""Registry-level browser extension router.

This module is the *agent-side* half of the browser-extension-control
feature: it decides, for one registry ``browser_*`` handler invocation,
whether the command is executed by an attached extension controller (via
the :mod:`gateway.browser_control_broker`) or by the existing legacy
browser backend.

Routing contract (exercised by ``tests/tools/test_browser_extension_router.py``):

- **Feature off ⇒ legacy, untouched.** When ``enabled`` is false the broker
  is never touched and ``fallback()`` is called exactly once. This is the
  default: ``browser.extension_control.enabled`` is false unless explicitly
  configured, so every real browser action keeps its exact legacy path.

- **No server-bound identity ⇒ legacy.** Generic Hermes callers keep the
  existing backend when no authenticated browser-controller identity is bound.

- **Bound identity ⇒ authoritative extension lane.** Once the gateway binds a
  browser-controller principal and transport family, missing/ambiguous scope,
  disconnect, or capability mismatch fail closed. A "control this tab" turn
  must never jump to an unrelated local/cloud browser backend.

- **Selected controller ⇒ authoritative.** Once a controller is selected the
  command is dispatched to it and its result returned; the legacy backend
  is *never* retried, even when the controller fails (timeout, cancellation,
  rejection, transport error all propagate to the caller).

- **Arguments are never mutated.** ``args`` is passed through untouched;
  the broker copies arguments into its command frame itself.

The lazy wrapper :func:`routed_browser_handler` is what the ``browser_*``
registry handlers call. It resolves the feature flag and the process-local
broker lazily on every invocation so importing this module (or
``tools.browser_tool``) never pulls in the gateway, and so a mid-process
config change is honored without restart.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def extension_controller_available(action: str) -> bool:
    """Whether this request owns one exact controller capable of ``action``.

    Tool-schema assembly runs inside the API request's session context, before
    a model can call a browser tool. The legacy browser backend's availability
    probe cannot decide whether the extension route is usable, so routeable
    tools consult the process-local broker directly. Missing server-bound
    identity, ambiguous scope, a detached controller, or a capability mismatch
    all fail closed.
    """
    try:
        from gateway.browser_control_broker import (
            browser_control_enabled,
            get_browser_control_broker,
        )
        from gateway.session_context import get_session_env

        if not browser_control_enabled():
            return False
        session_id = get_session_env("HERMES_SESSION_ID", "") or None
        principal_id = get_session_env("HERMES_BROWSER_CONTROL_PRINCIPAL", "") or None
        transport_family = get_session_env(
            "HERMES_BROWSER_CONTROL_TRANSPORT_FAMILY", ""
        ) or None
        if not session_id or not principal_id or not transport_family:
            return False
        broker = get_browser_control_broker()
        scope = broker.scope_for_session(
            session_id=session_id,
            principal_id=principal_id,
            transport_family=transport_family,
        )
        return scope is not None and broker.select(scope, action) is not None
    except Exception:
        logger.debug(
            "browser extension availability check failed for %s",
            action,
            exc_info=True,
        )
        return False


def route_browser_tool(
    action: str,
    args: Dict[str, Any],
    *,
    fallback: Callable[[], Any],
    broker: Any,
    enabled: bool,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    principal_id: Optional[str] = None,
    transport_family: Optional[str] = None,
    tool_call_id: Optional[str] = "",
) -> Any:
    """Route one browser action through the extension-control broker.

    Parameters
    ----------
    action:
        Registry tool name / controller capability, e.g. ``"browser_navigate"``.
    args:
        Tool arguments as received from the model. Never mutated.
    fallback:
        The existing backend handler, called exactly once when the feature is
        off or no server-bound controller identity exists. Must be a
        zero-argument callable.
    broker:
        Object exposing ``scope_for_session(**identity) -> scope|None``,
        ``select(scope, capability) -> controller|None`` and
        ``dispatch(scope, *, action, arguments, tool_call_id)``. The real
        implementation is ``gateway.browser_control_broker``.
    enabled:
        Feature flag; false bypasses the broker entirely.
    session_id/task_id:
        Caller session hints forwarded to ``scope_for_session``.
    principal_id/transport_family:
        Server-bound caller identity. Both are mandatory when the feature is
        enabled; missing values preserve the existing backend for generic
        Hermes callers.
    tool_call_id:
        Caller tool-call id forwarded verbatim to ``dispatch``.

    Returns
    -------
    The legacy backend's return value when falling back, or the controller's
    completion result when routed. Exceptions from a selected controller are
    propagated — the legacy backend is never retried after selection.
    """
    if not enabled:
        return fallback()

    if not str(principal_id or "").strip() or not str(transport_family or "").strip():
        return fallback()

    scope = broker.scope_for_session(
        session_id=session_id,
        task_id=task_id,
        principal_id=principal_id,
        transport_family=transport_family,
    )
    if scope is None:
        # A stamped identity alone does not make the extension lane
        # authoritative — authentication happens at transport auth, but the
        # lane only BINDS when a controller actually registers for it. If no
        # controller ever registered, generic callers keep the legacy
        # backend. Once a lane registered (even if the controller is
        # currently offline/ambiguous), fail closed: a "control this tab"
        # session must never silently jump to an unrelated browser.
        lane_bound = getattr(broker, "lane_registered", None)
        if callable(lane_bound) and not lane_bound(
            session_id=session_id,
            task_id=task_id,
            principal_id=principal_id,
            transport_family=transport_family,
        ):
            return fallback()
        from gateway.browser_control_broker import ControllerUnavailable

        raise ControllerUnavailable(
            f"bound browser controller unavailable for {action}"
        )

    controller = broker.select(scope, action)
    if controller is None:
        from gateway.browser_control_broker import ControllerUnavailable

        raise ControllerUnavailable(
            f"bound browser controller cannot execute {action}"
        )

    # A controller was selected: it is authoritative. Never retry through the
    # existing backend, whatever happens here. Registry handlers must return a
    # string (or the dedicated multimodal envelope), while controller transports
    # naturally complete with decoded JSON values. Preserve existing string
    # results byte-for-byte and serialize decoded values at this boundary.
    result = broker.dispatch(
        scope, action=action, arguments=args, tool_call_id=tool_call_id
    )
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def current_tool_call_id() -> str:
    """Return the active tool_call_id, or ``""`` when none is bound.

    The agent executor binds the id via
    ``tools.approval.set_current_observability_context`` immediately before
    registry dispatch, so the registry handler (and this router) can read it
    back from the same context. Bare/offline callers have no binding.
    """
    try:
        from tools.approval import _approval_tool_call_id

        return _approval_tool_call_id.get() or ""
    except Exception:
        return ""


def routed_browser_handler(
    action: str,
    args: Dict[str, Any],
    *,
    fallback: Callable[[], Any],
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    principal_id: Optional[str] = None,
    transport_family: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Any:
    """Lazy registry-handler route wrapper for ``browser_*`` tools.

    Resolves the feature flag and process-local broker lazily so the
    default (feature off) path costs one cached config read and an immediate
    fallback, and so importing ``tools.browser_tool`` never imports the
    gateway. When the gateway cannot be imported or the feature is off, the
    legacy handler runs unchanged.
    """
    try:
        from gateway.browser_control_broker import (
            browser_control_enabled,
            get_browser_control_broker,
        )
    except Exception as exc:  # pragma: no cover - defensive, gateway always present
        logger.debug(
            "browser extension router unavailable (%s); using legacy backend",
            exc,
        )
        return fallback()

    if not browser_control_enabled():
        return fallback()

    if tool_call_id is None:
        tool_call_id = current_tool_call_id()

    try:
        from gateway.session_context import get_session_env

        session_id = session_id or get_session_env("HERMES_SESSION_ID", "") or None
        principal_id = principal_id or get_session_env(
            "HERMES_BROWSER_CONTROL_PRINCIPAL", ""
        ) or None
        transport_family = transport_family or get_session_env(
            "HERMES_BROWSER_CONTROL_TRANSPORT_FAMILY", ""
        ) or None
    except Exception:
        pass

    return route_browser_tool(
        action,
        args,
        fallback=fallback,
        broker=get_browser_control_broker(),
        enabled=True,
        session_id=session_id,
        task_id=task_id,
        principal_id=principal_id,
        transport_family=transport_family,
        tool_call_id=tool_call_id,
    )
