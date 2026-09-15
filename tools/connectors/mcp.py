"""MCP targets of ``manage_connections``: the renderer runs install / enable / OAuth and answers
through ``connection.respond``; nothing else observes an MCP flow today (PR3 moves OAuth
observation server-side). Calls without an approval callback settle unavailable."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from tools.connectors.contract import Actor, SettleReason, TargetState
from tools.connectors.gateway.config import operation_session_key, session_platform
from tools.connectors.operation import ConnectionOperation, Target
from tools.connectors.run import Kind, run_operation
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# The renderer's answer vocabulary → target state. A failed approval stays open; a decline resolves.
_OUTCOME_STATES = {
    "installed": TargetState.connected, "enabled": TargetState.connected, "authorized": TargetState.connected,
    "connected": TargetState.connected,
    "declined": TargetState.skipped, "skipped": TargetState.skipped,
    "error": TargetState.failed, "failed": TargetState.failed,
}

# The intermediate `initiated` hop's actor depends on which state a target is leaving: a fresh
# report (still pending) is driven by the renderer flow itself, a retry after a failure is the
# user's explicit "Try again" click — the contract requires the matching actor for each (mirrors
# ``managed.mint``'s ``failed`` handling, which steps through the same hop on reinitiate).
_INITIATE_FROM = {
    TargetState.pending: Actor.renderer_flow,
    TargetState.failed: Actor.user,
}

UNAVAILABLE_HINT = "hermes mcp install {name} / hermes mcp login {name}"

NOTE = (
    "Settled once; do not re-ask for any target the user skipped or that timed out — continue "
    "without it or ask in chat. Tools of a newly installed or authorized server become available "
    "on your next turn."
)


def _catalog_names() -> List[str]:
    from hermes_cli.mcp_catalog import list_catalog

    return sorted(e.name for e in list_catalog())


def _configured_names() -> List[str]:
    from hermes_cli.mcp_catalog import installed_servers

    return sorted(installed_servers())


def validate_mcp_names(action: str, names: List[str]) -> Optional[str]:
    try:
        catalog = _catalog_names()
        configured = _configured_names()
    except Exception as exc:
        return f"could not read the MCP catalog: {exc}"
    allowed = set(catalog) if action == "install" else set(configured)
    unknown = [n for n in names if n not in allowed]
    if not unknown:
        return None
    if action == "install":
        return (
            f"unknown MCP server(s) for install: {', '.join(unknown)}. Install works for "
            f"catalog entries only: {', '.join(catalog) or '(empty catalog)'}."
            + (f" Already configured (use enable/authorize): {', '.join(configured)}." if configured else "")
        )
    return (
        f"unknown MCP server(s) for {action}: {', '.join(unknown)}. {action} works for servers "
        f"already in mcp_servers: {', '.join(configured) or '(none configured)'}."
        + (f" Catalog entries you can install: {', '.join(catalog)}." if catalog else "")
    )


def apply_answer(operation: ConnectionOperation, raw: str) -> None:
    """Fold the card's ``connection.respond`` payload into the operation. Settlement is derived
    from target states afterwards, never from the card's own ``settled_by`` claim."""
    try:
        answer = json.loads(raw)
    except (TypeError, ValueError):
        answer = {}
    if not isinstance(answer, dict):
        answer = {}
    for entry in answer.get("targets") or ():
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip().lower()
        state = _OUTCOME_STATES.get(str(entry.get("state") or entry.get("status") or "").lower())
        target = operation.target(name)
        if target is None or state is None:
            continue
        actor = Actor.user if state == TargetState.skipped else Actor.renderer_flow
        extra = {k: v for k, v in entry.items() if k in ("tools",)}
        initiate_actor = _INITIATE_FROM.get(target.state)
        if initiate_actor is not None and state != TargetState.skipped:
            operation.transition(name, TargetState.initiated, initiate_actor)
        operation.transition(name, state, actor, detail=str(entry.get("detail") or ""), **extra)
    if answer.get("settled_by") == SettleReason.continue_.value and not operation.all_resolved:
        operation.settle(SettleReason.continue_)


def _unavailable(operation: ConnectionOperation) -> str:
    for target in operation.targets:
        target.state = TargetState.unavailable
        target.detail = "no approval surface in this session"
        target.extra = {"hint": UNAVAILABLE_HINT.format(name=target.name)}
    operation.settle(SettleReason.unavailable)
    payload = operation.result()
    payload["status"] = "unavailable"
    payload["note"] = (
        "This session has no approval card, so local MCP servers cannot be set up here. Tell "
        "the user to run the terminal commands in each target's 'hint', then continue."
    )
    return json.dumps(payload, ensure_ascii=False)


def run_mcp_operation(
    names: List[str],
    action: str,
    *,
    connection_callback: Optional[Callable[[Dict[str, Any]], Optional[str]]],
    session_id: Optional[str],
    tool_call_id: Optional[str] = None,
) -> str:
    error = validate_mcp_names(action, names)
    if error:
        return tool_error(error)
    targets = [Target(n, "mcp", action) for n in names]
    session_key = operation_session_key(session_id)
    # The surface decides, not the callback: every tui_gateway session has the callback attached,
    # the Ink TUI included, and only the desktop renders the card.
    if session_platform() != "desktop" or connection_callback is None:
        return _unavailable(ConnectionOperation(targets, session_key=session_key))
    def prepare(operation: ConnectionOperation) -> None:
        pass

    return run_operation(
        targets, Kind(prepare=prepare, observe=lambda op: None, note=NOTE),
        session_key=session_key, tool_call_id=tool_call_id, connection_callback=connection_callback,
        with_urls_in_result=False,
    )
