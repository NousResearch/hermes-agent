"""MCP targets of ``manage_connections``: target normalization, action/target validation,
catalog validation, and the blocking approval leg that drives a :class:`ConnectionOperation`
through the GUI bridge.

The consent card lives in the desktop renderer. The tool reaches it through
``agent.connection_callback`` (wired per session by ``tui_gateway/agent_callbacks.py``), which
``registry.dispatch`` never forwards: the fold therefore runs through the agent-level inline
executor (``agent/inline_tool_executors.py``), and a registry-path call — every non-GUI
surface — degrades to a per-target ``unavailable`` result with the terminal commands.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.connections_tool_operation import (
    CONNECTED,
    FAILED,
    SETTLED_ALL_RESOLVED,
    SETTLED_CONTINUE,
    SETTLED_DEADLINE,
    SETTLED_INTERRUPT,
    SETTLED_UNAVAILABLE,
    SKIPPED,
    UNAVAILABLE,
    ConnectionOperation,
    Target,
    resolve_wait_timeout,
)
from tools.registry import tool_error

logger = logging.getLogger(__name__)

CONNECTOR_ACTIONS = ("status", "connect", "reconnect", "wait")
MCP_ACTIONS = ("install", "enable", "authorize")
ALL_ACTIONS = CONNECTOR_ACTIONS + MCP_ACTIONS

_TARGET_FIELDS = frozenset({"name", "mcp"})

# Renderer-reported per-target outcomes → operation states. ``declined`` is the renderer's
# word for the user's Not now; ``error`` is a recoverable failure (auth denied, install
# failed) that keeps the card live until Continue or the deadline.
_OUTCOME_STATES = {
    "installed": CONNECTED, "enabled": CONNECTED, "authorized": CONNECTED, "connected": CONNECTED,
    "declined": SKIPPED, "skipped": SKIPPED,
    "error": FAILED, "failed": FAILED,
}

UNAVAILABLE_HINT = "hermes mcp install {name} / hermes mcp login {name}"


def normalize_targets(raw: Any) -> Tuple[List[str], List[str], Optional[str]]:
    """``connectors`` → ``(managed names, mcp names, error)``.

    Bare strings and ``{"name": ...}`` are managed connectors; ``{"name": ..., "mcp": true}``
    is a locally configured MCP server. Any other field on a target object is an error, so a
    model cannot smuggle URLs, commands or credentials past the manifest.
    """
    if raw is None:
        return [], [], None
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return [], [], "'connectors' must be a list of names or {name, mcp} objects."
    managed: List[str] = []
    mcp: List[str] = []
    for item in raw:
        if isinstance(item, dict):
            unknown = sorted(set(item) - _TARGET_FIELDS)
            if unknown:
                return [], [], (
                    f"unknown target field(s) {', '.join(unknown)}: a target is "
                    "{\"name\": \"<slug>\"} or {\"name\": \"<server>\", \"mcp\": true}. Transport, "
                    "URLs and credentials come from the catalog manifest, never from the call."
                )
            name = str(item.get("name") or "").strip().lower()
            is_mcp = bool(item.get("mcp", False))
        else:
            name, is_mcp = str(item or "").strip().lower(), False
        if not name:
            return [], [], "every target needs a non-empty 'name'."
        bucket = mcp if is_mcp else managed
        if name not in bucket:
            bucket.append(name)
    return managed, mcp, None


def validate_action(action: str, managed: List[str], mcp: List[str]) -> Optional[str]:
    """MCP verbs need ``mcp:true`` targets; connector verbs need managed targets."""
    if action not in ALL_ACTIONS:
        return (
            f"action must be one of {', '.join(ALL_ACTIONS)}. "
            f"{', '.join(MCP_ACTIONS)} apply to local MCP servers "
            "(targets {\"name\": ..., \"mcp\": true}); the rest apply to managed connectors. "
            "Disconnecting an account is done by the user in the Nous Portal dashboard, not "
            "through this tool."
        )
    if action in MCP_ACTIONS:
        if managed:
            return (
                f"'{action}' is an MCP action: every target must carry \"mcp\": true "
                f"(got managed connector(s) {', '.join(managed)}). Managed connectors use "
                "connect / reconnect / wait / status."
            )
        if not mcp:
            return (
                f"'{action}' requires 'connectors': the MCP server name(s), e.g. "
                "[{\"name\": \"linear\", \"mcp\": true}]."
            )
    elif mcp:
        return (
            f"'{action}' is a managed-connector action; MCP targets ({', '.join(mcp)}) use "
            f"{', '.join(MCP_ACTIONS)}."
        )
    return None


def _catalog_names() -> List[str]:
    from hermes_cli.mcp_catalog import list_catalog

    return sorted(e.name for e in list_catalog())


def _configured_names() -> List[str]:
    from hermes_cli.mcp_catalog import installed_servers

    return sorted(installed_servers())


def validate_mcp_names(action: str, names: List[str]) -> Optional[str]:
    """``install`` is catalog-only; ``enable`` / ``authorize`` need a configured server.
    Unknown → a model-actionable error that lists what exists."""
    try:
        catalog = _catalog_names()
        configured = _configured_names()
    except Exception as exc:  # catalog on disk is malformed: still a model-visible answer
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


def _unavailable_result(operation: ConnectionOperation) -> str:
    for target in operation.targets:
        operation.record_target(
            target.name, UNAVAILABLE, "no approval surface in this session",
            hint=UNAVAILABLE_HINT.format(name=target.name),
        )
    operation.settle(SETTLED_UNAVAILABLE)
    payload = operation.result()
    payload["status"] = "unavailable"
    payload["note"] = (
        "This session has no approval card, so local MCP servers cannot be set up here. Tell "
        "the user to run the terminal commands in each target's 'hint', then continue."
    )
    return json.dumps(payload, ensure_ascii=False)


def _apply_answer(operation: ConnectionOperation, raw: str) -> Optional[str]:
    """Fold the renderer's answer into the operation; returns how it asked to settle (or None)."""
    try:
        answer = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(answer, dict):
        return None
    for entry in answer.get("targets") or ():
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip().lower()
        state = _OUTCOME_STATES.get(str(entry.get("state") or entry.get("status") or "").lower())
        if not name or state is None:
            continue
        extra = {k: v for k, v in entry.items() if k in ("tools",)}
        operation.record_target(name, state, str(entry.get("detail") or ""), **extra)
    settled_by = str(answer.get("settled_by") or "")
    return settled_by if settled_by in (SETTLED_ALL_RESOLVED, SETTLED_CONTINUE) else None


def run_mcp_operation(
    names: List[str],
    action: str,
    reason: str,
    *,
    connection_callback: Optional[Callable[[Dict[str, Any]], Optional[str]]],
    session_id: Optional[str],
    wait_seconds: Optional[float] = None,
) -> str:
    """One operation for the MCP targets of a call. Returns the tool's JSON string."""
    error = validate_mcp_names(action, names)
    if error:
        return tool_error(error)
    operation = ConnectionOperation(
        [Target(n, "mcp", action) for n in names],
        session_key=str(session_id or ""),
        wait_seconds=resolve_wait_timeout() if wait_seconds is None else wait_seconds,
    )
    if connection_callback is None:
        return _unavailable_result(operation)

    try:
        raw = connection_callback(operation.request_payload(reason))
    except Exception as exc:
        return tool_error(f"MCP approval flow failed: {exc}")

    settled_by = _apply_answer(operation, raw or "")
    if raw:
        operation.settle(settled_by or (SETTLED_ALL_RESOLVED if operation.all_resolved else SETTLED_CONTINUE))
    else:
        # The bridge returned empty: the deadline passed or the user interrupted the turn.
        from tools.interrupt import is_interrupted

        operation.settle(SETTLED_INTERRUPT if is_interrupted() else SETTLED_DEADLINE)

    payload = operation.result()
    payload["status"] = "settled"
    payload["note"] = (
        "Settled once; do not re-ask for any target the user skipped or that timed out — "
        "continue without it or ask in chat. Tools of a newly installed or authorized server "
        "become available on your next turn."
    )
    return json.dumps(payload, ensure_ascii=False)
