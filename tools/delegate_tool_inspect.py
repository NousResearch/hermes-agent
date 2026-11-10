"""Read-only recovery of a conversation's durable delegation result."""

import json

from agent.runtime_session_store import WorkerPersistenceError
from tools.async_delegation import _LIVE_STATES, get_durable_delegation
from tools.delegate_tool_registry import _owns_subagent_record
from tools.delegation_redaction import redact_child_result
from tools.registry import tool_error

_GOAL_CHARS = 1000
_SUMMARY_CHARS = 4000


def _bounded(value, cap):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= cap else text[:cap - 1] + "…"


def _inspect_results(row):
    result = redact_child_result(row.get("result") or {})
    entries = result.get("results", [result])
    return [{"task_index": entry.get("task_index", index), "status": entry.get("status", row["state"]),
             "summary": _bounded(entry.get("summary") or entry.get("error") or "", _SUMMARY_CHARS)}
            for index, entry in enumerate(entries)]


def inspect_delegation(delegation_id, parent_agent):
    """Recover an owned result without claiming delivery or exposing live output."""
    from gateway.session_context import get_session_env

    if not (delegation_id or "").strip():
        return tool_error("action='inspect' requires delegation_id (from the dispatch response).")
    not_found = tool_error("Delegation not found in this conversation.")
    try:
        row = get_durable_delegation(delegation_id.strip())
    except WorkerPersistenceError as exc:
        if str(exc) not in {"not_found", "permission_denied"}:
            raise
        return not_found
    if row is None:
        return not_found
    ui = get_session_env("HERMES_UI_SESSION_ID", "")
    if not ((ui and row.get("origin_ui_session_id") == ui) or _owns_subagent_record(
            {"owner_agent_session_id": row.get("parent_session_id")}, parent_agent)):
        return not_found
    if row["state"] in _LIVE_STATES:
        return json.dumps({"state": "running"})
    task = row.get("task") or {}
    goal = task.get("goal") or "; ".join(task.get("goals") or [])
    payload = {"delegation_id": row["delegation_id"], "state": row["state"],
               "goal": _bounded(goal, _GOAL_CHARS), "dispatched_at": row.get("dispatched_at"),
               "completed_at": row.get("completed_at"), "delivery_state": row.get("delivery_state"),
               "results": _inspect_results(row)}
    return json.dumps(payload, ensure_ascii=False)
