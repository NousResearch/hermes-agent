"""Persistent task admission bridge into the existing GoalManager lifecycle."""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from hermes_cli.goals import GoalContract, GoalManager
from tools.registry import registry


_CONTRACT_LIST_FIELDS = ("constraints", "boundaries", "stop_when")


TASK_COMMIT_SCHEMA = {
    "name": "task_commit",
    "description": (
        "Commit a clear execution task to Hermes' persistent Goal lifecycle when completion must be tracked "
        "across multiple actions, tool calls, background operations, or turns. Do not use for discovery, "
        "ordinary Q&A, or short work reliably completed in this turn. If a Goal already exists, use amend "
        "only for the same task, replace only for an explicitly changed objective, or do not call this tool "
        "for an unrelated side question. outcome + verification define DONE; stop_when defines conditions "
        "that require BLOCKED/human intervention. The tool never plans or executes the task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["create", "amend", "replace"],
                "description": "create a new Goal, amend the same Goal, or explicitly replace its objective.",
            },
            "objective": {"type": ["string", "null"], "description": "What must be accomplished."},
            "outcome": {
                "type": ["string", "null"],
                "description": "The end state that must be true when the task is complete.",
            },
            "verification": {
                "type": ["string", "null"],
                "description": "Concrete evidence that proves the outcome is satisfied.",
            },
            "constraints": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Conditions that must not be violated. None means preserve on amend.",
            },
            "boundaries": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Allowed files, systems, tools, or scope. None means preserve on amend.",
            },
            "stop_when": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Conditions that require BLOCKED/human intervention, never DONE.",
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}


def check_task_commit_requirements() -> bool:
    return True


def _text(value: Any, field: str, *, required: bool = False) -> Optional[str]:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    value = value.strip()
    if not value:
        raise ValueError(f"{field} cannot be empty")
    return value


def _items(value: Any, field: str) -> Optional[list[str]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array or null")
    cleaned: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{field} entries must be non-empty strings")
        item = raw.strip()
        if item not in cleaned:
            cleaned.append(item)
    if not cleaned:
        raise ValueError(f"{field} cannot be an empty array; V1 does not support clearing fields")
    return cleaned


def _field_items(value: str) -> list[str]:
    """Recover V1 list semantics while preserving legacy one-line GoalContract values."""
    if not value or not value.strip():
        return []
    items = []
    for raw in value.splitlines():
        item = raw.strip()
        if item.startswith("- "):
            item = item[2:].strip()
        if item and item not in items:
            items.append(item)
    return items or [value.strip()]


def _join_items(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _contract(outcome: str, verification: str, constraints, boundaries, stop_when) -> GoalContract:
    return GoalContract(
        outcome=outcome,
        verification=verification,
        constraints=_join_items(constraints or []),
        boundaries=_join_items(boundaries or []),
        stop_when=_join_items(stop_when or []),
    )


def _snapshot(manager: GoalManager) -> Optional[dict]:
    state = manager.state
    if state is None:
        return None
    return {
        "objective": state.goal,
        "status": state.status,
        "created_at": state.created_at,
        "turns_used": state.turns_used,
        "max_turns": state.max_turns,
        "contract": state.contract.to_dict(),
    }


def _same_contract(left: GoalContract, right: GoalContract) -> bool:
    return left.to_dict() == right.to_dict()


def _verify_persisted(session_id: str, expected_goal: str, expected_contract: GoalContract, expected_created_at: float) -> bool:
    reloaded = GoalManager(session_id=session_id).state
    return bool(
        reloaded is not None
        and reloaded.goal == expected_goal
        and reloaded.status in {"active", "paused"}
        and reloaded.created_at == expected_created_at
        and _same_contract(reloaded.contract, expected_contract)
    )


def task_commit(
    *,
    session_id: str,
    operation: str,
    objective: Optional[str] = None,
    outcome: Optional[str] = None,
    verification: Optional[str] = None,
    constraints: Optional[list[str]] = None,
    boundaries: Optional[list[str]] = None,
    stop_when: Optional[list[str]] = None,
) -> str:
    """Create, amend, or replace one session's persistent GoalContract."""
    try:
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("current session identity is unavailable")
        operation = str(operation or "").strip().lower()
        if operation not in {"create", "amend", "replace"}:
            raise ValueError("operation must be create, amend, or replace")

        manager = GoalManager(session_id=sid)
        current = manager.state
        has_goal = manager.has_goal()

        raw_lists = {"constraints": constraints, "boundaries": boundaries, "stop_when": stop_when}
        if operation in {"create", "replace"}:
            values = {
                "objective": _text(objective, "objective", required=True),
                "outcome": _text(outcome, "outcome", required=True),
                "verification": _text(verification, "verification", required=True),
            }
            list_values = {field: _items(raw_lists[field], field) for field in _CONTRACT_LIST_FIELDS}
            proposed = _contract(
                values["outcome"], values["verification"],
                list_values["constraints"], list_values["boundaries"], list_values["stop_when"],
            )

            if operation == "create" and has_goal:
                if current and current.goal == values["objective"] and _same_contract(current.contract, proposed):
                    return json.dumps({"success": True, "result": "idempotent_noop", "goal": _snapshot(manager)}, ensure_ascii=False)
                return json.dumps({
                    "success": False,
                    "result": "conflict",
                    "error": "a different active or paused Goal already exists; choose amend, replace, or no task_commit",
                    "goal": _snapshot(manager),
                }, ensure_ascii=False)

            previous_created_at = current.created_at if current is not None else 0.0
            state = manager.set(values["objective"], contract=proposed)
            if operation == "replace" and has_goal and state.created_at <= previous_created_at:
                # The timestamp is the V1 stale-event generation cutoff; keep it strictly monotonic
                # even if the wall clock stalls or steps backwards.
                state.created_at = previous_created_at + 0.000001
                manager._save()
            result = "replaced" if operation == "replace" and has_goal else "created"
            if not _verify_persisted(sid, state.goal, state.contract, state.created_at):
                raise RuntimeError("Goal persistence verification failed")
            return json.dumps({"success": True, "result": result, "goal": _snapshot(manager)}, ensure_ascii=False)

        if not has_goal or current is None:
            raise RuntimeError("no active or paused Goal exists to amend")

        amended_objective = _text(objective, "objective")
        if amended_objective is not None and amended_objective != current.goal:
            raise ValueError("amend cannot change objective; use replace")
        amended_outcome = _text(outcome, "outcome")
        amended_verification = _text(verification, "verification")
        supplied_lists = {field: _items(raw_lists[field], field) for field in _CONTRACT_LIST_FIELDS}

        old = current.contract
        merged = {}
        for field in _CONTRACT_LIST_FIELDS:
            existing = _field_items(getattr(old, field))
            additions = supplied_lists[field]
            merged[field] = existing if additions is None else existing + [item for item in additions if item not in existing]
        proposed = GoalContract(
            outcome=amended_outcome if amended_outcome is not None else old.outcome,
            verification=amended_verification if amended_verification is not None else old.verification,
            constraints=_join_items(merged["constraints"]),
            boundaries=_join_items(merged["boundaries"]),
            stop_when=_join_items(merged["stop_when"]),
        )
        if _same_contract(old, proposed):
            return json.dumps({"success": True, "result": "idempotent_noop", "goal": _snapshot(manager)}, ensure_ascii=False)
        created_at = current.created_at
        manager.set_contract(proposed)
        if not _verify_persisted(sid, current.goal, proposed, created_at):
            raise RuntimeError("Goal amendment persistence verification failed")
        return json.dumps({"success": True, "result": "amended", "goal": _snapshot(manager)}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


registry.register(
    name="task_commit",
    toolset="task_commit",
    schema=TASK_COMMIT_SCHEMA,
    handler=lambda args, **kwargs: task_commit(
        session_id=kwargs.get("session_id") or "",
        operation=args.get("operation", ""),
        objective=args.get("objective"),
        outcome=args.get("outcome"),
        verification=args.get("verification"),
        constraints=args.get("constraints"),
        boundaries=args.get("boundaries"),
        stop_when=args.get("stop_when"),
    ),
    check_fn=check_task_commit_requirements,
)
