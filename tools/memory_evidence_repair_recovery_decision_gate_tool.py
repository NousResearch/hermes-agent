"""Read-only Memory Evidence Repair Recovery Decision Gate tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_recovery_decision_gate import (
    build_evidence_repair_recovery_decision_gate,
)
from tools.memory_evidence_repair_rollback_drill_preview_tool import (
    MEMORY_EVIDENCE_REPAIR_ROLLBACK_DRILL_PREVIEW_SCHEMA,
    memory_evidence_repair_rollback_drill_preview_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_ROLLBACK_DRILL_PREVIEW_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_RECOVERY_DECISION_GATE_SCHEMA = {
    "name": "memory_evidence_repair_recovery_decision_gate",
    "description": (
        "Read-only recovery decision gate for memory evidence repair rollback "
        "drills. It chooses the next recovery route without mutating durable "
        "memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "rollback_drill": {
                "type": "object",
                "description": "Optional rollback drill preview report payload.",
                "additionalProperties": True,
            },
            "rollback_drill_preview": {
                "type": "object",
                "description": "Alias for rollback_drill.",
                "additionalProperties": True,
            },
            "rollback_drill_report": {
                "type": "object",
                "description": "Alias for rollback_drill.",
                "additionalProperties": True,
            },
            "format": {
                "type": "string",
                "enum": ["json", "markdown"],
                "description": "Optional response format helper. JSON is always returned; markdown adds a readable summary field.",
            },
            **_UPSTREAM_PROPERTIES,
        },
    },
}


def memory_evidence_repair_recovery_decision_gate_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    rollback_drill = _resolve_rollback_drill(args, kwargs.get("memory_manager"))
    if isinstance(rollback_drill, str):
        return rollback_drill

    payload = build_evidence_repair_recovery_decision_gate(
        rollback_drill=rollback_drill,
    ).to_dict()
    payload["success"] = True
    payload["read_only"] = True
    payload["read_only_memory"] = True
    payload["would_mutate_memory"] = False
    if str(args.get("format", "")).lower() == "markdown":
        payload["markdown"] = _render_markdown(payload)
    return json.dumps(payload, ensure_ascii=False)


def _validate_args(args: dict[str, Any]) -> str | None:
    object_fields = (
        "rollback_drill",
        "rollback_drill_preview",
        "rollback_drill_report",
        "post_commit_audit",
        "post_commit_audit_preview",
        "post_commit_audit_report",
        "executor_preview",
        "executor",
        "observed_memory_patches",
        "rollback_status",
        "write_lock_gate",
        "executor_lock_gate",
        "write_lock",
        "commit_receipt",
        "receipt",
        "token_gate",
        "approval_token_gate",
        "approval_token",
        "token",
        "dry_run",
        "snapshot_plan",
        "rollback_plan",
        "existing_snapshots",
        "pre_commit_snapshots",
        "ledger",
        "commit_gate",
        "preview",
        "approval",
        "plan",
        "gate",
        "audit",
        "diagnostics",
        "policy",
        "proposed_evidence",
    )
    array_fields = (
        "released_lock_ids",
        "active_locks",
        "locks",
        "already_used_token_ids",
        "tokens",
        "used_token_ids",
        "receipts",
        "requests",
        "steps",
        "entries",
        "decisions",
        "previews",
        "confirmed_preview_ids",
        "candidates",
        "approved_candidate_ids",
        "repairs",
    )
    for field_name in object_fields:
        value = args.get(field_name)
        if value is not None and not isinstance(value, dict):
            return tool_error(f"{field_name} must be an object when provided.", success=False)
    for field_name in ("existing_receipts", "recorded_receipts"):
        value = args.get(field_name)
        if value is not None and not isinstance(value, (dict, list)):
            return tool_error(
                f"{field_name} must be an object or array when provided.",
                success=False,
            )
    for field_name in array_fields:
        value = args.get(field_name)
        if value is not None and not isinstance(value, list):
            return tool_error(f"{field_name} must be an array when provided.", success=False)
    return None


def _resolve_rollback_drill(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    for field_name in (
        "rollback_drill",
        "rollback_drill_preview",
        "rollback_drill_report",
    ):
        explicit = args.get(field_name)
        if isinstance(explicit, dict):
            return explicit

    drill_args = dict(args)
    for field_name in (
        "rollback_drill",
        "rollback_drill_preview",
        "rollback_drill_report",
        "format",
    ):
        drill_args.pop(field_name, None)
    result = memory_evidence_repair_rollback_drill_preview_tool(
        drill_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error("rollback drill preview tool returned invalid JSON.", success=False)
    if not isinstance(payload, dict):
        return tool_error(
            "rollback drill preview tool returned a non-object payload.",
            success=False,
        )
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Recovery Decision Gate",
        f"- Status: {payload.get('status')}",
        f"- Decisions: {summary.get('decision_count', 0)}",
        f"- Manual rollback: {summary.get('manual_rollback_required_count', 0)}",
        f"- Isolation review: {summary.get('isolate_and_review_count', 0)}",
        f"- Preparedness review: {summary.get('preparedness_review_count', 0)}",
    ]
    for decision in payload.get("decisions", []):
        lines.append(f"- Decision: {decision.get('id')}")
        lines.append(f"  Route: {decision.get('route')}")
        lines.append(f"  Priority: {decision.get('priority')}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    for action in payload.get("required_actions", []):
        lines.append(f"- Required: {action}")
    return "\n".join(lines)


def check_memory_evidence_repair_recovery_decision_gate_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_recovery_decision_gate",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_RECOVERY_DECISION_GATE_SCHEMA,
    handler=memory_evidence_repair_recovery_decision_gate_tool,
    check_fn=check_memory_evidence_repair_recovery_decision_gate_requirements,
    emoji="🧠",
)
