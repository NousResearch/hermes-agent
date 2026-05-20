"""Read-only Memory Evidence Repair Recovery Execution Preview tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_recovery_execution_preview import (
    build_evidence_repair_recovery_execution_preview,
)
from tools.memory_evidence_repair_recovery_decision_gate_tool import (
    MEMORY_EVIDENCE_REPAIR_RECOVERY_DECISION_GATE_SCHEMA,
    memory_evidence_repair_recovery_decision_gate_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_RECOVERY_DECISION_GATE_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_RECOVERY_EXECUTION_PREVIEW_SCHEMA = {
    "name": "memory_evidence_repair_recovery_execution_preview",
    "description": (
        "Read-only recovery execution preview for memory evidence repair. "
        "It orders future manual recovery actions without mutating durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recovery_decision": {
                "type": "object",
                "description": "Optional recovery decision gate report payload.",
                "additionalProperties": True,
            },
            "recovery_decision_gate": {
                "type": "object",
                "description": "Alias for recovery_decision.",
                "additionalProperties": True,
            },
            "recovery_decision_report": {
                "type": "object",
                "description": "Alias for recovery_decision.",
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


def memory_evidence_repair_recovery_execution_preview_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    recovery_decision = _resolve_recovery_decision(args, kwargs.get("memory_manager"))
    if isinstance(recovery_decision, str):
        return recovery_decision

    payload = build_evidence_repair_recovery_execution_preview(
        recovery_decision=recovery_decision,
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
        "recovery_decision",
        "recovery_decision_gate",
        "recovery_decision_report",
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


def _resolve_recovery_decision(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    for field_name in (
        "recovery_decision",
        "recovery_decision_gate",
        "recovery_decision_report",
    ):
        explicit = args.get(field_name)
        if isinstance(explicit, dict):
            return explicit

    decision_args = dict(args)
    for field_name in (
        "recovery_decision",
        "recovery_decision_gate",
        "recovery_decision_report",
        "format",
    ):
        decision_args.pop(field_name, None)
    result = memory_evidence_repair_recovery_decision_gate_tool(
        decision_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error("recovery decision gate tool returned invalid JSON.", success=False)
    if not isinstance(payload, dict):
        return tool_error(
            "recovery decision gate tool returned a non-object payload.",
            success=False,
        )
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Recovery Execution Preview",
        f"- Status: {payload.get('status')}",
        f"- Previews: {summary.get('preview_count', 0)}",
        f"- Execution steps: {summary.get('execution_step_count', 0)}",
        f"- Future mutation steps: {summary.get('future_mutation_step_count', 0)}",
        f"- Manual rollback previews: {summary.get('manual_rollback_preview_count', 0)}",
    ]
    for preview in payload.get("previews", []):
        lines.append(f"- Preview: {preview.get('id')}")
        lines.append(f"  Route: {preview.get('route')}")
        lines.append(f"  Decision: {preview.get('decision_id')}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    for action in payload.get("required_actions", []):
        lines.append(f"- Required: {action}")
    return "\n".join(lines)


def check_memory_evidence_repair_recovery_execution_preview_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_recovery_execution_preview",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_RECOVERY_EXECUTION_PREVIEW_SCHEMA,
    handler=memory_evidence_repair_recovery_execution_preview_tool,
    check_fn=check_memory_evidence_repair_recovery_execution_preview_requirements,
    emoji="🧠",
)
