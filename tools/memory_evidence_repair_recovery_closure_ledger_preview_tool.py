"""Read-only Memory Evidence Repair Recovery Closure Ledger Preview tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_recovery_closure_ledger_preview import (
    build_evidence_repair_recovery_closure_ledger_preview,
)
from tools.memory_evidence_repair_recovery_closure_gate_tool import (
    MEMORY_EVIDENCE_REPAIR_RECOVERY_CLOSURE_GATE_SCHEMA,
    memory_evidence_repair_recovery_closure_gate_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_RECOVERY_CLOSURE_GATE_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_RECOVERY_CLOSURE_LEDGER_PREVIEW_SCHEMA = {
    "name": "memory_evidence_repair_recovery_closure_ledger_preview",
    "description": (
        "Read-only lifecycle ledger preview for a memory evidence repair recovery "
        "closure. It never records a ledger entry or mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recovery_closure_gate": {
                "type": "object",
                "description": "Optional recovery closure gate report payload.",
                "additionalProperties": True,
            },
            "recovery_closure_gate_report": {
                "type": "object",
                "description": "Alias for recovery_closure_gate.",
                "additionalProperties": True,
            },
            "closure_gate": {
                "type": "object",
                "description": "Alias for recovery_closure_gate.",
                "additionalProperties": True,
            },
            "recovery_closure": {
                "type": "object",
                "description": "Optional single recovery closure draft payload.",
                "additionalProperties": True,
            },
            "closure": {
                "type": "object",
                "description": "Alias for recovery_closure.",
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


def memory_evidence_repair_recovery_closure_ledger_preview_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    recovery_closure_gate = _resolve_recovery_closure_gate(
        args,
        kwargs.get("memory_manager"),
    )
    if isinstance(recovery_closure_gate, str):
        return recovery_closure_gate

    payload = build_evidence_repair_recovery_closure_ledger_preview(
        recovery_closure_gate=recovery_closure_gate,
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
        "recovery_closure_gate",
        "recovery_closure_gate_report",
        "closure_gate",
        "recovery_closure",
        "closure",
        "recovery_completion_audit",
        "recovery_completion_audit_preview",
        "recovery_completion_audit_report",
        "recovery_completion_receipt",
        "recovery_completion_receipt_report",
        "completion_receipt",
        "observed_recovery_steps",
        "post_recovery_audit_status",
        "contamination_status",
        "recovery_executor_preview",
        "recovery_executor_preview_report",
        "recovery_executor",
        "recovery_write_lock_gate",
        "recovery_executor_lock_gate",
        "recovery_write_lock",
        "recovery_token_gate",
        "recovery_approval_token_gate",
        "token_gate",
        "recovery_approval_token",
        "recovery_human_approval_token",
        "recovery_token",
        "token",
        "recovery_execution",
        "recovery_execution_preview",
        "recovery_execution_report",
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
        "approval_token_gate",
        "approval_token",
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
        "active_locks",
        "locks",
        "already_used_token_ids",
        "tokens",
        "used_token_ids",
        "released_lock_ids",
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


def _resolve_recovery_closure_gate(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    for field_name in (
        "recovery_closure_gate",
        "recovery_closure_gate_report",
        "closure_gate",
    ):
        explicit = args.get(field_name)
        if isinstance(explicit, dict):
            return explicit

    for field_name in ("recovery_closure", "closure"):
        explicit_closure = args.get(field_name)
        if isinstance(explicit_closure, dict):
            return {
                "status": "recovery_closure_ready",
                "summary": {"closure_count": 1},
                "closures": [explicit_closure],
                "read_only": True,
                "read_only_memory": True,
                "would_mutate_memory": False,
            }

    gate_args = dict(args)
    for field_name in (
        "recovery_closure_gate",
        "recovery_closure_gate_report",
        "closure_gate",
        "recovery_closure",
        "closure",
        "format",
    ):
        gate_args.pop(field_name, None)
    result = memory_evidence_repair_recovery_closure_gate_tool(
        gate_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error("recovery closure gate tool returned invalid JSON.", success=False)
    if not isinstance(payload, dict):
        return tool_error(
            "recovery closure gate tool returned a non-object payload.",
            success=False,
        )
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Recovery Closure Ledger Preview",
        f"- Status: {payload.get('status')}",
        f"- Checks: {summary.get('check_count', 0)}",
        f"- Entries: {summary.get('entry_count', 0)}",
        f"- Milestones: {summary.get('milestone_count', 0)}",
        f"- Ready: {summary.get('recovery_closure_ledger_ready', False)}",
    ]
    for entry in payload.get("entries", []):
        lines.append(f"- Ledger entry: {entry.get('id')}")
        lines.append(f"  Closure: {entry.get('closure_id')}")
        lines.append(f"  Audit: {entry.get('audit_preview_id')}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    for action in payload.get("required_actions", []):
        lines.append(f"- Required: {action}")
    return "\n".join(lines)


def check_memory_evidence_repair_recovery_closure_ledger_preview_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_recovery_closure_ledger_preview",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_RECOVERY_CLOSURE_LEDGER_PREVIEW_SCHEMA,
    handler=memory_evidence_repair_recovery_closure_ledger_preview_tool,
    check_fn=check_memory_evidence_repair_recovery_closure_ledger_preview_requirements,
    emoji="🧠",
)
