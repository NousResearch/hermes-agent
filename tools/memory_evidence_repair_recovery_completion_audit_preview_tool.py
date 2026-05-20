"""Read-only Memory Evidence Repair Recovery Completion Audit Preview tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_recovery_completion_audit_preview import (
    build_evidence_repair_recovery_completion_audit_preview,
)
from tools.memory_evidence_repair_recovery_completion_receipt_tool import (
    MEMORY_EVIDENCE_REPAIR_RECOVERY_COMPLETION_RECEIPT_SCHEMA,
    memory_evidence_repair_recovery_completion_receipt_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_RECOVERY_COMPLETION_RECEIPT_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_RECOVERY_COMPLETION_AUDIT_PREVIEW_SCHEMA = {
    "name": "memory_evidence_repair_recovery_completion_audit_preview",
    "description": (
        "Read-only recovery completion audit preview for future memory evidence "
        "repair manual recoveries. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recovery_completion_receipt": {
                "type": "object",
                "description": "Optional recovery completion receipt report payload.",
                "additionalProperties": True,
            },
            "recovery_completion_receipt_report": {
                "type": "object",
                "description": "Alias for recovery_completion_receipt.",
                "additionalProperties": True,
            },
            "completion_receipt": {
                "type": "object",
                "description": "Alias for recovery_completion_receipt.",
                "additionalProperties": True,
            },
            "observed_recovery_steps": {
                "type": "object",
                "description": "Optional observed recovery step signals keyed by recovery step id.",
                "additionalProperties": True,
            },
            "recorded_receipts": {
                "description": "Optional recorded recovery completion receipt report or receipt list.",
                "oneOf": [
                    {"type": "object", "additionalProperties": True},
                    {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                ],
            },
            "released_lock_ids": {
                "type": "array",
                "description": "Optional recovery write-lock ids observed as released.",
                "items": {"type": "string"},
            },
            "post_recovery_audit_status": {
                "type": "object",
                "description": "Optional observed post-recovery audit status.",
                "additionalProperties": True,
            },
            "contamination_status": {
                "type": "object",
                "description": "Optional observed secondary-memory-contamination status.",
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


def memory_evidence_repair_recovery_completion_audit_preview_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    recovery_completion_receipt = _resolve_recovery_completion_receipt(
        args,
        kwargs.get("memory_manager"),
    )
    if isinstance(recovery_completion_receipt, str):
        return recovery_completion_receipt

    payload = build_evidence_repair_recovery_completion_audit_preview(
        recovery_completion_receipt=recovery_completion_receipt,
        observed_recovery_steps=(
            args.get("observed_recovery_steps")
            if isinstance(args.get("observed_recovery_steps"), dict)
            else None
        ),
        recorded_receipts=_recorded_receipts(args),
        used_token_ids=(
            args.get("used_token_ids")
            if isinstance(args.get("used_token_ids"), list)
            else None
        ),
        released_lock_ids=(
            args.get("released_lock_ids")
            if isinstance(args.get("released_lock_ids"), list)
            else None
        ),
        post_recovery_audit_status=(
            args.get("post_recovery_audit_status")
            if isinstance(args.get("post_recovery_audit_status"), dict)
            else None
        ),
        contamination_status=(
            args.get("contamination_status")
            if isinstance(args.get("contamination_status"), dict)
            else None
        ),
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


def _resolve_recovery_completion_receipt(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    for field_name in (
        "recovery_completion_receipt",
        "recovery_completion_receipt_report",
        "completion_receipt",
    ):
        explicit = args.get(field_name)
        if isinstance(explicit, dict):
            return explicit

    receipt_args = dict(args)
    for field_name in (
        "recovery_completion_receipt",
        "recovery_completion_receipt_report",
        "completion_receipt",
        "observed_recovery_steps",
        "recorded_receipts",
        "used_token_ids",
        "released_lock_ids",
        "post_recovery_audit_status",
        "contamination_status",
        "format",
    ):
        receipt_args.pop(field_name, None)
    result = memory_evidence_repair_recovery_completion_receipt_tool(
        receipt_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error(
            "recovery completion receipt tool returned invalid JSON.",
            success=False,
        )
    if not isinstance(payload, dict):
        return tool_error(
            "recovery completion receipt tool returned a non-object payload.",
            success=False,
        )
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _recorded_receipts(args: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]] | None:
    explicit = args.get("recorded_receipts")
    if isinstance(explicit, (dict, list)):
        return explicit
    return None


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Recovery Completion Audit Preview",
        f"- Status: {payload.get('status')}",
        f"- Previews: {summary.get('preview_count', 0)}",
        f"- Audit steps: {summary.get('audit_step_count', 0)}",
        f"- Planned: {summary.get('planned_audit_step_count', 0)}",
        f"- Observed pass: {summary.get('observed_pass_step_count', 0)}",
        f"- Observed fail: {summary.get('observed_fail_step_count', 0)}",
    ]
    for preview in payload.get("previews", []):
        lines.append(f"- Audit: {preview.get('id')}")
        lines.append(f"  Receipt: {preview.get('receipt_id')}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    for action in payload.get("required_actions", []):
        lines.append(f"- Required: {action}")
    return "\n".join(lines)


def check_memory_evidence_repair_recovery_completion_audit_preview_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_recovery_completion_audit_preview",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_RECOVERY_COMPLETION_AUDIT_PREVIEW_SCHEMA,
    handler=memory_evidence_repair_recovery_completion_audit_preview_tool,
    check_fn=check_memory_evidence_repair_recovery_completion_audit_preview_requirements,
    emoji="🧠",
)
