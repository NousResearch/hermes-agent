"""Read-only Memory Evidence Repair Recovery Completion Receipt tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_recovery_completion_receipt import (
    build_evidence_repair_recovery_completion_receipt,
)
from tools.memory_evidence_repair_recovery_executor_preview_tool import (
    MEMORY_EVIDENCE_REPAIR_RECOVERY_EXECUTOR_PREVIEW_SCHEMA,
    memory_evidence_repair_recovery_executor_preview_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_RECOVERY_EXECUTOR_PREVIEW_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_RECOVERY_COMPLETION_RECEIPT_SCHEMA = {
    "name": "memory_evidence_repair_recovery_completion_receipt",
    "description": (
        "Read-only completion receipt draft and token/lock ledger view for a "
        "future memory evidence repair manual recovery. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recovery_executor_preview": {
                "type": "object",
                "description": "Optional recovery executor preview report payload.",
                "additionalProperties": True,
            },
            "recovery_executor_preview_report": {
                "type": "object",
                "description": "Alias for recovery_executor_preview.",
                "additionalProperties": True,
            },
            "recovery_executor": {
                "type": "object",
                "description": "Alias for recovery_executor_preview.",
                "additionalProperties": True,
            },
            "existing_receipts": {
                "description": "Optional existing recovery completion receipt report or receipt list.",
                "oneOf": [
                    {"type": "object", "additionalProperties": True},
                    {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                ],
            },
            "receipts": {
                "type": "array",
                "description": "Optional existing recovery completion receipt list.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "used_token_ids": {
                "type": "array",
                "description": "Optional used recovery approval token ids.",
                "items": {"type": "string"},
            },
            "released_lock_ids": {
                "type": "array",
                "description": "Optional released recovery write-lock ids.",
                "items": {"type": "string"},
            },
            "actor": {
                "type": "string",
                "description": "Human actor label for the future recovery completion receipt.",
            },
            "recovery_reason": {
                "type": "string",
                "description": "Reason for the future manual recovery completion.",
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


def memory_evidence_repair_recovery_completion_receipt_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    recovery_executor_preview = _resolve_recovery_executor_preview(
        args,
        kwargs.get("memory_manager"),
    )
    if isinstance(recovery_executor_preview, str):
        return recovery_executor_preview

    payload = build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview=recovery_executor_preview,
        existing_receipts=_existing_receipts(args),
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
        actor=str(args.get("actor") or "human"),
        recovery_reason=str(args.get("recovery_reason") or ""),
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


def _resolve_recovery_executor_preview(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    for field_name in (
        "recovery_executor_preview",
        "recovery_executor_preview_report",
        "recovery_executor",
    ):
        explicit = args.get(field_name)
        if isinstance(explicit, dict):
            return explicit

    executor_args = dict(args)
    for field_name in (
        "recovery_executor_preview",
        "recovery_executor_preview_report",
        "recovery_executor",
        "existing_receipts",
        "receipts",
        "used_token_ids",
        "released_lock_ids",
        "actor",
        "recovery_reason",
        "format",
    ):
        executor_args.pop(field_name, None)
    result = memory_evidence_repair_recovery_executor_preview_tool(
        executor_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error(
            "recovery executor preview tool returned invalid JSON.",
            success=False,
        )
    if not isinstance(payload, dict):
        return tool_error(
            "recovery executor preview tool returned a non-object payload.",
            success=False,
        )
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _existing_receipts(args: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]] | None:
    explicit = args.get("existing_receipts")
    if isinstance(explicit, (dict, list)):
        return explicit
    receipts = args.get("receipts")
    if isinstance(receipts, list):
        return receipts
    return None


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Recovery Completion Receipt",
        f"- Status: {payload.get('status')}",
        f"- Receipts: {summary.get('receipt_count', 0)}",
        f"- Used recovery tokens: {summary.get('used_recovery_token_count', 0)}",
        f"- Released recovery locks: {summary.get('released_recovery_lock_count', 0)}",
        f"- Blocks: {summary.get('blocking_reason_count', 0)}",
    ]
    for receipt in payload.get("receipts", []):
        lines.append(f"- Receipt: {receipt.get('id')}")
        lines.append(f"  Token: {receipt.get('token_id')}")
        lines.append(f"  Lock: {receipt.get('lock_id')}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    for action in payload.get("required_actions", []):
        lines.append(f"- Required: {action}")
    return "\n".join(lines)


def check_memory_evidence_repair_recovery_completion_receipt_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_recovery_completion_receipt",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_RECOVERY_COMPLETION_RECEIPT_SCHEMA,
    handler=memory_evidence_repair_recovery_completion_receipt_tool,
    check_fn=check_memory_evidence_repair_recovery_completion_receipt_requirements,
    emoji="🧠",
)
