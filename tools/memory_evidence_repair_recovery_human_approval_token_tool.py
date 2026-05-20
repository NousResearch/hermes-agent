"""Read-only Memory Evidence Repair Recovery Human Approval Token tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_recovery_human_approval_token import (
    build_evidence_repair_recovery_human_approval_token,
)
from tools.memory_evidence_repair_recovery_execution_preview_tool import (
    MEMORY_EVIDENCE_REPAIR_RECOVERY_EXECUTION_PREVIEW_SCHEMA,
    memory_evidence_repair_recovery_execution_preview_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_RECOVERY_EXECUTION_PREVIEW_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)
_UPSTREAM_PROPERTIES.pop("expires_in_minutes", None)

MEMORY_EVIDENCE_REPAIR_RECOVERY_HUMAN_APPROVAL_TOKEN_SCHEMA = {
    "name": "memory_evidence_repair_recovery_human_approval_token",
    "description": (
        "Read-only recovery human approval token draft for a ready memory "
        "evidence repair recovery execution preview. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recovery_execution": {
                "type": "object",
                "description": "Optional recovery execution preview report payload.",
                "additionalProperties": True,
            },
            "recovery_execution_preview": {
                "type": "object",
                "description": "Alias for recovery_execution.",
                "additionalProperties": True,
            },
            "recovery_execution_report": {
                "type": "object",
                "description": "Alias for recovery_execution.",
                "additionalProperties": True,
            },
            "approver": {
                "type": "string",
                "description": "Human approver label for the recovery token draft.",
            },
            "approval_reason": {
                "type": "string",
                "description": "Reason for drafting the recovery approval token.",
            },
            "expires_in_minutes": {
                "type": "integer",
                "description": "Draft recovery token expiry window in minutes. Defaults to 15.",
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


def memory_evidence_repair_recovery_human_approval_token_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    recovery_execution = _resolve_recovery_execution(
        args,
        kwargs.get("memory_manager"),
    )
    if isinstance(recovery_execution, str):
        return recovery_execution

    payload = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=recovery_execution,
        approver=str(args.get("approver") or "human"),
        approval_reason=str(args.get("approval_reason") or ""),
        expires_in_minutes=args.get("expires_in_minutes") or 15,
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


def _resolve_recovery_execution(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    for field_name in (
        "recovery_execution",
        "recovery_execution_preview",
        "recovery_execution_report",
    ):
        explicit = args.get(field_name)
        if isinstance(explicit, dict):
            return explicit

    execution_args = dict(args)
    for field_name in (
        "recovery_execution",
        "recovery_execution_preview",
        "recovery_execution_report",
        "approver",
        "approval_reason",
        "expires_in_minutes",
        "format",
    ):
        execution_args.pop(field_name, None)
    result = memory_evidence_repair_recovery_execution_preview_tool(
        execution_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error(
            "recovery execution preview tool returned invalid JSON.",
            success=False,
        )
    if not isinstance(payload, dict):
        return tool_error(
            "recovery execution preview tool returned a non-object payload.",
            success=False,
        )
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Recovery Human Approval Token",
        f"- Status: {payload.get('status')}",
        f"- Tokens: {summary.get('token_count', 0)}",
        f"- Blocks: {summary.get('blocking_reason_count', 0)}",
        f"- Required actions: {summary.get('required_action_count', 0)}",
    ]
    for token in payload.get("tokens", []):
        lines.append(f"- Token: {token.get('id')}")
        lines.append(f"  Confirm: {token.get('required_confirmation_text')}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    return "\n".join(lines)


def check_memory_evidence_repair_recovery_human_approval_token_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_recovery_human_approval_token",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_RECOVERY_HUMAN_APPROVAL_TOKEN_SCHEMA,
    handler=memory_evidence_repair_recovery_human_approval_token_tool,
    check_fn=check_memory_evidence_repair_recovery_human_approval_token_requirements,
    emoji="🧠",
)
