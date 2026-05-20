"""Read-only Memory Evidence Repair Recovery Approval Token Gate tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_recovery_approval_token_gate import (
    build_evidence_repair_recovery_approval_token_gate,
)
from tools.memory_evidence_repair_recovery_human_approval_token_tool import (
    MEMORY_EVIDENCE_REPAIR_RECOVERY_HUMAN_APPROVAL_TOKEN_SCHEMA,
    memory_evidence_repair_recovery_human_approval_token_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_RECOVERY_HUMAN_APPROVAL_TOKEN_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_RECOVERY_APPROVAL_TOKEN_GATE_SCHEMA = {
    "name": "memory_evidence_repair_recovery_approval_token_gate",
    "description": (
        "Read-only gate that verifies a recovery human approval token before "
        "any later manual memory evidence repair recovery. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recovery_approval_token": {
                "type": "object",
                "description": "Optional recovery approval token report payload.",
                "additionalProperties": True,
            },
            "recovery_human_approval_token": {
                "type": "object",
                "description": "Alias for recovery_approval_token.",
                "additionalProperties": True,
            },
            "recovery_token": {
                "type": "object",
                "description": "Optional single recovery approval token payload.",
                "additionalProperties": True,
            },
            "token": {
                "type": "object",
                "description": "Optional single recovery approval token payload.",
                "additionalProperties": True,
            },
            "tokens": {
                "type": "array",
                "description": "Optional recovery approval token list.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "confirmation_text": {
                "type": "string",
                "description": "Exact human confirmation text required by the recovery token.",
            },
            "used_token_ids": {
                "type": "array",
                "description": "Read-only ids already marked as used by an external ledger.",
                "items": {"type": "string"},
            },
            "token_generated_at": {
                "type": "string",
                "description": "Issue time for bare token payloads.",
            },
            "current_time": {
                "type": "string",
                "description": "Optional current time override for deterministic verification.",
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


def memory_evidence_repair_recovery_approval_token_gate_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    recovery_approval_token = _resolve_recovery_approval_token(
        args,
        kwargs.get("memory_manager"),
    )
    if isinstance(recovery_approval_token, str):
        return recovery_approval_token

    recovery_execution = _resolve_recovery_execution(args, recovery_approval_token)
    payload = build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=recovery_approval_token,
        recovery_execution=recovery_execution,
        confirmation_text=str(args.get("confirmation_text") or ""),
        used_token_ids=(
            args.get("used_token_ids")
            if isinstance(args.get("used_token_ids"), list)
            else None
        ),
        current_time=args.get("current_time"),
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
        "token_gate",
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
        "tokens",
        "used_token_ids",
        "released_lock_ids",
        "active_locks",
        "locks",
        "already_used_token_ids",
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


def _resolve_recovery_approval_token(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    for field_name in ("recovery_approval_token", "recovery_human_approval_token"):
        explicit_report = args.get(field_name)
        if isinstance(explicit_report, dict):
            if "generated_at" not in explicit_report and args.get("token_generated_at"):
                explicit_report = dict(explicit_report)
                explicit_report["generated_at"] = str(args.get("token_generated_at") or "")
            return explicit_report

    for field_name in ("recovery_token", "token"):
        explicit_token = args.get(field_name)
        if isinstance(explicit_token, dict):
            return {
                "generated_at": str(args.get("token_generated_at") or ""),
                "tokens": [explicit_token],
                "read_only": True,
                "read_only_memory": True,
                "would_mutate_memory": False,
            }

    explicit_tokens = args.get("tokens")
    if isinstance(explicit_tokens, list):
        return {
            "generated_at": str(args.get("token_generated_at") or ""),
            "tokens": explicit_tokens,
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }

    token_args = dict(args)
    for field_name in (
        "recovery_approval_token",
        "recovery_human_approval_token",
        "recovery_token",
        "token",
        "tokens",
        "confirmation_text",
        "used_token_ids",
        "token_generated_at",
        "current_time",
        "format",
    ):
        token_args.pop(field_name, None)
    result = memory_evidence_repair_recovery_human_approval_token_tool(
        token_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error(
            "recovery human approval token tool returned invalid JSON.",
            success=False,
        )
    if not isinstance(payload, dict):
        return tool_error(
            "recovery human approval token tool returned a non-object payload.",
            success=False,
        )
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _resolve_recovery_execution(
    args: dict[str, Any],
    recovery_approval_token: dict[str, Any],
) -> dict[str, Any]:
    for field_name in (
        "recovery_execution",
        "recovery_execution_preview",
        "recovery_execution_report",
    ):
        explicit = args.get(field_name)
        if isinstance(explicit, dict):
            return explicit
    token = _first_token(recovery_approval_token)
    source = token.get("source_execution_preview") if isinstance(token, dict) else {}
    return dict(source) if isinstance(source, dict) else {}


def _first_token(recovery_approval_token: dict[str, Any]) -> dict[str, Any]:
    tokens = recovery_approval_token.get("tokens")
    if isinstance(tokens, list):
        for token in tokens:
            if isinstance(token, dict):
                return token
    if isinstance(recovery_approval_token, dict) and recovery_approval_token.get("id"):
        return recovery_approval_token
    return {}


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Recovery Approval Token Gate",
        f"- Status: {payload.get('status')}",
        f"- Token: {payload.get('token_id') or 'none'}",
        f"- Checks: {summary.get('check_count', 0)}",
        f"- Passed: {summary.get('pass_count', 0)}",
        f"- Failed: {summary.get('fail_count', 0)}",
        f"- Manual recovery verified: {summary.get('manual_recovery_verified', False)}",
    ]
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    for action in payload.get("required_actions", []):
        lines.append(f"- Required: {action}")
    return "\n".join(lines)


def check_memory_evidence_repair_recovery_approval_token_gate_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_recovery_approval_token_gate",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_RECOVERY_APPROVAL_TOKEN_GATE_SCHEMA,
    handler=memory_evidence_repair_recovery_approval_token_gate_tool,
    check_fn=check_memory_evidence_repair_recovery_approval_token_gate_requirements,
    emoji="🧠",
)
