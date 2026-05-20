"""Read-only Memory Evidence Repair Recovery Write Lock Gate tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_recovery_write_lock_gate import (
    build_evidence_repair_recovery_write_lock_gate,
)
from tools.memory_evidence_repair_recovery_approval_token_gate_tool import (
    MEMORY_EVIDENCE_REPAIR_RECOVERY_APPROVAL_TOKEN_GATE_SCHEMA,
    memory_evidence_repair_recovery_approval_token_gate_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_RECOVERY_APPROVAL_TOKEN_GATE_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_RECOVERY_WRITE_LOCK_GATE_SCHEMA = {
    "name": "memory_evidence_repair_recovery_write_lock_gate",
    "description": (
        "Read-only recovery write-lock gate for memory evidence repair recovery. "
        "It drafts a future recovery write lock without mutating durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recovery_token_gate": {
                "type": "object",
                "description": "Optional recovery approval token gate report payload.",
                "additionalProperties": True,
            },
            "recovery_approval_token_gate": {
                "type": "object",
                "description": "Alias for recovery_token_gate.",
                "additionalProperties": True,
            },
            "token_gate": {
                "type": "object",
                "description": "Alias for recovery_token_gate.",
                "additionalProperties": True,
            },
            "active_locks": {
                "type": "array",
                "description": "Optional active recovery write-lock records to check for conflicts.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "locks": {
                "type": "array",
                "description": "Alias for active_locks.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "already_used_token_ids": {
                "type": "array",
                "description": "External used-token ids to check before drafting a recovery write lock.",
                "items": {"type": "string"},
            },
            "lock_owner": {
                "type": "string",
                "description": "Owner label for the future recovery write-lock draft.",
            },
            "lock_ttl_minutes": {
                "type": "integer",
                "description": "Recovery write-lock draft TTL in minutes. Defaults to 10.",
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


def memory_evidence_repair_recovery_write_lock_gate_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    recovery_token_gate = _resolve_recovery_token_gate(
        args,
        kwargs.get("memory_manager"),
    )
    if isinstance(recovery_token_gate, str):
        return recovery_token_gate

    payload = build_evidence_repair_recovery_write_lock_gate(
        recovery_token_gate=recovery_token_gate,
        active_locks=_active_locks(args),
        already_used_token_ids=(
            args.get("already_used_token_ids")
            if isinstance(args.get("already_used_token_ids"), list)
            else None
        ),
        lock_owner=str(args.get("lock_owner") or args.get("actor") or "human"),
        lock_ttl_minutes=args.get("lock_ttl_minutes") or 10,
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


def _resolve_recovery_token_gate(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    for field_name in ("recovery_token_gate", "recovery_approval_token_gate", "token_gate"):
        explicit = args.get(field_name)
        if isinstance(explicit, dict):
            return explicit

    gate_args = dict(args)
    for field_name in (
        "recovery_token_gate",
        "recovery_approval_token_gate",
        "token_gate",
        "active_locks",
        "locks",
        "already_used_token_ids",
        "lock_owner",
        "lock_ttl_minutes",
        "format",
    ):
        gate_args.pop(field_name, None)
    result = memory_evidence_repair_recovery_approval_token_gate_tool(
        gate_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error(
            "recovery approval token gate tool returned invalid JSON.",
            success=False,
        )
    if not isinstance(payload, dict):
        return tool_error(
            "recovery approval token gate tool returned a non-object payload.",
            success=False,
        )
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _active_locks(args: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(args.get("active_locks"), list):
        return args["active_locks"]
    if isinstance(args.get("locks"), list):
        return args["locks"]
    return []


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Recovery Write Lock Gate",
        f"- Status: {payload.get('status')}",
        f"- Checks: {summary.get('check_count', 0)}",
        f"- Locks: {summary.get('lock_count', 0)}",
        f"- Conflicts: {summary.get('active_lock_conflict_count', 0)}",
        f"- Recovery executor preview allowed: {summary.get('recovery_executor_preview_allowed', False)}",
    ]
    for lock in payload.get("locks", []):
        lines.append(f"- Lock: {lock.get('id')}")
        lines.append(f"  Token: {lock.get('token_id')}")
        lines.append(f"  Expires: {lock.get('expires_at')}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    for action in payload.get("required_actions", []):
        lines.append(f"- Required: {action}")
    return "\n".join(lines)


def check_memory_evidence_repair_recovery_write_lock_gate_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_recovery_write_lock_gate",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_RECOVERY_WRITE_LOCK_GATE_SCHEMA,
    handler=memory_evidence_repair_recovery_write_lock_gate_tool,
    check_fn=check_memory_evidence_repair_recovery_write_lock_gate_requirements,
    emoji="🧠",
)
