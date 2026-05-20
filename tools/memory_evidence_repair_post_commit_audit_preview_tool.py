"""Read-only Memory Evidence Repair Post-Commit Audit Preview tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_post_commit_audit_preview import (
    build_evidence_repair_post_commit_audit_preview,
)
from tools.memory_evidence_repair_executor_preview_tool import (
    MEMORY_EVIDENCE_REPAIR_EXECUTOR_PREVIEW_SCHEMA,
    memory_evidence_repair_executor_preview_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_EXECUTOR_PREVIEW_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_POST_COMMIT_AUDIT_PREVIEW_SCHEMA = {
    "name": "memory_evidence_repair_post_commit_audit_preview",
    "description": (
        "Read-only post-commit audit preview for future memory evidence repair "
        "manual commits. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "executor_preview": {
                "type": "object",
                "description": "Optional executor preview report payload.",
                "additionalProperties": True,
            },
            "executor": {
                "type": "object",
                "description": "Alias for executor_preview.",
                "additionalProperties": True,
            },
            "observed_memory_patches": {
                "type": "object",
                "description": "Optional observed post-commit patch signals keyed by operation id.",
                "additionalProperties": True,
            },
            "recorded_receipts": {
                "description": "Optional recorded receipt report or receipt list.",
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
                "description": "Optional write-lock ids observed as released after commit.",
                "items": {"type": "string"},
            },
            "rollback_status": {
                "type": "object",
                "description": "Optional observed rollback or snapshot readiness signal.",
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


def memory_evidence_repair_post_commit_audit_preview_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    executor_preview = _resolve_executor_preview(args, kwargs.get("memory_manager"))
    if isinstance(executor_preview, str):
        return executor_preview

    payload = build_evidence_repair_post_commit_audit_preview(
        executor_preview=executor_preview,
        observed_memory_patches=(
            args.get("observed_memory_patches")
            if isinstance(args.get("observed_memory_patches"), dict)
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
        rollback_status=(
            args.get("rollback_status")
            if isinstance(args.get("rollback_status"), dict)
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


def _resolve_executor_preview(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    explicit_preview = args.get("executor_preview")
    if isinstance(explicit_preview, dict):
        return explicit_preview
    explicit_preview = args.get("executor")
    if isinstance(explicit_preview, dict):
        return explicit_preview

    executor_args = dict(args)
    for field_name in (
        "executor_preview",
        "executor",
        "observed_memory_patches",
        "recorded_receipts",
        "released_lock_ids",
        "rollback_status",
        "format",
    ):
        executor_args.pop(field_name, None)
    result = memory_evidence_repair_executor_preview_tool(
        executor_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error("executor preview tool returned invalid JSON.", success=False)
    if not isinstance(payload, dict):
        return tool_error("executor preview tool returned a non-object payload.", success=False)
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
        "### Memory Evidence Repair Post-Commit Audit Preview",
        f"- Status: {payload.get('status')}",
        f"- Previews: {summary.get('preview_count', 0)}",
        f"- Audit steps: {summary.get('audit_step_count', 0)}",
        f"- Planned: {summary.get('planned_audit_step_count', 0)}",
        f"- Observed pass: {summary.get('observed_pass_step_count', 0)}",
        f"- Observed fail: {summary.get('observed_fail_step_count', 0)}",
    ]
    for preview in payload.get("previews", []):
        lines.append(f"- Audit: {preview.get('id')}")
        lines.append(f"  Executor: {preview.get('executor_preview_id')}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    for action in payload.get("required_actions", []):
        lines.append(f"- Required: {action}")
    return "\n".join(lines)


def check_memory_evidence_repair_post_commit_audit_preview_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_post_commit_audit_preview",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_POST_COMMIT_AUDIT_PREVIEW_SCHEMA,
    handler=memory_evidence_repair_post_commit_audit_preview_tool,
    check_fn=check_memory_evidence_repair_post_commit_audit_preview_requirements,
    emoji="🧠",
)
