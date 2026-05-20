"""Read-only Memory Evidence Repair Manual Commit Executor Preview tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_executor_preview import (
    build_evidence_repair_executor_preview,
)
from tools.memory_evidence_repair_write_lock_gate_tool import (
    MEMORY_EVIDENCE_REPAIR_WRITE_LOCK_GATE_SCHEMA,
    memory_evidence_repair_write_lock_gate_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_WRITE_LOCK_GATE_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_EXECUTOR_PREVIEW_SCHEMA = {
    "name": "memory_evidence_repair_executor_preview",
    "description": (
        "Read-only ordered executor preview for a future memory evidence repair "
        "manual commit. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "write_lock_gate": {
                "type": "object",
                "description": "Optional write-lock gate payload.",
                "additionalProperties": True,
            },
            "executor_lock_gate": {
                "type": "object",
                "description": "Alias for write_lock_gate.",
                "additionalProperties": True,
            },
            "write_lock": {
                "type": "object",
                "description": "Optional single write-lock draft payload.",
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


def memory_evidence_repair_executor_preview_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    write_lock_gate = _resolve_write_lock_gate(args, kwargs.get("memory_manager"))
    if isinstance(write_lock_gate, str):
        return write_lock_gate

    payload = build_evidence_repair_executor_preview(
        write_lock_gate=write_lock_gate,
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
    existing_receipts = args.get("existing_receipts")
    if existing_receipts is not None and not isinstance(existing_receipts, (dict, list)):
        return tool_error(
            "existing_receipts must be an object or array when provided.",
            success=False,
        )
    for field_name in array_fields:
        value = args.get(field_name)
        if value is not None and not isinstance(value, list):
            return tool_error(f"{field_name} must be an array when provided.", success=False)
    return None


def _resolve_write_lock_gate(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    explicit_gate = args.get("write_lock_gate")
    if isinstance(explicit_gate, dict):
        return explicit_gate
    explicit_gate = args.get("executor_lock_gate")
    if isinstance(explicit_gate, dict):
        return explicit_gate

    explicit_lock = args.get("write_lock")
    if isinstance(explicit_lock, dict):
        return {
            "status": "write_lock_ready_for_manual_commit",
            "summary": {"lock_count": 1},
            "locks": [explicit_lock],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }

    gate_args = dict(args)
    for field_name in (
        "write_lock_gate",
        "executor_lock_gate",
        "write_lock",
        "format",
    ):
        gate_args.pop(field_name, None)
    result = memory_evidence_repair_write_lock_gate_tool(
        gate_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error("write-lock gate returned invalid JSON.", success=False)
    if not isinstance(payload, dict):
        return tool_error("write-lock gate returned a non-object payload.", success=False)
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Executor Preview",
        f"- Status: {payload.get('status')}",
        f"- Previews: {summary.get('preview_count', 0)}",
        f"- Steps: {summary.get('step_count', 0)}",
        f"- Future write steps: {summary.get('future_write_step_count', 0)}",
        f"- Ready: {summary.get('manual_executor_preview_ready', False)}",
    ]
    for preview in payload.get("previews", []):
        lines.append(f"- Preview: {preview.get('id')}")
        lines.append(f"  Lock: {preview.get('lock_id')}")
        lines.append(f"  Failure policy: {preview.get('failure_policy')}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    for action in payload.get("required_actions", []):
        lines.append(f"- Required: {action}")
    return "\n".join(lines)


def check_memory_evidence_repair_executor_preview_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_executor_preview",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_EXECUTOR_PREVIEW_SCHEMA,
    handler=memory_evidence_repair_executor_preview_tool,
    check_fn=check_memory_evidence_repair_executor_preview_requirements,
    emoji="🧠",
)
