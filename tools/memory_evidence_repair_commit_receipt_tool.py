"""Read-only Memory Evidence Repair Manual Commit Receipt tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_commit_receipt import (
    build_evidence_repair_commit_receipt,
)
from tools.memory_evidence_repair_approval_token_gate_tool import (
    MEMORY_EVIDENCE_REPAIR_APPROVAL_TOKEN_GATE_SCHEMA,
    memory_evidence_repair_approval_token_gate_tool,
)
from tools.registry import registry, tool_error


_UPSTREAM_PROPERTIES = dict(
    MEMORY_EVIDENCE_REPAIR_APPROVAL_TOKEN_GATE_SCHEMA["parameters"]["properties"]
)
_UPSTREAM_PROPERTIES.pop("format", None)

MEMORY_EVIDENCE_REPAIR_COMMIT_RECEIPT_SCHEMA = {
    "name": "memory_evidence_repair_commit_receipt",
    "description": (
        "Read-only receipt draft and used-token ledger view for a verified "
        "memory evidence repair manual commit. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "token_gate": {
                "type": "object",
                "description": "Optional approval token gate payload.",
                "additionalProperties": True,
            },
            "approval_token_gate": {
                "type": "object",
                "description": "Alias for token_gate.",
                "additionalProperties": True,
            },
            "existing_receipts": {
                "description": "Optional existing receipt report or receipt list.",
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
                "description": "Optional existing receipt list.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "actor": {
                "type": "string",
                "description": "Human actor label for the future receipt.",
            },
            "commit_reason": {
                "type": "string",
                "description": "Reason for the future manual memory evidence repair commit.",
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


def memory_evidence_repair_commit_receipt_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    token_gate = _resolve_token_gate(args, kwargs.get("memory_manager"))
    if isinstance(token_gate, str):
        return token_gate

    payload = build_evidence_repair_commit_receipt(
        token_gate=token_gate,
        existing_receipts=_existing_receipts(args),
        used_token_ids=(
            args.get("used_token_ids")
            if isinstance(args.get("used_token_ids"), list)
            else None
        ),
        actor=str(args.get("actor") or "human"),
        commit_reason=str(args.get("commit_reason") or ""),
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


def _resolve_token_gate(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    explicit_gate = args.get("token_gate")
    if isinstance(explicit_gate, dict):
        return explicit_gate
    explicit_gate = args.get("approval_token_gate")
    if isinstance(explicit_gate, dict):
        return explicit_gate

    gate_args = dict(args)
    for field_name in (
        "token_gate",
        "approval_token_gate",
        "existing_receipts",
        "receipts",
        "actor",
        "commit_reason",
        "format",
    ):
        gate_args.pop(field_name, None)
    result = memory_evidence_repair_approval_token_gate_tool(
        gate_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error("approval token gate returned invalid JSON.", success=False)
    if not isinstance(payload, dict):
        return tool_error("approval token gate returned a non-object payload.", success=False)
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
        "### Memory Evidence Repair Commit Receipt",
        f"- Status: {payload.get('status')}",
        f"- Receipts: {summary.get('receipt_count', 0)}",
        f"- Used tokens: {summary.get('used_token_count', 0)}",
        f"- Blocks: {summary.get('blocking_reason_count', 0)}",
        f"- Required actions: {summary.get('required_action_count', 0)}",
    ]
    for receipt in payload.get("receipts", []):
        lines.append(f"- Receipt: {receipt.get('id')}")
        lines.append(f"  Token: {receipt.get('token_id')}")
        lines.append(f"  Operations: {len(receipt.get('operation_ids') or [])}")
    for reason in payload.get("blocking_reasons", []):
        lines.append(f"- Blocked: {reason}")
    return "\n".join(lines)


def check_memory_evidence_repair_commit_receipt_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_commit_receipt",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_COMMIT_RECEIPT_SCHEMA,
    handler=memory_evidence_repair_commit_receipt_tool,
    check_fn=check_memory_evidence_repair_commit_receipt_requirements,
    emoji="🧠",
)
