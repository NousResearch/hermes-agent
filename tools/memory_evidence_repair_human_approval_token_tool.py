"""Read-only Memory Evidence Repair Human Approval Token tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_human_approval_token import (
    build_evidence_repair_human_approval_token,
)
from tools.memory_evidence_repair_manual_commit_dry_run_tool import (
    memory_evidence_repair_manual_commit_dry_run_tool,
)
from tools.registry import registry, tool_error


MEMORY_EVIDENCE_REPAIR_HUMAN_APPROVAL_TOKEN_SCHEMA = {
    "name": "memory_evidence_repair_human_approval_token",
    "description": (
        "Read-only human approval token draft for a ready memory evidence "
        "repair manual commit dry-run. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "object",
                "description": "Optional manual commit dry-run payload.",
                "additionalProperties": True,
            },
            "approver": {
                "type": "string",
                "description": "Human approver label for the token draft.",
            },
            "approval_reason": {
                "type": "string",
                "description": "Reason for drafting the approval token.",
            },
            "expires_in_minutes": {
                "type": "integer",
                "description": "Draft token expiry window in minutes. Defaults to 30.",
            },
            "snapshot_plan": {
                "type": "object",
                "description": "Optional snapshot planner payload.",
                "additionalProperties": True,
            },
            "requests": {
                "type": "array",
                "description": "Optional snapshot requests.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "rollback_plan": {
                "type": "object",
                "description": "Optional rollback plan payload.",
                "additionalProperties": True,
            },
            "steps": {
                "type": "array",
                "description": "Optional rollback steps.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "existing_snapshots": {
                "type": "object",
                "description": "Optional existing pre-commit snapshots.",
                "additionalProperties": True,
            },
            "pre_commit_snapshots": {
                "type": "object",
                "description": "Alias for existing_snapshots.",
                "additionalProperties": True,
            },
            "ledger": {
                "type": "object",
                "description": "Optional commit ledger draft payload.",
                "additionalProperties": True,
            },
            "entries": {
                "type": "array",
                "description": "Optional commit ledger entries.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "commit_gate": {
                "type": "object",
                "description": "Optional commit gate report payload.",
                "additionalProperties": True,
            },
            "decisions": {
                "type": "array",
                "description": "Optional commit gate decisions.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "ledger_actor": {
                "type": "string",
                "description": "Optional actor label for generated ledger entries.",
            },
            "ledger_reason": {
                "type": "string",
                "description": "Optional reason for generated ledger entries.",
            },
            "preview": {
                "type": "object",
                "description": "Optional apply preview report payload.",
                "additionalProperties": True,
            },
            "previews": {
                "type": "array",
                "description": "Optional apply previews.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "confirmed_preview_ids": {
                "type": "array",
                "description": "Optional preview ids explicitly confirmed by the user.",
                "items": {"type": "string"},
            },
            "user_confirmed": {
                "type": "boolean",
                "description": "Whether the user explicitly confirmed these previews.",
            },
            "approval": {
                "type": "object",
                "description": "Optional approval candidate report payload.",
                "additionalProperties": True,
            },
            "candidates": {
                "type": "array",
                "description": "Optional approval candidates.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "approved_candidate_ids": {
                "type": "array",
                "description": "Optional candidate ids confirmed for apply preview generation.",
                "items": {"type": "string"},
            },
            "plan": {
                "type": "object",
                "description": "Optional evidence repair plan payload.",
                "additionalProperties": True,
            },
            "repairs": {
                "type": "array",
                "description": "Optional evidence repair items.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "gate": {
                "type": "object",
                "description": "Optional memory policy gate payload.",
                "additionalProperties": True,
            },
            "audit": {
                "type": "object",
                "description": "Optional memory recall audit payload.",
                "additionalProperties": True,
            },
            "diagnostics": {
                "type": "object",
                "description": "Optional memory recall diagnostics payload.",
                "additionalProperties": True,
            },
            "policy": {
                "type": "object",
                "description": "Optional memory auto-policy payload.",
                "additionalProperties": True,
            },
            "proposed_evidence": {
                "type": "object",
                "description": "Optional proposed evidence values keyed globally or by candidate id.",
                "additionalProperties": True,
            },
            "format": {
                "type": "string",
                "enum": ["json", "markdown"],
                "description": "Optional response format helper. JSON is always returned; markdown adds a readable summary field.",
            },
        },
    },
}


def memory_evidence_repair_human_approval_token_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    dry_run = _resolve_dry_run(args, kwargs.get("memory_manager"))
    if isinstance(dry_run, str):
        return dry_run

    payload = build_evidence_repair_human_approval_token(
        dry_run=dry_run,
        approver=str(args.get("approver") or "human"),
        approval_reason=str(args.get("approval_reason") or ""),
        expires_in_minutes=args.get("expires_in_minutes") or 30,
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
    for field_name in array_fields:
        value = args.get(field_name)
        if value is not None and not isinstance(value, list):
            return tool_error(f"{field_name} must be an array when provided.", success=False)
    return None


def _resolve_dry_run(args: dict[str, Any], memory_manager: Any) -> dict[str, Any] | str:
    explicit = args.get("dry_run")
    if isinstance(explicit, dict):
        return explicit

    dry_run_args = dict(args)
    dry_run_args.pop("dry_run", None)
    dry_run_args.pop("approver", None)
    dry_run_args.pop("approval_reason", None)
    dry_run_args.pop("expires_in_minutes", None)
    result = memory_evidence_repair_manual_commit_dry_run_tool(
        dry_run_args,
        memory_manager=memory_manager,
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return tool_error("manual commit dry-run returned invalid JSON.", success=False)
    if not isinstance(payload, dict):
        return tool_error("manual commit dry-run returned a non-object payload.", success=False)
    if payload.get("success") is False:
        return json.dumps(payload, ensure_ascii=False)
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Human Approval Token",
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


def check_memory_evidence_repair_human_approval_token_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_human_approval_token",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_HUMAN_APPROVAL_TOKEN_SCHEMA,
    handler=memory_evidence_repair_human_approval_token_tool,
    check_fn=check_memory_evidence_repair_human_approval_token_requirements,
    emoji="🧠",
)
