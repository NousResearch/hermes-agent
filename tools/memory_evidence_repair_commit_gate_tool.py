"""Read-only Memory Evidence Repair Commit Gate tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_apply_preview import (
    build_evidence_repair_apply_preview,
)
from agent.memory_evidence_repair_approval import (
    build_evidence_repair_approval_candidates,
)
from agent.memory_evidence_repair_commit_gate import (
    MemoryEvidenceRepairCommitGateReport,
    build_evidence_repair_commit_gate,
)
from agent.memory_evidence_repair_planner import build_evidence_repair_plan
from tools.registry import registry, tool_error


MEMORY_EVIDENCE_REPAIR_COMMIT_GATE_SCHEMA = {
    "name": "memory_evidence_repair_commit_gate",
    "description": (
        "Read-only gate that decides whether an evidence repair apply preview "
        "is eligible for a later manual memory write. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "preview": {
                "type": "object",
                "description": "Optional apply preview report payload.",
                "additionalProperties": True,
            },
            "previews": {
                "type": "array",
                "description": "Optional apply previews to gate.",
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
                "description": "Optional approval candidates used to build previews.",
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
                "description": "Optional repair items used to build candidates.",
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
            "limit": {
                "type": "integer",
                "description": "Maximum gate decisions to include. Defaults to all decisions.",
            },
        },
    },
}


def memory_evidence_repair_commit_gate_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    preview = _resolve_preview(args, kwargs.get("memory_manager"))
    if isinstance(preview, str):
        return preview

    report = build_evidence_repair_commit_gate(
        preview=preview,
        confirmed_preview_ids=(
            args.get("confirmed_preview_ids")
            if isinstance(args.get("confirmed_preview_ids"), list)
            else None
        ),
        user_confirmed=bool(args.get("user_confirmed")),
    )

    limit = _positive_int(args.get("limit"))
    if limit is not None:
        report = MemoryEvidenceRepairCommitGateReport(
            generated_at=report.generated_at,
            decisions=report.decisions[:limit],
        )

    payload = report.to_dict()
    if limit is not None:
        payload["limited_to"] = limit
    payload["success"] = True
    payload["read_only"] = True
    payload["read_only_memory"] = True
    payload["would_mutate_memory"] = False
    if str(args.get("format", "")).lower() == "markdown":
        payload["markdown"] = _render_markdown(payload)
    return json.dumps(payload, ensure_ascii=False)


def _validate_args(args: dict[str, Any]) -> str | None:
    object_fields = (
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


def _resolve_preview(args: dict[str, Any], memory_manager: Any) -> dict[str, Any] | str:
    explicit_preview = args.get("preview")
    explicit_previews = args.get("previews")
    if isinstance(explicit_preview, dict):
        if "previews" in explicit_preview:
            return explicit_preview
        return {
            "summary": {"preview_count": 1},
            "previews": [explicit_preview],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }
    if isinstance(explicit_previews, list):
        return {
            "summary": {"preview_count": len(explicit_previews)},
            "previews": explicit_previews,
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }

    approval = _resolve_approval(args, memory_manager)
    if isinstance(approval, str):
        return approval
    return build_evidence_repair_apply_preview(
        approval=approval,
        approved_candidate_ids=(
            args.get("approved_candidate_ids")
            if isinstance(args.get("approved_candidate_ids"), list)
            else None
        ),
        proposed_evidence=(
            args.get("proposed_evidence")
            if isinstance(args.get("proposed_evidence"), dict)
            else {}
        ),
    ).to_dict()


def _resolve_approval(args: dict[str, Any], memory_manager: Any) -> dict[str, Any] | str:
    explicit_approval = args.get("approval")
    explicit_candidates = args.get("candidates")
    if isinstance(explicit_approval, dict):
        return explicit_approval
    if isinstance(explicit_candidates, list):
        return {
            "summary": {"candidate_count": len(explicit_candidates)},
            "candidates": explicit_candidates,
            "read_only": True,
            "read_only_memory": True,
        }

    plan = _resolve_plan(args, memory_manager)
    if isinstance(plan, str):
        return plan
    return build_evidence_repair_approval_candidates(
        plan=plan,
        proposed_evidence=(
            args.get("proposed_evidence")
            if isinstance(args.get("proposed_evidence"), dict)
            else {}
        ),
    ).to_dict()


def _resolve_plan(args: dict[str, Any], memory_manager: Any) -> dict[str, Any] | str:
    explicit_plan = args.get("plan")
    explicit_repairs = args.get("repairs")
    if isinstance(explicit_plan, dict):
        return explicit_plan
    if isinstance(explicit_repairs, list):
        return {
            "summary": {"repair_count": len(explicit_repairs)},
            "repairs": explicit_repairs,
            "read_only": True,
        }

    gate = args.get("gate") if isinstance(args.get("gate"), dict) else _manager_dict(
        memory_manager,
        "last_memory_policy_gate",
    )
    audit = args.get("audit") if isinstance(args.get("audit"), dict) else _manager_dict(
        memory_manager,
        "last_recall_audit",
    )
    diagnostics = (
        args.get("diagnostics")
        if isinstance(args.get("diagnostics"), dict)
        else _manager_dict(memory_manager, "last_recall_diagnostics")
    )
    policy = args.get("policy") if isinstance(args.get("policy"), dict) else _manager_dict(
        memory_manager,
        "last_memory_auto_policy",
    )
    return build_evidence_repair_plan(
        gate=gate,
        audit=audit,
        diagnostics=diagnostics,
        policy=policy,
    ).to_dict()


def _manager_dict(memory_manager: Any, method_name: str) -> dict[str, Any]:
    if memory_manager is None or not hasattr(memory_manager, method_name):
        return {}
    try:
        value = getattr(memory_manager, method_name)()
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "### Memory Evidence Repair Commit Gate",
        f"- Decisions: {summary.get('decision_count', 0)}",
        f"- Allow: {summary.get('allow_count', 0)}",
        f"- Blocked: {summary.get('blocked_count', 0)}",
        f"- Needs confirmation: {summary.get('needs_confirmation_count', 0)}",
    ]
    for index, decision in enumerate(payload.get("decisions", []), start=1):
        provider = decision.get("provider") or "memory"
        lines.append(
            f"{index}. [{decision.get('decision')}] "
            f"{decision.get('repair_action')} provider={provider}"
        )
        reasons = decision.get("reasons") or []
        if reasons:
            lines.append(f"   Reasons: {', '.join(str(item) for item in reasons)}")
        required = decision.get("required_actions") or []
        if required:
            lines.append(f"   Required: {', '.join(str(item) for item in required)}")
    return "\n".join(lines)


def check_memory_evidence_repair_commit_gate_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_commit_gate",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_COMMIT_GATE_SCHEMA,
    handler=memory_evidence_repair_commit_gate_tool,
    check_fn=check_memory_evidence_repair_commit_gate_requirements,
    emoji="🧠",
)
