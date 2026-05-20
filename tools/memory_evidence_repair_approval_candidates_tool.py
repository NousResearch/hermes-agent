"""Read-only Memory Evidence Repair Approval Candidates tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_approval import (
    MemoryEvidenceRepairApprovalReport,
    build_evidence_repair_approval_candidates,
)
from agent.memory_evidence_repair_planner import build_evidence_repair_plan
from tools.registry import registry, tool_error


MEMORY_EVIDENCE_REPAIR_APPROVAL_CANDIDATES_SCHEMA = {
    "name": "memory_evidence_repair_approval_candidates",
    "description": (
        "Read-only planner that turns memory evidence repair plans into "
        "human-confirmable approval candidates. It never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "object",
                "description": "Optional evidence repair plan payload.",
                "additionalProperties": True,
            },
            "repairs": {
                "type": "array",
                "description": "Optional repair items to wrap as a plan.",
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
                "description": "Optional proposed evidence values keyed by required field.",
                "additionalProperties": True,
            },
            "format": {
                "type": "string",
                "enum": ["json", "markdown"],
                "description": "Optional response format helper. JSON is always returned; markdown adds a readable summary field.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum approval candidates to include. Defaults to all candidates.",
            },
        },
    },
}


def memory_evidence_repair_approval_candidates_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    resolved_plan = _resolve_plan(args, kwargs.get("memory_manager"))
    if isinstance(resolved_plan, str):
        return resolved_plan

    proposed_evidence = args.get("proposed_evidence")
    if proposed_evidence is not None and not isinstance(proposed_evidence, dict):
        return tool_error(
            "proposed_evidence must be an object when provided.",
            success=False,
        )

    report = build_evidence_repair_approval_candidates(
        plan=resolved_plan,
        proposed_evidence=proposed_evidence if isinstance(proposed_evidence, dict) else {},
    )
    limit = _positive_int(args.get("limit"))
    if limit is not None:
        report = MemoryEvidenceRepairApprovalReport(
            generated_at=report.generated_at,
            candidates=report.candidates[:limit],
        )

    payload = report.to_dict()
    if limit is not None:
        payload["limited_to"] = limit
    payload["success"] = True
    payload["read_only"] = True
    payload["read_only_memory"] = True
    if str(args.get("format", "")).lower() == "markdown":
        payload["markdown"] = _render_markdown(payload)
    return json.dumps(payload, ensure_ascii=False)


def _resolve_plan(args: dict[str, Any], memory_manager: Any) -> dict[str, Any] | str:
    explicit_plan = args.get("plan")
    explicit_repairs = args.get("repairs")
    if explicit_plan is not None:
        if not isinstance(explicit_plan, dict):
            return tool_error("plan must be an object when provided.", success=False)
        return explicit_plan
    if explicit_repairs is not None:
        if not isinstance(explicit_repairs, list):
            return tool_error("repairs must be an array when provided.", success=False)
        return {
            "summary": {"repair_count": len(explicit_repairs)},
            "repairs": explicit_repairs,
            "read_only": True,
        }

    raw_inputs = _resolve_raw_inputs(args, memory_manager)
    if isinstance(raw_inputs, str):
        return raw_inputs
    gate, audit, diagnostics, policy = raw_inputs
    return build_evidence_repair_plan(
        gate=gate,
        audit=audit,
        diagnostics=diagnostics,
        policy=policy,
    ).to_dict()


def _resolve_raw_inputs(
    args: dict[str, Any],
    memory_manager: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | str:
    explicit_gate = args.get("gate")
    explicit_audit = args.get("audit")
    explicit_diagnostics = args.get("diagnostics")
    explicit_policy = args.get("policy")
    for field_name, value in (
        ("gate", explicit_gate),
        ("audit", explicit_audit),
        ("diagnostics", explicit_diagnostics),
        ("policy", explicit_policy),
    ):
        if value is not None and not isinstance(value, dict):
            return tool_error(f"{field_name} must be an object when provided.", success=False)

    gate = explicit_gate if isinstance(explicit_gate, dict) else _manager_dict(
        memory_manager,
        "last_memory_policy_gate",
    )
    audit = explicit_audit if isinstance(explicit_audit, dict) else _manager_dict(
        memory_manager,
        "last_recall_audit",
    )
    diagnostics = (
        explicit_diagnostics
        if isinstance(explicit_diagnostics, dict)
        else _manager_dict(memory_manager, "last_recall_diagnostics")
    )
    policy = explicit_policy if isinstance(explicit_policy, dict) else _manager_dict(
        memory_manager,
        "last_memory_auto_policy",
    )
    return gate, audit, diagnostics, policy


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
        "### Memory Evidence Repair Approval Candidates",
        f"- Candidates: {summary.get('candidate_count', 0)}",
        f"- Requires confirmation: {summary.get('requires_user_confirmation', False)}",
        f"- By priority: {summary.get('by_priority', {})}",
    ]
    for index, candidate in enumerate(payload.get("candidates", []), start=1):
        provider = candidate.get("provider") or "memory"
        lines.append(
            f"{index}. [{candidate.get('priority')}] "
            f"{candidate.get('repair_action')} provider={provider}"
        )
        question = candidate.get("confirmation_question")
        if question:
            lines.append(f"   Confirm: {question}")
        required = candidate.get("required_evidence") or []
        if required:
            lines.append(f"   Required: {', '.join(str(item) for item in required)}")
    return "\n".join(lines)


def check_memory_evidence_repair_approval_candidates_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_approval_candidates",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_APPROVAL_CANDIDATES_SCHEMA,
    handler=memory_evidence_repair_approval_candidates_tool,
    check_fn=check_memory_evidence_repair_approval_candidates_requirements,
    emoji="🧠",
)
