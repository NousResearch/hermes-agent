"""Read-only Memory Evidence Repair Planner tool."""

from __future__ import annotations

import json
from typing import Any

from agent.memory_evidence_repair_planner import (
    build_evidence_repair_plan,
    empty_evidence_repair_plan,
)
from tools.registry import registry, tool_error


MEMORY_EVIDENCE_REPAIR_PLANNER_SCHEMA = {
    "name": "memory_evidence_repair_planner",
    "description": (
        "Read-only planner for memory evidence repair. It turns policy gate, "
        "recall audit, diagnostics, and auto-policy signals into provenance "
        "repair candidates. The tool never mutates durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
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
            "format": {
                "type": "string",
                "enum": ["json", "markdown"],
                "description": "Optional response format helper. JSON is always returned; markdown adds a readable summary field.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum repair candidates to include. Defaults to all repairs.",
            },
        },
    },
}


def memory_evidence_repair_planner_tool(args: dict[str, Any], **kwargs) -> str:
    resolved = _resolve_inputs(args, kwargs.get("memory_manager"))
    if isinstance(resolved, str):
        return resolved
    gate, audit, diagnostics, policy = resolved
    payload = build_evidence_repair_plan(
        gate=gate,
        audit=audit,
        diagnostics=diagnostics,
        policy=policy,
    ).to_dict()

    limit = _positive_int(args.get("limit"))
    if limit is not None:
        payload["repairs"] = list(payload.get("repairs", []))[:limit]
        payload["limited_to"] = limit

    payload["success"] = True
    payload["read_only"] = True
    payload["read_only_memory"] = True
    if str(args.get("format", "")).lower() == "markdown":
        payload["markdown"] = _render_markdown(payload)
    return json.dumps(payload, ensure_ascii=False)


def _resolve_inputs(
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
        "### Memory Evidence Repair Planner",
        f"- Repairs: {summary.get('repair_count', 0)}",
        f"- Blocked by gate: {summary.get('blocked_count', 0)}",
        f"- By priority: {summary.get('by_priority', {})}",
    ]
    for index, repair in enumerate(payload.get("repairs", []), start=1):
        provider = repair.get("provider") or "memory"
        lines.append(
            f"{index}. [{repair.get('priority')}] {repair.get('action')} "
            f"provider={provider}"
        )
        reason = repair.get("reason")
        if reason:
            lines.append(f"   Reason: {reason}")
        required = repair.get("required_evidence") or []
        if required:
            lines.append(f"   Required: {', '.join(str(item) for item in required)}")
    return "\n".join(lines)


def check_memory_evidence_repair_planner_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_planner",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_PLANNER_SCHEMA,
    handler=memory_evidence_repair_planner_tool,
    check_fn=check_memory_evidence_repair_planner_requirements,
    emoji="🧠",
)
