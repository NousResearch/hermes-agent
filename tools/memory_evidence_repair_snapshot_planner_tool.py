"""Read-only Memory Evidence Repair Snapshot Planner tool."""

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
    build_evidence_repair_commit_gate,
)
from agent.memory_evidence_repair_commit_ledger import (
    build_evidence_repair_commit_ledger,
)
from agent.memory_evidence_repair_planner import build_evidence_repair_plan
from agent.memory_evidence_repair_rollback_plan import (
    build_evidence_repair_rollback_plan,
)
from agent.memory_evidence_repair_snapshot_planner import (
    MemoryEvidenceRepairSnapshotPlan,
    build_evidence_repair_snapshot_plan,
)
from tools.registry import registry, tool_error


MEMORY_EVIDENCE_REPAIR_SNAPSHOT_PLANNER_SCHEMA = {
    "name": "memory_evidence_repair_snapshot_planner",
    "description": (
        "Read-only pre-commit snapshot planner for memory evidence repair. "
        "It identifies evidence fields to capture before a future manual write."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "rollback_plan": {
                "type": "object",
                "description": "Optional rollback plan payload.",
                "additionalProperties": True,
            },
            "steps": {
                "type": "array",
                "description": "Optional rollback steps to turn into snapshot requests.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "existing_snapshots": {
                "type": "object",
                "description": "Optional existing pre-commit snapshots keyed by ledger, candidate, preview, decision, or patch digest id.",
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
                "description": "Optional commit ledger entries used to build rollback steps.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "commit_gate": {
                "type": "object",
                "description": "Optional commit gate report payload.",
                "additionalProperties": True,
            },
            "decisions": {
                "type": "array",
                "description": "Optional commit gate decisions used to build a ledger.",
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
                "description": "Optional apply previews used to build a commit gate.",
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
                "description": "Maximum snapshot requests to include. Defaults to all requests.",
            },
        },
    },
}


def memory_evidence_repair_snapshot_planner_tool(
    args: dict[str, Any],
    **kwargs,
) -> str:
    if not isinstance(args, dict):
        return tool_error("args must be an object.", success=False)

    validated = _validate_args(args)
    if validated is not None:
        return validated

    existing_snapshots = _existing_snapshots(args)
    rollback_plan = _resolve_rollback_plan(args, kwargs.get("memory_manager"))
    if isinstance(rollback_plan, str):
        return rollback_plan

    report = build_evidence_repair_snapshot_plan(
        rollback_plan=rollback_plan,
        existing_snapshots=existing_snapshots,
    )

    limit = _positive_int(args.get("limit"))
    if limit is not None:
        report = MemoryEvidenceRepairSnapshotPlan(
            generated_at=report.generated_at,
            requests=report.requests[:limit],
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


def _existing_snapshots(args: dict[str, Any]) -> dict[str, Any]:
    if isinstance(args.get("existing_snapshots"), dict):
        return args["existing_snapshots"]
    if isinstance(args.get("pre_commit_snapshots"), dict):
        return args["pre_commit_snapshots"]
    return {}


def _resolve_rollback_plan(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    explicit_plan = args.get("rollback_plan")
    explicit_steps = args.get("steps")
    if isinstance(explicit_plan, dict):
        if "steps" in explicit_plan:
            return explicit_plan
        return {
            "summary": {"rollback_step_count": 1},
            "steps": [explicit_plan],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }
    if isinstance(explicit_steps, list):
        return {
            "summary": {"rollback_step_count": len(explicit_steps)},
            "steps": explicit_steps,
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }

    ledger = _resolve_ledger(args, memory_manager)
    if isinstance(ledger, str):
        return ledger
    return build_evidence_repair_rollback_plan(
        ledger=ledger,
        pre_commit_snapshots=_existing_snapshots(args),
    ).to_dict()


def _resolve_ledger(args: dict[str, Any], memory_manager: Any) -> dict[str, Any] | str:
    explicit_ledger = args.get("ledger")
    explicit_entries = args.get("entries")
    if isinstance(explicit_ledger, dict):
        if "entries" in explicit_ledger:
            return explicit_ledger
        return {
            "summary": {"entry_count": 1},
            "entries": [explicit_ledger],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }
    if isinstance(explicit_entries, list):
        return {
            "summary": {"entry_count": len(explicit_entries)},
            "entries": explicit_entries,
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }

    commit_gate = _resolve_commit_gate(args, memory_manager)
    if isinstance(commit_gate, str):
        return commit_gate
    return build_evidence_repair_commit_ledger(
        commit_gate=commit_gate,
        ledger_actor=str(args.get("ledger_actor") or "hermes"),
        ledger_reason=str(args.get("ledger_reason") or ""),
    ).to_dict()


def _resolve_commit_gate(
    args: dict[str, Any],
    memory_manager: Any,
) -> dict[str, Any] | str:
    explicit_gate = args.get("commit_gate")
    explicit_decisions = args.get("decisions")
    if isinstance(explicit_gate, dict):
        if "decisions" in explicit_gate:
            return explicit_gate
        return {
            "summary": {"decision_count": 1},
            "decisions": [explicit_gate],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }
    if isinstance(explicit_decisions, list):
        return {
            "summary": {"decision_count": len(explicit_decisions)},
            "decisions": explicit_decisions,
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }

    preview = _resolve_preview(args, memory_manager)
    if isinstance(preview, str):
        return preview
    return build_evidence_repair_commit_gate(
        preview=preview,
        confirmed_preview_ids=(
            args.get("confirmed_preview_ids")
            if isinstance(args.get("confirmed_preview_ids"), list)
            else None
        ),
        user_confirmed=bool(args.get("user_confirmed")),
    ).to_dict()


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
        "### Memory Evidence Repair Snapshot Planner",
        f"- Requests: {summary.get('snapshot_request_count', 0)}",
        f"- Capture required: {summary.get('capture_required_count', 0)}",
        f"- Already available: {summary.get('already_available_count', 0)}",
        f"- Blocking: {summary.get('blocking_count', 0)}",
    ]
    for index, request in enumerate(payload.get("requests", []), start=1):
        provider = request.get("provider") or "memory"
        lines.append(
            f"{index}. [{request.get('status')}] "
            f"{request.get('snapshot_action')} provider={provider}"
        )
        fields = request.get("evidence_fields") or []
        if fields:
            lines.append(f"   Capture: {', '.join(str(item) for item in fields)}")
        verify = request.get("verification_steps") or []
        if verify:
            lines.append(f"   Verify: {', '.join(str(item) for item in verify)}")
    return "\n".join(lines)


def check_memory_evidence_repair_snapshot_planner_requirements() -> bool:
    return True


registry.register(
    name="memory_evidence_repair_snapshot_planner",
    toolset="memory",
    schema=MEMORY_EVIDENCE_REPAIR_SNAPSHOT_PLANNER_SCHEMA,
    handler=memory_evidence_repair_snapshot_planner_tool,
    check_fn=check_memory_evidence_repair_snapshot_planner_requirements,
    emoji="🧠",
)
