"""Read-only Skill Governance proposal storage and Curator dry-run import.

This module is the pre-mutation side of the Skill Governance Loop.  It stores
PM-reviewable proposal cards under ``get_hermes_home()`` using one directory per
proposal.  It intentionally does not call ``skill_manage`` or mutate skills.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
DECISION_STATUSES = {
    "pending",
    "approved",
    "rejected",
    "deferred",
    "needs_changes",
    "bad_test_target",
}
DECISION_STATUS_ORDER = (
    "pending",
    "approved",
    "needs_changes",
    "deferred",
    "bad_test_target",
    "rejected",
)
CODEX_REVIEW_STATUSES = {
    "not_requested",
    "pending",
    "approved",
    "changes_requested",
    "blocked",
    "failed",
}
RISK_LEVELS = {"low", "medium", "high", "unknown"}
MAX_ARTIFACT_TEXT_CHARS = 200_000

# No ``applied`` state in the MVP.  The dashboard records decisions only; safe
# apply remains a later CLI/manual wrapper.
_DECISION_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"pending", "approved", "rejected", "deferred", "needs_changes", "bad_test_target"},
    "approved": {"approved", "pending", "rejected", "deferred", "needs_changes", "bad_test_target"},
    "rejected": {"rejected", "pending", "deferred", "needs_changes", "bad_test_target"},
    "deferred": {"deferred", "pending", "approved", "rejected", "needs_changes", "bad_test_target"},
    "needs_changes": {"needs_changes", "pending", "approved", "rejected", "deferred", "bad_test_target"},
    "bad_test_target": {"bad_test_target", "pending", "deferred", "rejected"},
}

PROPOSAL_FIELDS = (
    "schema_version",
    "proposal_id",
    "created_at",
    "updated_at",
    "source",
    "source_run_id",
    "source_paths",
    "action",
    "title",
    "pm_summary",
    "rationale",
    "risk_level",
    "impact_summary",
    "pin_policy_status",
    "target_skills",
    "created_skills",
    "archived_skills",
    "affected_skills",
    "pinned_skills",
    "evidence",
    "artifact_paths",
    "diff_path",
    "backup_path",
    "rollback_path",
    "codex_review_status",
    "codex_review_path",
    "codex_review_summary",
    "codex_review_warnings",
    "codex_review_errors",
    "recommended_decision",
    "decision_status",
    "decision_by",
    "decision_at",
    "decision_note",
    "decision_history",
    "metadata",
)

DASHBOARD_HUDUI_SKILLS = [
    "hermes-dashboard-cron-operations",
    "hermes-dashboard-feature-fixture-layer",
    "hermes-dashboard-gateway-channels",
    "hermes-dashboard-home-overview",
    "hermes-dashboard-knowledge-overview",
    "hermes-dashboard-lan-access",
    "hermes-dashboard-operations",
    "hermes-dashboard-ui-customization",
    "hermes-hudui-frontend-workflow",
    "hermes-hudui-lan-access",
]
LLM_TRAINING_SKILLS = [
    "grpo-rl-training",
    "peft-fine-tuning",
    "pytorch-fsdp",
    "modal-serverless-gpu",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-_")
    return slug or "unknown"


def _default_proposals_root() -> Path:
    return get_hermes_home() / "skill-governance" / "proposals"


def _resolve_root(root: str | os.PathLike[str] | None = None) -> Path:
    return Path(root) if root is not None else _default_proposals_root()


def _proposal_dir(proposal_id: str, *, proposals_root: str | os.PathLike[str] | None = None) -> Path:
    return _resolve_root(proposals_root) / _safe_component(proposal_id)


def _proposal_json_path(proposal_id: str, *, proposals_root: str | os.PathLike[str] | None = None) -> Path:
    return _proposal_dir(proposal_id, proposals_root=proposals_root) / "proposal.json"


def _read_proposal_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _read_safe_detail_text(path_value: Any, *, root: Path) -> str | None:
    """Read dashboard detail text only when it stays under the proposal root.

    Proposal JSON is mutable local state, so do not trust stored artifact or diff
    paths blindly. A tampered path must not expose arbitrary local files through
    the dashboard.
    """

    if not path_value:
        return None
    raw = Path(str(path_value)).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    if not _is_relative_to(candidate, root):
        return None
    try:
        text = candidate.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None
    if len(text) > MAX_ARTIFACT_TEXT_CHARS:
        return text[:MAX_ARTIFACT_TEXT_CHARS] + "\n\n[detail text truncated for dashboard governance view]\n"
    return text


def _is_canonical_proposal_id(proposal_id: str) -> bool:
    return bool(proposal_id) and proposal_id == _safe_component(proposal_id)


def _proposal_view_from_storage(proposal: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a dashboard-safe proposal view from stored JSON, or skip stale/tampered data."""

    if not proposal:
        return None
    proposal_id = str(proposal.get("proposal_id") or "")
    if not _is_canonical_proposal_id(proposal_id):
        return None
    try:
        decision_status = _validate_decision_status(str(proposal.get("decision_status") or "pending"))
        _validate_codex_review_status(str(proposal.get("codex_review_status") or "not_requested"))
    except ValueError:
        return None
    view = {field: proposal.get(field) for field in PROPOSAL_FIELDS}
    view["allowed_decision_statuses"] = _allowed_decision_statuses(decision_status)
    return view


def _allowed_decision_statuses(current_status: str) -> list[str]:
    allowed = _DECISION_TRANSITIONS.get(current_status, set())
    return [status for status in DECISION_STATUS_ORDER if status in allowed]


def _proposal_detail_from_storage(
    proposal: Mapping[str, Any] | None,
    *,
    proposals_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    view = _proposal_view_from_storage(proposal)
    if view is None:
        return None

    proposal_id = str(view.get("proposal_id") or "")
    proposal_dir = _proposal_dir(proposal_id, proposals_root=proposals_root)
    artifacts_dir = proposal_dir / "artifacts"

    artifact_texts: dict[str, str] = {}
    for raw_name, path_value in (view.get("artifact_paths") or {}).items():
        artifact_name = _safe_artifact_name(str(raw_name))
        text = _read_safe_detail_text(path_value, root=artifacts_dir)
        if text is not None:
            artifact_texts[artifact_name] = text

    view["artifact_texts"] = artifact_texts
    view["diff_text"] = _read_safe_detail_text(view.get("diff_path"), root=proposal_dir)
    return view


def _write_proposal_file(path: Path, proposal: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(proposal), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _as_string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(val) for key, val in value.items() if val is not None}


def _unique_ordered(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _validate_decision_status(status: str) -> str:
    if status not in DECISION_STATUSES:
        raise ValueError(
            f"Invalid Skill Governance decision status {status!r}; expected one of {sorted(DECISION_STATUSES)}"
        )
    return status


def _validate_decision_transition(current: str, target: str) -> None:
    allowed = _DECISION_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ValueError(f"Invalid Skill Governance transition {current!r} -> {target!r}")


def _validate_codex_review_status(status: str) -> str:
    if status not in CODEX_REVIEW_STATUSES:
        raise ValueError(
            f"Invalid Codex review status {status!r}; expected one of {sorted(CODEX_REVIEW_STATUSES)}"
        )
    return status


def _validate_risk_level(risk_level: str) -> str:
    if risk_level not in RISK_LEVELS:
        return "unknown"
    return risk_level


def _safe_artifact_name(name: str) -> str:
    return _safe_component(Path(name).name)


def _write_artifacts(
    proposal_id: str,
    artifact_texts: Mapping[str, str] | None,
    *,
    proposals_root: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    if not artifact_texts:
        return {}
    artifacts_dir = _proposal_dir(proposal_id, proposals_root=proposals_root) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for raw_name, raw_text in artifact_texts.items():
        artifact_name = _safe_artifact_name(str(raw_name))
        text = str(raw_text)
        if len(text) > MAX_ARTIFACT_TEXT_CHARS:
            text = text[:MAX_ARTIFACT_TEXT_CHARS] + "\n\n[artifact truncated for dashboard governance storage]\n"
        path = artifacts_dir / artifact_name
        path.write_text(text, encoding="utf-8")
        written[artifact_name] = str(path)
    return written


def _normalize_proposal(
    data: Mapping[str, Any],
    *,
    existing: Mapping[str, Any] | None = None,
    artifact_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    now = _now_iso()
    raw_proposal_id = str(data.get("proposal_id") or "").strip()
    if raw_proposal_id:
        proposal_id = _safe_component(raw_proposal_id)
    else:
        basis = f"{data.get('source', 'manual')}-{data.get('source_run_id', '')}-{data.get('title', now)}"
        proposal_id = _safe_component(basis)

    target_skills = _as_string_list(data.get("target_skills"))
    created_skills = _as_string_list(data.get("created_skills"))
    archived_skills = _as_string_list(data.get("archived_skills"))
    affected_skills = _unique_ordered(
        _as_string_list(data.get("affected_skills")) + target_skills + created_skills + archived_skills
    )

    existing_decision_status = str(existing.get("decision_status", "pending")) if existing else "pending"
    incoming_decision_status = data.get("decision_status")
    if existing and existing_decision_status != "pending":
        decision_status = _validate_decision_status(existing_decision_status)
        decision_by = existing.get("decision_by")
        decision_at = existing.get("decision_at")
        decision_note = existing.get("decision_note")
        decision_history = list(existing.get("decision_history") or [])
    else:
        decision_status = _validate_decision_status(str(incoming_decision_status or existing_decision_status or "pending"))
        decision_by = data.get("decision_by", existing.get("decision_by") if existing else None)
        decision_at = data.get("decision_at", existing.get("decision_at") if existing else None)
        decision_note = data.get("decision_note", existing.get("decision_note") if existing else None)
        decision_history = list(data.get("decision_history") or (existing.get("decision_history") if existing else []) or [])

    merged_artifact_paths = dict(existing.get("artifact_paths") or {}) if existing else {}
    merged_artifact_paths.update(dict(artifact_paths or {}))

    codex_review_status = str(
        data.get("codex_review_status")
        or (existing.get("codex_review_status") if existing else None)
        or "not_requested"
    )

    proposal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "proposal_id": proposal_id,
        "created_at": data.get("created_at") or (existing.get("created_at") if existing else None) or now,
        "updated_at": now,
        "source": str(data.get("source") or (existing.get("source") if existing else None) or "manual"),
        "source_run_id": data.get("source_run_id") or (existing.get("source_run_id") if existing else None),
        "source_paths": _as_string_map(data.get("source_paths") or (existing.get("source_paths") if existing else None)),
        "action": str(data.get("action") or (existing.get("action") if existing else None) or "review"),
        "title": str(data.get("title") or (existing.get("title") if existing else None) or proposal_id),
        "pm_summary": data.get("pm_summary") or (existing.get("pm_summary") if existing else None) or "",
        "rationale": data.get("rationale") or (existing.get("rationale") if existing else None) or "",
        "risk_level": _validate_risk_level(str(data.get("risk_level") or (existing.get("risk_level") if existing else None) or "unknown")),
        "impact_summary": data.get("impact_summary") or (existing.get("impact_summary") if existing else None) or "",
        "pin_policy_status": data.get("pin_policy_status") or (existing.get("pin_policy_status") if existing else None) or "not_checked",
        "target_skills": target_skills,
        "created_skills": created_skills,
        "archived_skills": archived_skills,
        "affected_skills": affected_skills,
        "pinned_skills": _as_string_list(data.get("pinned_skills") or (existing.get("pinned_skills") if existing else None)),
        "evidence": _as_string_list(data.get("evidence") or (existing.get("evidence") if existing else None)),
        "artifact_paths": merged_artifact_paths,
        "diff_path": data.get("diff_path") or (existing.get("diff_path") if existing else None),
        "backup_path": data.get("backup_path") or (existing.get("backup_path") if existing else None),
        "rollback_path": data.get("rollback_path") or (existing.get("rollback_path") if existing else None),
        "codex_review_status": _validate_codex_review_status(codex_review_status),
        "codex_review_path": data.get("codex_review_path") or (existing.get("codex_review_path") if existing else None),
        "codex_review_summary": data.get("codex_review_summary") or (existing.get("codex_review_summary") if existing else None) or "",
        "codex_review_warnings": _as_string_list(data.get("codex_review_warnings") or (existing.get("codex_review_warnings") if existing else None)),
        "codex_review_errors": _as_string_list(data.get("codex_review_errors") or (existing.get("codex_review_errors") if existing else None)),
        "recommended_decision": data.get("recommended_decision") or (existing.get("recommended_decision") if existing else None) or "review_first",
        "decision_status": decision_status,
        "decision_by": decision_by,
        "decision_at": decision_at,
        "decision_note": decision_note,
        "decision_history": decision_history,
        "metadata": dict(data.get("metadata") or (existing.get("metadata") if existing else {}) or {}),
    }
    return {field: proposal.get(field) for field in PROPOSAL_FIELDS}


def create_or_update_skill_governance_proposal(
    proposal: Mapping[str, Any],
    *,
    proposals_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Create or idempotently update a read-only governance proposal.

    Existing non-pending human/policy decisions are preserved when the same
    proposal is imported again.  Optional ``artifact_texts`` in ``proposal`` are
    written under the proposal artifact directory but are not stored inline.
    """

    raw_id = str(proposal.get("proposal_id") or "").strip()
    probe_id = raw_id or _safe_component(
        f"{proposal.get('source', 'manual')}-{proposal.get('source_run_id', '')}-{proposal.get('title', '')}"
    )
    path = _proposal_json_path(probe_id, proposals_root=proposals_root)
    existing = _read_proposal_file(path)
    artifact_paths = _write_artifacts(
        probe_id,
        proposal.get("artifact_texts") if isinstance(proposal.get("artifact_texts"), Mapping) else None,
        proposals_root=proposals_root,
    )
    normalized = _normalize_proposal(proposal, existing=existing, artifact_paths=artifact_paths)
    final_path = _proposal_json_path(normalized["proposal_id"], proposals_root=proposals_root)
    _write_proposal_file(final_path, normalized)
    return normalized


def list_skill_governance_proposals(
    decision_status: str | None = None,
    limit: int = 50,
    *,
    proposals_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Return governance proposals, newest first."""

    root = _resolve_root(proposals_root)
    if not root.exists():
        return []
    if decision_status is not None:
        _validate_decision_status(decision_status)

    proposals: list[dict[str, Any]] = []
    for path in root.glob("*/proposal.json"):
        proposal = _proposal_view_from_storage(_read_proposal_file(path))
        if not proposal:
            continue
        if decision_status is not None and proposal.get("decision_status") != decision_status:
            continue
        proposals.append(proposal)

    proposals.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    status_rank = {
        "pending": 0,
        "needs_changes": 1,
        "approved": 2,
        "deferred": 3,
        "bad_test_target": 4,
        "rejected": 5,
    }
    proposals.sort(key=lambda item: status_rank.get(str(item.get("decision_status") or ""), 9))
    return proposals[: max(0, limit)]


def get_skill_governance_proposal(
    proposal_id: str,
    *,
    proposals_root: str | os.PathLike[str] | None = None,
    include_artifacts: bool = False,
) -> dict[str, Any] | None:
    """Return one governance proposal by id."""

    raw_id = str(proposal_id or "").strip()
    if not _is_canonical_proposal_id(raw_id):
        return None
    stored = _read_proposal_file(_proposal_json_path(raw_id, proposals_root=proposals_root))
    if include_artifacts:
        return _proposal_detail_from_storage(stored, proposals_root=proposals_root)
    return _proposal_view_from_storage(stored)


def record_skill_governance_decision(
    proposal_id: str,
    status: str,
    *,
    note: str | None = None,
    decided_by: str = "hermes-user",
    proposals_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Record a PM decision on a proposal without applying it."""

    target_status = _validate_decision_status(status)
    raw_id = str(proposal_id or "").strip()
    if not _is_canonical_proposal_id(raw_id):
        return None
    path = _proposal_json_path(raw_id, proposals_root=proposals_root)
    proposal = _read_proposal_file(path)
    if proposal is None:
        return None

    current_status = _validate_decision_status(str(proposal.get("decision_status") or "pending"))
    _validate_decision_transition(current_status, target_status)

    now = _now_iso()
    history = list(proposal.get("decision_history") or [])
    history.append(
        {
            "previous_status": current_status,
            "status": target_status,
            "note": note,
            "decided_by": decided_by,
            "decided_at": now,
        }
    )
    proposal["updated_at"] = now
    proposal["decision_status"] = target_status
    proposal["decision_by"] = decided_by
    proposal["decision_at"] = now
    proposal["decision_note"] = note
    proposal["decision_history"] = history
    proposal = {field: proposal.get(field) for field in PROPOSAL_FIELDS}
    _write_proposal_file(path, proposal)
    return proposal


def _load_run_json(run_json_path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if run_json_path is None:
        return {}
    try:
        data = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_consolidations(report_text: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    pattern = re.compile(r"-\s+from:\s+([A-Za-z0-9_.-]+)\s+\n\s+into:\s+([A-Za-z0-9_.-]+)", re.MULTILINE)
    for match in pattern.finditer(report_text):
        source, target = match.group(1), match.group(2)
        grouped.setdefault(target, []).append(source)
    return {target: _unique_ordered(sources) for target, sources in grouped.items()}


def _fallback_skills(report_text: str, umbrella: str, defaults: Sequence[str]) -> list[str]:
    lower = report_text.lower()
    found = [skill for skill in defaults if skill.lower() in lower]
    if found:
        return found
    return list(defaults) if umbrella.lower() in lower else []


def _extract_section(report_text: str, heading_fragment: str) -> str:
    lines = report_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if heading_fragment.lower() in line.lower():
            start = idx
            break
    if start is None:
        return report_text
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("### ") or lines[idx].startswith("## "):
            end = idx
            break
    return "\n".join(lines[start:end]).strip() + "\n"


def _run_id_from_paths(report_path: Path, run_json: Mapping[str, Any]) -> str:
    started_at = run_json.get("started_at")
    if isinstance(started_at, str) and started_at:
        return _safe_component(report_path.parent.name or started_at)
    return _safe_component(report_path.parent.name or "curator-run")


def import_curator_dry_run(
    report_path: str | os.PathLike[str],
    *,
    run_json_path: str | os.PathLike[str] | None = None,
    proposals_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Import a Curator dry-run report into reviewable proposals.

    The import is intentionally conservative and idempotent.  It currently
    recognizes the two umbrella proposals surfaced by the 2026-05-06 dry-run:
    dashboard/HUDUI consolidation and LLM training consolidation.  Unknown report
    content yields an empty list rather than a guessed mutation proposal.
    """

    report = Path(report_path)
    report_text = report.read_text(encoding="utf-8")
    run_json = _load_run_json(run_json_path)
    run_id = _run_id_from_paths(report, run_json)
    grouped = _parse_consolidations(report_text)
    source_paths = {"report_path": str(report)}
    if run_json_path is not None:
        source_paths["run_json_path"] = str(Path(run_json_path))

    dashboard_sources = grouped.get("hermes-dashboard-development") or _fallback_skills(
        report_text, "hermes-dashboard-development", DASHBOARD_HUDUI_SKILLS
    )
    training_sources = grouped.get("llm-training-workflows") or _fallback_skills(
        report_text, "llm-training-workflows", LLM_TRAINING_SKILLS
    )

    imported: list[dict[str, Any]] = []
    if dashboard_sources:
        imported.append(
            create_or_update_skill_governance_proposal(
                {
                    "proposal_id": f"curator-{run_id}-hermes-dashboard-development",
                    "source": "curator_dry_run",
                    "source_run_id": run_id,
                    "source_paths": source_paths,
                    "action": "consolidate",
                    "title": "Consolidate dashboard/HUDUI skills into hermes-dashboard-development",
                    "pm_summary": "Curator proposes one dashboard/HUDUI umbrella skill so recurring dashboard recipes become easier to find without losing detailed references.",
                    "rationale": "The dry-run found repeated dashboard, HUDUI, operations, LAN, and UI-customization recipes that fit one class-level workflow skill.",
                    "risk_level": "medium",
                    "impact_summary": "Would create one umbrella and later archive absorbed narrow skills only after backup, diff, review, and explicit approval.",
                    "pin_policy_status": "not_checked",
                    "target_skills": dashboard_sources,
                    "created_skills": ["hermes-dashboard-development"],
                    "archived_skills": dashboard_sources,
                    "evidence": [
                        "Curator dry-run reported no mutating actions.",
                        "Structured summary proposed consolidation into hermes-dashboard-development.",
                        "User selected dashboard/HUDUI as the first meaningful validation target.",
                    ],
                    "recommended_decision": "review_first",
                    "metadata": {"curator_model": run_json.get("model"), "curator_provider": run_json.get("provider")},
                    "artifact_texts": {"curator_report_excerpt.md": _extract_section(report_text, "hermes-dashboard-development")},
                },
                proposals_root=proposals_root,
            )
        )

    if training_sources:
        imported.append(
            create_or_update_skill_governance_proposal(
                {
                    "proposal_id": f"curator-{run_id}-llm-training-workflows",
                    "source": "curator_dry_run",
                    "source_run_id": run_id,
                    "source_paths": source_paths,
                    "action": "consolidate",
                    "title": "Defer llm-training-workflows consolidation as first Curator test",
                    "pm_summary": "Curator proposed an LLM training umbrella, but this is a poor first validation target because the user does not actively use that cluster.",
                    "rationale": "A technically plausible consolidation is not necessarily a good activation test when the maintainer cannot judge source fidelity and retained edges.",
                    "risk_level": "medium",
                    "impact_summary": "No skill mutation should be attempted for this proposal during the first governance-console validation pass.",
                    "pin_policy_status": "not_checked",
                    "target_skills": training_sources,
                    "created_skills": ["llm-training-workflows"],
                    "archived_skills": training_sources,
                    "evidence": [
                        "Curator dry-run proposed llm-training-workflows.",
                        "User decision: do not use this as the first real Curator test.",
                    ],
                    "recommended_decision": "defer",
                    "decision_status": "bad_test_target",
                    "decision_by": "hermes-policy",
                    "decision_at": _now_iso(),
                    "decision_note": "Marked from user decision: llm-training-workflows is a poor first validation target for Curator activation.",
                    "metadata": {"curator_model": run_json.get("model"), "curator_provider": run_json.get("provider")},
                    "artifact_texts": {"curator_report_excerpt.md": _extract_section(report_text, "llm-training-workflows")},
                },
                proposals_root=proposals_root,
            )
        )

    return imported


__all__ = [
    "SCHEMA_VERSION",
    "DECISION_STATUSES",
    "CODEX_REVIEW_STATUSES",
    "create_or_update_skill_governance_proposal",
    "list_skill_governance_proposals",
    "get_skill_governance_proposal",
    "record_skill_governance_decision",
    "import_curator_dry_run",
]
