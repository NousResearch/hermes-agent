"""Canonical contracts for Hermes research/decision task engines.

This module is intentionally deterministic. It owns the stage schema,
role/model bindings, fail-closed validation, and final report rendering for
the three heavy task modes. It does not change the default chat model.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ENGINE_RESEARCH = "RESEARCH"
ENGINE_DECISION = "DECISION"
ENGINE_RESEARCH_DECISION = "RESEARCH_DECISION"
ENGINE_MODES = {ENGINE_RESEARCH, ENGINE_DECISION, ENGINE_RESEARCH_DECISION}

PIPELINE_COMPLETE = "PIPELINE_COMPLETE"
PIPELINE_INCOMPLETE = "PIPELINE_INCOMPLETE"
PIPELINE_BLOCKED = "PIPELINE_BLOCKED"

GEMINI_HIGH = "Gemini 3.5 Flash (High)"
GEMINI_PRO_HIGH = "Gemini 3.1 Pro (High)"
DDGS_MODEL = "DDGS"
QWEN72B = "Qwen72B"
NEMOTRON120B = "Nemotron-120B"
LLAMA70B = "Llama70B"
GEMMA431B = "Gemma-4-31B"
R1_32B = "R1-32B"
GPT_OR_GEMINI_EXTERNAL = "GPT Bridge or Gemini/agy"
CONTROLLER_ACCEPTANCE = "controller_acceptance"
FINAL_CONTROLLER = "final_controller_report"

DIVERGENCE_ROLES = {
    "structure_mapper",
    "evidence_judge",
    "premise_auditor",
    "alternative_generator",
    "insight_harvester",
}

R1_ALLOWED_STAGES = {"L3_r1_synthesis", "convergence_report"}

FORBIDDEN_MARKDOWN_TOKENS = (
    "web_search",
    "api_call",
    "codex_exec",
    "delegate_task",
    "persona:",
    "R1 convergence",
)


@dataclass(frozen=True)
class StageSpec:
    stage_name: str
    owner: str
    model: str
    required_outputs: tuple[str, ...] = ("artifact_path",)


@dataclass
class StageRecord:
    stage_name: str
    owner: str
    model: str
    executor_model: str
    artifact_path: str
    created_in_current_run: bool
    legacy_contaminated: bool
    valid_for_pipeline: bool
    outputs: dict[str, str] = field(default_factory=dict)
    status: str = "planned"


RESEARCH_STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        "L1_gemini_search",
        GEMINI_HIGH,
        GEMINI_HIGH,
        ("source_candidates.json",),
    ),
    StageSpec(
        "L2_ddgs_supplement",
        "DDGS",
        DDGS_MODEL,
        ("ddgs_gap_sources.json",),
    ),
    StageSpec(
        "L2_5_codex_evidence_organizer",
        "Hermes-Codex handoff",
        "Codex",
        (
            "source_candidates.json",
            "ddgs_gap_sources.json",
            "evidence_runner_*.request.md",
            "evidence_runner_*.request.json",
            "sources.csv",
            "evidence.csv",
            "claims.md",
            "gaps.md",
        ),
    ),
    StageSpec("L3_r1_synthesis", R1_32B, R1_32B, ("r1_synthesis.md",)),
    StageSpec("L4_gemini_audit", GEMINI_PRO_HIGH, GEMINI_PRO_HIGH, ("gemini_audit_report.md",)),
    StageSpec(
        "L5_deepseek_acceptance",
        CONTROLLER_ACCEPTANCE,
        CONTROLLER_ACCEPTANCE,
        ("research_evidence_packet.md",),
    ),
)

DECISION_STAGES: tuple[StageSpec, ...] = (
    StageSpec("intelligence_layer", GEMINI_HIGH, GEMINI_HIGH, ("intelligence_layer_report.md",)),
    StageSpec("supplementary_search", "DDGS", DDGS_MODEL, ("parent_training_supplement.md",)),
    StageSpec("structure_mapper", QWEN72B, QWEN72B, ("structure_mapper.md",)),
    StageSpec("evidence_judge", NEMOTRON120B, NEMOTRON120B, ("evidence_judge.md",)),
    StageSpec("premise_auditor", LLAMA70B, LLAMA70B, ("premise_auditor.md",)),
    StageSpec("alternative_generator", GEMMA431B, GEMMA431B, ("alternative_generator.md",)),
    StageSpec("insight_harvester", GEMMA431B, GEMMA431B, ("insight_harvester.md",)),
    StageSpec("convergence_report", R1_32B, R1_32B, ("convergence_report.md",)),
    StageSpec("external_calibration", GPT_OR_GEMINI_EXTERNAL, GPT_OR_GEMINI_EXTERNAL, ("external_calibration.md",)),
    StageSpec("final_controller_report", "Controller", FINAL_CONTROLLER, ("final_decision_report.md",)),
)

CANONICAL_STAGES: dict[str, tuple[StageSpec, ...]] = {
    ENGINE_RESEARCH: RESEARCH_STAGES,
    ENGINE_DECISION: DECISION_STAGES,
    ENGINE_RESEARCH_DECISION: RESEARCH_STAGES + DECISION_STAGES,
}


def canonical_schema(mode: str) -> dict[str, Any]:
    """Return the machine-readable schema for a task engine mode."""
    normalized = normalize_mode(mode)
    return {
        "mode": normalized,
        "controller_model": QWEN72B,
        "ordinary_chat_model_replaced": False,
        "stages": [asdict(stage) for stage in CANONICAL_STAGES[normalized]],
        "output_policy": {
            "body_stage": "final_controller_report"
            if normalized != ENGINE_RESEARCH
            else "L5_deepseek_acceptance",
            "include_compact_pipeline_trace": True,
            "required_machine_markers": (
                "entered_engine_run_pipeline=true",
                f"pipeline_mode={normalized}",
                "pipeline_status=PIPELINE_COMPLETE",
                "pipeline_validation.valid=true",
                "delegation_used=false",
            ),
        },
    }


def build_engine_contract(mode: str, user_query: str) -> dict[str, Any]:
    """Build the clean 72B-first controller contract for one engine run."""
    schema = canonical_schema(mode)
    return {
        "contract_version": "hermes-task-engine/v1",
        "mode": schema["mode"],
        "user_query": user_query,
        "controller": {
            "model": QWEN72B,
            "scope": "task_engine_contract_and_execution_control_only",
            "must_not_replace_ordinary_chat": True,
        },
        "schema": schema,
        "fail_closed": {
            "missing_stage": PIPELINE_INCOMPLETE,
            "wrong_model": PIPELINE_BLOCKED,
            "missing_artifact": PIPELINE_BLOCKED,
            "legacy_contamination": PIPELINE_BLOCKED,
        },
    }


def build_dry_run_plan(mode: str, *, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Create canonical StageRecord plans without invoking any executor."""
    normalized = normalize_mode(mode)
    base = Path(base_dir) if base_dir is not None else Path("<artifact_root>")
    stages = [
        asdict(make_stage_record(spec, base_dir=base, created=False, valid=False, status="planned"))
        for spec in CANONICAL_STAGES[normalized]
    ]
    return {
        "mode": normalized,
        "execution_mode": "dry-run",
        "model_calls_made": False,
        "stages": stages,
        "stage_count": len(stages),
    }


def make_stage_record(
    spec: StageSpec,
    *,
    base_dir: str | Path,
    created: bool,
    valid: bool,
    status: str = "ok",
    artifact_path: str | Path | None = None,
    outputs: dict[str, str] | None = None,
    legacy_contaminated: bool = False,
    executor_model: str | None = None,
) -> StageRecord:
    """Build a StageRecord from the canonical StageSpec binding."""
    base = Path(base_dir)
    resolved_outputs = outputs or planned_outputs(spec, base)
    artifact = artifact_path
    if artifact is None:
        artifact = _planned_artifact_path(spec, base, resolved_outputs)
    return StageRecord(
        stage_name=spec.stage_name,
        owner=spec.owner,
        model=spec.model,
        executor_model=executor_model or spec.model,
        artifact_path=str(artifact),
        created_in_current_run=created,
        legacy_contaminated=legacy_contaminated,
        valid_for_pipeline=valid,
        outputs=resolved_outputs,
        status=status,
    )


def planned_outputs(spec: StageSpec, base_dir: str | Path) -> dict[str, str]:
    """Return deterministic artifact filenames expected for a stage."""
    base = Path(base_dir)
    stage_dir = base / spec.stage_name
    outputs: dict[str, str] = {}
    for required in spec.required_outputs:
        if required == "artifact_path":
            continue
        if "*" in required:
            filename = required.replace("*", "001")
        else:
            filename = required
        outputs[required] = str(stage_dir / filename)
    if not outputs:
        outputs["report.md"] = str(stage_dir / "report.md")
    return outputs


def normalize_mode(mode: str) -> str:
    value = (mode or "").strip().upper().replace("-", "_")
    if value not in ENGINE_MODES:
        raise ValueError(f"Unknown task engine mode: {mode!r}")
    return value


def detect_task_engine_mode(text: str) -> str | None:
    """Classify only the three heavy task modes; ordinary chat returns None."""
    lowered = (text or "").lower()
    has_research = any(
        token in lowered
        for token in ("研究", "research", "最新进展", "latest research", "evidence")
    )
    has_decision = any(
        token in lowered
        for token in ("决策", "是否", "要不要", "到什么程度", "decision", "should i")
    )
    if has_research and has_decision:
        return ENGINE_RESEARCH_DECISION
    if has_research:
        return ENGINE_RESEARCH
    if has_decision:
        return ENGINE_DECISION
    return None


def validate_pipeline(mode: str, run: dict[str, Any], *, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Validate a completed run against the canonical contract.

    Validation is fail-closed: any absent stage, wrong binding, missing artifact,
    legacy contamination marker, or invalid handoff output blocks final reporting.
    """
    normalized = normalize_mode(mode)
    specs = CANONICAL_STAGES[normalized]
    base = Path(base_dir) if base_dir is not None else None
    stage_records = _stage_records(run)
    by_name = {record.get("stage_name"): record for record in stage_records}

    errors: list[str] = []
    warnings: list[str] = []

    if len(by_name) != len(stage_records):
        errors.append("duplicate_stage_name")

    expected_names = [spec.stage_name for spec in specs]
    actual_names = [record.get("stage_name") for record in stage_records]
    if actual_names != expected_names:
        errors.append(
            "stage_order_mismatch:"
            + json.dumps({"expected": expected_names, "actual": actual_names}, ensure_ascii=False)
        )

    for spec in specs:
        record = by_name.get(spec.stage_name)
        if not record:
            errors.append(f"missing_stage:{spec.stage_name}")
            continue

        _require_equal(errors, spec.stage_name, "owner", record.get("owner"), spec.owner)
        _require_equal(errors, spec.stage_name, "model", record.get("model"), spec.model)

        if record.get("created_in_current_run") is not True:
            errors.append(f"{spec.stage_name}:created_in_current_run_not_true")
        if record.get("legacy_contaminated") is not False:
            errors.append(f"{spec.stage_name}:legacy_contaminated_not_false")
        if record.get("valid_for_pipeline") is not True:
            errors.append(f"{spec.stage_name}:valid_for_pipeline_not_true")

        model = str(record.get("model") or "")
        if "R1-32B" in model and spec.stage_name not in R1_ALLOWED_STAGES:
            errors.append(f"{spec.stage_name}:r1_forbidden_here")
        if spec.stage_name == "external_calibration" and (
            "Nemotron" in model or "Controller" in model
        ):
            errors.append("external_calibration:forbidden_model")

        artifact_path = record.get("artifact_path")
        if not artifact_path:
            errors.append(f"{spec.stage_name}:missing_artifact_path")
            continue
        artifact = _resolve_path(artifact_path, base)
        if not artifact.exists():
            errors.append(f"{spec.stage_name}:artifact_path_not_found:{artifact}")
        elif spec.stage_name == "L5_deepseek_acceptance":
            _validate_l5_acceptance_artifact(errors, artifact)

        for required in spec.required_outputs:
            if required == "artifact_path":
                continue
            if not _required_output_exists(record, required, base):
                errors.append(f"{spec.stage_name}:required_output_missing:{required}")

    if normalized in {ENGINE_DECISION, ENGINE_RESEARCH_DECISION}:
        _validate_divergence_models(errors, by_name)

    status = PIPELINE_COMPLETE if not errors else PIPELINE_BLOCKED
    return {
        "valid": not errors,
        "pipeline_status": status,
        "errors": errors,
        "warnings": warnings,
        "stage_count": len(stage_records),
        "expected_stage_count": len(specs),
        "divergence_unique_model_count": _divergence_unique_model_count(by_name),
    }


def render_final_markdown(
    mode: str,
    run: dict[str, Any],
    validation: dict[str, Any] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> str:
    """Render only the accepted final controller report as user-visible body."""
    normalized = normalize_mode(mode)
    validation = validation or validate_pipeline(normalized, run, base_dir=base_dir)
    markers = [
        "entered_engine_run_pipeline=true",
        f"pipeline_mode={normalized}",
        f"pipeline_status={validation['pipeline_status']}",
        f"pipeline_validation.valid={str(bool(validation['valid'])).lower()}",
        "delegation_used=false",
    ]

    if not validation["valid"]:
        return "\n".join(
            markers
            + [
                "",
                validation["pipeline_status"],
                "",
                "Pipeline validation failed closed. No final report was produced.",
                "",
                "Compact Pipeline Trace:",
                _compact_trace(run),
                "",
                "Validation Errors:",
                "\n".join(f"- {error}" for error in validation["errors"]),
            ]
        )

    final_stage = "final_controller_report" if normalized != ENGINE_RESEARCH else "L5_deepseek_acceptance"
    final_text = _read_stage_artifact_text(run, final_stage, base_dir=base_dir)
    if not final_text.strip():
        blocked = dict(validation)
        blocked["valid"] = False
        blocked["pipeline_status"] = PIPELINE_BLOCKED
        blocked["errors"] = [*validation.get("errors", []), f"{final_stage}:empty_final_artifact"]
        return render_final_markdown(normalized, run, blocked, base_dir=base_dir)

    leaked = [token for token in FORBIDDEN_MARKDOWN_TOKENS if token in final_text]
    if leaked:
        blocked = dict(validation)
        blocked["valid"] = False
        blocked["pipeline_status"] = PIPELINE_BLOCKED
        blocked["errors"] = [*validation.get("errors", []), f"final_markdown_forbidden_tokens:{','.join(leaked)}"]
        return render_final_markdown(normalized, run, blocked, base_dir=base_dir)

    return "\n".join(
        markers
        + [
            "",
            final_text.strip(),
            "",
            "Compact Pipeline Trace:",
            _compact_trace(run),
        ]
    )


def _stage_records(run: dict[str, Any]) -> list[dict[str, Any]]:
    stages = run.get("stages")
    if isinstance(stages, list):
        return [stage for stage in stages if isinstance(stage, dict)]
    if isinstance(stages, dict):
        return [stage for stage in stages.values() if isinstance(stage, dict)]
    return []


def _planned_artifact_path(spec: StageSpec, base_dir: Path, outputs: dict[str, str]) -> Path:
    stage_dir = base_dir / spec.stage_name
    if spec.stage_name == "L2_5_codex_evidence_organizer":
        return stage_dir
    if spec.required_outputs == ("artifact_path",):
        return stage_dir / "report.md"
    first_required = spec.required_outputs[0]
    return Path(outputs.get(first_required, stage_dir / first_required))


def _require_equal(errors: list[str], stage_name: str, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{stage_name}:{field}_mismatch:{actual!r}!={expected!r}")


def _resolve_path(value: str | Path, base_dir: Path | None) -> Path:
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _artifact_dir(record: dict[str, Any], base_dir: Path | None) -> Path | None:
    artifact_path = record.get("artifact_path")
    if not artifact_path:
        return None
    artifact = _resolve_path(artifact_path, base_dir)
    return artifact if artifact.is_dir() else artifact.parent


def _required_output_exists(record: dict[str, Any], required: str, base_dir: Path | None) -> bool:
    outputs = record.get("outputs")
    candidates: list[Path] = []
    if isinstance(outputs, dict):
        value = outputs.get(required)
        if isinstance(value, str):
            candidates.append(_resolve_path(value, base_dir))
        for value in outputs.values():
            if isinstance(value, str) and fnmatch.fnmatch(Path(value).name, required):
                candidates.append(_resolve_path(value, base_dir))
    elif isinstance(outputs, list):
        for value in outputs:
            if isinstance(value, str) and fnmatch.fnmatch(Path(value).name, required):
                candidates.append(_resolve_path(value, base_dir))

    stage_dir = _artifact_dir(record, base_dir)
    if stage_dir is not None:
        if any(ch in required for ch in "*?[]"):
            candidates.extend(stage_dir.glob(required))
        else:
            candidates.append(stage_dir / required)

    return any(candidate.exists() for candidate in candidates)


def _validate_divergence_models(errors: list[str], by_name: dict[str, dict[str, Any]]) -> None:
    missing = sorted(role for role in DIVERGENCE_ROLES if role not in by_name)
    if missing:
        errors.append("divergence_roles_missing:" + ",".join(missing))
        return
    count = _divergence_unique_model_count(by_name)
    if count < 4:
        errors.append(f"divergence_unique_models_lt_4:{count}")


def _validate_l5_acceptance_artifact(errors: list[str], artifact: Path) -> None:
    text = artifact.read_text(encoding="utf-8", errors="replace")
    required_stages = (
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
    )
    lowered = text.lower()
    if "verdict: accepted" not in lowered:
        errors.append("L5_deepseek_acceptance:verdict_not_accepted")
    if "accepted: true" not in lowered:
        errors.append("L5_deepseek_acceptance:accepted_not_true")
    ready_value = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("evidence_packet_ready_for_decision:"):
            ready_value = line.split(":", 1)[1].strip().lower()
            break
    ready_true = ready_value == "true"
    ready_conditional = ready_value == "conditional"
    if not (ready_true or ready_conditional):
        errors.append("L5_deepseek_acceptance:evidence_packet_not_ready")
    for stage_name in required_stages:
        if stage_name not in text:
            errors.append(f"L5_deepseek_acceptance:checked_stage_missing:{stage_name}")
    if "final_controller_report" in lowered:
        errors.append("L5_deepseek_acceptance:forbidden_final_controller_report")
    for token in ("artifact_path", "executor_model", "valid_for_pipeline", "stage_name", "owner="):
        if token in lowered:
            errors.append(f"L5_deepseek_acceptance:raw_metadata:{token}")
    required_sections = (
        "evidence_strength",
        "claim_table",
        "controversy",
        "evidence_gap",
        "evidence_supported",
        "reasonable_inference",
        "foresight_hypothesis",
    )
    missing_sections = [section for section in required_sections if f"## {section}" not in lowered]
    if missing_sections:
        errors.append("L5_deepseek_acceptance:missing_evidence_packet_sections:" + ",".join(missing_sections))
    for section in required_sections:
        body = _markdown_section_body(text, section)
        if f"## {section}" in lowered and len(body) < 60:
            errors.append(f"L5_deepseek_acceptance:thin_evidence_packet_section:{section}")
    combined_sections = "\n".join(_markdown_section_body(text, section).lower() for section in required_sections)
    if not combined_sections.strip():
        errors.append("L5_deepseek_acceptance:acceptance_summary_only")
    elif "accepted" in combined_sections and not any(
        term in combined_sections
        for term in ("evidence", "证据", "inference", "推断", "hypothesis", "假设", "gap", "缺口", "controvers", "争议")
    ):
        errors.append("L5_deepseek_acceptance:acceptance_summary_only")
    claim_table_body = _markdown_section_body(text, "claim_table")
    if "claim_id:" not in claim_table_body:
        errors.append("L5_deepseek_acceptance:missing_claim_table")
    if "source_anchors:" not in claim_table_body:
        errors.append("L5_deepseek_acceptance:claim_table_missing_source_anchors")
    if "requires_full_text_verification" in lowered and ready_true:
        errors.append("L5_deepseek_acceptance:unconditional_ready_with_verification_required")
    if ready_conditional and "handoff_caveats:" not in lowered:
        errors.append("L5_deepseek_acceptance:conditional_ready_missing_handoff_caveats")


def _markdown_section_body(text: str, heading: str) -> str:
    marker = f"## {heading}"
    lowered = (text or "").lower()
    start = lowered.find(marker.lower())
    if start < 0:
        return ""
    body_start = start + len(marker)
    next_index = lowered.find("\n## ", body_start)
    if next_index < 0:
        next_index = len(text or "")
    return (text or "")[body_start:next_index].strip()


def _divergence_unique_model_count(by_name: dict[str, dict[str, Any]]) -> int:
    return len(
        {
            str(by_name[role].get("model"))
            for role in DIVERGENCE_ROLES
            if role in by_name and by_name[role].get("model")
        }
    )


def _read_stage_artifact_text(run: dict[str, Any], stage_name: str, *, base_dir: str | Path | None) -> str:
    base = Path(base_dir) if base_dir is not None else None
    for record in _stage_records(run):
        if record.get("stage_name") != stage_name:
            continue
        artifact = _resolve_path(record.get("artifact_path", ""), base)
        if artifact.is_dir():
            artifact = artifact / "report.md"
        try:
            return artifact.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def _compact_trace(run: dict[str, Any]) -> str:
    lines = []
    for record in _stage_records(run):
        stage = record.get("stage_name", "")
        owner = record.get("owner", "")
        executor_model = record.get("executor_model", "")
        artifact = record.get("artifact_path", "")
        valid = record.get("valid_for_pipeline", "")
        lines.append(f"- {stage} | owner={owner} | executor_model={executor_model} | artifact_path={artifact} | valid_for_pipeline={valid}")
    return "\n".join(lines)


__all__ = [
    "CANONICAL_STAGES",
    "DIVERGENCE_ROLES",
    "ENGINE_DECISION",
    "ENGINE_MODES",
    "ENGINE_RESEARCH",
    "ENGINE_RESEARCH_DECISION",
    "GEMINI_HIGH",
    "GEMINI_PRO_HIGH",
    "PIPELINE_BLOCKED",
    "PIPELINE_COMPLETE",
    "PIPELINE_INCOMPLETE",
    "StageRecord",
    "StageSpec",
    "build_engine_contract",
    "build_dry_run_plan",
    "canonical_schema",
    "detect_task_engine_mode",
    "make_stage_record",
    "planned_outputs",
    "render_final_markdown",
    "validate_pipeline",
]
