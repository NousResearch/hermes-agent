"""Formal scoring calibration rules for Hermes task-engine outputs.

The functions in this module are deterministic and sample-independent. They
codify the post-baseline quality gates that keep final-controller and
production-readiness decisions from passing on stale artifacts, raw dumps, or
over-broad legal/compliance conclusions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CALIBRATION_THRESHOLDS = {
    "external_calibration_min_body_chars": 1500,
    "external_calibration_min_field_chars": 20,
    "readiness_min_source_consumption_yes": 3,
}

EXTERNAL_CALIBRATION_MINIMUM_FIELDS = (
    "calibration_verdict",
    "agreement_points",
    "disagreement_or_risk_points",
    "missing_considerations",
    "final_adjustment_recommendation",
)

STRENGTH_LABEL_TERMS = (
    "supported",
    "plausible",
    "speculative",
    "contradicted",
    "支持",
    "可信",
    "推测",
    "矛盾",
)

PRODUCTION_READINESS_PASS_TERMS = (
    "production_readiness: pass",
    "production_readiness：pass",
    "production_readiness=pass",
    "production_ready: true",
    "production_ready=true",
    "ready for production",
    "生产就绪",
)

LEGAL_RISK_TERMS = (
    "legal",
    "lawyer",
    "law firm",
    "contract",
    "case law",
    "international law",
    "maritime law",
    "law of the sea",
    "sovereignty dispute",
    "unclos",
    "regulatory",
    "法律",
    "律师",
    "法务",
    "合同",
    "案例",
    "诉讼",
    "监管",
    "国际法",
    "海洋法",
    "主权争议",
    "争议海域",
)

AUDIT_FINANCE_TERMS = (
    "audit",
    "auditor",
    "finance",
    "financial",
    "accounting",
    "审计",
    "审计底稿",
    "财务异常",
    "财务分析",
    "管理层讨论分析",
    "初级审计员",
    "企业财务分析师",
)

RAW_OR_STALE_FINAL_TERMS = (
    "persona raw",
    "r1 convergence body",
    "raw artifact",
    "artifact dump",
    "claim_strength_table | claim",
    "```json\n{\n  \"stage_name\"",
    "external_calibration executor_model",
    "fallback_reasons:",
    "evidence judge - decision stage",
    "evidence judge – decision stage",
    "convergence decision framework",
    "convergence_fixed_section_digest",
    "artifact (stage)",
    "strength / quality of evidence",
    "\n## key_drivers",
    "\n## mechanism_chain",
    "\n## certainty_levels",
    "\n## uncertainty_boundary",
    "created_in_current_run: false",
    "legacy_contaminated: true",
    "valid_for_pipeline: false",
)

STALE_FINAL_STATE_TERMS = (
    "created_in_current_run: false",
    "legacy_contaminated: true",
    "valid_for_pipeline: false",
)

STALE_PATH_PATTERNS = (
    re.compile(r"/\.hermes_task_engine_runs/[^ \n]+", re.IGNORECASE),
    re.compile(r"/work/stress_runs/[^ \n]+", re.IGNORECASE),
    re.compile(r"/(?:old|legacy|baseline|round\d+)[^ \n]*/final_controller_report/", re.IGNORECASE),
)


@dataclass(frozen=True)
class CalibrationResult:
    """Machine-readable scoring calibration decision."""

    status: str
    reason: str = ""
    details: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def blocked(self) -> bool:
        return self.status == "block"


def pass_result(*details: str) -> CalibrationResult:
    return CalibrationResult("pass", "", tuple(detail for detail in details if detail))


def block_result(reason: str, *details: str) -> CalibrationResult:
    return CalibrationResult("block", reason, tuple(detail for detail in details if detail))


def external_calibration_quality_body(text: str) -> str:
    metadata_prefixes = (
        "executor_model:",
        "fallback_reasons:",
        "artifact_path:",
        "valid_for_pipeline:",
        "stage_name:",
        "owner=",
        "owner:",
        "model:",
        "metadata:",
        "error:",
        "error_summary:",
        "attempt:",
        "fallback_used:",
        "created_in_current_run:",
        "artifact_state:",
    )
    metadata_key_pattern = re.compile(
        r'^\s*"?(?:executor_model|fallback_reasons|artifact_path|valid_for_pipeline|'
        r'stage_name|owner|model|metadata|error|error_summary|attempt|fallback_used|'
        r'created_in_current_run|artifact_state)"?\s*[:=]',
        re.IGNORECASE,
    )
    skipped_exact = {"external_calibration", "metadata", "diagnostic"}
    kept: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if not line:
            if kept and kept[-1].strip():
                kept.append("")
            continue
        if lowered in skipped_exact:
            continue
        if any(lowered.startswith(prefix) for prefix in metadata_prefixes):
            continue
        if metadata_key_pattern.match(line):
            continue
        kept.append(raw_line)
    return "\n".join(kept).strip()


def external_calibration_has_verdict_body(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        term in lowered
        for term in (
            "calibration_verdict",
            "calibration verdict",
            "calibration-verdict",
            "verdict:",
            "verdict：",
            "校准结论",
            "final calibration sentence",
            "final calibrated conclusion",
        )
    )


def assess_external_calibration_text(text: str) -> CalibrationResult:
    value = external_calibration_quality_body(text or "")
    lowered = value.lower()
    if external_calibration_has_minimum_fields(value):
        missing_body = external_calibration_header_only_fields(value)
        if missing_body:
            return block_result("external_calibration_header_only", *missing_body)
        if not any(term in lowered for term in STRENGTH_LABEL_TERMS):
            return block_result("external_calibration_missing_strength_labels")
        return pass_result("external_calibration_required_fields_present")
    if len(value) < CALIBRATION_THRESHOLDS["external_calibration_min_body_chars"]:
        if any(term in lowered for term in ("calibration_scope", "claim_strength_table", "calibration_verdict")):
            return block_result("external_calibration_header_only")
        return block_result("external_calibration_too_short")
    tail = normalized_tail(value)
    if tail_looks_truncated(tail):
        return block_result("truncated_tail")
    if not any(term in lowered for term in STRENGTH_LABEL_TERMS):
        return block_result("external_calibration_missing_strength_labels")
    if not external_calibration_has_verdict_body(value):
        return block_result("external_calibration_missing_verdict")
    if "claim_strength_table" in lowered and "calibration" not in lowered[lowered.rfind("claim_strength_table"):]:
        return block_result("external_calibration_header_only")
    return pass_result("external_calibration_long_form_complete")


def external_calibration_has_minimum_fields(text: str) -> bool:
    lowered = (text or "").lower()
    return all(field in lowered for field in EXTERNAL_CALIBRATION_MINIMUM_FIELDS)


def external_calibration_header_only_fields(text: str) -> list[str]:
    missing: list[str] = []
    for field_name in EXTERNAL_CALIBRATION_MINIMUM_FIELDS:
        body = markdown_section_body(text, field_name) or colon_or_plain_section_body(text, field_name)
        if len(" ".join(body.split())) < CALIBRATION_THRESHOLDS["external_calibration_min_field_chars"]:
            missing.append(field_name)
    return missing


def classify_compliance_domain(query: str) -> str:
    value = query or ""
    lowered = value.lower()
    has_audit_finance = any(term in value or term in lowered for term in AUDIT_FINANCE_TERMS)
    has_legal = any(term in value or term in lowered for term in LEGAL_RISK_TERMS)
    if has_audit_finance:
        return "audit_finance_compliance"
    if has_legal:
        return "legal_compliance"
    if "compliance" in lowered or "合规" in value:
        return "general_compliance"
    return "general"


def assess_stage_freshness(
    stage_records: list[dict[str, Any]],
    *,
    base_dir: str | Path | None = None,
    expected_stages: tuple[str, ...] | None = None,
) -> CalibrationResult:
    if expected_stages is not None:
        actual = tuple(str(record.get("stage_name") or "") for record in stage_records)
        if actual != expected_stages:
            return block_result("stale_decision_stage_order_mismatch", f"actual={actual!r}")
    base = Path(base_dir).resolve() if base_dir is not None else None
    for record in stage_records:
        stage_name = str(record.get("stage_name") or "<unknown>")
        if record.get("created_in_current_run") is not True:
            return block_result("stale_decision_not_current_run", stage_name)
        if record.get("legacy_contaminated") is not False:
            return block_result("stale_decision_legacy_contaminated", stage_name)
        if record.get("valid_for_pipeline") is not True:
            return block_result("stale_decision_invalid_stage", stage_name)
        if base is not None:
            for key in ("artifact_path",):
                path = str(record.get(key) or "")
                if path and not _path_is_under_base(path, base):
                    return block_result("stale_decision_path_outside_current_run", stage_name, key)
            outputs = record.get("outputs") or {}
            if isinstance(outputs, dict):
                for key, path in outputs.items():
                    if isinstance(path, str) and path and not _path_is_under_base(path, base):
                        return block_result("stale_decision_output_path_outside_current_run", stage_name, str(key))
    return pass_result("stage_records_are_current_run")


def assess_final_controller_text(text: str, *, query: str = "") -> CalibrationResult:
    value = (text or "").strip()
    lowered = value.lower()
    if any(term in lowered for term in STALE_FINAL_STATE_TERMS):
        return block_result("stale_final_or_raw_intermediate_dump")
    if any(term in lowered for term in RAW_OR_STALE_FINAL_TERMS):
        return block_result("raw_intermediate_dump")
    for pattern in STALE_PATH_PATTERNS:
        if pattern.search(value):
            return block_result("stale_final_old_run_path")
    if looks_like_raw_markdown_table_dump(value):
        return block_result("raw_table_dump")
    if tail_looks_truncated(normalized_tail(value)):
        return block_result("truncated_tail")
    readiness = assess_production_readiness(value, query=query)
    if readiness.blocked:
        return readiness
    return pass_result("final_controller_text_calibrated")


def assess_final_controller_packet(packet: dict[str, Any], text: str) -> CalibrationResult:
    query = str(packet.get("query") or "")
    text_result = assess_final_controller_text(text, query=query)
    if text_result.blocked:
        return text_result
    trace = packet.get("stage_trace")
    base_dir = packet.get("base_dir")
    if isinstance(trace, list) and trace and any(
        any(key in item for key in ("created_in_current_run", "legacy_contaminated", "valid_for_pipeline"))
        for item in trace
        if isinstance(item, dict)
    ):
        freshness = assess_stage_freshness(
            [item for item in trace if isinstance(item, dict)],
            base_dir=str(base_dir) if base_dir else None,
        )
        if freshness.blocked:
            return freshness
    return pass_result("final_controller_packet_calibrated")


def assess_production_readiness(text: str, *, query: str = "") -> CalibrationResult:
    value = text or ""
    lowered = value.lower()
    claims_pass = any(term in lowered or term in value for term in PRODUCTION_READINESS_PASS_TERMS)
    if not claims_pass:
        return pass_result("no_production_readiness_pass_claim")

    missing = [
        term
        for term in ("evidence_strength", "controversy", "evidence_gap")
        if term not in lowered and {
            "evidence_strength": "证据强度",
            "controversy": "争议",
            "evidence_gap": "缺口",
        }[term] not in value
    ]
    if missing:
        return block_result("production_readiness_over_broad", *missing)

    source_yes_count = len(re.findall(r"=\s*yes\b", lowered))
    if "source_consumption_check" in lowered and source_yes_count < CALIBRATION_THRESHOLDS["readiness_min_source_consumption_yes"]:
        return block_result("production_readiness_missing_source_consumption")

    domain = classify_compliance_domain(query)
    if domain in {"legal_compliance", "audit_finance_compliance", "general_compliance"}:
        risk_terms = ("risk", "liability", "regulatory", "supervision", "责任", "监管", "复核", "风险", "授权")
        if not any(term in lowered or term in value for term in risk_terms):
            return block_result("production_readiness_compliance_without_risk_bucket", domain)
    return pass_result("production_readiness_claim_bounded")


def markdown_section_body(text: str, heading: str) -> str:
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


def colon_or_plain_section_body(text: str, field_name: str) -> str:
    lines = (text or "").splitlines()
    start = -1
    lowered_field = field_name.lower()
    known = set(EXTERNAL_CALIBRATION_MINIMUM_FIELDS)
    for index, line in enumerate(lines):
        stripped = line.strip().lower().lstrip("#").strip()
        key = stripped.split(":", 1)[0].strip()
        if key == lowered_field:
            start = index
            break
    if start < 0:
        return ""
    collected: list[str] = []
    first = lines[start].split(":", 1)
    if len(first) == 2 and first[1].strip():
        collected.append(first[1].strip())
    for line in lines[start + 1:]:
        stripped = line.strip().lower().lstrip("#").strip()
        key = stripped.split(":", 1)[0].strip()
        if key in known:
            break
        collected.append(line)
    return "\n".join(collected).strip()


def normalized_tail(text: str, *, limit: int = 180) -> str:
    return " ".join((text or "").strip().split())[-limit:].strip().lower()


def tail_looks_truncated(tail: str) -> bool:
    if not tail:
        return True
    exact_or_suffixes = (
        "claim_strength_table",
        "strength_by_claim",
        "alternative c:",
        "alternative c: proactive ne",
        "third-grad",
        "evidence pac",
        "causing",
        "deficit neutr",
        "low – inten",
        "low - inten",
        "最危险的错误培养路径 ... 过",
    )
    if any(tail.endswith(fragment) for fragment in exact_or_suffixes):
        return True
    if tail.endswith(("|", "-", "###", "##", ":")):
        return True
    return False


def looks_like_raw_markdown_table_dump(text: str) -> bool:
    rows = [line.strip() for line in (text or "").splitlines() if line.strip().startswith("|")]
    if len(rows) < 3:
        return False
    joined = "\n".join(rows[:12]).lower()
    stageish = "artifact" in joined or "stage" in joined or "evidence" in joined or "claim" in joined
    has_separator = any("---" in row for row in rows[:6])
    return stageish and has_separator


def _path_is_under_base(path_value: str, base: Path) -> bool:
    path = Path(path_value)
    if not path.is_absolute():
        path = base / path
    try:
        path.resolve().relative_to(base)
        return True
    except ValueError:
        return False
