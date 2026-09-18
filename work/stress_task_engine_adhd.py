#!/usr/bin/env python3
"""Automated ADHD task-engine stress loop.

Runs bounded local stress rounds for the Hermes task engine. The loop never
relaxes validation and never advances real stages after a fail-closed block.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.task_engine_executors import (  # noqa: E402
    run_research_decision_alternative_generator_smoke,
    run_research_decision_convergence_smoke,
    run_research_decision_evidence_judge_smoke,
    run_research_decision_external_calibration_smoke,
    run_research_decision_final_controller_smoke,
    run_research_decision_intelligence_smoke,
    run_research_decision_insight_harvester_smoke,
    run_research_decision_premise_auditor_smoke,
    run_research_decision_structure_mapper_smoke,
    run_research_decision_supplementary_search_smoke,
    run_research_l2_5_codex_handoff_smoke,
    run_research_l3_synthesis_smoke,
    run_research_l4_gemini_audit_smoke,
    run_research_l5_acceptance_smoke,
)
from tools.task_engine_runner import task_engine_runner  # noqa: E402


ADHD_PROMPT = """这是一个研究决策任务。

ADHD 儿童最新的研究进展和治疗方案；
与我儿子情况相匹配的梳理；
一些建议，以及长期发展的路线。
我想知道是否要主动干预，要主动干预到什么程度？
我最后补充一个，我儿子主要是注意力缺陷，但他不多动。孩子的注意力是被大脑放空、发呆，他脑子里面自己想的，他内在的东西太多，而不是被外部分心。
我需要详细地去讲以下家长行为培训。
怎么样为三年级做准备比较好？
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--out", default="work/stress_runs/task_engine_adhd")
    args = parser.parse_args()

    max_rounds = max(1, min(args.max_rounds, 5))
    out_dir = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    repeated_external_blocker = 0
    last_external_signature = ""

    for round_number in range(1, max_rounds + 1):
        round_id = f"round_{round_number:02d}"
        round_dir = out_dir / round_id
        round_dir.mkdir(parents=True, exist_ok=True)

        pytest_result = _run_pytest()
        dry_run = _runner_json("dry-run", round_dir / "dry_run")
        simulated_run = _runner_json("simulated-run", round_dir / "simulated_run")

        agy_preflight: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        smoke_l1_l2: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        l2_5: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        l3: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        l4: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        l5: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        intelligence: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        supplementary: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        structure: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        evidence: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        premise: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        alternative: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        insight: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        convergence: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        calibration: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        final_report: dict[str, Any] = {"status": "skipped", "pipeline_status": "SKIPPED"}
        blocked_stage = ""
        blocked_reason = ""
        any_contract_violation = False
        next_action = "continue"

        if not pytest_result["passed"]:
            blocked_stage = "pytest"
            blocked_reason = pytest_result["stderr"][:1000] or pytest_result["stdout"][:1000]
            any_contract_violation = True
            next_action = "fix_pytest_failure_then_rerun"
        elif dry_run.get("status") != "ok":
            blocked_stage = "dry-run"
            blocked_reason = json.dumps(dry_run, ensure_ascii=False)[:1000]
            any_contract_violation = True
            next_action = "fix_dry_run_contract_then_rerun"
        elif simulated_run.get("status") != "ok" or simulated_run.get("pipeline_status") != "PIPELINE_COMPLETE":
            blocked_stage = "simulated-run"
            blocked_reason = json.dumps(simulated_run.get("validation", simulated_run), ensure_ascii=False)[:1000]
            any_contract_violation = True
            next_action = "fix_simulated_run_contract_then_rerun"
        else:
            agy_preflight = _runner_json("agy-preflight", round_dir / "agy_preflight")
            if agy_preflight.get("status") != "AGY_OK":
                blocked_stage = "agy_preflight"
                blocked_reason = str(agy_preflight.get("blocked_reason") or agy_preflight.get("status") or "AGY_PREFLIGHT_BLOCKED")
                next_action = "blocked_complete_agy_antigravity_auth_then_rerun"
            else:
                smoke_l1_l2 = _runner_json("smoke-research-l1-l2", round_dir / "real_l1_l2")
            if not blocked_stage and smoke_l1_l2.get("status") == "ok":
                l2_5 = run_research_l2_5_codex_handoff_smoke(
                    smoke_l1_l2.get("run", {}),
                    base_dir=round_dir / "real_l1_l2",
                )
                if l2_5.get("status") == "ok":
                    l3 = run_research_l3_synthesis_smoke(
                        l2_5.get("run", {}),
                        base_dir=round_dir / "real_l1_l2",
                    )
                    if l3.get("status") == "ok":
                        l4 = run_research_l4_gemini_audit_smoke(
                            l3.get("run", {}),
                            base_dir=round_dir / "real_l1_l2",
                        )
                        if l4.get("status") == "ok":
                            l5 = run_research_l5_acceptance_smoke(
                                l4.get("run", {}),
                                base_dir=round_dir / "real_l1_l2",
                            )
                            if l5.get("status") == "ok":
                                intelligence = run_research_decision_intelligence_smoke(
                                    l5.get("run", {}),
                                    query=ADHD_PROMPT,
                                    base_dir=round_dir / "real_l1_l2",
                                )
                                if intelligence.get("status") == "ok":
                                    supplementary = run_research_decision_supplementary_search_smoke(
                                        intelligence.get("run", {}),
                                        query=ADHD_PROMPT,
                                        base_dir=round_dir / "real_l1_l2",
                                    )
                                    if supplementary.get("status") == "ok":
                                        structure = run_research_decision_structure_mapper_smoke(
                                            supplementary.get("run", {}),
                                            query=ADHD_PROMPT,
                                            base_dir=round_dir / "real_l1_l2",
                                        )
                                        if structure.get("status") == "ok":
                                            evidence = run_research_decision_evidence_judge_smoke(
                                                structure.get("run", {}),
                                                query=ADHD_PROMPT,
                                                base_dir=round_dir / "real_l1_l2",
                                            )
                                            if evidence.get("status") == "ok":
                                                premise = run_research_decision_premise_auditor_smoke(
                                                    evidence.get("run", {}),
                                                    query=ADHD_PROMPT,
                                                    base_dir=round_dir / "real_l1_l2",
                                                )
                                                if premise.get("status") == "ok":
                                                    alternative = run_research_decision_alternative_generator_smoke(
                                                        premise.get("run", {}),
                                                        query=ADHD_PROMPT,
                                                        base_dir=round_dir / "real_l1_l2",
                                                    )
                                                    if alternative.get("status") == "ok":
                                                        insight = run_research_decision_insight_harvester_smoke(
                                                            alternative.get("run", {}),
                                                            query=ADHD_PROMPT,
                                                            base_dir=round_dir / "real_l1_l2",
                                                        )
                                                        if insight.get("status") == "ok":
                                                            convergence = run_research_decision_convergence_smoke(
                                                                insight.get("run", {}),
                                                                query=ADHD_PROMPT,
                                                                base_dir=round_dir / "real_l1_l2",
                                                            )
                                                            if convergence.get("status") == "ok":
                                                                calibration = run_research_decision_external_calibration_smoke(
                                                                    convergence.get("run", {}),
                                                                    query=ADHD_PROMPT,
                                                                    base_dir=round_dir / "real_l1_l2",
                                                                )
                                                                if calibration.get("status") == "ok":
                                                                    final_report = run_research_decision_final_controller_smoke(
                                                                        calibration.get("run", {}),
                                                                        query=ADHD_PROMPT,
                                                                        base_dir=round_dir / "real_l1_l2",
                                                                    )
                                                                    if final_report.get("status") == "ok" and final_report.get("pipeline_status") == "PIPELINE_COMPLETE":
                                                                        next_action = "final_controller_report_passed_pipeline_complete"
                                                                    else:
                                                                        blocked_stage = str(final_report.get("blocked_stage") or "final_controller_report")
                                                                        blocked_reason = _extract_blocked_reason(final_report)
                                                                        next_action = "blocked_fix_final_controller_report_then_rerun"
                                                                else:
                                                                    blocked_stage = str(calibration.get("blocked_stage") or "external_calibration")
                                                                    blocked_reason = _extract_blocked_reason(calibration)
                                                                    next_action = "blocked_fix_external_calibration_bridge_then_rerun"
                                                            else:
                                                                blocked_stage = str(convergence.get("blocked_stage") or "convergence_report")
                                                                blocked_reason = _extract_blocked_reason(convergence)
                                                                next_action = "blocked_fix_convergence_report_r1_then_rerun"
                                                        else:
                                                            blocked_stage = str(insight.get("blocked_stage") or "insight_harvester")
                                                            blocked_reason = _extract_blocked_reason(insight)
                                                            next_action = "blocked_fix_insight_harvester_gemma431b_then_rerun"
                                                    else:
                                                        blocked_stage = str(alternative.get("blocked_stage") or "alternative_generator")
                                                        blocked_reason = _extract_blocked_reason(alternative)
                                                        next_action = "blocked_fix_alternative_generator_gemma431b_then_rerun"
                                                else:
                                                    blocked_stage = str(premise.get("blocked_stage") or "premise_auditor")
                                                    blocked_reason = _extract_blocked_reason(premise)
                                                    next_action = "blocked_fix_premise_auditor_llama70b_then_rerun"
                                            else:
                                                blocked_stage = str(evidence.get("blocked_stage") or "evidence_judge")
                                                blocked_reason = _extract_blocked_reason(evidence)
                                                next_action = "blocked_fix_evidence_judge_nemotron_then_rerun"
                                        else:
                                            blocked_stage = str(structure.get("blocked_stage") or "structure_mapper")
                                            blocked_reason = _extract_blocked_reason(structure)
                                            next_action = "blocked_fix_structure_mapper_qwen72b_then_rerun"
                                    else:
                                        blocked_stage = str(supplementary.get("blocked_stage") or "supplementary_search")
                                        blocked_reason = _extract_blocked_reason(supplementary)
                                        next_action = "blocked_fix_supplementary_search_ddgs_then_rerun"
                                else:
                                    blocked_stage = str(intelligence.get("blocked_stage") or "intelligence_layer")
                                    blocked_reason = _extract_blocked_reason(intelligence)
                                    next_action = "blocked_fix_intelligence_layer_agy_then_rerun"
                            else:
                                blocked_stage = str(l5.get("blocked_stage") or "L5_deepseek_acceptance")
                                blocked_reason = _extract_blocked_reason(l5)
                                next_action = "blocked_fix_l5_acceptance_then_rerun"
                        else:
                            blocked_stage = str(l4.get("blocked_stage") or "L4_gemini_audit")
                            blocked_reason = _extract_blocked_reason(l4)
                            next_action = "blocked_fix_l4_agy_gemini_audit_then_rerun"
                    else:
                        blocked_stage = str(l3.get("blocked_stage") or "L3_r1_synthesis")
                        blocked_reason = _extract_blocked_reason(l3)
                        next_action = "blocked_fix_r1_omlx_then_rerun"
                else:
                    blocked_stage = str(l2_5.get("blocked_stage") or "L2_5_codex_evidence_organizer")
                    blocked_reason = _extract_blocked_reason(l2_5)
                    next_action = "fix_l2_5_codex_handoff_smoke_then_rerun"
            elif not blocked_stage:
                blocked_stage = str(smoke_l1_l2.get("blocked_stage") or "smoke_l1_l2")
                blocked_reason = _extract_blocked_reason(smoke_l1_l2)
                next_action = _next_action_for_block(blocked_reason)

        external_signature = _external_block_signature(blocked_stage, blocked_reason)
        if external_signature and external_signature == last_external_signature:
            repeated_external_blocker += 1
        elif external_signature:
            repeated_external_blocker = 1
        else:
            repeated_external_blocker = 0
        last_external_signature = external_signature

        summary = {
            "round_id": round_id,
            "git_diff_summary": _git_diff_summary(),
            "pytest": {
                "passed": pytest_result["passed"],
                "returncode": pytest_result["returncode"],
            },
            "dry_run": _compact_runner_status(dry_run),
            "simulated_run": _compact_runner_status(simulated_run),
            "agy_preflight": _compact_runner_status(agy_preflight),
            "smoke_l1_l2": _compact_runner_status(smoke_l1_l2),
            "l2_5_codex_handoff_smoke": _compact_runner_status(l2_5),
            "l3_r1_synthesis_smoke": _compact_runner_status(l3),
            "l4_gemini_audit_smoke": _compact_runner_status(l4),
            "l5_deepseek_acceptance_smoke": _compact_runner_status(l5),
            "intelligence_layer_smoke": _compact_runner_status(intelligence),
            "supplementary_search_smoke": _compact_runner_status(supplementary),
            "structure_mapper_smoke": _compact_runner_status(structure),
            "evidence_judge_smoke": _compact_runner_status(evidence),
            "premise_auditor_smoke": _compact_runner_status(premise),
            "alternative_generator_smoke": _compact_runner_status(alternative),
            "insight_harvester_smoke": _compact_runner_status(insight),
            "convergence_report_smoke": _compact_runner_status(convergence),
            "external_calibration_smoke": _compact_runner_status(calibration),
            "final_controller_report_smoke": _compact_runner_status(final_report),
            "blocked_stage": blocked_stage,
            "blocked_reason": blocked_reason,
            "any_contract_violation": any_contract_violation,
            "next_action": next_action,
        }

        if repeated_external_blocker >= 2:
            summary["blocked_status"] = "BLOCKED_STATUS"
            summary["next_action"] = "stop_external_permission_or_agy_alias_block_repeated"
            summaries.append(summary)
            _write_round(round_dir, summary, pytest_result, dry_run, simulated_run, agy_preflight, smoke_l1_l2, l2_5, l3, l4, l5, intelligence, supplementary, structure, evidence, premise, alternative, insight, convergence, calibration, final_report)
            print(json.dumps(summary, ensure_ascii=False))
            break

        summaries.append(summary)
        _write_round(round_dir, summary, pytest_result, dry_run, simulated_run, agy_preflight, smoke_l1_l2, l2_5, l3, l4, l5, intelligence, supplementary, structure, evidence, premise, alternative, insight, convergence, calibration, final_report)
        print(json.dumps(summary, ensure_ascii=False))

        if next_action.startswith("fix_"):
            # The runner does not self-modify code. It records the first clear
            # failing cause and lets the engineering agent make one targeted fix.
            break

        if not blocked_stage:
            break

        time.sleep(0.2)

    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if summaries and summaries[-1]["pytest"]["passed"] else 1


def _run_pytest() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/tools/test_task_engine_contracts.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _runner_json(action: str, base_dir: Path) -> dict[str, Any]:
    try:
        raw = task_engine_runner(
            query=ADHD_PROMPT,
            mode="AUTO" if action != "smoke-research-l1-l2" else "RESEARCH",
            action=action,
            base_dir=str(base_dir),
        )
        return json.loads(raw)
    except Exception as exc:
        return {"status": "blocked", "pipeline_status": "PIPELINE_BLOCKED", "error": str(exc)}


def _compact_runner_status(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "status": result.get("status"),
        "pipeline_status": result.get("pipeline_status"),
    }
    if "mode" in result:
        compact["mode"] = result.get("mode")
    if "plan" in result:
        compact["stage_count"] = result["plan"].get("stage_count")
    if "validation" in result:
        compact["valid"] = result["validation"].get("valid")
        compact["stage_count"] = result["validation"].get("stage_count")
    if "blocked_stage" in result:
        compact["blocked_stage"] = result.get("blocked_stage")
    if "blocked_reason" in result:
        compact["blocked_reason"] = result.get("blocked_reason")
    if "missing_models" in result:
        compact["missing_models"] = result.get("missing_models")
    return compact


def _extract_blocked_reason(result: dict[str, Any]) -> str:
    if result.get("error"):
        return str(result["error"])[:1000]
    stages = result.get("run", {}).get("stages", [])
    for stage in reversed(stages):
        if isinstance(stage, dict) and stage.get("error"):
            return str(stage["error"])[:1000]
    return str(result.get("message") or result)[:1000]


def _next_action_for_block(reason: str) -> str:
    lowered = reason.lower()
    if "agy_model_alias_blocked" in lowered or "defaulting to ccpa" in lowered or "not in local config" in lowered:
        return "blocked_configure_hard_agy_alias_map_no_ccpa_fallback"
    if "operation not permitted" in lowered or "bind" in lowered or ".gemini" in lowered:
        return "blocked_fix_agy_local_permissions_or_run_outside_sandbox"
    if "ddgs" in lowered:
        return "fix_ddgs_real_search_adapter_then_rerun"
    if "structure_mapper" in lowered or "qwen72b" in lowered:
        return "blocked_fix_structure_mapper_qwen72b_then_rerun"
    if "evidence_judge" in lowered or "nemotron" in lowered:
        return "blocked_fix_evidence_judge_nemotron_then_rerun"
    if "premise_auditor" in lowered or "llama70b" in lowered:
        return "blocked_fix_premise_auditor_llama70b_then_rerun"
    if "insight_harvester" in lowered:
        return "blocked_fix_insight_harvester_gemma431b_then_rerun"
    if "convergence_report" in lowered:
        return "blocked_fix_convergence_report_r1_then_rerun"
    if "external_calibration" in lowered or "gpt bridge" in lowered:
        return "blocked_fix_external_calibration_bridge_then_rerun"
    if "final_controller_report" in lowered:
        return "blocked_fix_final_controller_report_then_rerun"
    if "alternative_generator" in lowered or "gemma" in lowered:
        return "blocked_fix_alternative_generator_gemma431b_then_rerun"
    if "omlx" in lowered or "r1" in lowered:
        return "blocked_fix_r1_omlx_then_rerun"
    if "intelligence_layer" in lowered:
        return "blocked_fix_intelligence_layer_agy_then_rerun"
    if "supplementary_search" in lowered:
        return "blocked_fix_supplementary_search_ddgs_then_rerun"
    if "agy" in lowered or "gemini" in lowered:
        return "blocked_fix_l4_agy_gemini_audit_then_rerun"
    return "inspect_smoke_l1_l2_block_then_rerun"


def _external_block_signature(stage: str, reason: str) -> str:
    lowered = reason.lower()
    if stage == "L1_gemini_search" and (
        "agy_model_alias_blocked" in lowered
        or "operation not permitted" in lowered
        or "defaulting to ccpa" in lowered
        or "not in local config" in lowered
        or "bind" in lowered
    ):
        if "agy_model_alias_blocked" in lowered or "defaulting to ccpa" in lowered or "not in local config" in lowered:
            return "L1_gemini_search:agy_alias"
        return "L1_gemini_search:agy_permission"
    return ""


def _git_diff_summary() -> str:
    diff = subprocess.run(
        ["git", "diff", "--stat"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    parts = []
    if diff.stdout.strip() or diff.stderr.strip():
        parts.append((diff.stdout or diff.stderr).strip())
    untracked = "\n".join(
        line for line in status.stdout.splitlines() if line.startswith("?? ")
    )
    if untracked:
        parts.append("Untracked:\n" + untracked)
    return "\n\n".join(parts)


def _write_round(
    round_dir: Path,
    summary: dict[str, Any],
    pytest_result: dict[str, Any],
    dry_run: dict[str, Any],
    simulated_run: dict[str, Any],
    agy_preflight: dict[str, Any],
    smoke_l1_l2: dict[str, Any],
    l2_5: dict[str, Any],
    l3: dict[str, Any],
    l4: dict[str, Any],
    l5: dict[str, Any],
    intelligence: dict[str, Any],
    supplementary: dict[str, Any],
    structure: dict[str, Any],
    evidence: dict[str, Any],
    premise: dict[str, Any],
    alternative: dict[str, Any],
    insight: dict[str, Any],
    convergence: dict[str, Any],
    calibration: dict[str, Any],
    final_report: dict[str, Any],
) -> None:
    payloads = {
        "summary.json": summary,
        "pytest.json": pytest_result,
        "dry_run.json": dry_run,
        "simulated_run.json": simulated_run,
        "agy_preflight.json": agy_preflight,
        "smoke_l1_l2.json": smoke_l1_l2,
        "l2_5_codex_handoff_smoke.json": l2_5,
        "l3_r1_synthesis_smoke.json": l3,
        "l4_gemini_audit_smoke.json": l4,
        "l5_deepseek_acceptance_smoke.json": l5,
        "intelligence_layer_smoke.json": intelligence,
        "supplementary_search_smoke.json": supplementary,
        "structure_mapper_smoke.json": structure,
        "evidence_judge_smoke.json": evidence,
        "premise_auditor_smoke.json": premise,
        "alternative_generator_smoke.json": alternative,
        "insight_harvester_smoke.json": insight,
        "convergence_report_smoke.json": convergence,
        "external_calibration_smoke.json": calibration,
        "final_controller_report_smoke.json": final_report,
    }
    for name, payload in payloads.items():
        (round_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
