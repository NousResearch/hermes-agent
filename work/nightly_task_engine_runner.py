#!/usr/bin/env python3
"""Bounded nightly stress runner for the Hermes task engine."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSONL = ROOT / "work" / "nightly_task_engine_summary.jsonl"
FINAL_JSON = ROOT / "work" / "nightly_task_engine_final.json"
STRESS_SUMMARY_JSON = ROOT / "work" / "stress_runs" / "task_engine_adhd" / "summary.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=int, default=20 * 60)
    args = parser.parse_args()

    started = time.time()
    max_seconds = min(max(args.max_hours, 0.1), 6.0) * 3600
    max_rounds = max(1, min(args.max_rounds, 12))
    SUMMARY_JSONL.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSONL.write_text("", encoding="utf-8")

    last_blocker = ""
    repeated_blockers = 0
    latest: dict[str, Any] = {}
    stopped_reason = "max_rounds"

    for round_number in range(1, max_rounds + 1):
        elapsed = time.time() - started
        if elapsed >= max_seconds:
            stopped_reason = "max_hours"
            break

        summary = run_round(round_number, started)
        latest = summary
        blocker_signature = _blocker_signature(summary)
        if blocker_signature:
            repeated_blockers = repeated_blockers + 1 if blocker_signature == last_blocker else 1
            last_blocker = blocker_signature
        else:
            repeated_blockers = 0
            last_blocker = ""

        if _is_external_blocker(summary):
            summary["should_continue"] = False
            summary["next_action"] = "stop_external_blocker"
            stopped_reason = "external_blocker"
        elif repeated_blockers >= 2:
            summary["should_continue"] = False
            summary["next_action"] = "stop_repeated_blocker"
            stopped_reason = "repeated_blocker"

        _append_summary(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

        if not summary["should_continue"]:
            break

        if round_number < max_rounds:
            remaining = max_seconds - (time.time() - started)
            if remaining <= 0:
                stopped_reason = "max_hours"
                break
            time.sleep(min(args.sleep_seconds, max(0, remaining)))
    else:
        stopped_reason = "max_rounds"

    final = {
        "stopped_reason": stopped_reason,
        "latest_blocker": {
            "blocked_stage": latest.get("blocked_stage", ""),
            "blocked_reason": latest.get("blocked_reason", ""),
        },
        "tests_passed": bool(
            latest.get("py_compile_status") == "passed"
            and latest.get("pytest_status") == "passed"
        ),
        "files_changed": latest.get("changed_files", []),
        "tomorrow_next_step": _tomorrow_next_step(stopped_reason, latest),
    }
    FINAL_JSON.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return 0


def run_round(round_number: int, started: float) -> dict[str, Any]:
    before = _changed_files()
    parity = _legacy_parity_diff()
    py_compile = _run(["python", "-m", "py_compile", "tools/task_engine_contracts.py", "tools/task_engine_executors.py", "tools/task_engine_runner.py"], timeout=60)
    pytest = _run(["pytest", "-q", "tests/tools/test_task_engine_contracts.py"], timeout=180)
    stress = _run(["python", "work/stress_task_engine_adhd.py", "--max-rounds", "1"], timeout=1800)
    after = _changed_files()
    stress_summary = _parse_last_json_line(stress["stdout"])
    stress_summary_source = "stdout" if stress_summary else ""
    if not stress_summary and stress.get("timed_out"):
        stress_summary = _latest_stress_summary_from_file()
        stress_summary_source = "summary_file" if stress_summary else ""
    blocked_stage = stress_summary.get("blocked_stage", "") if stress_summary else ""
    blocked_reason = stress_summary.get("blocked_reason", "") if stress_summary else ""
    child_process_timeout = bool(stress.get("timed_out"))
    pipeline_blocked = bool(blocked_stage)
    external_blocker = _is_external_blocker_data(blocked_stage, blocked_reason)

    contract_violation = (
        not parity["ok"]
        or py_compile["returncode"] != 0
        or pytest["returncode"] != 0
        or bool(stress_summary and stress_summary.get("any_contract_violation"))
    )
    next_action = "continue_after_sleep"
    should_continue = True
    if not parity["ok"]:
        next_action = "fix_legacy_parity_diff"
        should_continue = False
    elif py_compile["returncode"] != 0:
        next_action = "fix_py_compile_failure"
        should_continue = False
    elif pytest["returncode"] != 0:
        next_action = "fix_pytest_failure"
        should_continue = False
    elif stress["returncode"] != 0 and not stress_summary:
        next_action = "inspect_child_process_timeout" if child_process_timeout else "inspect_stress_runner_failure"
        should_continue = False
        blocked_reason = _tail(stress["stderr"] or stress["stdout"])
    elif blocked_stage:
        next_action = _next_action_for_blocker(blocked_stage, blocked_reason)
        should_continue = False if external_blocker else True
    elif stress_summary:
        next_action = stress_summary.get("next_action") or next_action
        if child_process_timeout:
            next_action = "inspect_runner_timeout_after_completed_stress"

    return {
        "round": round_number,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "changed_files": after,
        "patch_summary": _patch_summary(before, after),
        "legacy_parity_status": "passed" if parity["ok"] else "failed",
        "legacy_parity_diff": parity["diff"],
        "py_compile_status": "passed" if py_compile["returncode"] == 0 else "failed",
        "pytest_status": "passed" if pytest["returncode"] == 0 else "failed",
        "stress_status": "passed" if stress_summary and stress_summary.get("pytest", {}).get("passed") else "failed",
        "stress_summary_source": stress_summary_source,
        "runner_timeout": child_process_timeout,
        "child_process_timeout": child_process_timeout,
        "child_stdout_tail": _tail(stress.get("stdout", "")),
        "child_stderr_tail": _tail(stress.get("stderr", "")),
        "pipeline_blocked": pipeline_blocked,
        "external_blocker": external_blocker,
        "blocked_stage": blocked_stage,
        "blocked_reason": blocked_reason,
        "contract_violation": contract_violation,
        "next_action": next_action,
        "should_continue": should_continue,
    }


def _legacy_parity_diff() -> dict[str, Any]:
    try:
        sys.path.insert(0, str(ROOT))
        from tools.task_engine_contracts import CANONICAL_STAGES, ENGINE_RESEARCH, ENGINE_RESEARCH_DECISION

        research = {stage.stage_name: stage.model for stage in CANONICAL_STAGES[ENGINE_RESEARCH]}
        research_decision = {stage.stage_name: stage.model for stage in CANONICAL_STAGES[ENGINE_RESEARCH_DECISION]}
        expected = {
            "L1_gemini_search": "Gemini 3.5 Flash (High)",
            "L4_gemini_audit": "Gemini 3.1 Pro (High)",
        }
        diff = []
        for stage, model in expected.items():
            if research.get(stage) != model:
                diff.append(f"RESEARCH:{stage}:{research.get(stage)!r}!={model!r}")
            if research_decision.get(stage) != model:
                diff.append(f"RESEARCH_DECISION:{stage}:{research_decision.get(stage)!r}!={model!r}")
        return {"ok": not diff, "diff": diff}
    except Exception as exc:
        return {"ok": False, "diff": [str(exc)]}


def _run(command: list[str], *, timeout: int) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_seconds": round(time.time() - started, 1),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        stdout, stderr = proc.communicate()
        stdout = (stdout or "") + _decode(exc.stdout)
        stderr = (stderr or "") + _decode(exc.stderr)
        return {
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr + f"\ntimeout after {timeout}s",
            "elapsed_seconds": round(time.time() - started, 1),
            "timed_out": True,
        }


def _parse_last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed((text or "").splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "round_id" in parsed:
            return parsed
    return None


def _latest_stress_summary_from_file() -> dict[str, Any] | None:
    try:
        parsed = json.loads(STRESS_SUMMARY_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[-1], dict):
        return parsed[-1]
    if isinstance(parsed, dict) and "round_id" in parsed:
        return parsed
    return None


def _changed_files() -> list[str]:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    files = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        files.append(line[3:] if len(line) > 3 else line.strip())
    return sorted(files)


def _patch_summary(before: list[str], after: list[str]) -> str:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    if not added and not removed:
        return "none"
    parts = []
    if added:
        parts.append("new/changed: " + ", ".join(added[:8]))
    if removed:
        parts.append("removed: " + ", ".join(removed[:8]))
    return "; ".join(parts)


def _is_external_blocker(summary: dict[str, Any]) -> bool:
    return _is_external_blocker_data(summary.get("blocked_stage", ""), summary.get("blocked_reason", ""))


def _is_external_blocker_data(stage: str, reason: str) -> bool:
    text = f"{stage}\n{reason}".lower()
    external_terms = (
        "agy_model_alias_blocked",
        "not logged into antigravity",
        "failed to get oauth token",
        "operation not permitted",
        "ddgs",
        "network",
        "omlx",
        "codex bridge",
        "gpt bridge",
        "auth",
    )
    return bool(stage) and any(term in text for term in external_terms)


def _blocker_signature(summary: dict[str, Any]) -> str:
    if not summary.get("blocked_stage"):
        return ""
    reason = str(summary.get("blocked_reason", "")).lower()
    if "ddgs" in reason:
        kind = "ddgs"
    elif "agy" in reason or "gemini" in reason:
        kind = "agy"
    elif "codex" in reason:
        kind = "codex"
    else:
        kind = reason[:120]
    return f"{summary['blocked_stage']}:{kind}"


def _next_action_for_blocker(stage: str, reason: str) -> str:
    if _is_external_blocker_data(stage, reason):
        return "stop_external_blocker"
    return "inspect_blocker_one_small_patch_max"


def _tomorrow_next_step(stopped_reason: str, latest: dict[str, Any]) -> str:
    if latest.get("runner_timeout"):
        return "Inspect nightly runner child-process capture/cleanup; stress summary was read separately from the pipeline result."
    if stopped_reason == "external_blocker":
        return "Resolve external dependency/auth/network blocker, then rerun nightly runner."
    if stopped_reason == "repeated_blocker":
        return "Inspect repeated blocker and apply one small targeted patch."
    if latest.get("blocked_stage"):
        return "Inspect latest blocked_stage and blocked_reason before continuing implementation."
    if latest.get("next_action") == "l4_smoke_passed_stop_before_l5":
        return "Continue from L5 acceptance wiring; L1/L2/L2.5/L3/L4 smoke remained healthy."
    if latest.get("next_action") == "l5_smoke_passed_research_complete_stop_before_decision":
        return "Continue from Decision phase wiring; RESEARCH L1-L5 smoke completed and stopped before Decision."
    if latest.get("next_action") == "intelligence_smoke_passed_stop_before_stage8":
        return "Continue from Decision Stage 8 supplementary_search wiring; intelligence_layer smoke completed and stopped before Stage 8."
    if latest.get("next_action") == "supplementary_search_passed_stop_before_structure_mapper":
        return "Continue from Decision Stage 9 structure_mapper wiring; supplementary_search smoke completed and stopped before structure_mapper."
    if latest.get("next_action") == "structure_mapper_passed_stop_before_evidence_judge":
        return "Continue from Decision Stage 10 evidence_judge wiring; structure_mapper smoke completed and stopped before evidence_judge."
    if latest.get("next_action") == "evidence_judge_passed_stop_before_premise_auditor":
        return "Continue from Decision Stage 11 premise_auditor wiring; evidence_judge smoke completed and stopped before premise_auditor."
    if latest.get("next_action") == "premise_auditor_passed_stop_before_alternative_generator":
        return "Continue from Decision Stage 12 alternative_generator wiring; premise_auditor smoke completed and stopped before alternative_generator."
    if latest.get("next_action") == "alternative_generator_passed_stop_before_insight_harvester":
        return "Continue from Decision Stage 13 insight_harvester wiring; alternative_generator smoke completed and stopped before insight_harvester."
    if latest.get("next_action") == "insight_harvester_passed_stop_before_convergence_report":
        return "Continue from Decision Stage 14 convergence_report wiring; insight_harvester smoke completed and stopped before convergence_report."
    if latest.get("next_action") == "convergence_report_passed_stop_before_external_calibration":
        return "Continue from Decision Stage 15 external_calibration wiring; convergence_report smoke completed and stopped before external_calibration."
    if latest.get("next_action") == "external_calibration_passed_stop_before_final_controller_report":
        return "Continue from final_controller_report wiring; external_calibration smoke completed and stopped before final_controller_report."
    if latest.get("next_action") == "final_controller_report_passed_pipeline_complete":
        return (
            "Archived RESEARCH_DECISION integration-test pipeline completed; "
            "production path is RESEARCH full -> DECISION full with research_packet_path. "
            "Continue with final output QA/regression only."
        )
    return "Continue from L3 R1 synthesis wiring; L1/L2/L2.5 smoke remained healthy."


def _append_summary(summary: dict[str, Any]) -> None:
    with SUMMARY_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")


def _tail(text: str, limit: int = 1200) -> str:
    return (text or "")[-limit:]


def _decode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
