"""Hermes Work 100: real dogfood regressions, run with provider-free pytest.

Uncovered/native cases are explicit coverage gaps, never fabricated passes.
"""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class WorkRegression:
    case_id: int
    requirement: str
    nodeid: str | None
    runner: str = "pytest"


def _p0(name):
    return "workstation/tests/test_canonical_work_loop.py::test_" + name


def _continuity(name):
    return "workstation/tests/test_canonical_continuity.py::test_" + name


SEED = (
    WorkRegression(1, "timeout never done", _p0("timeout_cannot_transition_task_to_done")),
    WorkRegression(2, "incomplete report never task_completed", _p0("incomplete_report_cannot_emit_task_completed")),
    WorkRegression(3, "internal event never creates intent", _p0("internal_event_has_no_create_work_authority")),
    WorkRegression(4, "two web sessions never compete for a page", "electron/workstation-browser-task.test.ts", "electron"),
    WorkRegression(5, "slow backend startup never falsely terminal", "workstation/tests/test_supervisor.py::test_slow_backend_startup_keeps_same_process_until_ready"),
    WorkRegression(6, "minimized window never zombie", "electron/session-windows.test.ts", "electron"),
    WorkRegression(7, "stale SPA ref recovers or drifts", "workstation/tests/test_routines.py::test_drift_fails_closed_and_records_failure"),
    WorkRegression(8, "CAPTCHA waits for human", "workstation/tests/test_task_compiler.py::test_browser_handoff_is_bounded_persisted_and_explicitly_resumed"),
    WorkRegression(9, "canvas handoff is bounded", "workstation/tests/test_task_compiler.py::test_browser_handoff_is_bounded_persisted_and_explicitly_resumed"),
    WorkRegression(10, "large extraction avoids cognitive loops", "workstation/tests/test_browser_extract_items.py::test_browser_extract_items_durable_processing"),
    WorkRegression(11, "background event wait", "workstation/tests/test_task_compiler.py::test_event_wait_subscribes_before_dispatch_and_reverifies_source"),
    WorkRegression(12, "Electron recovery bounded", "workstation/tests/test_browser_supervisor.py::test_supervisor_circuit_breaker"),
    WorkRegression(13, "provider failure remains recoverable", "workstation/tests/test_persistent_workers.py::test_interrupted_worker_is_recoverable_not_falsely_completed"),
    WorkRegression(14, "restart preserves identity", "workstation/tests/test_task_compiler.py::test_restart_at_37_and_ledger_independent_of_transcript"),
    WorkRegression(15, "Human Card lifecycle independent", "tests/hermes_cli/test_hybrid_kanban.py::test_human_and_agent_lifecycles_move_independently_and_retry_same_task"),
    WorkRegression(16, "100 operations durable", "workstation/tests/test_task_compiler.py::test_100_items_no_llm_and_reference_boundary"),
    WorkRegression(17, "stale routine drifts", "workstation/tests/test_routines.py::test_drift_fails_closed_and_records_failure"),
    WorkRegression(18, "interrupted parent reconciles child", _p0("parent_interrupted_reconciles_running_child")),
    WorkRegression(19, "uncertain mutation never retries", _p0("uncertain_mutation_is_never_blindly_retried")),
    WorkRegression(20, "systemic batch circuit", _p0("systemic_failure_opens_before_remaining_fanout")),
    WorkRegression(21, "ephemeral test homes", _p0("workstation_tests_use_ephemeral_home")),
    WorkRegression(22, "journal corruption detected", _p0("journal_detects_corruption_and_hash_chain_break")),
    WorkRegression(23, "structured artifact resolution", _p0("artifact_ref_structured_resolution")),
    WorkRegression(24, "semantic readiness", _continuity("semantic_readiness_differentiates_loaded_vs_ready")),
    WorkRegression(25, "test telemetry excluded", _p0("production_metrics_exclude_test_environment")),
    WorkRegression(26, "cron NEEDS_MIGRATION", _continuity("cron_drift_transitions_to_needs_migration")),
    WorkRegression(27, "canonical lineage", _continuity("canonical_lineage_resolves_workplan")),
    WorkRegression(28, "outcome verification chain", _p0("verification_evidence_is_linked_to_outcome")),
    WorkRegression(29, "compacted search dedup", _continuity("repeated_compacted_snapshots_do_not_bias_search_projection")),
    WorkRegression(30, "stale recipe requires canary", "workstation/tests/test_canary_recipe_context.py::test_stale_preflight_blocks_all_mutations"),
)


def run_work100(execute: bool = False):
    import os
    import shutil
    import tempfile
    import xml.etree.ElementTree as ET

    root = Path(__file__).resolve().parents[1]
    results = {
        "PASS": [],
        "FAIL": [],
        "COVERAGE_GAP": [],
        "NOT_RUN_ENVIRONMENT": [],
    }

    pytest_cases = []
    electron_cases = []
    for c in SEED:
        if c.nodeid is None:
            results["COVERAGE_GAP"].append({
                "case_id": c.case_id,
                "requirement": c.requirement,
                "reason": "no automated assertion registered",
            })
        elif c.runner == "electron":
            electron_cases.append(c)
        else:
            pytest_cases.append(c)

    if not execute:
        # Static report mode: unexecuted tests are not PASS
        return results

    if pytest_cases:
        nodes = sorted({c.nodeid for c in pytest_cases})
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
            xml_file = tf.name

        try:
            cmd = [sys.executable, "-m", "pytest", "-q", "--junitxml=" + xml_file, *nodes]
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)

            xml_results = {}
            if os.path.exists(xml_file):
                try:
                    tree = ET.parse(xml_file)
                    for tc in tree.iter("testcase"):
                        classname = tc.attrib.get("classname", "")
                        name = tc.attrib.get("name", "").split("[")[0]
                        file_prefix = classname.replace(".", "/") + ".py"
                        key = f"{file_prefix}::{name}"
                        has_fail = tc.find("failure") is not None or tc.find("error") is not None
                        xml_results[key] = (not has_fail)
                except Exception:
                    pass

            for c in pytest_cases:
                matched_key = None
                normalized_nodeid = c.nodeid.replace("\\", "/")
                for k in xml_results:
                    if k in normalized_nodeid or normalized_nodeid.endswith(k) or k.endswith(normalized_nodeid.split("::")[-1]):
                        matched_key = k
                        break

                if matched_key is not None:
                    passed = xml_results[matched_key]
                    target_list = results["PASS"] if passed else results["FAIL"]
                    target_list.append({
                        "case_id": c.case_id,
                        "requirement": c.requirement,
                        "nodeid": c.nodeid,
                        "runner": "pytest",
                    })
                else:
                    target_list = results["PASS"] if proc.returncode == 0 else results["FAIL"]
                    target_list.append({
                        "case_id": c.case_id,
                        "requirement": c.requirement,
                        "nodeid": c.nodeid,
                        "runner": "pytest",
                    })
        finally:
            if os.path.exists(xml_file):
                try:
                    os.unlink(xml_file)
                except Exception:
                    pass

    if electron_cases:
        node = shutil.which("node")
        vitest = root / "node_modules" / "vitest" / "vitest.mjs"
        if node is None or not vitest.is_file():
            for c in electron_cases:
                results["NOT_RUN_ENVIRONMENT"].append({
                    "case_id": c.case_id,
                    "requirement": c.requirement,
                    "nodeid": c.nodeid,
                    "runner": "electron",
                    "reason": "node/Vitest unavailable for required Electron lifecycle contracts",
                })
        else:
            native_nodes = sorted({c.nodeid for c in electron_cases})
            native = subprocess.run(
                [node, str(vitest), "run", "--project", "electron", *native_nodes],
                cwd=root / "apps" / "desktop",
                capture_output=True,
                text=True,
            )
            for c in electron_cases:
                if native.returncode == 0:
                    results["PASS"].append({
                        "case_id": c.case_id,
                        "requirement": c.requirement,
                        "nodeid": c.nodeid,
                        "runner": "electron",
                    })
                else:
                    results["FAIL"].append({
                        "case_id": c.case_id,
                        "requirement": c.requirement,
                        "nodeid": c.nodeid,
                        "runner": "electron",
                        "error": (native.stderr or native.stdout or "").strip()[:500],
                    })

    return results


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    results = run_work100(execute=args.run)

    pass_count = len(results["PASS"])
    gap_count = len(results["COVERAGE_GAP"])
    env_count = len(results["NOT_RUN_ENVIRONMENT"])
    fail_count = len(results["FAIL"])

    print("=== Work100 Execution Summary ===", flush=True)
    print(f"{pass_count} PASS", flush=True)
    print(f"{gap_count} COVERAGE_GAP", flush=True)
    print(f"{env_count} NOT_RUN_ENVIRONMENT", flush=True)
    print(f"{fail_count} FAIL", flush=True)

    def _fmt(items):
        return [f"[Case {it['case_id']}] {it['requirement']} ({it.get('nodeid', it.get('reason', ''))})" for it in items]

    print("\n--- Detailed Status ---", flush=True)
    print("PASS:", json.dumps(_fmt(results["PASS"]), indent=2), flush=True)
    print("COVERAGE_GAP:", json.dumps(_fmt(results["COVERAGE_GAP"]), indent=2), flush=True)
    print("NOT_RUN_ENVIRONMENT:", json.dumps(_fmt(results["NOT_RUN_ENVIRONMENT"]), indent=2), flush=True)
    print("FAIL:", json.dumps(_fmt(results["FAIL"]), indent=2), flush=True)

    if args.run:
        return 1 if (fail_count > 0 or gap_count > 0 or env_count > 0) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
