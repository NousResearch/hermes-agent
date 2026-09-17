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


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    gaps = [{"case_id": c.case_id, "requirement": c.requirement} for c in SEED if c.nodeid is None]
    print(json.dumps({"seed_cases": len(SEED), "coverage_gaps": gaps}, indent=2), flush=True)
    if args.run:
        root = Path(__file__).resolve().parents[1]
        nodes = sorted({c.nodeid for c in SEED if c.nodeid and c.runner == "pytest"})
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", *nodes], cwd=root)
        native_nodes = sorted({c.nodeid for c in SEED if c.nodeid and c.runner == "electron"})
        native_code = 0
        if native_nodes:
            import shutil
            node = shutil.which("node")
            vitest = root / "node_modules" / "vitest" / "vitest.mjs"
            if node is None or not vitest.is_file():
                print("NOT_RUN_ENVIRONMENT: node/Vitest unavailable for required Electron lifecycle contracts", flush=True)
                native_code = 1
            else:
                native = subprocess.run([node, str(vitest), "run", "--project", "electron", *native_nodes],
                                        cwd=root / "apps" / "desktop")
                native_code = native.returncode
        return result.returncode or native_code or int(bool(gaps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
