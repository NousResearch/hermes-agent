# Phase 4 baseline reconciliation ledger — P4.0

First-run evidence: `/tmp/evidence-spine-p34/baselines/p4-first-run.txt`
Foundation first-run evidence: `/tmp/evidence-spine-p34/baselines/foundation-first-run.txt`

Base: `90bbba7be8538f48cd350ee2ec4e3c95d2467145`
Phase 4 selector at base: 192 collected — 178 passed / 14 failed, all 14 in
`tests/test_phase_d_audit_wiring.py` (reproduces the independently reported input exactly).

Classification policy per handoff §6 P4.0: `FIX_CODE` only when source violates the
current contract; `FIX_TEST` for stale assertions; weakening of the executable
`decompose-tasks.json` manifest contract and the execute/pr+qa evidence gates is
prohibited (no SCRAP used to make an optional subsystem mandatory).

## Failure classifications

| # | Test node | Fresh failure | Source seam | Classification | Reason | Minimal correction |
|---|-----------|---------------|-------------|----------------|--------|--------------------|
| 1 | TestDecomposeAndDocumentGates::test_decompose_gate_passes_with_ws1_sections | AssertionError (gate returns manifest error) | `feature_pipeline.validate_decompose_artifact` | FIX_TEST | Test writes only `decompose-output.md`; the current contract additionally requires the `decompose-tasks.json` sidecar (handoff: must not weaken decompose-tasks.json contracts). | Test also writes a valid manifest. |
| 2 | TestPassThroughStages::test_execute_and_pr_qa_in_gate_functions | AssertionError | `feature_pipeline.GATE_FUNCTIONS` | FIX_TEST | Legacy prose expectation; current locked contract states "Every execution-bearing stage has an evidence gate" — execute/pr+qa must HAVE gates. | Assert the opposite: execute and pr+qa ARE gated. |
| 3–6 | TestDecomposeChildTasks::test_parser_accepts_all_gate_heading_variants[4 variants] | AttributeError: no `_parse_decompose_children` | `kanban_db._create_decompose_child_tasks` | FIX_TEST | Prose-parsing function deliberately removed; markdown is never parsed into task rows (documented design decision in `_create_decompose_child_tasks`). | Replace with manifest-based creation test. |
| 7 | TestDecomposeChildTasks::test_parser_returns_empty_on_missing_artifact | AttributeError | same | FIX_TEST | Same. | Assert `_create_decompose_child_tasks` raises/returns empty for missing parent, and manifest load fails cleanly on missing dir. |
| 8 | TestDecomposeChildTasks::test_create_children_returns_empty_on_missing_parent | ValueError raised | same | FIX_TEST | Current contract raises ValueError (defensive hard error), which is correct behaviour; test asserted legacy silent-empty. | Expect ValueError. |
| 9 | TestDecomposeChildTasks::test_create_children_links_to_parent | Called with only `decompose-output.md` fixture | same | FIX_TEST | Needs executable manifest with key/title/owner/role/body/dependencies + parent match. | Write full manifest fixture (scratch workspace). |
| 10 | TestDispatcherNewStages::test_decompose_artifact_present_advances | pipeline_stage did not reach execute; also expected prose-derived children | `kanban_db.dispatch_once` + `_create_decompose_child_tasks` | FIX_TEST | Needs manifest; children titles come from manifest, roles validated; runtime child graph must have implementation task. | Provide manifest with implementation role; assert titles from manifest and tier inheritance. |
| 11 | TestDispatcherNewStages::test_decompose_children_not_duplicated_on_redispatch | Called `_create_decompose_child_tasks` with prose-only dir → error | same | FIX_TEST | Idempotence is via `decompose_children_created` event; needs manifest. | Provide manifest; call twice; assert same ids, 2 children. |
| 12 | TestDispatcherNewStages::test_passthrough_stage_execute_auto_advances | AssertionError (execute is now gated) | `feature_pipeline.GATE_FUNCTIONS` / `validate_execute_artifact` | FIX_TEST | execute requires execution-evidence.json per locked contract. | Provide valid execution evidence manifest (children done, digests) and materialised children; expect advance to pr+qa. |
| 13 | TestDispatcherNewStages::test_audit_conditional_creates_followup_and_advances | AssertionError: stage == 'spec' not 'final_sign_off' | `kanban_db.dispatch_once` audit branch + `_validate_pipeline_runtime_state` | FIX_TEST | `_validate_pipeline_runtime_state` requires a materialised decomposition child graph with done audit-role child before audit gate passes; test had no children. | Materialise children (decompose children event) with audit task done + runtime validation satisfied; then CONDITIONAL path runs. |
| 14 | TestDispatcherNewStages::test_audit_pass_advances_no_followup | same as 13 | same | FIX_TEST | Same reason. | Same fixture; PASS path runs, no followup. |

## What is NOT changed

- `decompose-tasks.json` executable manifest contract — unchanged.
- `validate_execute_artifact` / `validate_pr_qa_artifact` evidence gates — unchanged.
- `_validate_pipeline_runtime_state` canonical-state-over-artifact rule — unchanged.
- `feature_pipeline.GATE_FUNCTIONS` — unchanged.

All 14 nodes: source verified correct at exact base SHA; tests were written against
the pre-manifest prose pipeline and were never updated after the manifest contract
landed. No FIX_CODE, no DEFER, no SCRAP required.