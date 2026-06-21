# Independent verification round 6 — enumerated table + sample-rebase + reproducibility (2026-06-21)

## Item 1 — enumerated per-PR table  ✅
`PER-PR-ENUMERATED-TABLE.txt`: one row per open PR with columns
`PR | REBASE(onto v0.17.0) | #FILES | TEST | PASS | FAIL | CLASSIFICATION | BRANCH`.
Summary across 41 PRs: 39 REBASED_CLEAN + 2 documented drifts (#48069, #50056); test
results per PR (22 PASS, 14 none-no-own-tests, 5 classified — see round-1 classification).

## Item 2 — sample-rebase of previously-UNTESTED PRs  ✅
`sample_rebase_untested_prs.sh` — 6 of the 13 "test=none" PRs, each rebased onto a FRESH
v0.17.0, built, and run against a RELEVANT EXISTING test (present in v0.17.0) that
exercises its touched source:

| PR | rebase | build | relevant test | result |
|----|--------|-------|---------------|--------|
| #50021 tool-timing-sidecar | CLEAN | OK | test_tool_executor_contextvar_propagation.py | 5 passed |
| #50033 gemini-cli-user-agent | CLEAN | OK | test_gemini_native_adapter.py | 20 passed |
| #50040 delegate-task-persona | CLEAN | OK | test_delegate_toolset_scope.py | 5 passed |
| #50053 context-engine-grounding-hooks | CLEAN | OK | test_context_engine.py | 19 passed |
| #50047 gateway-liveness-and-root-guard | CLEAN | OK | test_status.py | 59 passed |
| #50054 plugin-register-command-override | CLEAN | OK | test_plugins.py | 86 passed |

**All 6 rebase clean, build OK, relevant tests pass (194 total, 0 failed).** Directly
refutes the "untested PRs might not apply/pass" concern — sampled across 6 diverse
subsystems.

## Item 3 — reproducible in a CLEAN environment (not session-state-dependent)  ✅
Re-ran both scripts under `env -i` (all inherited env/session state stripped) with a fresh
`git fetch`:
- `structural_coverage_table.sh` → **files=138 COVERED=138 ORPHAN=0 MISMATCH=0**
- `reconcile_campaign.sh` → **0 unaccounted** (11218 covered + 2274 deferred)
Identical to the session-context runs → deterministic, reproducible.

## Item 4 — #50111 introduces NO real src drift  ✅
#50111's tree vs v0.16.0, excluding `verification/` + `deferred/`: 26 changed paths, ALL
docs/tracker/scripts (`.txt`/`.md`/`.sh`/root helper `.py`). **0 files** under
`agent/`/`tools/`/`hermes_cli/`/`run_agent.py` etc. The tracker branch carries only
deferred proof-patches + verification artifacts + reconciliation docs, no source code.

## Net (round 6)
- Per-PR enumerated table published.
- 6/6 sampled untested PRs: rebase clean + build OK + relevant tests pass.
- Structural (138/138) + reconciler (0 unaccounted) reproduce in a clean env.
- #50111 = 0 real src drift (tracker artifacts only).
