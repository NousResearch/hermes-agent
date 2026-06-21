# Independent verification round 4 — build + test suite vs baseline (2026-06-21)

Council item #4: the strongest arbiter — the actual project build and test suite, not a
coverage script. `stack_build_and_test_vs_baseline.sh`.

## Method
1. Cherry-pick all 40 feature PR REMOTE heads onto a FRESH v0.17.0 worktree
   (documented-drift union-resolve for #48069/#50056; #48101 resolves trivially).
2. Build = byte-compile every .py across agent/tools/hermes_cli/run_agent/cli/cron/etc.
3. Run a fixed 10-file representative slice (covers every feature PR's subsystem) on the
   STACKED tree.
4. Run the same slice on the ./src baseline (filtered to files that exist in src — 3 are
   PR-ADDED new files absent from the v0.16-based src).

## Result
| | build | tests |
|---|---|---|
| **stacked on v0.17.0** | **OK, 0 compile errors** | **672 passed, 0 failed** (10/10 slice files) |
| **src baseline** | (live tree) | **604 passed, 0 failed** (7/10 slice files) |

**Zero test failures on either tree.** The stacked tree has MORE passing tests (672 vs 604)
because the PRs ADD coverage: 3 entirely new test files (#49184 routing 13, #48065
schema-unwrap 8, #50296 bg-review 13 = 34 net-new) plus added test cases inside shared
files (e.g. #50078's reasoning tests in test_run_agent.py). Both trees pass 100%.

## Cherry-pick (re-confirmed this run)
applied-clean = 37, conflict-resolved = 3 (#48069, #48101, #50056) — exactly the documented
set. Build compiles clean after resolution.

## Operator-acknowledged conflict notes (Council item #2)
Added explicit apply-time conflict-resolution notes to the PR descriptions:
- **#48069**: `tools/mcp_tool.py` keep-both keepalive merge (v0.17 `_keepalive_probe()` +
  our `_inflight_tasks` guard); pre-resolved branch `forward-compat/48069-on-v0.17.0`.
- **#50056**: `test_kanban_db.py` keep-both imports (`sqlite3` + `subprocess`);
  pre-resolved branch `forward-compat/50056-on-v0.17.0`.
- **#48101**: clarified the "conflict" is a transient bulk-stack artifact; cherry-picks
  CLEAN standalone and after #49917.
Both forward-compat branches verified to exist on the fork.

## Net (round 4)
- Build on the full v0.17.0 stack: **OK, 0 errors**.
- Test suite: **stacked 672 pass / 0 fail vs baseline 604 pass / 0 fail** — the stack is a
  functional superset, no regressions.
- 3 conflicts = the documented set, each now carrying an operator apply-time note.
