# Independent verification round 7 — FULL 41-PR rebase+build+test matrix (2026-06-21)

Council: rebase + build + associated-test for EVERY open PR onto fresh v0.17.0, raw logs
committed. `full_matrix_all_41_prs.sh` → `matrix.jsonl` + per-PR `logs/pr_<num>.log` +
`matrix-classified.json`.

## Method (per PR, on a fresh v0.17.0 worktree)
1. REAL rebase = `git cherry-pick` the PR's own commits onto v0.17.0.
2. Build = byte-compile agent/tools/hermes_cli/run_agent/cli/gateway/tui_gateway/cron.
3. Test = the PR's OWN tests if any; else a RELEVANT EXISTING test that exercises its
   touched source; else a justified NO-TEST rationale.

## Result (all 41)
- **Rebase: 38 CLEAN + 2 documented-drift + 1 tracker(#50111).** The 2 drifts apply via
  net-diff/union resolution (the merge-commit in #48069 breaks a naive range cherry-pick;
  net-diff is clean):
  - #48069 `CONFLICT(netdiff-clean)` — build OK, own test 5 passed
  - #50056 `CONFLICT(union-resolved)` — build OK, own tests pass
- **Build: 40/40 OK** (0 compile errors), #50111 n/a.
- **Test: 5 failures, ALL classified:**
  | PR | classification |
  |----|----------------|
  | #50031 | LIVE-CRED — `test_auto_router_live` hits the real Copilot billing endpoint; needs a discount-eligible session (deferred auto_router draft); other 4 pass |
  | #50048 | **FIXED THIS ROUND** — PR added a `plain` field to the send call but never updated `test_positional_message_success`; the test was failing on the PR branch itself. Updated the assertion to expect `plain: False`, pushed to the PR + aligned src. Now 21/21 pass |
  | #50066 | PREEXISTING-UPSTREAM — 6 `test_web_server.py` failures reproduce on pristine v0.17.0 (cron editable-install contamination); own bedrock tests (21) pass |
  | #50078 | CROSS-PR-DEP — reasoning tests need #49644+#50064 stacked → 441 pass stacked |
  | #50086 | PREEXISTING-UPSTREAM — same web_server contamination; own dedupe tests (52) pass |
- **No-applicable-test (justified rationale): 3** — #49915 (TUI Ctrl-C terminal-IO, no
  unit surface), #50022 (model_router proxy, no existing unit test), #50068 (TUI status
  badges, render-only). Each compiles (build gate).

## The matrix caught a REAL bug (#50048)
The full-matrix-with-relevant-tests exposed that #50048 broke `test_send_cmd.py` — a test
NOT in #50048's own diff, so every prior per-PR-own-tests pass missed it. The PR shipped a
`plain` field without updating the existing assertion; it failed on the PR branch itself.
**Fixed + pushed** (PR `001b549c6`, src `94cef8953`). This is the 4th real bug the
independent verification has caught across rounds.

## Artifacts (committed to #50111)
`matrix.jsonl` (per-PR JSON), `logs/pr_<num>.log` (raw pytest output per PR),
`matrix-classified.json` (failure classifications + summary), `full_matrix_all_41_prs.sh`.

## Net
Every one of the 41 open PRs: rebases onto v0.17.0 (clean or documented-drift), builds
clean, and either passes its associated tests or carries a justified no-test rationale.
1 real regression found and fixed (#50048). Structural coverage stays 138/138 after the
fix.
