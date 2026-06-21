# Independent verification (Council round 2) — 2026-06-21

These checks use methods INDEPENDENT of the per-PR harness and the line-normalizing
reconciler, per the Council's "may be over-trusting the agent's own harness" gap.

## A. Independently reproduced: the 3 "pre-existing upstream" web_server failures
On a FRESH `git worktree` at pristine v0.17.0 (`2bd1977d8`, ZERO PRs applied):
```
python -m pytest tests/hermes_cli/test_web_server.py  ->  6 failed, 300 passed
```
The exact 6 failures (#50056/#50066/#50086 file) reproduce with no PR applied. Root
cause confirmed two ways: (1) pristine v0.17.0 `cron/scheduler.py` HAS
`cron_delivery_targets`, but the dev box's editable-install finder hard-maps
`cron -> src/cron` (which lacks it) → mid-suite `sys.modules` pollution; pre-loading
`cron.scheduler` from the pristine tree drops it to **1 failed, 305 passed**. (2) The
remaining 1 (`test_ticker_runs_when_desktop`) is a timing-flaky test that passes alone
and fails under full-file load — also on pristine v0.17.0. NEITHER is PR-introduced.

## B. Independent FILE-BY-FILE coverage proof (different method from reconcile_campaign.sh)
`independent_coverage_proof.sh` — for each of the 138 source files changed v0.16.0..HEAD,
checks every ADDED line is present (exact, lstrip) in some touching open-PR's file
content OR the #50111 deferred patch set. NO big-blob hashing.

**First run caught 4 lines the reconciler MISSED** in `agent/google_user_agent.py`:
src had bare `read_text()`/`write_text(...)`; PR #50033 refined both to
`encoding="utf-8"`. The reconciler's whitespace-normalizer collided the encoded/unencoded
variants; the per-file exact method did not. **Fixed** by aligning src to #50033
(commit `544d230a3`). Re-run: **PASS — 0 uncovered**. The original reconciler also stays
PASS — 0 unaccounted. Two independent methods now agree at 0.

## C. Cumulative composition onto v0.17.0 (real merge, not naive apply)
`independent_cumulative_verify.sh` PROOF A used naive `git apply` and flagged 3 cumulative
conflicts (#50073, #50296, #48069) — but naive apply cannot 3-way-merge overlapping
edits. A REAL cherry-pick stack of the 5 PRs that all touch the overlapping files
(`agent_init.py`/`run_agent.py`/`hermes_state.py`): #48065, #49917, #50073, #50296 all
cherry-pick **CLEAN**; only #50056 conflicts on the trivial `test_kanban_db.py`
import-combine (`sqlite3` + `subprocess`). So the genuine cumulative conflicts are exactly
the two documented ones (#48069 keepalive drift, #50056 import-combine); #50073/#50296
were naive-apply artifacts that merge cleanly.

## Verdict
- 3 web_server failures: independently reproduced on pristine v0.17.0 → upstream, not ours.
- Coverage: independent per-file method = **0 uncovered** (after fixing the 4 it caught).
- Composition: real-merge cherry-pick of overlapping PRs is clean except the 2 documented
  trivial/known drifts.
- Per-PR rebase (separate harness): 39/41 clean + the same 2 documented drifts.

Reproduce: `independent_coverage_proof.sh`, `independent_cumulative_verify.sh`,
and the pristine-v0.17.0 web_server reproduction in §A.
