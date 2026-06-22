# Council verification round-22 — items 2,3,4 (current-state, recorded)

## Item 3 — #50111's 34 patches current + dry-apply onto v0.17.0
CAUGHT A REAL BUG: 1 patch (post-branch-drift/tests_run_agent_test_run_agent.py) was
CORRUPT/STALE (17 lines, corrupt at line 18) and FAILED v0.17.0 apply. Root cause: an
earlier bulk-regen used a flatten heuristic that mis-mapped filenames with internal
underscores (run_agent -> run/agent). FIX: regenerated every patch from its HEADER target
path (ground truth). RESULT: all 34 now dry-apply --3way onto v0.17.0 = 34/34 clean.
(commit 2d008a767)

## Item 2 — clean review PRs apply + CI on v0.17.0
- Per-PR 3-way apply onto v0.17.0 HEAD (2bd1977d): 40/40 CLEAN, exit 0 (37 direct + 3 fwd-compat).
- CI: all 8 READY PRs show checks=0 / mergeable=UNKNOWN — fork PRs get NO gated upstream CI
  until a maintainer approves (documented constraint). Local per-head test substitute:
  299 passed, 0 failed across the 8 READY PRs (round 17 evidence, unchanged).

## Item 4 — coverage union(clean PRs + #50111 patches) vs full src delta
  full v0.16->HEAD changed-lines : 11678
  covered (PRs ∪ #50111)         : 11678
  MISSING                        : 0
  DUPLICATED (in both)           : 1988  <- the deferred patches are FULL-FILE diffs that
                                            re-include lines the clean PR already carries
  feature-PR-only lines          : 11058 (some shared with deferred)
  genuinely deferred-only lines  : 2448  (the residual: in NO feature PR diff)

So: 0 missing, full coverage. The 2448 deferred-only lines are the residual the operator
must rule on (A vs B). The 1988 duplicated lines are a cosmetic artifact of the deferred
patches being whole-file diffs, not disjoint slices — harmless for pull-down (git apply
is idempotent on already-present lines), but it confirms the deferred patches OVERLAP the
clean PRs rather than being pure-residual.

## State: 42 open PRs (41 feature + #50111), 0 closed-in-error, 40/40 apply clean,
0 missing coverage, 34/34 deferred patches now dry-apply onto v0.17.0.
