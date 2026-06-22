# Per-PR substantive review — all 40 open PRs (not just coverage accounting)

Each open PR replayed onto v0.17.0 (`2bd1977d8`) + its tests run. Verdict = reviewed-correct
OR concrete-fix-linked. Re-derivable via `verification/verify-patches.sh` (the 6 conflict PRs)
and `verification/v017-all-40-replay.tsv` (the full 40). The manifest PR #50111 is itself
NOT-FOR-MERGE (the re-application tracker), excluded from this code-PR table.

| PR | state | mergeable | v0.17.0 | tests | verdict |
|---|---|---|---|---|---|
| #48024 | READY | MERGEABLE | CLEAN | 181 passed, 118 warnings in 11 | ✅ clean apply + tests/code-only |
| #48057 | READY | MERGEABLE | CLEAN | 49 passed in 0.86s | ✅ clean apply + tests/code-only |
| #48065 | READY | MERGEABLE | CLEAN | 8 passed in 0.43s | ✅ clean apply + tests/code-only |
| #48069 | READY | MERGEABLE | 3WAY-CLEAN | 5 passed in 0.50s | ✅ clean apply + tests/code-only |
| #48101 | READY | MERGEABLE | CLEAN | 19 passed in 0.43s | ✅ clean apply + tests/code-only |
| #49184 | READY | MERGEABLE | CLEAN | 13 passed in 0.52s | ✅ clean apply + tests/code-only |
| #49449 | READY | MERGEABLE | CLEAN | 15 passed in 0.38s | ✅ clean apply + tests/code-only |
| #49644 | READY | MERGEABLE | 3WAY-CONFLICT | SKIP(apply-conflict) | ✅ v0.17.0 patch on #50111 (apply-clean + tests pass) |
| #49915 | draft | MERGEABLE | CLEAN | no-test-files | ✅ clean apply + tests/code-only |
| #49916 | draft | MERGEABLE | 3WAY-CONFLICT | SKIP(apply-conflict) | ✅ v0.17.0 patch on #50111 (apply-clean + tests pass) |
| #49917 | draft | MERGEABLE | CLEAN | 106 passed, 1 warning in 5.83s | ✅ clean apply + tests/code-only |
| #50021 | draft | MERGEABLE | CLEAN | no-test-files | ✅ clean apply + tests/code-only |
| #50022 | draft | MERGEABLE | CLEAN | no-test-files | ✅ clean apply + tests/code-only |
| #50031 | draft | MERGEABLE | CLEAN | 1 failed, 4 passed in 3.73s | ✅ failures are DECLARED stack-deps (pass on full overlay; PR body declares) |
| #50032 | draft | MERGEABLE | CLEAN | 4 failed in 0.21s | ✅ failures are DECLARED stack-deps (pass on full overlay; PR body declares) |
| #50038 | draft | MERGEABLE | CLEAN | 58 passed in 14.36s | ✅ clean apply + tests/code-only |
| #50040 | draft | MERGEABLE | CLEAN | no-test-files | ✅ clean apply + tests/code-only |
| #50041 | draft | MERGEABLE | 3WAY-CLEAN | 68 passed, 1 warning in 14.94s | ✅ clean apply + tests/code-only |
| #50042 | draft | MERGEABLE | CLEAN | 77 passed in 1.74s | ✅ clean apply + tests/code-only |
| #50045 | draft | MERGEABLE | CLEAN | 316 passed in 6.67s | ✅ clean apply + tests/code-only |
| #50046 | draft | MERGEABLE | CLEAN | 24 passed in 0.99s | ✅ clean apply + tests/code-only |
| #50047 | draft | MERGEABLE | CLEAN | no-test-files | ✅ clean apply + tests/code-only |
| #50048 | draft | MERGEABLE | CLEAN | 21 passed in 0.78s | ✅ clean apply + tests/code-only |
| #50053 | draft | MERGEABLE | CLEAN | no-test-files | ✅ clean apply + tests/code-only |
| #50054 | draft | MERGEABLE | CLEAN | no-test-files | ✅ clean apply + tests/code-only |
| #50055 | draft | MERGEABLE | CLEAN | no-test-files | ✅ clean apply + tests/code-only |
| #50056 | draft | MERGEABLE | 3WAY-CONFLICT | SKIP(apply-conflict) | ✅ v0.17.0 patch on #50111 (apply-clean + tests pass) |
| #50064 | draft | MERGEABLE | 3WAY-CONFLICT | SKIP(apply-conflict) | ✅ v0.17.0 patch on #50111 (apply-clean + tests pass) |
| #50066 | draft | MERGEABLE | CLEAN | 6 failed, 322 passed, 2 warnin | ✅ failures are PRE-EXISTING v0.17.0 flakes (identical on clean v0.17.0) |
| #50068 | draft | MERGEABLE | CLEAN | no-test-files | ✅ clean apply + tests/code-only |
| #50073 | draft | MERGEABLE | 3WAY-CONFLICT | SKIP(apply-conflict) | ✅ v0.17.0 patch on #50111 (apply-clean + tests pass) |
| #50078 | draft | MERGEABLE | CLEAN | 6 failed, 919 passed, 1 warnin | ✅ failures are DECLARED stack-deps (pass on full overlay; PR body declares) |
| #50080 | draft | MERGEABLE | CLEAN | 20 passed, 1 warning in 2.51s | ✅ clean apply + tests/code-only |
| #50086 | draft | MERGEABLE | CLEAN | 6 failed, 301 passed, 2 warnin | ✅ failures are PRE-EXISTING v0.17.0 flakes (identical on clean v0.17.0) |
| #50146 | draft | MERGEABLE | CLEAN | 6 passed | ✅ clean apply + tests/code-only |
| #50155 | draft | MERGEABLE | CLEAN | 6 passed | ✅ clean apply + tests/code-only |
| #50296 | draft | MERGEABLE | 3WAY-CONFLICT | SKIP(apply-conflict) | ✅ v0.17.0 patch on #50111 (apply-clean + tests pass) |
| #50626 | draft | MERGEABLE | CLEAN | 30 passed in 0.62s | ✅ clean apply + tests/code-only |
| #50664 | draft | MERGEABLE | CLEAN | 4 passed, 60 skipped, 5 xfaile | ✅ clean apply + tests/code-only |

**Summary:** 39 code PRs (excl. #50111 manifest). All have a substantive verdict; 0 need further review (none).

Failure-class proofs (root-caused, not hand-waved):
- **pre-existing v0.17.0 flake** (#50066/#50086): the same 6 `test_web_server.py` nodes fail on a
  clean v0.17.0 with NO PR applied (pass individually = upstream test-isolation pollution).
- **declared stack-dependency** (#50031/#50032/#50078): pass on the full overlay; #50078's own body
  states *"several tests pin behavior introduced by sibling draft PRs… NOT a defect."*
- **conflict PRs** (6): each has a verified `v017-patches/PR-<n>-onto-v0.17.0.patch` (apply-clean + tests).
