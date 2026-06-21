# Independent verification round 5 — structural per-file table + forward-compat branches (2026-06-21)

Council round-5: the explicit structural artifact (per-file src↔PR table) + verify the
forward-compat branches actually APPLY and PASS (not just exist).

## Item 1+2 — per-file structural coverage table  ✅
`structural_coverage_table.sh` — for EVERY file where src@HEAD differs from v0.16.0
(`3c231eb`), reads the covering PR's REMOTE head content and checks every src-added line
is present there (or in the #50111 deferred patch set). Full table in
`structural-coverage-table.txt`.

```
files=138  COVERED=138  DEFERRED-ONLY=0  ORPHAN=0  MISMATCH=0
RESULT: PASS — 0 orphan, 0 mismatch
```
Every one of the 138 src-changed files maps to ≥1 covering open PR with **0 uncovered
lines**. 0 orphan src changes; 0 mismatches. The table prints `file | covering_PR#(s) |
added | uncovered | verdict` per row.

## Item 3 — drift-detection re-run = 0 new findings  ✅
This per-line containment check IS the method that previously caught the `copilot_auth`
and `google_user_agent` `encoding="utf-8"` drifts. Both now show **COVERED, 0 uncovered**
(the fixes are in src AND match their PR heads #50064/#50033). No new drift found across
all 138 files.

## Item 4 — forward-compat branches APPLY cleanly AND pass  ✅ (caught + fixed a real bug)
Both `forward-compat/{48069,50056}-on-v0.17.0` are built directly on v0.17.0 (it's an
ancestor), have 0 conflict markers, and add 1 clean commit.

**The check caught a real staleness bug:** the OLD `forward-compat/48069` predated the
`getattr(server, "_inflight_tasks", None)` hardening that's in the current #48069 head, so
it FAILED 5 `test_mcp_structured_content.py` tests (the `SimpleNamespace` mock lacks
`_inflight_tasks`). **Fixed:** regenerated the branch from the current #48069 head
(net-diff 3-way apply onto v0.17.0 — clean; the merge-commit in the PR was what broke a
naive range cherry-pick), keep-both resolution of the complementary `__slots__`/`__init__`
overlap with v0.17.0's `_pending_call_context`. Re-verified:

| branch | built on v0.17.0 | conflict markers | tests |
|---|---|---|---|
| forward-compat/48069 | YES | 0 | **205 passed, 0 failed** |
| forward-compat/50056 | YES | 0 | **496 passed, 0 failed** |

## Net (round 5)
- Structural per-file table: **138/138 COVERED, 0 orphan, 0 mismatch**.
- Drift re-run: **0 new findings** (prior 2 bugs fixed, no new drift).
- Both forward-compat branches: **apply clean on v0.17.0 + tests green** (fixed a stale
  branch the check exposed).
- The independent method continues to earn its place — this round it caught and fixed a
  stale pre-resolved branch.
