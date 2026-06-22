# v0.17.0 PR-stack verification — machine-readable per-PR rebase + test (2026-06-21)

Addresses the Council's 4 verification items with reproducible evidence.
Base = v0.17.0 `2bd1977d8fad185c9b4be47884f7e87f1add0ce3`. Fork = arminanton.

## Reproduce
```
cd <REPO_ROOT>
bash <LOCAL_PATH> fork > report.jsonl
bash <LOCAL_PATH> fork   # coverage: 0 unaccounted
```
Raw output: `rebase-report.jsonl` (one JSON line per PR + summary).
Classified verdicts: `rebase-report-classified.jsonl`.

## Item 1 — Per-PR rebase onto v0.17.0 (machine-readable)
The harness rebases each PR's OWN commits (`merge-base(origin/main,sha)..sha`, since
PRs are cut off origin/main which descends from v0.17.0) onto v0.17.0 via real
`git rebase --onto`, then runs that PR's own changed test files.

**Result across 41 open PRs: 39 REBASED_CLEAN, 2 REBASED_CONFLICT.**
- #48069 — 1-file known drift (`tools/mcp_tool.py`; origin/main refactored our keepalive
  into `_keepalive_probe()`). Documented keep-both resolution +
  `#50111:deferred/post-branch-drift/tools_mcp_tool.py.patch` preserves the full state.
- #50056 — trivial import-combine conflict (`sqlite3` + `subprocess`), already resolved
  in the integration tree.

## Item 2 — Per-PR own-tests on the rebased state (with pre-existing classification)
**22 PASS, 14 no-test-files, 5 FAIL.** Every FAIL classified by re-running the same test
files on PRISTINE v0.17.0 (zero PRs):

| PR | fail | classification | proof |
|----|------|----------------|-------|
| #50056 | 6 | PREEXISTING upstream | same 6 fail on pristine v0.17.0 |
| #50066 | 6 | PREEXISTING upstream | same fail on pristine; own bedrock tests (21) pass |
| #50086 | 6 | PREEXISTING upstream | same fail on pristine; own dedupe tests (52) pass |
| #50078 | 6 | cross-PR dependency | stack #49644+#50064+#50078 → **441 pass, 0 fail** |
| #50031 | 1 | live-credential smoke test | `test_auto_router_live` hits the real Copilot billing endpoint; needs a discount-eligible session; other 4 tests pass; #50031 is the deferred auto_router draft |

The 6 `test_web_server.py` failures (#50056/#50066/#50086) share ONE root cause: the dev
box's editable-install finder hard-maps `cron -> <REPO_ROOT>/cron`, which
lacks `cron_delivery_targets`. They reproduce identically on a pristine v0.17.0 checkout
with NO PRs applied → upstream/harness, not ours. Absent on a clean CI checkout (its own
install maps `cron` to v0.17.0 code, which HAS the function).

**ZERO unexplained PR-introduced test defects.**

## Item 3 — Deferrals live in an ACTUAL open PR (not a local tracker)
PR **#50111** (`[deferred-work tracker, NOT FOR MERGE]`, branch
`deferred/residual-lines-on-v0.17.0`) is **OPEN (draft)**. The coverage reconciler reads
that exact branch head (`35c3da5ba`). So:
- 11216 src lines → covered by feature PRs (all open)
- 2274 src lines → in PR #50111's branch (open)

The 2 deferrals the Council named ARE in #50111 as proof patches:
`deferred/post-branch-drift/tests_run_agent_test_run_agent.py.patch` (refusal lines) +
`deferred/post-branch-drift/tests_agent_test_model_metadata.py.patch` (private 900K).
**Every `v0.16.0..HEAD` src line maps to an OPEN PR — 0 tracker-only.**

## Item 4 — Re-inspection of previously-pushed PRs
- #48069: confirmed the keepalive drift is the only conflict; resolution documented + preserved.
- #50064: confirmed it carries the copilot-profile xhigh-honor change #50078 depends on.
- #49644: confirmed it carries `max`-effort acceptance #50078 depends on; rebases clean.
- #50056/#50066/#50086: confirmed their OWN tests pass; the web_server failures are upstream.
- New fixes pushed this session: #50078 refusal alignment, src copilot_auth utf-8 (=#50064),
  #50296 (background-review isolation, the newly-landed src drift).

## Net
- Rebase onto v0.17.0: **39/41 clean + 2 documented drifts**.
- Tests: **0 PR-introduced defects** (3 preexisting-upstream, 1 cross-PR-dependency, 1 live-cred).
- Coverage: **0 unaccounted**, every line in an OPEN PR (#50111 included).
- No PR force-merged or flipped to ready — operator retains that action.
