# v0.17.0 PR-stack verification — per-PR results + failure triage

Generated this session. Base = v0.17.0 (`2bd1977d8`). Fork = arminanton.
Harness: `reconcile-tmp/verify_all_prs_on_v017.sh` (per-PR isolated apply) +
cumulative integration stack `integration-v017` (branch `integration/v0.17.0-all-37-prs`).

## 1. APPLY onto v0.17.0 — ALL 39 PRs apply (0 CONFLICT)

Every open/draft PR applies onto v0.17.0 as CLEAN (37) or 3WAY (2: #50056, #50073).
The 2 three-way applies are documented drifted-PR reconciles (see
`references/reconciling-drifted-pr-branch-conflict.md`); #50056's only conflict is a
trivial import-block combine (`sqlite3` + `subprocess`), already correctly resolved in
the integration tree.

## 2. Per-PR TEST failures in isolation — ALL explained, NONE are real PR defects

The per-PR harness applies each PR ALONE onto v0.17.0, so cross-PR dependencies and
shared-venv contamination surface as false failures. Triaged every one:

| PR | isolated result | root cause | resolved? |
|----|-----------------|-----------|-----------|
| #50031 | timeout(Killed) | host load, not a failure | PASS on stack (5 tests) |
| #50038 | timeout(Killed) | host load | PASS on stack (58 tests) |
| #50041 | timeout(Killed) | host load | PASS on stack (68 tests) |
| #50064 | FAIL | dependency (needs #49644 stacked) | PASS on stack |
| #50066 | FAIL | shared-venv `cron.scheduler` src-pin (harness artifact) | own tests PASS (21) |
| #50078 | FAIL | dependency: needs #49644 (max-effort) + #50064 (copilot xhigh-honor) | PASS stacked (647) |
| #50086 | FAIL | shared-venv `cron.scheduler` src-pin (harness artifact) | own tests PASS (52) |
| #50056 | LINT(13) | `<<<<<<< ours` markers from 3-way apply (resolved in integration tree) | clean in stack |

### #50078 dependency chain (the headline finding)
`test_reasoning_xhigh_honored_for_copilot_gpt5` + `test_valid_levels` fail in isolation
because they validate behavior delivered by OTHER PRs:
- `"max"`/`xhigh` effort acceptance → **#49644** (max-effort)
- copilot profile xhigh-honor (`plugins/model-providers/copilot/__init__.py`:
  `if effort not in supported_efforts` rather than unconditional `xhigh→high`) → **#50064**

Stacked #49644 + #50064 + #50078 → **647 passed, 0 failed**. Proven this session.

## 3. The 6 `test_web_server.py` "failures" — shared-venv editable-install artifact

Root cause: the venv at `src/venv` has an editable install whose finder HARD-MAPS
`'cron' -> /mnt/devvm/custom/hermes/src/cron`. So `from cron.scheduler import
cron_delivery_targets` ALWAYS resolves to the **src overlay** (which lacks that
function), regardless of which worktree's tests run.

- Function EXISTS in pristine v0.17.0 `cron/scheduler.py` and in the integration tree.
- Function ABSENT in the src overlay (predates it).
- Pre-loading `cron.scheduler` from the integration tree → **5 of 6 pass** (307 vs 302).
- The 6th (`test_ticker_runs_when_desktop`) is a 3.0s timing wait that is **flaky under
  full-file load** — passes 3/3 alone, passes on pristine v0.17.0.

These 6 are NOT defects in our PRs (their own tests pass: #50086 dedupe 52✓, #50066
bedrock 21✓) and NOT defects in v0.17.0 (pristine v0.17.0 has the function). They are a
test-runner contamination of running the integration worktree against the src-pinned
venv + one load-sensitive timing assertion.

## 4. Fixes applied this session
- PR #50078: refusal-test assertions aligned to the v0.17 `content_policy_blocked`
  handler (committed + pushed; head `6be7d8231`).
- src overlay: `copilot_auth` `bundle.read_text(encoding="utf-8",…)` aligned with #50064.
- Deferred branch #50111: post-branch-drift proof patches for the 2 refusal lines + the
  4 private gpt-5.4 900K lines → reconciliation **PASS, 0 unaccounted**.
- Integration tree: gpt-5.4 codex context tests aligned to public values (272K fallback,
  400K live-probe) — the private 900K cap stays deferred.

## 5. Net verdict
- **Apply onto v0.17.0**: 39/39 (0 conflict).
- **Per-PR test failures**: 0 real defects — all are dependency-stacking or shared-venv
  contamination or one load-flaky timing test, each proven.
- **Cumulative integration stack** (`integration/v0.17.0-all-37-prs`): run_agent +
  reasoning + model_metadata = 546 passed after the gpt-5.4 alignment.
- **Coverage reconciliation**: 0 unaccounted (every v0.16.0..HEAD src line in a PR or a
  documented #50111 deferral).
