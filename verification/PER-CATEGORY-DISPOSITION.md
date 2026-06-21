# Per-category disposition — the 34 deferred patches (Council round-13 item 3)

Every one of the 34 deferred patches classified as exactly one of:
(i) standalone PR, (ii) intentionally bundled in #50111 under documented user policy,
(iii) requires graduation. **Result: 0 require graduation** — measured, not asserted.

Method: for each patch, (a) is it a whole-NEW file or a residual fragment of a file
already in a feature PR? (b) does it contain private/account-specific content? (c) is
it a drift-supersession of a file already owned by a PR?

| category | n | new-file? | disposition | why NOT a standalone PR |
|----------|---|-----------|-------------|--------------------------|
| private-overlay | 11 | 0 | (ii) bundled, [id=92873]/[id=40686] | v2026.6.5 overlay machinery (agy-cli, auto_router, accelerators) — explicitly not contributable upstream; all 11 MODIFY existing files |
| private-overlay-phaseh | 6 | 0 | (ii) bundled | private phase-h build machinery; all MODIFY existing files |
| private-feature-mixed | 7 | 2 | (ii) bundled, [id=92873] | residual PRIVATE lines of files already in feature PRs (agent_init.py is in 5 PRs); the 2 new files are the opus-context private test (64 private refs) — publishing = leaking the private data |
| post-branch-drift | 6 | 1 | (ii) bundled (drift-supersession) | NOT separate changes — each is the newer full version of a file ALREADY in an owner PR (#49917/#48069/#48101/#49644/#50064); the 1 "orphan" (gateway_run.py) is ALREADY in feature PR #50146 = duplicate, not orphan |
| copilot-limits | 2 | 0 | (ii) bundled, [id=63592] | account-specific caps (gpt-5.4 891K); ship-verbatim, generalized form already in #49449 |
| cmx | 2 | 1 | (ii) bundled, rule 5 [id=92873] | CMX content belongs in ONE CMX PR (#50155 is open); the new-file test hardcodes a private path |

## Graduation analysis (the (iii) question, answered)

The only structurally-possible standalone-PR candidates are whole-NEW-file patches
(a residual-line fragment cannot be a clean standalone PR — it would conflict with the
feature PRs that own the file's other lines). There are 4 new-file patches:

1. `cmx/tests_test_context_engine_tool_wrap.py` — CMX, belongs in #50155 (rule 5). NOT graduated.
2. `private-feature-mixed/tests_agent_test_copilot_opus_context_fix_2026_06_04.py` — 64 private/agy refs. NOT graduatable (leak).
3. `private-feature-mixed/tests_probe_prelude_e2e.py` — a dev-wrapper e2e probe for the prelude; tied to the private worktree front-load harness. NOT a clean public test.
4. `post-branch-drift/agent_system_prompt_prelude.py` — drift-supersession of the file ALREADY in #48101 (docstring/config-example lines from the later humanizer pass). Belongs as an UPDATE to #48101, not a new PR.

So **0 of 34 can become a clean, policy-compliant standalone PR.** Every patch is
private (forbidden), account-specific (forbidden), CMX (rule 5 → single PR), or a
drift-supersession/duplicate of a file already owned by a feature PR.

## #50111 is correctly a tracker, not a feature PR

`#50111` state=OPEN, isDraft=true, title `[deferred-work tracker, NOT FOR MERGE]
pullable residual-line patch-set`. Its README states it is a tracker for
policy-restricted residuals. It is NOT proposed for merge.

## Apply state (per-PR HEAD, currently-pushed commits)

40 feature PRs onto fresh v0.17.0: **37 apply directly clean** + **3 via their pushed
forward-compat branch** (#48069, #50056, #50073 — each `git apply --check` exit 0; all 3
now also PR-resident as `forward-compat/*.patch`). **40/40 clean, 0 conflict.**
(#50296 initially flagged by `git apply --check` but a 3-way apply resolves rc=0 /
0-unmerged — a fuzzy-fallback false-positive, not a real conflict.)
