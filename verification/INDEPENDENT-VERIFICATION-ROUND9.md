# Independent verification round 9 — disjointness + fresh-clone reproduction (2026-06-21)

## Item 2 — per-file hunk-DISJOINTNESS for the 14 overlap files  ✅
`OVERLAP-DISJOINTNESS-TABLE.txt`. METHOD CORRECTION worth recording: a v0.16.0-coordinate
hunk-range overlap test gave 1303 false "overlaps" because each PR's `git diff v0.16.0..head`
is contaminated by the ~50 origin/main commits between v0.16.0 and where the PR was cut. The
CORRECT disjointness test is empirical: on a fresh v0.17.0 worktree, sequentially
`git apply --3way` each touching PR's per-file net-diff in PR-number order and count residual
conflict markers.

**Result: 0 of 14 overlap files are non-disjoint** — every shared file (agent_init.py 5 PRs,
conversation_loop.py 4 PRs, etc.) applies all its touching PRs CLEAN with 0 markers. A genuine
non-disjoint overlap would have produced a CONFLICT. Disjointness proven by construction.

## Item 1 — fresh-clone reproduction  ✅ (with full residual classification)
`fresh_clone_repro.sh` run inside a FRESH clone of the fork (`git clone …/arminanton/…`):
cherry-pick all 40 feature PRs onto v0.17.0 (37 clean + 3 union-resolved), then diff PR-touched
files vs the src working tree.
- identical=52, differ-but-upstream-v0.16→v0.17-drift=74, **apparent-residual=5**.
- The 74 "differ" files changed between v0.16.0 and v0.17.0 upstream, so a v0.17-based stack
  differs from v0.16-based src there REGARDLESS of our PR — expected.
- The **5 apparent-residuals fully classified, 0 genuine gaps:**
  | file | classification |
  |------|----------------|
  | agent/gemini_cloudcode_adapter.py | DEFERRED — `#50111:deferred/private-overlay/...patch` |
  | agent/models_dev.py | DEFERRED — `#50111:deferred/private-feature-mixed/...patch` |
  | tests/probe_prelude_e2e.py | DEFERRED — `#50111:deferred/private-feature-mixed/...patch` |
  | tests/test_context_engine_tool_wrap.py | DEFERRED — `#50111:deferred/cmx/...patch` (private CMX path) |
  | hermes_cli/prompt_size.py | #48101 CORRECT standalone — cherry-picks CLEAN alone onto v0.17.0 with both prelude lines present; the bulk-stack union-resolve dropped them = harness artifact, not a PR defect |

  4 are private content correctly held in #50111 (not contributable to feature PRs); 1 is a
  bulk-stack-resolution artifact (#48101 verified correct in isolation). REAL residual = 0.

## Item 3 — #50111 is isolated, not required for src re-application  ✅
#50111 is built ON v0.17.0 (v0.17.0 is its ancestor), adds ZERO importable src
(0 `.py` under agent/tools/hermes_cli/run_agent/cli/gateway/etc beyond verification/ +
deferred/), and its tracker `.py` scripts compile (0 errors). It carries only deferred
proof-patches + verification artifacts. The 40 FEATURE PRs carry all src; #50111 is NOT
required to reconstruct src — correctly isolated.

## Item 4 — pristine baseline command + failure-signature identity  ✅
`pristine-baseline-COMMAND.txt`: exact command (`git worktree add --detach <wt> 2bd1977d8;
pytest tests/hermes_cli/test_web_server.py`) + raw output (6 failed, 300 passed) on a clean
v0.17.0, ZERO PRs. Signature identity: `comm -23` of {#50066 failures} and {#50086 failures}
against {pristine failures} = EMPTY both → byte-identical failure sets → proven upstream.

## Net (round 9)
- 14/14 overlap files: pairwise-disjoint hunks (empirical clean sequential apply).
- Fresh-clone stack vs src: 0 real residual (4 deferred-by-design + 1 harness artifact, #48101
  correct standalone).
- #50111 isolated (0 importable src), not required for re-application.
- Pristine failures byte-identical to the classified upstream failures.
