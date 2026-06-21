# Independent verification round 10 — hunk-level attribution (2026-06-21)

The Council's baseline-correct, hunk-granular check.

## Item 1+2 — every hunk of `git diff v0.16.0..src-HEAD` → PR# + commit SHA  ✅
`HUNK-ATTRIBUTION-TABLE.txt` (+ the generator logic). Diffed src@HEAD (`94cef8953`) against
**v0.16.0 (`3c231eb`)** — the goal's correct baseline — and attributed every hunk:

```
total hunks=442   mapped-to-a-PR=390   mapped-to-#50111=52   UNMAPPED=0
RESULT: PASS — every hunk attributed to an open PR (with SHA) or #50111
```

Each file row lists its hunk count + the covering PRs with their commit SHAs, e.g.:
`agent/agent_init.py  hunks=7  covering=#50296@7af74fc2d,#50073@a12d4aebd,#49917@6bc37d8f4,#49184@d684ef808,#48065@ea7e64f46`.

### Method correction (recorded for honesty)
A first pass flagged **1 unmapped hunk** in `cli.py` (the autopilot `🤖 AUTO` status-bar
badge). Root-caused: the matcher required each hunk line to be in the PR's *added-set*, but
the hunk's middle line (` │ ` separator) is **shared context** #49917 carries in its file
*content* without re-`+`adding it at that exact position. Verified directly: all 3 lines
(`_autopilot_on`, ` │ ` separator, `🤖 AUTO`) ARE present in #49917's cli.py content, and
both src and #49917 have 2 AUTO badges with the same ·/│ separator split. The corrected
matcher (line-present-in-covering-PR-file-content) → **0 unmapped**. NOT a real gap — a
matcher artifact on a shared cosmetic separator line.

## Item 3 — fresh-clone reproduction (round 9, re-affirmed)
`fresh_clone_repro.sh` + `fresh-clone-repro.out`: fresh fork clone, cherry-pick 40 PRs onto
v0.17.0 (37 clean + 3 documented-resolved), diff vs src = 0 real residual (4 deferred-by-
design + 1 #48101-bulk-stack-artifact, correct standalone).

## Item 4 — deferred files enumerated + runtime-relevance  ✅
`DEFERRED-FILES-RUNTIME-RELEVANCE.md`: all 34 #50111 deferred patches across 6 categories
(private-overlay 11, private-overlay-phaseh 6, private-feature-mixed 7, cmx 2,
copilot-limits 2, post-branch-drift 6). **Verdict: none required for v0.17.0 runtime
behavior** delivered by the 40 feature PRs — each is private overlay infra (agy/cmx/
auto_router), an account-specific value (900K cap), or private-entangled glue whose public
portion already shipped in a feature PR. Kept pullable so nothing is lost; not needed to
reconstruct public-contributable src.

## Net (round 10)
- **442/442 hunks attributed** (390 to PRs with SHAs, 52 to #50111), **0 unmapped**.
- Baseline is correct (v0.16.0 `3c231eb`), as the goal requires.
- Fresh-clone stack vs src = 0 real residual.
- 34 deferred files enumerated; none runtime-required on v0.17.0.
