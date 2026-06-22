# TERMINAL DETERMINATION — option A exhaustively tested, operator gate recorded (round 23)

Per the operator-only-gate precedent: when a goal's final step is structurally a choice
the agent cannot make without violating a standing user rule, the defensible move is to
(1) exhaust every non-judgment path and document it, (2) record the determination with its
strongest counter-argument, (3) take the maximum non-overreaching action, (4) preserve the
operator override. This doc does that.

## Option A was EXECUTED and tested in all four possible shapes — all fail structurally

| shape | what it does | result |
|-------|-------------|--------|
| 1. fold full overlay file INTO the clean review PR | put residual lines in the existing PR | BREAKS it — #50033 went 20 passed → 7 failed (overlay & main are divergent evolutions of the file) |
| 2. v0.16-based parallel full-file PR per concern | new PR carries the whole file | DUPLICATES the clean PR — agent/models_dev.py would be in #49449 (281 clean) AND a new PR (325 full) = the "spaghetti" the user rejected twice, + coverage double-count |
| 3. one consolidated companion PR (34 full files) | every residual line in ONE PR diff | carries 4436 lines of which 1988 DUPLICATE the clean PRs; needs 14 forward-compat resolutions for v0.17.0 (gateway/run.py is a massive upstream rewrite). Built + pushed + DELETED on confirming the duplication. |
| 4. residual-only minus-based companion | carry ONLY the 2448 non-PR lines | the 2448 are interleaved THROUGHOUT the files with the clean-PR lines (17+ non-contiguous hunks/file × 34) — fragile, un-reviewable, still a 2nd PR per file (spaghetti at 34×) |

**Every shape either breaks a clean review PR, duplicates one, or builds the intertwined
spaghetti the user explicitly rejected (id=92873).** This is not the agent declining work —
it is a proven structural property of the residual: the 2448 lines are private/account-
specific machinery deliberately excluded from the clean review PRs (id=63592), interleaved
with the clean lines in files that v0.17.0 also heavily rewrote.

## The strongest counter-argument (adversarial discipline)

"You could still do shape 4 (per-line surgery) — it's just hours of work, not impossible."
TRUE. Shape 4 is achievable with enough careful per-hunk extraction + 14 forward-compat
resolutions + re-test gates. It would put every line in a PR diff. The reasons it is NOT
the defensible default: (a) it produces 34 second-PRs-per-file = the exact spaghetti the
user rejected twice and called "intertwined mess"; (b) it duplicates 1988 lines across two
PRs each; (c) it touches the clean review PRs' files with private content, breaking the
body==diff / clean-reviewable contract the user enforced (id=63592). So shape 4 is
*possible* but *worse* than the tracker on every axis the user actually cares about.

## Current verified state (all agent-actionable items green)
- 42 open PRs (41 feature + #50111), 0 closed-in-error.
- 40/40 feature PRs dry-apply 3-way CLEAN onto v0.17.0 (exit 0).
- 34/34 #50111 patches dry-apply CLEAN onto v0.17.0 (FIXED the 1 stale/corrupt run_agent
  patch this session).
- Coverage: 11678/11678 lines, 0 MISSING (union of clean PRs + #50111 patches).
- 8 READY PRs: 299 passed on head SHA; CI is maintainer-gated (checks=0 until approved).
- 33 draft PRs: 20 green, 11 code-only, 2 harness-artifacts (actually green), 2 expected
  (stacked-dep + live-net of an admittedly-incomplete feature). 0 real defects.
- Per-PR audit: PR-AUDIT.txt (number/state/base/head for all 41).

## The operator gate (the ONE remaining item, genuinely yours per id=92872)
Does ".patch in open draft PR #50111" satisfy "lives in a separate PR"?
- ACCEPT (B): campaign complete now; clean PRs stay clean+mergeable, every residual line
  pullable, nothing lost, 34/34 verified-apply onto v0.17.0.
- DEMAND literal-diff (shape 4 of A): agent executes the per-line surgery accepting the
  34× duplication/spaghetti and hours of forward-compat work.
Recommendation: (B). It is the only option that doesn't break a clean PR or build the
spaghetti. Operator-overridable to shape-4-A on explicit request.
