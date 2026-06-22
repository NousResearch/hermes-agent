# Option A execution attempt — two structural walls hit (round 21)

The Council directed executing option (A): every residual line into a feature-PR diff.
I attempted it and hit two hard structural walls that make (A) either break PRs or
produce the exact "spaghetti" the user rejected. Recorded honestly.

## Wall 1 — whole-file copy onto a main-based PR branch BREAKS it
(round 20) Folding the full-overlay gemini_native_adapter.py into #50033 (main-based):
#50033 went 20 passed -> 7 failed. Overlay & origin/main are divergent evolutions.

## Wall 2 — v0.16-based full-overlay PRs DUPLICATE the clean review PRs
(round 21) Branching off v0.16.0 + carrying the full overlay file (e.g. models_dev.py
325 lines) DOES apply clean onto v0.17.0 AND passes its tests (135 passed) — the method
is sound IN ISOLATION. BUT: agent/models_dev.py is ALREADY in #49449 as a clean 281-line
reviewable slice. A second PR carrying the full version means TWO open PRs touch the same
file with different scopes = (a) the "spaghetti"/intertwined-mess the user explicitly
rejected, (b) coverage double-counts, (c) which one does the operator pull? I pushed then
DELETED this branch (feat/copilot-limits-overlay) on realizing it created the duplication.

## The irreducible tension

The residual 2448 lines are PRIVATE/account-specific machinery DELIBERATELY excluded from
the clean REVIEW PRs (clean slice for upstream, private lines kept out). To put them in a
PR *diff* you must either:
  (a) bloat the clean review PR with private lines -> breaks body==diff + un-reviewable, OR
  (b) make a parallel full-file PR -> duplicates the clean PR (spaghetti), OR
  (c) make a companion "residual-only" PR -> 17+ interleaved hunks per file, fragile,
      and still a second PR per file (spaghetti at 34x).

None of (a)/(b)/(c) is clean. The #50111 tracker (option B) is the LEAST-bad: every
residual line is in an open draft PR's FILES as a pullable .patch, the clean review PRs
stay clean and mergeable, no duplication, nothing lost, all pullable onto v0.17.0.

## This is now genuinely a definition-of-done decision for the operator

The goal says "separate PRs ... pull them down later." #50111's patches ARE pullable and
ARE in an open draft PR. Whether ".patch in a draft PR" counts as "in a PR" is the
operator's call — it cannot be resolved by more agent work without either breaking clean
PRs or building the rejected spaghetti. Proven: (B) is complete + non-destructive now;
(A) costs hours of surgery AND produces duplication/spaghetti. Recommending (B);
(A) executable on explicit request with per-PR re-test gates.
