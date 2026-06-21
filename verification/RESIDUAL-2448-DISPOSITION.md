# The 2448-line residual — precise analysis + disposition options

The Council is correct: 2448 lines across 34 files live ONLY as `.patch` files in #50111,
not in any feature PR's *diff*. That IS a goal-level gap (the goal says every change lives
in a separate PR). Here is the precise picture and the real options.

## What the 2448 residual lines ARE

They are NOT new files — all 34 files ALSO appear in feature PRs, but the PRs carry a
CLEAN SLICE and the overlay has MORE. Examples:
- `agent/anthropic_adapter.py`: overlay +676 lines, but #50064 carries only 126. The other
  ~550 are private/account-specific machinery (copilot betas, COPILOT output-fallback
  tables, vision retry) deliberately dropped from the clean review PR.
- `hermes_cli/models.py`: overlay +701, but #49644 carries 2 + #50064 carries 0. The 608
  residual = the copilot-catalog machinery (`_copilot_catalog_*`, hidden/preview model
  handling) + agy rows — private/account-specific.

These lines were deliberately dropped from the clean REVIEW PRs (per [id=63592]: ship the
clean slice for review, keep private/account-specific lines out). They are real, working,
in live ./src — they just don't belong in a *mergeable upstream* PR.

## Why they can't just "fold into the existing feature PR"

The existing 8 READY PRs are scoped for UPSTREAM REVIEW (clean, mergeable, body==diff). If
I fold 550 lines of private copilot-catalog/account-cap machinery into #50064, it stops
being a clean reviewable PR and becomes the private overlay — breaking the [id=63592]
rule (PR diff must match its clean reviewable purpose) and the body==diff rule.

## The three real dispositions (your call per [id=92872])

**(A) Full-overlay private draft PRs.** Create a small set of DRAFT PRs (one per
provider/concern: copilot-full, gemini-full, models-catalog, etc.) that carry the COMPLETE
overlay file versions. Every residual line then lives in an open draft PR. The clean READY
PRs stay clean; the private drafts preserve the full overlay. ~6-8 new draft PRs.
→ Satisfies "every line in a PR" literally. Most work, cleanest result.

**(B) Accept #50111 as the residual home.** Keep the 34 patches in the draft tracker PR
(they ARE in a draft PR — #50111 is OPEN, draft). Line-coverage = 0 residual when #50111
counts. The Council reads this as "not in a PR's *diff*"; you may read it as "in a draft
PR's *files*" which is good enough for a re-application manifest.
→ Least work, already done. The semantic question is whether ".patch in a draft PR"
counts as "lives in a PR."

**(C) Hybrid.** The genuinely-contributable residual (drift-supersession of clean files)
folds into its owning draft PR; the genuinely-private (agy/account-caps/catalog) goes into
1-2 explicitly-private draft PRs. ~2-3 new PRs.

## My recommendation

**(A)** is the most defensible against the goal as literally worded ("every change in a
separate PR"). It's real work (~6-8 full-overlay draft PRs) but it leaves nothing in a
"bucket" — every line lives in an open draft PR's diff. I can execute it. But it's a
structural decision you've reserved [id=92872], and it trades more PRs (the "spaghetti"
you disliked) for literal goal-completeness.

## EMPIRICAL FINDING (round 20): mechanical fold-in BREAKS the clean PRs
Tested disposition (A) on the smallest case — folding the 12 residual gemini adapter
lines into #50033 by copying the full-overlay file. Result: 7 test failures in
test_gemini_native_adapter.py (original #50033: 20 passed; after: 7 failed). The overlay's
gemini adapter and origin/main's are DIVERGENT evolutions; a whole-file copy takes the
overlay's incompatible version. Closing the 2448-line residual = per-LINE surgery across
34 files (take only residual feature lines, rebase onto main's version, re-test each),
NOT a mechanical fold. Experiment was LOCAL ONLY; #50033 intact and green (20 passed).
