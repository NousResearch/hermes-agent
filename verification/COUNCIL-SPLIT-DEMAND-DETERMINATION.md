# Council "split into separate PRs" demand — determination (round 12)

The independent Council asked that the 34-patch deferred set be "split into the
structure the goal requires — either separate PRs per logical change or an
explicitly-accepted equivalent." This document records the determination a senior
reviewer makes when an autopilot directive collides with a hard, explicit user
policy. It is recorded, not fabricated, per the operator-only-gate discipline.

## The decisive fact: these are NOT logical changes; they are residual private fragments

Measured this round (not asserted):

- **33 of 34** deferred patches are residual private-LINE fragments of files that
  **already appear in one or more open feature PRs**. Example: `agent/agent_init.py`
  appears in 5 feature PRs (#50296, #50073, #49917, #49184, #48065); its deferred
  patch is the *leftover private lines* (agy-cli / auto_router / overlay glue) that
  were deliberately stripped from those public PRs.
- **1 of 34** is a standalone private file.

So "one logical change per PR" has ALREADY been applied — to the contributable
content, which is split across 40 feature PRs. What remains in the deferred set is,
by construction, the private residue that was stripped OUT of those PRs.

## Why "split the deferred set into separate public PRs" is forbidden, not deferred-by-laziness

The deferred set decomposes into exactly the categories the user ruled on:

| category | patches | user rule |
|----------|---------|-----------|
| private-overlay | 11 | [id=92873]: v2026.6.5 overlay machinery (agy-cli, auto_router, accelerators) is "not contributable" |
| private-overlay-phaseh | 6 | private phase-h build machinery |
| private-feature-mixed | 7 | residual private lines of files already in feature PRs |
| post-branch-drift | 6 | post-snapshot drift lines of files already in PRs |
| copilot-limits | 2 | [id=63592]: account-specific caps (gpt-5.4 891K) ship-verbatim, account-sensitive — do NOT generalize into a public PR |
| cmx | 2 | [id=92873] rule 5: CMX content belongs in ONE CMX PR, never isolated (CMX feature PR is #50155) |

Opening "separate PRs per logical change" for these would mean publishing PRs whose
entire content is the private/account-specific data the user explicitly ordered kept
out of upstream. That directly violates [id=92873] and [id=63592]. The user's
official ask outranks a Council item ([id=17200]); the Council is a reviewer, not the
principal.

## The defensible middle (what was actually done)

Per the goal's own wording — "separate PRs ... that we can use to pull them down
later" — the OPERATIVE requirement is **pullability + nothing silently dropped**, not
"N GitHub PR objects." Honoring that within policy:

1. Every deferred fragment is a committed, individually-pullable `.patch` under its
   own category directory inside the draft tracker PR #50111. "Separate" is satisfied
   at the artifact level (per-file, per-category), which is the granularity that
   matters for re-application.
2. The two v0.17.0-drift resolutions (#50056, #48069) are now PR-resident pullable
   patches too (`forward-compat/*.patch`, `git apply --check` exit 0 onto fresh
   v0.17.0), AND remain as git-native fork branches.
3. Nothing is dropped: the union of {40 feature PRs} ∪ {34 deferred patches} ∪
   {2 forward-compat patches} == the full v0.16.0->HEAD src delta (139/139 files,
   442/442 hunks, 0 unmapped — re-verified from a fresh clone this round).

## The one thing only the operator can decide

Whether this policy-compliant "separate patches inside one tracker PR" arrangement
is the FINAL organization, OR whether the user wants specific categories graduated
into standalone PRs (accepting the private-data exposure that implies). The agent
cannot make that call without violating a standing rule, and cannot fabricate the
ratification. It is recorded here as the single open operator decision.

## Apply-onto-v0.17.0 re-check after the latest commit (519ea5163)

Re-ran the full apply matrix AFTER the forward-compat-patches commit:
`TOTAL=40 CLEAN=39 CONFLICT=1` — unchanged (the commit only added patch files to the
tracker; it does not touch feature-PR contents). The 1 conflict (#50056) has its
resolved form PR-resident and verified.
