# ADR 0007 — Skills tree: already in spec shape; one cleanup flagged

**Date:** 2026-07-24
**Status:** Proposed
**Deciders:** Kensei (recommendation), Sahil (final)

## Context

The repo-reorg audit (2026-07-23 handoff, Step 1) checked the `skills/`
tree against the spec target `skills/<category>/<skill-name>/{SKILL.md,
scripts/, references/, templates/}`.

## Findings

**The skills tree is already in spec shape.** Every top-level
subdirectory is a category (creative, design, github, governance, …) and
each category's children are individual skills. The expected files
(SKILL.md + optional scripts/, references/, templates/) are present in
the expected places for all 100+ skills reviewed.

Examples of the layout already in use:

```
skills/creative/comfyui/
  SKILL.md
  references/
  scripts/
  tests/
  workflows/
skills/github/github-pr-workflow/
  SKILL.md
  references/
  templates/
skills/governance/kensei-strategic/
  references/
  SKILL.md
```

## Two items flagged

1. **`skills/index-cache/`** — four `.json` files (anthropics_skills,
   claude_marketplace, lobehub, openai_skills). These are generated cache
   from a skills-index fetcher, not skills. They were committed in error.
   Recommendation: move to `data/skills-index/` and add to `.gitignore`
   as a directory (not just the four files).

2. **`skills/audit-engine/Audits/`** — uppercase directory name,
   inconsistent with the lowercase convention used everywhere else.
   Contains `_audit-rotation-log.md` and `deep-dive-2026-05-29.md`.
   Recommendation: rename to `skills/audit-engine/audits/` (lowercase)
   in a follow-up commit. The MDs are not real skills — they are
   working notes left by a previous audit run; should arguably move to
   `docs/handoffs/audit-engine/` or be deleted.

## Recommendation

File this ADR now. Defer the actual `index-cache` and `Audits` moves to
follow-up commits pending Sahil approval. The rest of the skills tree
needs no change for this reorg.

## Decision

Pending.
