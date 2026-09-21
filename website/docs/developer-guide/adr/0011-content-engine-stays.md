# ADR 0011 — content_engine/ stays at top level (cron + scripts coupled)

**Date:** 2026-07-24
**Status:** Accepted
**Deciders:** Kensei

## Context

`content_engine/` is a 200+ file Python application that powers Kensei's
daily content pipeline. It is a Kensei-only addition, not part of
upstream Hermes. The spec target has no slot for it.

## Why it stays at the top level

1. **Cron job hardcoded paths.** `cron/jobs.snapshot.json` contains
   prompts that explicitly do:
   - `cd /home/kensei/repos/KenseiAgent/content_engine`
   - `export PYTHONPATH=/home/kensei/repos/KenseiAgent/content_engine`
   - `python3 content_engine.py <subcommand>`

   These are the prompts the daily CeeCee + content cron jobs run
   against. Moving `content_engine/` to any other location (e.g.
   `services/content_engine/`) would break all of them in a single
   shot, with no path-relative fallback. The prompt strings are
   inlined in the snapshot, so fixing them is a find-and-replace
   across hundreds of cron entries.

2. **Kensei-side scripts import it.** `scripts/ceecee_approval_handler.py`
   does `from content_engine.X import Y`. Moving the engine would
   force a coordinated rename in the import + every cron job + every
   README that references the path.

3. **Tests live in `content_engine/tests/`.** Per spec rule #5 these
   should move to `tests/content_engine/` to mirror source. They
   don't, but pytest finds them via the `test_*.py` glob regardless
   of the parent dir. Functionally fine, structurally inconsistent.

4. **It's a self-contained app.** The engine has its own config,
   database (`content_engine.db`), output dirs, and tooling. It does
   not depend on or import from the Hermes core. Treating it as a
   top-level "service" (rather than folding it into another
   subdirectory) reflects how it operates.

## Decision

Keep `content_engine/` at the top level. Document the spec mismatch
in this ADR. If a future cron refactor moves the prompts out of
`jobs.snapshot.json` into named skill files, the path-relative
coupling eases and a move becomes feasible.

## What this means in practice

- The spec target is silent on where a Kensei-only application should
  live; `content_engine/` is a reasonable answer.
- Future content_engine code lands under `content_engine/` as before.
- If/when the spec evolves to add a `services/` or `apps/` slot, revisit.

## Revisit triggers

- A cron refactor moves prompts out of `jobs.snapshot.json` into named
  skills → reconsider.
- A new Kensei-only app of similar size appears → colocate or
  introduce a `services/` slot.
- The `tests/content_engine/` mirror is desired → add a follow-up
  commit (cosmetic; pytest already finds the tests).
