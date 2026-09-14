# Skill health score provenance — P3.4 (01/09/2026)

## What is verified (read-only, live system)

Live ledger: `~/.hermes/governance/profile-activity-ledger.sqlite`
(sha256 `08b718d2bf9a63b5…` at build preflight).

- The database contains a `skill_health_scores` table with columns:
  `id INTEGER PK AUTOINCREMENT, window_start TEXT, window_end TEXT,
   skill_id TEXT NOT NULL, n_samples INTEGER, trigger_score REAL,
   compliance_score REAL, boundary_score REAL, su_score REAL,
   flags_json TEXT, created_at INTEGER NOT NULL`.
- Exactly two rows exist, ids 1 and 2, skill_ids `test-skill-pass` and
  `test-skill-fail`, both windowed 2026-08-11 and `n_samples = 6`.
  These are unowned historical/test-like scaffold: they were not
  written by any reviewed source, are not accepted as real scorecard
  evidence, and remain untouched.  Deletion or migration of the live
  rows is out of scope for this build.

## What is absent from source

- Reviewed KenseiAgent source at the frozen bases (Phase 1–2) contains
  **no** writer, reader, or schema migration for `skill_health_scores`
  (`grep -rn skill_health_scores hermes_cli/ scripts/ tools/` → no
  matches at the candidate).  The live table is therefore an orphaned
  scaffold with no authoritative owner in source.

## Skill evidence taxonomy actually present in the ledger

`skill.loaded` (19,955), `skill.access.blocked` (410), `skill.borrowed`
(115), `skill.revoked` (115), `skill.quarantined` (2).  There is **no**
event type proving "required procedure followed" (compliance) or
"forbidden action did not occur" (boundary).  The Trigger dimension is
partially supportable (loaded/use events); Compliance and Boundary are
NOT current-taxonomy supportable and honest implementations must emit
`insufficient_evidence` for them.

## Canonical contract (this build)

`hermes_cli/skill_evidence.py` is now the single canonical schema and
writer for skill health evidence (`skill-evidence` schema, version 1).
Rules enforced by the writer:

- schema versioned; values computed only from explicit evidence refs;
- `insufficient_evidence` whenever the taxonomy cannot support a
  dimension; no fabricated 0/1/100%/neutral scores;
- every value carries `window_start`, `window_end`, `skill_id`,
  `n_samples`, `evidence_refs`, `method`, `version`;
- writes go only to explicitly supplied database paths — never to the
  live ledger during the build (verified by tests); the live
  `skill_health_scores` scaffold is superseded by this source contract
  but not migrated in this phase;
- repeated execution for the same window is idempotent.

Supersession: the unowned live `skill_health_scores` table has no
source authority and MUST NOT be read as scorecard evidence by Phase 3+.
Canonical reads go through `hermes_cli/skill_evidence.py` operating on
explicit database paths.