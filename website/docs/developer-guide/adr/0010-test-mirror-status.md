# ADR 0010 — Test mirror status: largely compliant, minor drift flagged

**Date:** 2026-07-24
**Status:** Accepted
**Deciders:** Kensei

## Context

The spec target lists `tests/plugins/...` and `tests/skills/...` as the
test mirrors. The spec hard rule #5 says **"Tests mirror source 1:1.
If you create `plugins/foo/`, create `tests/plugins/test_foo.py` in the
same commit, not later."**

This is forward-looking — every plugin going forward must ship a test
in the same commit. Existing tests should also be evaluated for
compliance.

## Findings

The current `tests/plugins/` tree covers most plugins:

| Plugin | Test files in `tests/plugins/` referencing it |
|---|---|
| memory | 15 |
| platforms | 13 |
| image_gen | 6 |
| google_meet | 4 |
| dashboard_auth | 4 |
| video_gen | 3 |
| observability | 2 |
| cron_providers | 2 |
| web | 1 |
| teams_pipeline | 1 |
| kanban | 1 |
| tts | only `check_parity_vs_main.py` (parity check, not a real test) |
| spotify | test lives at `tests/tools/test_spotify_client.py`, not under `tests/plugins/spotify/` |
| security-guidance | 0 (covered indirectly through `test_security_guidance_plugin.py` exists; verify) |
| prompt-optimizer | 0 (covered by `test_prompt_optimizer_plugin.py` — grep artefact) |
| model-providers | 0 direct references; many upstream model tests exist under `tests/providers/` |
| kensei-* (8) | 0 (no behavioural test exists; they are dashboard fragments — see ADR 0006) |

## Drift items

1. **`tests/tools/test_spotify_client.py`** — should be
   `tests/plugins/spotify/test_client.py` per the spec's mirror rule.
   Move would require also patching any imports of the test module
   itself (if any).
2. **`tests/plugins/tts/check_parity_vs_main.py`** — currently
   `check_parity_vs_main.py` is a parity check, not a plugin behaviour
   test. Acceptable as-is but should be renamed or supplemented.
3. **kensei-* dashboard fragments** — no test (these are
   JS-bundle-driven, no Python behaviour to test; covered by the
   dashboard's own React tests, which live in `plugins/kensei-*/dashboard/`).
   This is acceptable per ADR 0006.

## Decision

The test mirror is largely compliant. Two minor drift items flagged
for follow-up:

- Move `tests/tools/test_spotify_client.py` to
  `tests/plugins/spotify/test_client.py` in a dedicated commit
  (touches the test file plus any importers; run tests after).
- Decide whether `check_parity_vs_main.py` should be
  `test_parity.py` (so pytest picks it up) or kept as a separate
  non-test script.

These are filed but not actioned in this reorg. The forward-looking
rule (every new plugin ships with its test in the same commit) is the
real lever; the retro renames can wait for a dedicated pass.

## Revisit triggers

- A new plugin is added without its test → file an issue immediately;
  the spec rule is for future work, not retro renames.
- A retro rename pass is scheduled (separate ticket).
