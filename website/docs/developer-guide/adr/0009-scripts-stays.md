# ADR 0009 — scripts/ stays; infra/ created for forward-only additions

**Date:** 2026-07-24
**Status:** Accepted (default)
**Deciders:** Kensei

## Context

The spec target lists `infra/{systemd/, setup/, backup/}`. The current
repo has these scattered:

- `scripts/systemd/*.service` (4 files)
- `scripts/backup-*.py` (4 files)
- `scripts/backup-*.sh` (3 files)
- `setup-hermes.sh` (top level)
- `scripts/lib/node-bootstrap.sh`
- `scripts/ci/classify_changes.py`, `scripts/ci/timings_report.py`

A naïve read of the spec would consolidate them all into `infra/`.

## Why we did not move them

Many of these are referenced by **upstream** code that must not be patched
under the handoff's CRITICAL CONSTRAINTS:

| Path | Referenced by | Upstream? |
|---|---|---|
| `scripts/lib/node-bootstrap.sh` | `hermes_cli/main.py:1599`, `hermes_cli/uninstall.py:137`, `hermes_constants.py:415` | yes |
| `scripts/install.sh` | `hermes_cli/uninstall.py:137` (POSIX installer) | yes |
| `scripts/ci/timings_report.py` | `tests/ci/test_classify_changes.py:1` (test header), `scripts/ci/timings_report.py:13,16` (self-docstring) | yes (test in core) |
| `setup-hermes.sh` | `tests/hermes_cli/test_setup_hermes_script.py:6`, `README.es.md:185,190`, `README.ur-pk.md:216,223` | yes |
| `scripts/systemd/*.service` | `hermes_cli/kanban.py:2745` (stub) | yes (stub in core) |

Patching any of the upstream callers to point to the new `infra/` path
would create permanent fork drift on every upstream merge.

The backup-*.py scripts in `scripts/` are not referenced by upstream, but
they ARE referenced by the wesker profile's `SKILL.md` (which is in the
profile we already moved to `agents/wesker/...`) and the
`backup-health-check.sh` is the production watchdog entry. Moving the
backup-*.py files to `infra/backup/` would force a coordinated rename
across at least the SKILL.md and the cron job definitions that call them
by name. The risk/reward of doing that in this reorg is poor.

## Decision

Keep `scripts/systemd/`, `scripts/lib/`, `scripts/ci/`, `setup-hermes.sh`,
and `scripts/backup-*` where they are. The `scripts/` directory is the
project's de facto infra home, wired into upstream and documented in
READMEs. Forcing it to match the spec target would require patching
upstream code or coordinating many renames that are out of scope for
this reorg.

The `infra/` directory **will be created** as a forward-only target for
any NEW infra (systemd unit, backup script, setup helper) that does not
have an upstream dependency. Existing infra remains in `scripts/`.

## What this means in practice

- `scripts/` continues to host all executable tooling.
- New systemd units land in `infra/systemd/` (only if they don't conflict
  with the `scripts/systemd/` names).
- New backup scripts land in `infra/backup/`.
- New setup helpers land in `infra/setup/`.
- ADRs for each new addition should explain why it does not live in
  `scripts/` (typically: it has no upstream coupling AND is large
  enough to warrant its own subdir).

## Revisit triggers

- If upstream renames the systemd/lib/ci paths, consolidate atomically.
- If a Kensei fork-patch policy is adopted for any of the upstream
  callers listed above, revisit.
- If a long-term migration to a single `infra/` home is desired,
  schedule a dedicated pass with full cross-reference renames.
