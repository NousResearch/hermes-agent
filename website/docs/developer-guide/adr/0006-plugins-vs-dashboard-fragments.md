# ADR 0006 — Plugins vs dashboard fragments: where each "kensei-*" lives

**Date:** 2026-07-24
**Status:** Proposed
**Deciders:** Kensei (recommendation), Sahil (final)

## Context

The repo-reorg audit (2026-07-23 handoff, Step 1) found 10 plugin-shaped
directories that do not actually follow the spec's plugin contract
(`plugin.yaml` + `__init__.py` with `register(ctx)`):

| Path | Contents | Real nature |
|---|---|---|
| `plugins/kanban/` | `dashboard/dist/`, `dashboard/plugin_api.py`, `dashboard/manifest.json`, `systemd/hermes-kanban-dispatcher.service` | Kanban UI + dispatcher backend |
| `plugins/hermes-achievements/` | `dashboard/`, `docs/`, `tests/`, `LICENSE`, `README.md` | Achievements system + tests |
| `plugins/kensei-brand/` | `dashboard/dist/`, `dashboard/manifest.json` | Dashboard brand skin |
| `plugins/kensei-console/` | `dashboard/dist/`, `dashboard/manifest.json` | Dashboard console view |
| `plugins/kensei-overview/` | `dashboard/dist/`, `dashboard/manifest.json` | Dashboard overview view |
| `plugins/kensei-postiz/` | `dashboard/dist/`, `dashboard/manifest.json` | Postiz widget |
| `plugins/kensei-pulse/` | `dashboard/dist/`, `dashboard/manifest.json` | Pulse widget |
| `plugins/kensei-settings/` | `dashboard/dist/`, `dashboard/manifest.json` | Settings view |
| `plugins/kensei-theme-assets/` | `dashboard/dist/`, `dashboard/manifest.json` | Theme assets |
| `plugins/kensei-triage/` | `dashboard/dist/`, `dashboard/manifest.json` | Triage widget |

None of them have a `plugin.yaml` or a top-level `__init__.py` with a
`register(ctx)`. They are **dashboard fragments** — TypeScript/JS bundles
served by the Hermes dashboard backend, registered through
`dashboard/manifest.json`, with a Python `plugin_api.py` exposing API
endpoints to the React UI.

The spec's target structure is:

```
dashboard/{src/, package.json}
```

…and the spec's hard rule #1 is **"Plugins never modify core Hermes files.
If a capability is missing, flag it as needing a new hook/ctx method —
do not hardcode around it."** Forcing these fragments to fit the plugin
shape would mean either fabricating a `register(ctx)` that only registers
dashboard assets, or stuffing `dashboard/manifest.json` into a plugin
that does no actual plugin work.

## Recommendation (revised after dependency check)

**Do not move the dashboard fragments.** The runtime API contract is
hardcoded by upstream:

```python
# hermes_cli/web_server.py:15963-15967
"""Scan plugins/*/dashboard/manifest.json for dashboard extensions.

1. User plugins:    ~/.hermes/plugins/<name>/dashboard/manifest.json
2. Bundled plugins: <repo>/plugins/<name>/dashboard/manifest.json
```

Tests assert the same contract:
- `tests/hermes_cli/test_web_server.py:5974` — `GET /api/plugins/hermes-achievements/overview`
- `tests/plugins/test_achievements_plugin.py:13` — `plugins/hermes-achievements/tests/`
- `tests/plugins/test_plugin_dashboard_auth_contract.py:55` — bundled plugins expected

The handoff's CRITICAL CONSTRAINTS forbid patching `hermes_cli/`. So
moving the dashboard fragments under `dashboard/src/<name>/` would
silently break the dashboard at runtime.

## Decision

Keep `plugins/kanban/`, `plugins/hermes-achievements/`, and the 8
`plugins/kensei-*/` directories in place. They continue to ship as
plugin-shaped entries but their `plugin.yaml` (if added) would be a
no-op — they exist solely as the bundle unit the dashboard loader
expects.

If a future Kensei fork-patch policy is adopted for
`hermes_cli/web_server.py:_discover_dashboard_plugins`, the move
becomes possible atomically. Until then, the spec target
`dashboard/{src/, package.json}` is forward-only for any NEW
dashboard work that does not need plugin discovery.

## Revisit triggers

- Upstream renames the discovery path → consolidate.
- Kensei fork-patch policy for `hermes_cli/web_server.py` adopted → move atomically.
- A test that specifically asserts `dashboard/src/<name>/manifest.json`
  discovery is added (no such test exists today) → reconsider.

## Alternative considered

**Keep them as plugins and add a `plugin.yaml` that does nothing.** Rejected.
That violates "Plugins never modify core Hermes files" by making plugins
that exist only to ship dashboard assets — the plugin shape is wrong for
this content.

## Decision

Pending. Kensei will hold the dashboard move until Sahil confirms.
