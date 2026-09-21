# ADR 0008 — optional-mcps/ kept at top level (mcp_catalog.py is upstream)

**Date:** 2026-07-24
**Status:** Accepted (default)
**Deciders:** Kensei

## Context

The 2026-07-23 repo-reorg handoff's spec target lists `mcp/<inbox-name>.yaml`
at the top level. The current repo has `optional-mcps/<name>/manifest.yaml`
instead. The natural read is: flatten the `optional-mcps/<name>/manifest.yaml`
into `mcp/<name>.yaml` to match the spec.

## Why we did not move them

`hermes_cli/mcp_catalog.py` (upstream) hardcodes the catalog location:

```python
# hermes_cli/mcp_catalog.py:131
return get_optional_mcps_dir(Path(__file__).parent.parent / "optional-mcps")
```

The repo-reorg handoff's CRITICAL CONSTRAINTS say:

> "This is KenseiAgent — a fork of Hermes Agent. The existing codebase has a
> specific structure that Hermes expects. Do NOT move core Hermes files
> (agent/, gateway/, hermes_cli/, cli.py, run_agent.py, etc.) — those are
> upstream files and must stay where they are."

`hermes_cli/mcp_catalog.py` is an upstream file. Patching it would create
a permanent fork drift that would need to be re-applied on every upstream
merge.

## Decision

Leave `optional-mcps/<name>/manifest.yaml` at the top level. Document the
spec mismatch in this ADR. If the upstream layout ever changes (or if
someone chooses to maintain a fork patch on `mcp_catalog.py`), revisit.

## What this means in practice

- `mcp/<inbox-name>.yaml` from the spec is the future shape for **inbox**
  MCP configs (separate concern from the catalog). No such files exist in
  the current repo, so no `mcp/` directory is created.
- The catalog (`optional-mcps/`) stays put.
- The test fixture at `tests/hermes_cli/test_mcp_catalog.py:41` (which
  builds an isolated `optional-mcps` dir for testing) continues to work
  unchanged.

## Revisit triggers

- If upstream renames `optional-mcps` to `mcp` in a future release, delete
  this ADR and move the catalog.
- If Kensei adopts a permanent fork-patch policy for `mcp_catalog.py`,
  move the catalog and patch the upstream loader atomically.
