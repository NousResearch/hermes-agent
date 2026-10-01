# vault-retrieval — token-efficient Obsidian Vault retrieval

Profile-scoped Hermes user plugin that enforces the locked contract in
`99 System/Hermes Token-Efficient Vault Retrieval Architecture - 2026-10-01.md`.

## What it does

Three surfaces, registered together:

1. **`vault_context` tool** — read-only retrieval with per-turn character
   budgets. Returns a JSON envelope with `status`, `candidates`,
   `extracts`, `conflicts`, `redactions`, `hold`, `log_ref`.
2. **Bounded system-prompt section** (`vault-retrieval-usage`) — tells the
   agent to use `vault_context` for any Vault evidence and what its
   constraints are.
3. **`pre_tool_call` hook** — in `enforce` mode, blocks direct `read_file`
   and `search_files` calls whose target resolves inside the configured
   Vault and points the caller at `vault_context`. In `audit` mode the
   hook passes through and only logs would-block events. In `off` mode
   the hook is a no-op.

Operational guard, not a security sandbox. Arbitrary terminal/Python can
still read files; if hostile-tool bypass prevention becomes required,
open a separate upstream core/sandbox change.

## Configuration

Configuration lives under `plugins.entries.vault-retrieval.settings` in
the active profile's `config.yaml`:

```yaml
plugins:
  entries:
    vault-retrieval:
      enabled: true
      settings:
        vault_root: "/root/Documents/Obsidian Vault"
        mode: enforce          # enforce | audit | off
        default_budget_chars: 12000
        hard_ceiling_chars: 24000   # hard cap = 24,000 (not configurable above)
        large_file_chars: 20000
        candidate_limit: 20
        max_primary_extracts: 3
        max_expansion_extracts: 2
        max_range_lines: 120
        max_range_chars: 8000
        query_log_enabled: true
        log_raw_query_terms: false   # MUST stay false (spec)
        snapshots_enabled: false
        fts5_enabled: false
        block_direct_file_reads: true
```

Invalid or unsafe values are rejected at plugin load — never silently
overridden.

## State and log paths

| Path | Purpose | Mode |
|---|---|---|
| `<HERMES_HOME>/state/vault-retrieval/query-log.jsonl` | Metadata-only audit log | `0600` |
| `<HERMES_HOME>/state/vault-retrieval/` | Parent dir | `0700` |
| `<HERMES_HOME>/cache/vault-retrieval/snapshots/` | Stage 2 snapshots (pilot, off by default) | not created until enabled |
| `<HERMES_HOME>/cache/vault-retrieval/index.sqlite3` | Stage 2 FTS5 (pilot, off by default) | not created until enabled |

`<HERMES_HOME>` is the active profile's `get_hermes_home()`.

## Install

```bash
# Copy the versioned plugin into the active profile
cp -r vault-retrieval/ "$(hermes --print-home)/plugins/vault-retrieval/"

# Verify discovery
hermes plugins list | grep vault-retrieval

# Roll out
# 1. Canary profile: set mode: audit for 48 hours; observe would-block events.
# 2. Switch to enforce.
# 3. Copy to every participating profile's ~/.hermes/plugins/.
```

## Rollback

Set `mode: off` in the plugin config OR remove the plugin directory.
Long-lived gateway/worker processes pick up the new mode on next
start; for immediate effect, restart them. Built-in file tools remain
unaffected when the hook is `off`.

## Development

```bash
# Unit tests (no Hermes core required)
pytest vault-retrieval/tests/ -v

# Acceptance contract tests
pytest vault-retrieval/tests/ -v -k Acceptance
```

The plugin does not depend on the Hermes runtime at import time — the
`register()` entrypoint takes a `PluginContext` and is invoked by
`PluginManager` after discovery.

## Architecture memo

This plugin implements the locked contract in
`/root/Documents/Obsidian Vault/99 System/Hermes Token-Efficient Vault Retrieval Architecture - 2026-10-01.md`.
Always read the contract first; do not introduce behaviour outside the
locked spec without updating the memo.
