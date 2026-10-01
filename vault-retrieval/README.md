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

The plugin is a profile-scoped user plugin. Each Hermes profile owns
its own `plugins/` directory under `get_hermes_home()`, so installs
must target the correct profile home — there is no single global
"home" because the active profile can vary (`hermes -p <name>`).

Resolve the active profile's home with the supported profile command
(installed CLI): either run the slash command interactively, or invoke
the same handler directly:

```bash
# Active profile (the one the current shell is using):
python -m hermes_cli.main profile
# Prints e.g.  Active profile: software-eng
#               Path:           /root/.hermes/profiles/software-eng

# A specific profile:
python -m hermes_cli.main -p default profile
# Prints e.g.  Active profile: default
#               Path:           /root/.hermes
```

Copy the versioned plugin into the chosen profile's `plugins/`
directory:

```bash
# Replace <profile_home> with the path printed by the profile command above.
PROFILE_HOME="$(python -m hermes_cli.main profile | awk '/Path:/ {print $2}')"
cp -r vault-retrieval/ "${PROFILE_HOME}/plugins/vault-retrieval/"

# Verify the active profile's config enables it (this is the supported,
# profile-scoped allow-list):
python -m hermes_cli.main plugins list | grep vault-retrieval
```

If `hermes` is on the executable PATH in this environment, the same
discovery commands work without the `python -m` prefix:

```bash
hermes profile
hermes plugins list | grep vault-retrieval
```

Do NOT use `hermes --print-home` — that flag does not exist in the
installed CLI; the previous version of this README did and the
adoption-gate review (2026-10-01 23:16 +07) blocked on it.

Then enable the plugin in the profile's `config.yaml` so the loader's
opt-in allow-list picks it up:

```yaml
plugins:
  enabled:
    - vault-retrieval
  entries:
    vault-retrieval:
      enabled: true
      settings:
        vault_root: "/root/Documents/Obsidian Vault"
        mode: enforce          # enforce | audit | off
        # ...other defaults match the locked contract...
```

Roll out:
1. Canary profile: set `mode: audit` for 48 hours; observe would-block events.
2. Switch to `enforce`.
3. Copy to every participating profile's `plugins/` directory and verify
   discovery per profile.

## Rollback

Set `mode: "off"` (quoted) in the plugin config — the quote is required
because YAML 1.1 parses unquoted `mode: off` as boolean `False`, which the
plugin normalizes to `"off"` defensively but operators should still quote
the value for clarity — OR remove the plugin directory. Long-lived
gateway/worker processes pick up the new mode on next start; for immediate
effect, restart them. Built-in file tools remain unaffected when the hook
is `off`.

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
