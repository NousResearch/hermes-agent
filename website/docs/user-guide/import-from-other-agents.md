---
sidebar_position: 9
title: "Import from Other Agents"
description: "One-command import of an OpenClaw (~/.openclaw), Claude Code (~/.claude), OpenAI Codex CLI (~/.codex) or Cursor (~/.cursor) setup into Hermes — instructions, rules, allowlists, MCP servers, skills, and memories."
---

# Import from Other Agents

`hermes import-agent` imports your existing **OpenClaw**, **Claude Code**, **OpenAI Codex CLI** or **Cursor** setup into Hermes with one command. You always see a per-item plan before anything is written, and `--dry-run` never touches disk.

The command comes from the official **`agent-importers` plugin** from the [plugin catalog](./features/plugins.md), maintained by Nous Research in [NousResearch/hermes-official-plugins](https://github.com/NousResearch/hermes-official-plugins/tree/main/agent-importers).

## The easy way: `hermes setup`

On a fresh install, `hermes setup` checks for `~/.openclaw` (or legacy `~/.clawdbot` / `~/.moltbot`), `~/.claude`, `~/.codex` and `~/.cursor` before anything else. When it finds one it asks **"Import your setup from Claude Code?"** (naming what it found); with several found you then pick which from a checklist, all checked. Say yes and setup installs the `agent-importers` plugin from the catalog, unless you already have it, and runs the import below for each pick, preview and confirmation included. For OpenClaw it passes `--migrate-secrets`, as Hermes' old built-in OpenClaw step did, so your API keys and bot tokens come over and the provider and messaging sections that follow are already filled in. If you then pick Quick Setup and the import brought a model along with its key, setup asks whether to keep that model (the default) or switch to Nous Portal. Say no, or have none of these folders, and nothing is installed. Already set up? Use the commands below.

## By hand

Install the plugin once per profile:

```bash
hermes plugins install agent-importers
```

```bash
hermes import-agent                    # auto-detect ~/.openclaw, ~/.claude, ~/.codex or ~/.cursor
hermes import-agent openclaw           # import from ~/.openclaw (see the OpenClaw guide)
hermes import-agent claude-code        # import from ~/.claude
hermes import-agent codex              # import from ~/.codex
hermes import-agent cursor             # import from ~/.cursor
hermes import-agent claude-code --dry-run          # preview only
hermes import-agent codex --source /path/to/.codex # custom location
hermes import-agent claude-code --overwrite --yes  # replace conflicts, skip prompts
```

## What gets imported

### OpenClaw (`~/.openclaw`)

OpenClaw has its own guide, **[Migrate from OpenClaw](../guides/migrate-from-openclaw.md)**, with the full mapping (persona, memory, skills, model providers, messaging platforms, agent behaviour, MCP servers and more), the `--preset`, `--migrate-secrets`, `--workspace-target` and `--skill-conflict` flags, the pre-migration backup and `hermes claw cleanup`. The older spelling `hermes claw migrate` still works and runs the same import.

### Claude Code (`~/.claude`)

| Claude Code | Hermes |
|---|---|
| `CLAUDE.md` (global instructions) | Memory entries in `~/.hermes/memories/MEMORY.md` |
| `settings.json` → `permissions.allow` (`Bash(...)` rules) | `command_allowlist` in `config.yaml` |
| `settings.json` → `permissions.deny` (`Bash(...)` rules) | `approvals.deny` in `config.yaml` |
| `mcpServers` (from `~/.claude.json` and `settings.json`) | `mcp_servers` in `config.yaml` |
| `skills/<name>/` (dirs with `SKILL.md`) | `~/.hermes/skills/claude-code-imports/<name>/` |
| `commands/*.md` (slash commands) | Skipped with a note — convert them into skills |

Claude's `Bash(npm run test:*)` prefix rules become `npm run test*` globs. Non-`Bash` permission rules (`Read(...)`, `WebFetch`, ...) gate Claude-specific tools and are reported as unmapped rather than imported.

### Codex CLI (`~/.codex`)

| Codex CLI | Hermes |
|---|---|
| `AGENTS.md` (global instructions) | Memory entries in `~/.hermes/memories/MEMORY.md` |
| `config.toml` → `[mcp_servers.*]` | `mcp_servers` in `config.yaml` |
| `memories/*.md` | Memory entries in `~/.hermes/memories/MEMORY.md` |
| `skills/<name>/` (dirs with `SKILL.md`) | `~/.hermes/skills/codex-imports/<name>/` |

### Cursor (`~/.cursor`)

| Cursor | Hermes |
|---|---|
| `AGENTS.md` (global instructions) | Memory entries in `~/.hermes/memories/MEMORY.md` |
| `rules/*.md`, `rules/*.mdc` (User Rules) | Memory entries in `~/.hermes/memories/MEMORY.md` |
| `mcp.json` → `mcpServers` | `mcp_servers` in `config.yaml` |
| `skills/**/<name>/` (dirs with `SKILL.md`, category folders allowed) | `~/.hermes/skills/cursor-imports/<name>/` |

`.mdc` frontmatter (`description`, `globs`, `alwaysApply`) is routing metadata for Cursor's rule engine and is stripped; only the rule body is imported. A skill is the folder holding `SKILL.md`, so `skills/shipping/land-it/` imports as `cursor-imports/land-it/`; two skills with the same name in different folders are reported as a conflict.

## What is never imported

**API keys and credentials.** (OpenClaw is the one exception: its allowlisted API keys and bot tokens come over only when you pass `--migrate-secrets`.) Credential files (`~/.claude/.credentials.json`, `~/.codex/auth.json`, `~/.cursor/cli-config.json`) are never read, and MCP server environment variables or headers with secret-looking names (`*_TOKEN`, `*_API_KEY`, `Authorization`, ...) are stripped and listed in the report so you can re-add them deliberately. Run `hermes setup` to configure providers, or add secrets to `~/.hermes/.env`.

## Behavior notes

- **Preview first, always.** The command prints the full plan before applying; in non-interactive sessions it stops at the preview unless you pass `--yes`.
- **Merges, not replaces.** Memory entries are deduplicated against your existing `MEMORY.md`; allowlist/denylist patterns merge with what's already in `config.yaml`.
- **Conflicts are skipped by default.** An MCP server or skill that already exists in Hermes is reported as a conflict; pass `--overwrite` to replace it.
- **Malformed files don't abort the run.** A broken `settings.json` or `config.toml` becomes a per-item error in the report while everything else still imports.

## Keeping imports in sync

Every successful Claude Code, Codex CLI or Cursor import registers its source (and the digest of everything it read) in `~/.hermes/import-sync.json`. When the other agent's setup changes later — new skills, edited `CLAUDE.md`/`AGENTS.md`, added MCP servers — pull the changes in with:

```bash
hermes import-agent --sync            # re-import every changed source
hermes import-agent --sync --dry-run  # preview what a sync would do
```

Sync is prompt-free and cheap: sources whose files are unchanged are skipped by digest comparison, so it is safe to run on a schedule (e.g. a daily [cron job](features/cron.md)). Rules:

- **Memory and config merges stay deduplicating** — a sync never duplicates entries or patterns you already have.
- **Skills previously imported by `import-agent` are refreshed in place** when the source copy changes.
- **Skills you created or modified under the import category yourself are never clobbered** — they keep normal conflict semantics (use `--overwrite` on a manual run to force).
- **Credential files never trigger a sync** — token refreshes in `~/.claude/.credentials.json` or `~/.codex/auth.json` are ignored by the digest, and secrets are still stripped from anything imported.

This mirrors ChatGPT Work's *Settings > Import* automatic updates, adapted to an explicit, inspectable command instead of a background service.

The manifest is per profile. A profile created with `hermes profile create <name> --clone --sync-imports` carries it over, so `hermes -p <name> import-agent --sync` keeps pulling from the same external trees (see [Profiles](./profiles.md#keep-a-clones-imported-agent-setups-synced---sync-imports)).

## Upgrading from a version with `import-agent` built in

`hermes import-agent` and `hermes claw` used to ship inside Hermes core. The commands, their flags and the `import-sync.json` manifest are unchanged in the plugin:

- **Profiles with a registered sync source** (an `import-sync.json`, which scheduled `--sync` runs read) get the plugin installed automatically by `hermes update`, or on the first start after updating (this honours `security.allow_lazy_installs`). If you later remove it (`hermes plugins remove agent-importers`), it stays removed.
- **Everyone else** installs it when they next need it: a fresh `hermes setup` that finds one of these agents offers it, and running `hermes import-agent` or `hermes claw` without the plugin prints the install command.
