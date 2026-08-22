---
sidebar_position: 9
title: "Import from Other Agents"
description: "One-command import of a Claude Code (~/.claude), OpenAI Codex CLI (~/.codex), or Cursor (~/.cursor) setup into Hermes — instructions, rules, allowlists, MCP servers, skills, and memories."
---

# Import from Other Agents

`hermes import-agent` imports your existing **Claude Code**, **OpenAI Codex CLI**, or **Cursor** setup into Hermes with one command. It follows the same preview-first pattern as [`hermes claw migrate`](../guides/migrate-from-openclaw.md): you always see a per-item plan before anything is written, and `--dry-run` never touches disk.

```bash
hermes import-agent                    # auto-detect ~/.claude, ~/.codex, or ~/.cursor
hermes import-agent claude-code        # import from ~/.claude
hermes import-agent codex              # import from ~/.codex
hermes import-agent cursor             # import from ~/.cursor
hermes import-agent claude-code --dry-run          # preview only
hermes import-agent codex --source /path/to/.codex # custom location
hermes import-agent claude-code --overwrite --yes  # replace conflicts, skip prompts
```

## What gets imported

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
| `skills/**/<name>/` (dirs with `SKILL.md`, nested categories supported) | `~/.hermes/skills/cursor-imports/<name>/` |

`.mdc` rule frontmatter (`description`, `globs`, `alwaysApply`) is routing metadata for Cursor's rule engine and is stripped — only the instruction body is imported. Cursor allows nested skill category folders (`skills/shipping/land-it/SKILL.md`); per Cursor's own semantics the skill's identity is the folder containing `SKILL.md`, so nested skills import flat as `cursor-imports/land-it/`. Duplicate skill names across categories are reported as conflicts.

## What is never imported

**API keys and credentials.** Credential files (`~/.claude/.credentials.json`, `~/.codex/auth.json`, `~/.cursor/cli-config.json`) are never read, and MCP server environment variables or headers with secret-looking names (`*_TOKEN`, `*_API_KEY`, `Authorization`, ...) are stripped and listed in the report so you can re-add them deliberately. Run `hermes setup` to configure providers, or add secrets to `~/.hermes/.env`.

## Behavior notes

- **Preview first, always.** The command prints the full plan before applying; in non-interactive sessions it stops at the preview unless you pass `--yes`.
- **Merges, not replaces.** Memory entries are deduplicated against your existing `MEMORY.md`; allowlist/denylist patterns merge with what's already in `config.yaml`.
- **Conflicts are skipped by default.** An MCP server or skill that already exists in Hermes is reported as a conflict; pass `--overwrite` to replace it.
- **Malformed files don't abort the run.** A broken `settings.json` or `config.toml` becomes a per-item error in the report while everything else still imports.
- Coming from OpenClaw instead? Use [`hermes claw migrate`](../guides/migrate-from-openclaw.md).
