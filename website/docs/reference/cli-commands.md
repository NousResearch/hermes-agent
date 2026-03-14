---
sidebar_position: 1
title: "CLI Commands Reference"
description: "Authoritative reference for Hermes terminal commands and command families"
---

# CLI Commands Reference

This page covers the **terminal commands** you run from your shell.

For in-chat slash commands, see [Slash Commands Reference](./slash-commands.md).

## Global entrypoint

```bash
hermes [global-options] <command> [subcommand/options]
```

### Global options

| Option | Description |
|--------|-------------|
| `--version`, `-V` | Show version and exit. |
| `--resume <session>`, `-r <session>` | Resume a previous session by ID or title. |
| `--continue [name]`, `-c [name]` | Resume the most recent session, or the most recent session matching a title. |
| `--worktree`, `-w` | Start in an isolated git worktree for parallel-agent workflows. |
| `--yolo` | Bypass dangerous-command approval prompts. |
| `--pass-session-id` | Include the session ID in the agent's system prompt. |

## Top-level commands

| Command | Purpose |
|---------|---------|
| `hermes chat` | Interactive or one-shot chat with the agent. |
| `hermes model` | Interactively choose the default provider and model. |
| `hermes gateway` | Run or manage the messaging gateway service. |
| `hermes setup` | Interactive setup wizard for all or part of the configuration. |
| `hermes whatsapp` | Configure and pair the WhatsApp bridge. |
| `hermes login` / `logout` | Authenticate with OAuth-backed providers. |
| `hermes status` | Show agent, auth, and platform status. |
| `hermes cron` | Inspect and tick the cron scheduler. |
| `hermes doctor` | Diagnose config and dependency issues. |
| `hermes config` | Show, edit, migrate, and query configuration files. |
| `hermes pairing` | Approve or revoke messaging pairing codes. |
| `hermes skills` | Browse, install, publish, audit, and configure skills. |
| `hermes honcho` | Manage Honcho cross-session memory integration. |
| `hermes acp` | Run Hermes as an ACP server for editor integration. |
| `hermes tools` | Configure enabled tools per platform. |
| `hermes sessions` | Browse, export, prune, rename, and delete sessions. |
| `hermes insights` | Show token/cost/activity analytics. |
| `hermes claw` | OpenClaw migration helpers. |
| `hermes version` | Show version information. |
| `hermes update` | Pull latest code and reinstall dependencies. |
| `hermes uninstall` | Remove Hermes from the system. |

## `hermes chat`

```bash
hermes chat [options]
```

Common options:

| Option | Description |
|--------|-------------|
| `-q`, `--query "..."` | One-shot, non-interactive prompt. |
| `-m`, `--model <model>` | Override the model for this run. |
| `-t`, `--toolsets <csv>` | Enable a comma-separated set of toolsets. |
| `--provider <provider>` | Force a provider: `auto`, `openrouter`, `nous`, `openai-codex`, `anthropic`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`. |
| `-v`, `--verbose` | Verbose output. |
| `-Q`, `--quiet` | Programmatic mode: suppress banner/spinner/tool previews. |
| `--resume <session>` / `--continue [name]` | Resume a session directly from `chat`. |
| `--worktree` | Create an isolated git worktree for this run. |
| `--checkpoints` | Enable filesystem checkpoints before destructive file changes. |
| `--yolo` | Skip approval prompts. |
| `--pass-session-id` | Pass the session ID into the system prompt. |

Examples:

```bash
hermes
hermes chat -q "Summarize the latest PRs"
hermes chat --provider openrouter --model anthropic/claude-sonnet-4.6
hermes chat --toolsets web,terminal,skills
hermes chat --quiet -q "Return only JSON"
hermes chat --worktree -q "Review this repo and open a PR"
```

## `hermes model`

Interactive provider + model selector.

```bash
hermes model
```

Use this when you want to:
- switch default providers
- log into OAuth-backed providers during model selection
- pick from provider-specific model lists
- save the new default into config

## `hermes gateway`

```bash
hermes gateway <subcommand>
```

Subcommands:

| Subcommand | Description |
|------------|-------------|
| `run` | Run the gateway in the foreground. |
| `start` | Start the installed gateway service. |
| `stop` | Stop the service. |
| `restart` | Restart the service. |
| `status` | Show service status. |
| `install` | Install as a user service (`systemd` on Linux, `launchd` on macOS). |
| `uninstall` | Remove the installed service. |
| `setup` | Interactive messaging-platform setup. |

## `hermes setup`

```bash
hermes setup [model|terminal|gateway|tools|agent] [--non-interactive] [--reset]
```

Use the full wizard or jump into one section:

| Section | Description |
|---------|-------------|
| `model` | Provider and model setup. |
| `terminal` | Terminal backend and sandbox setup. |
| `gateway` | Messaging platform setup. |
| `tools` | Enable/disable tools per platform. |
| `agent` | Agent behavior settings. |

Options:

| Option | Description |
|--------|-------------|
| `--non-interactive` | Use defaults / environment values without prompts. |
| `--reset` | Reset configuration to defaults before setup. |

## `hermes whatsapp`

```bash
hermes whatsapp
```

Runs the WhatsApp pairing/setup flow, including mode selection and QR-code pairing.

## `hermes login` / `hermes logout`

```bash
hermes login [--provider nous|openai-codex] [--portal-url ...] [--inference-url ...]
hermes logout [--provider nous|openai-codex]
```

`login` supports:
- Nous Portal OAuth/device flow
- OpenAI Codex OAuth/device flow

Useful options for `login`:
- `--no-browser`
- `--timeout <seconds>`
- `--ca-bundle <pem>`
- `--insecure`

## `hermes status`

```bash
hermes status [--all] [--deep]
```

| Option | Description |
|--------|-------------|
| `--all` | Show all details in a shareable redacted format. |
| `--deep` | Run deeper checks that may take longer. |

## `hermes cron`

```bash
hermes cron <list|status|tick>
```

| Subcommand | Description |
|------------|-------------|
| `list` | Show scheduled jobs. |
| `status` | Check whether the cron scheduler is running. |
| `tick` | Run due jobs once and exit. |

## `hermes doctor`

```bash
hermes doctor [--fix]
```

| Option | Description |
|--------|-------------|
| `--fix` | Attempt automatic repairs where possible. |

## `hermes config`

```bash
hermes config <subcommand>
```

Subcommands:

| Subcommand | Description |
|------------|-------------|
| `show` | Show current config values. |
| `edit` | Open `config.yaml` in your editor. |
| `set <key> <value>` | Set a config value. |
| `path` | Print the config file path. |
| `env-path` | Print the `.env` file path. |
| `check` | Check for missing or stale config. |
| `migrate` | Add newly introduced options interactively. |

## `hermes pairing`

```bash
hermes pairing <list|approve|revoke|clear-pending>
```

| Subcommand | Description |
|------------|-------------|
| `list` | Show pending and approved users. |
| `approve <platform> <code>` | Approve a pairing code. |
| `revoke <platform> <user-id>` | Revoke a user's access. |
| `clear-pending` | Clear pending pairing codes. |

## `hermes skills`

```bash
hermes skills <subcommand>
```

Subcommands:

| Subcommand | Description |
|------------|-------------|
| `browse` | Paginated browser for skill registries. |
| `search` | Search skill registries. |
| `install` | Install a skill. |
| `inspect` | Preview a skill without installing it. |
| `list` | List installed skills. |
| `check` | Check installed hub skills for upstream updates. |
| `update` | Reinstall hub skills with upstream changes when available. |
| `audit` | Re-scan installed hub skills. |
| `uninstall` | Remove a hub-installed skill. |
| `publish` | Publish a skill to a registry. |
| `snapshot` | Export/import skill configurations. |
| `tap` | Manage custom skill sources. |
| `config` | Interactive enable/disable configuration for skills by platform. |

Common examples:

```bash
hermes skills browse
hermes skills browse --source official
hermes skills search react --source skills-sh
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect official/security/1password
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install official/migration/openclaw-migration
hermes skills install skills-sh/anthropics/skills/pdf --force
hermes skills check
hermes skills update
hermes skills config
```

Notes:
- `--force` can override non-dangerous policy blocks for third-party/community skills.
- `--force` does not override a `dangerous` scan verdict.
- `--source skills-sh` searches the public `skills.sh` directory.
- `--source well-known` lets you point Hermes at a site exposing `/.well-known/skills/index.json`.

## `hermes honcho`

```bash
hermes honcho <subcommand>
```

Subcommands:

| Subcommand | Description |
|------------|-------------|
| `setup` | Interactive Honcho setup wizard. |
| `status` | Show current Honcho config and connection status. |
| `sessions` | List known Honcho session mappings. |
| `map` | Map the current directory to a Honcho session name. |
| `peer` | Show or update peer names and dialectic reasoning level. |
| `mode` | Show or set memory mode: `hybrid`, `honcho`, or `local`. |
| `tokens` | Show or set token budgets for context and dialectic. |
| `identity` | Seed or show the AI peer identity representation. |
| `migrate` | Migration guide from openclaw-honcho to Hermes Honcho. |

## `hermes acp`

```bash
hermes acp
```

Starts Hermes as an ACP (Agent Client Protocol) stdio server for editor integration.

Related entrypoints:

```bash
hermes-acp
python -m acp_adapter
```

Install support first:

```bash
pip install -e '.[acp]'
```

See [ACP Editor Integration](../user-guide/features/acp.md) and [ACP Internals](../developer-guide/acp-internals.md).

## `hermes tools`

```bash
hermes tools [--summary]
```

| Option | Description |
|--------|-------------|
| `--summary` | Print the current enabled-tools summary and exit. |

Without `--summary`, this launches the interactive per-platform tool configuration UI.

## `hermes sessions`

```bash
hermes sessions <subcommand>
```

Subcommands:

| Subcommand | Description |
|------------|-------------|
| `list` | List recent sessions. |
| `browse` | Interactive session picker with search and resume. |
| `export <output> [--session-id ID]` | Export sessions to JSONL. |
| `delete <session-id>` | Delete one session. |
| `prune` | Delete old sessions. |
| `stats` | Show session-store statistics. |
| `rename <session-id> <title>` | Set or change a session title. |

## `hermes insights`

```bash
hermes insights [--days N] [--source platform]
```

| Option | Description |
|--------|-------------|
| `--days <n>` | Analyze the last `n` days (default: 30). |
| `--source <platform>` | Filter by source such as `cli`, `telegram`, or `discord`. |

## `hermes claw`

```bash
hermes claw migrate
```

Used to migrate settings, memories, skills, and keys from OpenClaw to Hermes.

## Maintenance commands

| Command | Description |
|---------|-------------|
| `hermes version` | Print version information. |
| `hermes update` | Pull latest changes and reinstall dependencies. |
| `hermes uninstall [--full] [--yes]` | Remove Hermes, optionally deleting all config/data. |

## See also

- [Slash Commands Reference](./slash-commands.md)
- [CLI Interface](../user-guide/cli.md)
- [Sessions](../user-guide/sessions.md)
- [Skills System](../user-guide/features/skills.md)
- [Skins & Themes](../user-guide/features/skins.md)
