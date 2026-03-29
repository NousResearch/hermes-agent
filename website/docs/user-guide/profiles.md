---
sidebar_position: 2
---

# Profiles: Running Multiple Agents

Run multiple independent Hermes agents on the same machine — each with its own config, memory, sessions, and gateway.

## What are profiles?

A profile is a fully isolated Hermes environment. Each profile gets its own `HERMES_HOME` directory containing its own `config.yaml`, `.env`, `SOUL.md`, memories, sessions, skills, and state database. Profiles let you run separate agents for different purposes — a personal assistant, a work bot, a dev agent — without any cross-contamination.

Each profile also gets a shell alias (e.g., `hermes-work`) so you can launch it directly without flags.

## Quick start

```bash
# Create a profile called "work"
hermes profile create work

# Switch to it as the default
hermes profile use work

# Launch — now everything uses the "work" environment
hermes
```

That's it. From now on, `hermes` uses the "work" profile until you switch back.

## Creating a profile

### Blank profile

```bash
hermes profile create mybot
```

Creates a fresh, empty profile. You'll need to run `hermes setup` (or `hermes-mybot setup`) to configure it from scratch — provider, model, gateway tokens, etc.

### Clone config only (`--clone`)

```bash
hermes profile create work --clone
```

Copies your current profile's `config.yaml`, `.env`, and `SOUL.md` into the new profile. This gives you the same provider/model setup without copying memories, sessions, or skills. Useful when you want a second agent with the same API keys but different personality or gateway tokens.

### Clone everything (`--clone-all`)

```bash
hermes profile create backup --clone-all
```

Copies **everything** — config, memories, sessions, skills, state database, the lot. This is a full snapshot of your current profile. Useful for creating a backup or forking an agent that already has learned context.

## Using profiles

### Shell aliases

Every profile gets an alias installed to `~/.local/bin/`:

```bash
hermes-work       # Runs hermes with the "work" profile
hermes-mybot      # Runs hermes with the "mybot" profile
hermes-backup     # Runs hermes with the "backup" profile
```

These aliases work with all subcommands:

```bash
hermes-work chat -q "Check my calendar"
hermes-work gateway start
hermes-work skills list
```

### Sticky default (`hermes profile use`)

```bash
hermes profile use work
```

Sets "work" as the active profile. Now plain `hermes` uses the work profile — no alias or flag needed. The active profile is stored in `~/.hermes/active_profile`.

Switch back to the default profile:

```bash
hermes profile use default
```

### One-off with `-p` flag

```bash
hermes -p work chat -q "Summarize my inbox"
hermes -p mybot gateway status
```

The `-p` / `--profile` flag overrides the sticky default for a single command without changing it.

## Running gateways

Each profile runs its own independent gateway. This means you can have multiple bots online simultaneously — for example, a personal Telegram bot and a team Discord bot:

```bash
hermes-personal gateway start    # Starts personal bot's gateway
hermes-work gateway start        # Starts work bot's gateway
```

Each gateway uses the tokens and platform config from its own profile's `config.yaml` and `.env`. There are no port or token conflicts because each profile is fully isolated.

:::warning
Each bot token (Telegram, Discord, etc.) can only be used by **one** profile at a time. If two profiles try to use the same token, the second gateway will fail to connect. Use a separate bot token per profile.
:::

## Configuring profiles

Each profile has its own independent configuration files:

```
~/.hermes/profiles/work/
├── config.yaml        # Model, provider, gateway settings
├── .env               # API keys, bot tokens
├── SOUL.md            # Personality / system prompt
├── skills/            # Installed skills
├── memories/          # Agent memories
├── state.db           # Sessions, conversation history
└── logs/              # Gateway and agent logs
```

Edit a profile's config directly:

```bash
hermes-work config edit          # Opens work profile's config.yaml
hermes -p work setup             # Run setup wizard for work profile
```

Or edit the files manually:

```bash
nano ~/.hermes/profiles/work/config.yaml
nano ~/.hermes/profiles/work/.env
nano ~/.hermes/profiles/work/SOUL.md
```

The default profile lives at `~/.hermes/` (not in the `profiles/` subdirectory).

## Updating

```bash
hermes update
```

`hermes update` pulls the latest code and reinstalls dependencies once. It then syncs the updated skills to **all** profiles automatically. You don't need to run update separately for each profile — one update covers everything.

## Managing profiles

### List profiles

```bash
hermes profile list
```

Shows all profiles with their status. The active profile is marked with an asterisk:

```
  default
* work
  mybot
  backup
```

### Show profile details

```bash
hermes profile show work
```

Displays the profile's home directory, config path, active model, configured platforms, and other details.

### Rename a profile

```bash
hermes profile rename mybot assistant
```

Renames the profile directory and updates the shell alias from `hermes-mybot` to `hermes-assistant`.

### Export a profile

```bash
hermes profile export work ./work-backup.tar.gz
```

Packages the entire profile into a portable archive. Useful for backups or transferring to another machine.

### Import a profile

```bash
hermes profile import ./work-backup.tar.gz work-restored
```

Imports a previously exported profile archive as a new profile.

## Deleting a profile

```bash
hermes profile delete mybot
```

Removes the profile directory and its shell alias. You'll be prompted to confirm. This permanently deletes all config, memories, sessions, and skills for that profile.

:::warning
Deletion is irreversible. Export the profile first if you might need it later: `hermes profile export mybot ./mybot-backup.tar.gz`
:::

You cannot delete the currently active profile. Switch to a different one first:

```bash
hermes profile use default
hermes profile delete mybot
```

## Tab completion

Enable shell completions for profile names and subcommands:

```bash
# Generate completions for your shell
hermes completion bash >> ~/.bashrc
hermes completion zsh >> ~/.zshrc
hermes completion fish > ~/.config/fish/completions/hermes.fish

# Reload your shell
source ~/.bashrc   # or ~/.zshrc
```

After setup, `hermes profile <TAB>` autocompletes subcommands and `hermes -p <TAB>` autocompletes profile names.

## How it works

Under the hood, each profile is just a separate `HERMES_HOME` directory. When you run `hermes -p work` or `hermes-work`, Hermes sets `HERMES_HOME=~/.hermes/profiles/work` before starting. Everything — config loading, memory access, session storage, gateway operation — reads from and writes to that directory.

The sticky default (`hermes profile use`) writes the profile name to `~/.hermes/active_profile`. On startup, if no `-p` flag is given, Hermes checks this file and sets `HERMES_HOME` accordingly.

Profile aliases in `~/.local/bin/` are thin wrapper scripts that set `HERMES_HOME` and exec the real `hermes` binary. This means profiles work with all existing Hermes commands, flags, and features without any special handling.
