---
sidebar_position: 2
---

# Profiles: Running Multiple Agents

Run multiple independent Hermes agents on the same machine — each with its own config, API keys, memory, sessions, skills, and gateway state.

## What are profiles?

A profile is a separate Hermes home directory. Each profile gets its own directory containing its own `config.yaml`, `.env`, `SOUL.md`, memories, sessions, skills, cron jobs, and state database. Profiles let you run separate agents for different purposes — a coding assistant, a personal bot, a research agent — without mixing up Hermes state.

When you create a profile, it automatically becomes its own command. Create a profile called `coder` and you immediately have `coder chat`, `coder setup`, `coder gateway start`, etc.

## Quick start

```bash
hermes profile create coder       # creates profile + "coder" command alias
coder setup                       # configure API keys and model
coder chat                        # start chatting
```

That's it. `coder` is now its own Hermes profile with its own config, memory, and state.

## Creating a profile

:::tip
Quickest setup: run `hermes setup --portal` inside the new profile to wire up models + tools at once. See [Nous Portal](/integrations/nous-portal).
:::

### Blank profile

```bash
hermes profile create mybot
```

Creates a fresh profile with bundled skills seeded. Run `mybot setup` to configure API keys, model, and gateway tokens.

If you plan to use this profile as a kanban worker (or want the kanban orchestrator to route work to it), pass `--description "<role>"` at create time so the orchestrator knows what it's good at:

```bash
hermes profile create researcher --description "Reads source code and external docs, writes findings."
```

You can also set or auto-generate the description later with `hermes profile describe` — see the [Kanban guide](./features/kanban#auto-vs-manual-orchestration) for the full routing model.

### Clone config only (`--clone`)

```bash
hermes profile create work --clone
```

Copies your current profile's `config.yaml`, `.env`, `SOUL.md`, and skills into the new profile. Same API keys, model, and capabilities, but fresh sessions and memory. Edit `~/.hermes/profiles/work/.env` for different API keys, or `~/.hermes/profiles/work/SOUL.md` for a different personality.

### Clone everything (`--clone-all`)

```bash
hermes profile create backup --clone-all
```

Copies **everything** — config, API keys, personality, all memories, skills, cron jobs, plugins. A complete working snapshot. Per-profile history is excluded (session history, `state.db`, `backups/`, `state-snapshots/`, `checkpoints/`) — these belong to the source profile and can reach tens of GB. For a full backup including history, use `hermes profile export` or `hermes backup` instead.

### Clone from a specific profile

```bash
hermes profile create work --clone-from coder
```

`--clone-from <source>` selects the source profile directly and implies a config/skills/SOUL clone. Combine it with `--clone-all` when you want a full copy of that source profile:

```bash
hermes profile create work-backup --clone-from coder --clone-all
```

:::tip Honcho memory + profiles
When Honcho is enabled, clone operations automatically create a dedicated AI peer for the new profile while sharing the same user workspace. Each profile builds its own observations and identity. See [Honcho -- Multi-agent / Profiles](./features/memory-providers.md#honcho) for details.
:::

## Using profiles

### Command aliases

Every profile automatically gets a command alias at `~/.local/bin/<name>`:

```bash
coder chat                    # chat with the coder agent
coder setup                   # configure coder's settings
coder gateway start           # start coder's gateway
coder doctor                  # check coder's health
coder skills list             # list coder's skills
coder config set model.default anthropic/claude-sonnet-4
```

The alias works with every hermes subcommand — it's just `hermes -p <name>` under the hood.

### The `-p` flag

You can also target a profile explicitly with any command:

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # works in any position
```

### Sticky default (`hermes profile use`)

```bash
hermes profile use coder
hermes chat                   # now targets coder
hermes tools                  # configures coder's tools
hermes profile use default    # switch back
```

Sets a default so plain `hermes` commands target that profile. Like `kubectl config use-context`.

### Knowing where you are

The CLI always shows which profile is active:

- **Prompt**: `coder ❯` instead of `❯`
- **Banner**: Shows `Profile: coder` on startup
- **`hermes profile`**: Shows current profile name, path, model, gateway status

## Profiles vs workspaces vs sandboxing

Profiles are often confused with workspaces or sandboxes, but they are different things:

- A **profile** gives Hermes its own state directory: `config.yaml`, `.env`, `SOUL.md`, sessions, memory, logs, cron jobs, and gateway state.
- A **workspace** or **working directory** is where terminal commands start. That is controlled separately by `terminal.cwd`.
- A **sandbox** is what limits filesystem access. Profiles do **not** sandbox the agent.

On the default `local` terminal backend, the agent still has the same filesystem access as your user account. A profile does not stop it from accessing folders outside the profile directory.

If you want a profile to start in a specific project folder, set an explicit absolute `terminal.cwd` in that profile's `config.yaml`:

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

Using `cwd: "."` on the local backend means "the directory Hermes was launched from", not "the profile directory".

Also note:

- `SOUL.md` can guide the model, but it does not enforce a workspace boundary.
- Changes to `SOUL.md` take effect cleanly on a new session. Existing sessions may still be using the old prompt state.
- Asking the model "what directory are you in?" is not a reliable isolation test. If you need a predictable starting directory for tools, set `terminal.cwd` explicitly.

## Running gateways

Each profile runs its own gateway as a separate process with its own bot token:

```bash
coder gateway start           # starts coder's gateway
assistant gateway start       # starts assistant's gateway (separate process)
```

### Different bot tokens

Each profile has its own `.env` file. Configure a different Telegram/Discord/Slack bot token in each:

```bash
# Edit coder's tokens
nano ~/.hermes/profiles/coder/.env

# Edit assistant's tokens
nano ~/.hermes/profiles/assistant/.env
```

### Safety: token locks

If two profiles accidentally use the same bot token, the second gateway will be blocked with a clear error naming the conflicting profile. Supported for Telegram, Discord, Slack, WhatsApp, and Signal.

### Persistent services

```bash
coder gateway install         # creates hermes-gateway-coder systemd/launchd service
assistant gateway install     # creates hermes-gateway-assistant service
```

Each profile gets its own service name. They run independently.

:::note Inside the official Docker image
Per-profile gateways are supervised by [s6-overlay](https://github.com/just-containers/s6-overlay) (PID 1 in the container), so `hermes profile create <name>` automatically registers an s6 service slot at `/run/service/gateway-<name>/`. `hermes -p <name> gateway start/stop/restart` dispatches to `s6-svc` instead of spawning a bare process — crashes are auto-restarted and `docker restart` preserves the previously-running set of gateways. See [Per-profile gateway supervision](/user-guide/docker#per-profile-gateway-supervision) for details.
:::

## Configuring profiles

Each profile has its own:

- **`config.yaml`** — model, provider, toolsets, all settings
- **`.env`** — API keys, bot tokens
- **`SOUL.md`** — personality and instructions

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

If you want this profile to work in a specific project by default, also set its own `terminal.cwd`:

```bash
coder config set terminal.cwd /absolute/path/to/project
```

### From the dashboard

The [web dashboard](features/web-dashboard.md#managing-multiple-profiles)
is a machine-level surface that can manage **any** profile's config, API
keys, skills, MCPs, and model via the profile switcher in its sidebar — no
per-profile dashboard needed. `coder dashboard` routes to the machine
dashboard with the `coder` profile preselected. The dashboard's Chat tab
also follows the switcher, spawning a conversation under the selected
profile's home.

Note: "Set as active" on the dashboard's Profiles page is the sticky
default for **future CLI/gateway runs** (same as `hermes profile use`) —
to edit a profile from the dashboard, use the switcher instead.

## Updating

`hermes update` pulls code once (shared) and syncs new bundled skills to **all** profiles automatically:

```bash
hermes update
# → Code updated (12 commits)
# → Skills synced: default (up to date), coder (+2 new), assistant (+2 new)
```

User-modified skills are never overwritten.

## Managing profiles

```bash
hermes profile list           # show all profiles with status
hermes profile show coder     # detailed info for one profile
hermes profile rename coder dev-bot   # rename (updates alias + service)
hermes profile export coder   # export to coder.tar.gz
hermes profile import coder.tar.gz   # import from archive
```

Profiles can also have a human-friendly nickname, which other agents will be able to find and recognize when referenced. Set it in the profile's `profile.yaml`, for example `~/.hermes/profiles/coder/profile.yaml`:

```yaml
nickname: Artemis
```

## Deleting a profile

```bash
hermes profile delete coder
```

This stops the gateway, removes the systemd/launchd service, removes the command alias, and deletes all profile data. You'll be asked to type the profile name to confirm.

Use `--yes` to skip confirmation: `hermes profile delete coder --yes`

:::note
You cannot delete the default profile (`~/.hermes`). To remove everything, use `hermes uninstall`.
:::

## Tab completion

```bash
# Bash
eval "$(hermes completion bash)"

# Zsh
eval "$(hermes completion zsh)"
```

Add the line to your `~/.bashrc` or `~/.zshrc` for persistent completion. Completes profile names after `-p`, profile subcommands, and top-level commands.

## How it works

Profiles use the `HERMES_HOME` environment variable. When you run `coder chat`, the wrapper script sets `HERMES_HOME=~/.hermes/profiles/coder` before launching hermes. Since 119+ files in the codebase resolve paths via `get_hermes_home()`, Hermes state automatically scopes to the profile's directory — config, sessions, memory, skills, state database, gateway PID, logs, and cron jobs.

This is separate from terminal working directory. Tool execution starts from `terminal.cwd` (or the launch directory when `cwd: "."` on the local backend), not automatically from `HERMES_HOME`.

On host installs, tool subprocesses keep your real OS-user `HOME` by default so
existing CLI credentials under `~` keep working across profiles. Profile data is
isolated by `HERMES_HOME`, not by changing `HOME`. Container backends still use
`{HERMES_HOME}/home` for persistent tool state, and host users who need strict
per-profile tool config can opt in with `terminal.home_mode: profile`.

This means two things that are easy to mix up:

- `HERMES_HOME` is the profile boundary. It controls Hermes config, `.env`,
  memory, sessions, skills, logs, cron jobs, gateway state, and other Hermes
  data.
- `HOME` is the operating-system/user home that external CLIs expect. On host
  installs, Hermes keeps it as the real user home by default so tools like
  `git`, `ssh`, `gh`, `az`, `npm`, Claude Code, and Codex find the same
  credentials they use in your normal shell.

The tradeoff is that host profiles share normal user-level CLI state by default.
If you need separate CLI identities per profile, set `terminal.home_mode:
profile` in that profile's `config.yaml`. In that mode Hermes launches tool
subprocesses with `HOME={HERMES_HOME}/home`; you then need to initialize or link
the profile-specific `~/.ssh`, `~/.gitconfig`, `~/.config/gh`, cloud CLI auth,
Claude/Codex auth, npm state, and similar files inside that profile home.

Hermes also exposes `HERMES_REAL_HOME` to subprocesses so scripts can still find
the actual account home when `home_mode: profile` is active.

The default profile is simply `~/.hermes` itself. No migration needed — existing installs work identically.

## Sharing profiles as distributions

A profile you built on one machine can be packaged as a **git repository** and installed with one command on another machine — your own workstation, a teammate's laptop, or a community user's environment. The shared package includes the SOUL, config, skills, cron jobs, and MCP connections. Credentials, memories, and sessions stay per-machine.

```bash
# Install a whole agent from a git repo
hermes profile install github.com/you/research-bot --alias

# Update later when the author ships a new version (keeps your memories + .env)
hermes profile update research-bot
```

See **[Profile Distributions: Share a Whole Agent](./profile-distributions.md)** for the full guide — authoring, publishing, update semantics, security model, and use cases.
