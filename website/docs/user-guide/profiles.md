---
sidebar_position: 2
---

# Profiles: Running Multiple Agents

Run multiple independent Hermes agents on the same machine — each with its own config, API keys, memory, sessions, skills, and gateway state.

## What are profiles?

A profile is a separate Hermes home directory. Each profile gets its own directory containing its own `config.yaml`, `.env`, `SOUL.md`, memories, sessions, skills, cron jobs, and state database. Hermes recognises a directory under `~/.hermes/profiles/` as a profile only when it carries one of those identity files (`config.yaml`, `.env`, `SOUL.md`, `profile.yaml`, `auth.json`, `state.db`); a bare directory left behind by logging or cron is ignored by `profile list`, gateways and `-p`. Profiles let you run separate agents for different purposes — a coding assistant, a personal bot, a research agent — without mixing up Hermes state.

:::caution Give every agent its own profile
Never point two agent processes at the same profile (the same Hermes home). Both write memory automatically, and each loads the other's writes into its system prompt at session start — so two writers on one home compound each other's state until it stops being anything you configured. Profiles exist exactly to prevent this; agents that need shared memory should use an [external memory provider](./features/memory-providers.md) instead.
:::

When you create a profile, it automatically becomes its own command. Create a profile called `coder` and you immediately have `coder chat`, `coder setup`, `coder gateway start`, etc.

### Profiles, agents, and bots

These terms describe different parts of Hermes:

- **Profile** is the persistent home for an assistant's configuration and data.
  It keeps the same state across conversations and restarts.

- **Agent** is the running Hermes assistant that uses that configuration and
  state. "Hermes Agent" also names the product.

- **Bot Mode bot** is a profile presented as a named entry in the desktop's
  [Bot Mode](./bot-mode.md) roster, with an avatar and a persistent Bot Chat.
  The same profile remains accessible from the CLI. Every Bot is a profile, but
  not every profile is a Bot: a profile becomes a Bot when Bot Mode stores its
  roster presentation (title, avatar, section, hidden state) in the profile's
  metadata and pins its canonical Bot Chat. A profile you only ever drive from
  the CLI, Docker, or a gateway and never add to the roster stays a plain profile.

- **Messaging bot** is an account on a platform such as Telegram, Discord, or
  Slack, connected to Hermes through the gateway. Its
  [bot token](#different-bot-tokens) identifies that platform account.

- **Subagent** is a child assistant spawned by
  [`delegate_task`](./features/delegation.md) for a task, with a fresh
  conversation. A separate conversation is different from a separate profile.

## Quick start

```bash
hermes profile create coder       # creates profile + "coder" command alias
coder setup                       # configure API keys and model
coder chat                        # start chatting
```

That's it. `coder` is now its own Hermes profile with its own config, memory, and state.

## Creating a profile

:::tip
Quickest setup: run `hermes setup --portal` inside the new profile to wire up models + tools at once. See [Nous Portal](../integrations/nous-portal.md).
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

### Clone configuration and identity (`--clone`)

```bash
hermes profile create work --clone
```

Copies your current profile's `config.yaml`, `.env`, `SOUL.md`, skills, and the curated memory files `memories/MEMORY.md` and `memories/USER.md` into the new profile — memory is treated as part of the agent's identity, like `SOUL.md`. It also preserves installed marketplace package files, as described below. Sessions, `state.db`, cron jobs and other runtime state start empty. For a blank memory as well, create the profile without `--clone` or delete the two files afterwards; the agent never falls back to another profile's memory when they are absent. Edit `~/.hermes/profiles/work/.env` for different API keys, or `~/.hermes/profiles/work/SOUL.md` for a different personality.

#### Keep a clone's imported agent setups synced (`--sync-imports`)

If the source profile has run [`hermes import-agent`](./import-from-other-agents.md), its `import-sync.json` records which external Claude Code / Codex trees it imported. `--clone` leaves that manifest behind, so the clone gets a one-off copy of those skills and memories and stops there. Add `--sync-imports` to carry the manifest over:

```bash
hermes profile create work --clone --sync-imports
hermes -p work import-agent --sync        # pulls changes from the same ~/.claude / ~/.codex
```

This is explicit, opt-in and one-directional, and it links the clone to the **external agent trees only** — never to the source profile. Both profiles stay independent islands: editing the source's `config.yaml`, `SOUL.md` or skills afterwards never reaches the clone. `--clone-all` copies the manifest as part of the full copy.

### Clone the broader profile tree (`--clone-all`)

```bash
hermes profile create backup --clone-all
```

Copies the broader profile tree — config, permitted API keys, personality, memories and skills — with the exclusions below. Installed marketplace packages use the same inventory-verified package copy as `--clone`. This is not a complete backup or a guarantee that every plugin is ready to run. Per-profile history is excluded (session history, `state.db`, `backups/`, `state-snapshots/`, `checkpoints/`) — these belong to the source profile and can reach tens of GB. When cloning from the default profile, the local-model runtime trees (`models/`, `runtimes/`, `node/` — downloaded weights and managed binaries, re-fetched on demand) are skipped too, as `hermes backup` already does. **Cron jobs are not cloned** either: they are scheduled work bound to the source profile and its delivery channel, and a clone that inherited them would run every job twice (two gateways, same job ids). The new profile starts with an empty `cron/`. For a full backup including history and cron jobs, use `hermes profile export` or `hermes backup` instead.

:::note Supported single-use OAuth logins are stripped
Anthropic (Claude Pro/Max), OpenAI Codex, and xAI OAuth logins use **single-use refresh tokens** — a copy of one is not a second credential, it is the same credential with two owners, and the first profile to refresh it revokes it for every other copy. `--clone-all` (and the dashboard's credential mirroring) therefore drops those OAuth rows from the clone. Static API keys are copied as usual. Sign the new profile into the OAuth provider itself: `hermes -p <name> auth add <provider>` (or `hermes -p <name> model`).
:::

### Installed marketplace packages in both clone modes

Both `--clone` and `--clone-all` preserve installed marketplace packages under
the source profile's `plugins/` directory, including inactive memory providers
and general plugins. Bundled providers, project-local plugins and pip-installed
packages outside that directory are not copied into private installations.

Each package needs a matching installer-owned file inventory in
`plugins/.install-metadata.json` and matching catalog provenance in
`.hermes-catalog.json`. The installer captures the selected package's pristine
checked-out files before scanning or publication, preserving Git's built-in
line-ending and working-tree-encoding conversions. Updates capture the baseline
before restoring local edits; those edits are not adopted as pristine.
Cloning verifies those files' hashes and modes, then
copies them as independent regular files. A `.git` directory is **not required**:
repository `#subdir` installations and packages from an earlier profile clone
work when their inventory and catalog metadata remain intact.

- The installed revision, saved source (including any `#subdir` selector), pin
  state and catalog identity are preserved. The copied config retains plugin
  enablement and memory-provider selection; cloning does not newly enable a
  package or select a different provider.
- Only inventoried package files and their installation/catalog metadata are
  copied. Git history is not copied. Unlisted files — including runtime state,
  caches and installed dependency trees — stay behind. Files retain their pristine
  recorded POSIX permissions, including `0600`/`0700` from a restrictive umask;
  Windows preserves the recorded executable
  intent in metadata but checks writable files rather than POSIX mode bits.
  Installation/catalog metadata uses `0600` on POSIX. Ownership is determined by
  the inventory, not by guessing from filenames or scanning for embedded secrets.
- Missing, stale or ambiguous installation metadata, missing inventories,
  changed owned bytes or modes, unsafe owned paths, links or special files, and
  missing or conflicting catalog provenance refuse the clone. Packages matching
  the available offline removed-plugin lists are also refused. Unlisted runtime
  trees inside a known package are ignored, not adopted as package content.
- Package copying does not fetch code, refresh the catalog, install dependencies
  or grant plugin consent. Local integrity metadata is **not authenticated
  provenance or a security review**. It does not establish runtime readiness.

The destination retains its own catalog update/remove identity despite having
no Git history. Updating or removing the package in one profile does not change
another profile's copy. Review dependencies and follow the package's setup
instructions separately before using it.

#### Repair a legacy or changed installation

Legacy installations without an installer-owned inventory require an explicit
reinstall in the **source profile** before cloning; Hermes does not reconstruct
ownership from a used package directory, even if it still has Git metadata.
Back up local edits and private state outside `plugins/` first: a force reinstall
replaces the package. Use the exact saved source and installed revision from
`plugins/.install-metadata.json`:

```bash
hermes -p <profile> plugins install '<saved-source-including-#subdir>' --ref <saved-40-character-commit-SHA> --force
```

Keep any `#subdir` selector; do not replace a subdirectory package with its
repository root or silently move to the latest revision. If catalog provenance
is missing or disagrees with the saved source/revision, inspect the install
record and `.hermes-catalog.json` against the intended catalog entry first.
Restore the correct catalog installation through the installer at that same
source/selector and revision; do not invent a sidecar or mark a mismatch as
reviewed. A direct-source reinstall cannot establish missing catalog provenance
by itself. Retry cloning only after the installation metadata agrees.

#### Read the clone report

Every successful clone writes a private `.clone-report.json` in the destination
home (`0600` on POSIX). It is a **package-only receipt** containing
`plugins.copied` (package keys) and `plugins.warnings`. Warnings explain that only
installer-owned files were copied, without Git history, runtime state or
installed dependencies, and that catalog update/remove remain available.

```bash
python3 -m json.tool ~/.hermes/profiles/work/.clone-report.json
```

Replace `work` with the destination profile. This creation receipt is not a live
dependency or authentication health check, and later setup or updates do not
rewrite it.

The source profile's cooperative plugin-installation lock is held through
package copying and publication. Hermes installation writers participate;
arbitrary editors, runtime writers and third-party installers do not. The lock
may create or update `.plugin-installation.lock`, which is not copied. It is
not a transaction over every source file or a filesystem sandbox.

### External memory settings after cloning

Package preservation does not migrate native provider identities or prepare
provider-owned configuration. Existing clone behavior for those files is
unchanged: `--clone` copies the standard config/`.env` files, not arbitrary native
provider files; `--clone-all` copies native files with the broader profile tree,
subject to its existing exclusions. A copied native file is not rewritten to
create a new remote identity, relocate private paths or sanitize provider grants.
Cloning does not invoke a provider's `clone.py`.

The existing CLI-only Honcho post-create helper is unchanged and remains
separate from package copying; HTTP and WebSocket creation do not gain that
helper. Cross-surface parity here covers installed package preservation, not
native provider setup, authentication or remote identity guarantees.

An unchanged external provider that depends on a native configuration file may
therefore need setup in the destination after `--clone`, just as before. Follow
its documented setup flow, for example `hermes -p <name> memory setup`. This is
not a new clone-time readiness guard: a missing native file does not by itself
refuse a clone. With `--clone-all`, review copied native settings and intentional
remote sharing before starting the provider. Successful package copying alone
says nothing about credentials, services or remote identity isolation.

### Every profile owns its credentials

A named profile resolves providers from **its own** `auth.json` and `.env` only. It never inherits the root profile's logins or API keys, and a token refresh inside a profile never writes to the root store. A profile with no provider of its own is asked to set one up (`hermes -p <name> model`, or `hermes -p <name> auth add <provider>`) — it does not silently act as the owner. `hermes update` lists every profile that has no provider of its own so a bot never goes quiet unannounced.

### Clone from a specific profile

```bash
hermes profile create work --clone-from coder
```

`--clone-from <source>` selects the source profile directly and implies `--clone`, including its installed marketplace packages. Combine it with `--clone-all` when you want a full copy of that source profile:

```bash
hermes profile create work-backup --clone-from coder --clone-all
```

### Messaging channels are never cloned (`--clone-channels` to opt in)

Every clone — `--clone`, `--clone-from`, `--clone-all`, and the dashboard / Desktop / TUI
"clone from profile" option — copies the source **without its messaging channels**: bot tokens
and allowlists (`TELEGRAM_BOT_TOKEN`, `DISCORD_ALLOWED_USERS`, `WHATSAPP_ENABLED`,
`API_SERVER_KEY`, `WEBHOOK_SECRET`, …), the `platforms:` / `telegram:` / `discord:` sections of
`config.yaml`, `gateway.multiplex_profiles` / `profile_routes`, and (for `--clone-all`) the
pairing store, WhatsApp session and other per-bot state. Provider and tool API keys, the model
block, memory settings, skills and `SOUL.md` are copied as before. The command prints which
platforms were left behind.

The reason is that a bot can only belong to one profile: two standalone gateways holding the
same token fight over its long-poll, and a [multiplexed gateway](./multi-profile-gateways.md)
parks the duplicate adapter (and `hermes gateway migrate --multiplex` refuses with one
duplicate-credential blocker per platform). Configure the new profile's own bots with
`hermes -p <name> setup` or the dashboard Messaging page.

```bash
hermes profile create twin --clone --clone-channels   # keep the source's bots anyway
```

`--clone-channels` is refused when a running multiplexed gateway already serves the source
(the copy would be parked immediately) — from the CLI, the dashboard and the TUI alike — and
otherwise prints a warning naming the platforms now shared with the source. It is an error
without a clone flag. `hermes profile list` prints the same warning for any existing
profile whose bot credential is byte-identical to the default's, so older clones surface
before they bite.

**What counts as a channel setting** — the inventory is ownership-based and is judged in the
*source* profile's plugin scope (its private `plugins/` adapters included):

- every key an adapter declares (tokens, app/client ids, allowlists, allow-all switches, home
  channels) and every key under its `<PLATFORM>_` prefix, including historical aliases
  (`WECOM_*`, `SMS_*`/`TWILIO_*`, `QQ_*`, `HASS_*`, `EMAIL_*`);
- gateway-wide channel policy `GATEWAY_ALLOW_ALL_USERS` / `GATEWAY_ALLOWED_USERS` and the
  relay enrollment identity `GATEWAY_RELAY_ID` / `GATEWAY_RELAY_SECRET` / `GATEWAY_RELAY_DELIVERY_KEY`;
- for `--clone-all`, per-bot state files **and directories** (`platforms/`, pairing ledgers,
  `google_chat_user_tokens/`, `<platform>_*`).

Credentials that a messaging adapter shares with a non-channel capability — `HASS_TOKEN`/`HASS_URL`
(also the Home Assistant tool), `TWILIO_*` (also the telephony skill), `EMAIL_*` (also
mail-sending scripts) — are stripped **only when the source's gateway would run that adapter**
(the platform is enabled in its `config.yaml`, or its credential set is complete and not
explicitly disabled). A source with `platforms.homeassistant.enabled: false` uses `HASS_TOKEN`
as a tool key, so the clone keeps it. Allowlists and ports under those prefixes are always
channel-only and always stripped.

Clones are built in a hidden staging directory inside `profiles/`, beside the final
profile directory, and published with one rename after package copying and channel stripping,
so a running multiplexer (which rescans `profiles/` on create) can
never start adapters on a half-copied tree. A symlinked source `.env`/`config.yaml` is
materialized as a private copy first — the clone never writes through to the source.
If package validation or clone creation fails, staging is removed and no clone is published.

:::tip Local isolation is not remote isolation
Memory providers may intentionally share a remote user or workspace across profiles. Copying a package does not create a separate remote identity or memory store. Review [external memory settings after cloning](#external-memory-settings-after-cloning) before starting the clone.
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
Per-profile gateways are supervised by [s6-overlay](https://github.com/just-containers/s6-overlay) (PID 1 in the container), so `hermes profile create <name>` automatically registers an s6 service slot at `/run/service/gateway-<name>/`. `hermes -p <name> gateway start/stop/restart` dispatches to `s6-svc` instead of spawning a bare process — crashes are auto-restarted and `docker restart` preserves the previously-running set of gateways. See [Per-profile gateway supervision](./docker.md#per-profile-gateway-supervision) for details.
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
hermes profile migrate-identity coder dev-bot   # retry a rename's identity migration
hermes profile purge-identity dev-bot   # retry a delete's identity purge
hermes profile export coder   # pack into coder.tar.gz (shareable; keys stripped)
hermes profile import coder.tar.gz   # install an archive as a new profile
```

In chat, the same two live as `/export` and `/import` — and in the desktop app as **⌘K → Export/Import profile…**. See [Sharing a profile](#sharing-a-profile).

### Naming the default profile

The default profile's internal ID is always `default` — it can't be truly
renamed because `~/.hermes` is the installation root. Renaming it instead
sets a **display name**, which UI surfaces show in place of the bare ID:

```bash
hermes profile rename default Harumesu   # Unicode fine: 小助手
```

The display name appears in `hermes profile list`/`show`, the `/profile`
chat command, the dashboard, and the desktop app (including the Bot Mode
roster). It is presentation-only: `-p default`, service names, cron jobs,
and every other reference keep using the canonical `default` ID. It is
stored as `display_name` in `~/.hermes/profile.yaml`; remove that line to
revert. Named profiles can carry a `display_name` too (it survives a real
rename), but `rename` for them still renames the profile itself.

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

## Sharing a profile

A profile you built on one machine can go to another — your own workstation, a teammate's laptop, or the community. Two paths:

**Send a file.** `/export` packs the profile into one `.tar.gz` — skills, memory, persona, crons, plugins, settings, and (from the desktop) your theme and layout. API keys are stripped. The recipient runs `/import`.

```bash
# In chat, run /export, hand over the file, and they run /import on it
hermes profile export coder
hermes profile import ./coder.tar.gz --name coder
```

**Publish a distribution.** Package the profile as a **git repository** so recipients install it with one command and pull versioned updates later. Carries the SOUL, config, skills, cron jobs, and MCP connections; credentials, memories, and sessions stay per-machine.

```bash
# Install a whole agent from a git repo
hermes profile install github.com/you/research-bot --alias

# Update later when the author ships a new version (keeps your memories + .env)
hermes profile update research-bot
```

Use an export file for a one-time handoff or a move; use a distribution for an agent you'll keep shipping. See **[Profile Distributions: Share a Whole Agent](./profile-distributions.md)** for both — the comparison table, authoring, publishing, update semantics, and the security model.
