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

Copies your current profile's `config.yaml`, `.env`, `SOUL.md`, skills, and the curated memory files `memories/MEMORY.md` and `memories/USER.md` into the new profile — memory is treated as part of the agent's identity, like `SOUL.md`. It also snapshots verified installed plugins and prepares supported memory-provider settings, as described below. Sessions, `state.db`, cron jobs and other runtime state start empty. For a blank memory as well, create the profile without `--clone` or delete the two files afterwards; the agent never falls back to another profile's memory when they are absent. Edit `~/.hermes/profiles/work/.env` for different API keys, or `~/.hermes/profiles/work/SOUL.md` for a different personality.

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

Copies the broader profile tree — config, permitted API keys, personality, memories and skills — with the exclusions below. Installed plugins use the same verified code-only snapshot path as `--clone`; supported memory providers prepare their local settings before publication. This is not a complete backup or a guarantee that every plugin is ready to run. Per-profile history is excluded (session history, `state.db`, `backups/`, `state-snapshots/`, `checkpoints/`) — these belong to the source profile and can reach tens of GB. When cloning from the default profile, the local-model runtime trees (`models/`, `runtimes/`, `node/` — downloaded weights and managed binaries, re-fetched on demand) are skipped too, as `hermes backup` already does. **Cron jobs are not cloned** either: they are scheduled work bound to the source profile and its delivery channel, and a clone that inherited them would run every job twice (two gateways, same job ids). The new profile starts with an empty `cron/`. For a full backup including history and cron jobs, use `hermes profile export` or `hermes backup` instead.

:::note Supported single-use OAuth logins are stripped
Anthropic (Claude Pro/Max), OpenAI Codex, and xAI OAuth logins use **single-use refresh tokens** — a copy of one is not a second credential, it is the same credential with two owners, and the first profile to refresh it revokes it for every other copy. `--clone-all` (and the dashboard's credential mirroring) therefore drops those OAuth rows from the clone. Static API keys are copied as usual. Sign the new profile into the OAuth provider itself: `hermes -p <name> auth add <provider>` (or `hermes -p <name> model`).
:::

### Installed plugins in both clone modes

Both `--clone` and `--clone-all` snapshot the source profile's installed
`plugins/` packages, including **inactive memory providers and general plugins**.
They do not just copy the currently selected memory provider. Bundled providers,
project-local plugins, and pip-installed packages outside that directory are not
turned into private installations by cloning.

A package must have matching installation metadata and either a clean standalone
Git checkout or an intact Hermes code-snapshot manifest from an earlier clone.
Hermes verifies the recorded revision, owned files and executable modes; repeat
clones verify the snapshot manifest and file hashes. Git-owned code keeps its
`0644` or `0755` mode; snapshot and installation metadata use `0600` on POSIX.
The clone gets independent
regular files, not links back to the source. Updating or removing a package in
one profile does not update or remove the other's copy.

- Git internals and recognized runtime/secret-like paths, including `.env`, caches,
  `.venv/` and `node_modules/`, are excluded. A tracked file matching these path
  rules is refused rather than silently treated as code. These are ownership and
  path checks, not a content scanner that detects every embedded secret.
- Source, installed revision, pin state and applicable catalog provenance are
  retained in `plugins/.install-metadata.json` and, when applicable, the package's
  `.hermes-catalog.json`. Each code snapshot also has a `.hermes-snapshot.json`
  file/hash manifest. A catalog record that disagrees with the installed source
  or revision is kept
  as **historical** metadata with a warning, not represented as current approval.
- Unknown/manual installations, missing or ambiguous metadata, subdirectory-only
  installs, dirty or unowned code, unsupported links and snapshots without a
  valid integrity manifest fail closed. Packages matching the available offline
  removed-plugin lists are also refused. The clone is not published on failure.
- Cloning does not fetch packages, refresh the catalog, install dependencies,
  grant plugin consent or newly enable plugins. Local integrity metadata proves
  consistency with the recorded local snapshot, **not authenticated provenance
  or a security audit**. Dependency readiness is not established.

The snapshot has no Git checkout, so normal Git updates are not available until
you explicitly reinstall the package in the destination. Repair is a separate,
potentially networked action, never an automatic part of cloning. Back up local
changes and private data outside `plugins/` first, then follow the refusal's
repair instructions. For a supported standalone repository, reinstall in the
profile being repaired with
`hermes -p <profile> plugins install <source> --ref <40-character commit SHA> --force`.
For nested or subdirectory-only installations, first move the package outside
`plugins/` and remove only its matching `.install-metadata.json` entry; do not
reinstall the same unsupported subdirectory layout. The installer creates a flat
`plugins/<package>` installation: update configuration referring to an old nested
path, and remove an old category directory only if empty. For a subdirectory-only
source, omit its `#subdir` selector. The repository root must
itself be a supported plugin, otherwise obtain a standalone package from its author.

Review and install the destination package's dependencies separately using its
documented setup. A copied package or a successful reinstall alone does not prove
its dependencies, credentials, services or tools are ready.

#### Read the clone report

Every successful clone writes a private `.clone-report.json` in the destination
home (`0600` on POSIX). It records `needs_auth` (provider names), `plugins.copied`
(package keys), and `plugins.warnings`. Each copied package gets a warning that
it is a code-only snapshot without Git history or copied dependencies, plus
repair guidance for restoring updates. Historical catalog provenance produces
an additional warning. Read this report even when creation succeeds:

```bash
python3 -m json.tool ~/.hermes/profiles/work/.clone-report.json
```

Replace `work` with the destination profile. The CLI prints authentication needs;
read the file for the full package warnings and repair instructions. The report
is a creation receipt, not a live dependency or authentication health check: a
later sign-in or reinstall does not update it. Follow the separate reinstall and
dependency steps above only for packages that need repair; clone creation itself
does not perform those actions.

Cloning holds the source profile's cooperative plugin-installation lock through
snapshot, provider preparation and publication. Hermes installation writers
participate; arbitrary editors and third-party installers do not. This is not a
transaction over every source file or protection against arbitrary concurrent
filesystem mutation. The source's `.plugin-installation.lock` may be created or
updated for coordination and is not copied. This lock file is the exception to
the clone's no-source-data-writes contract; source configuration, credentials and
plugin code are not rewritten by clone preparation.

### Memory-provider preparation and limits

An optional provider-owned `clone.py` prepares native settings and removes
source-owned run state **before** the staged profile becomes visible. This runs
for installed providers, including inactive ones, without initializing a runtime
provider, refreshing OAuth or provisioning remote resources. Providers can report
`needs_auth`; sign in from the destination profile when prompted.

Companion Python code and relative helpers are pinned to a content generation.
After a package replacement changes those Python bytes at the same path, the
next companion load uses a new namespace; already-loaded companions retain their
old Python generation, including delayed relative imports. Existing runtime
provider objects are not reloaded or replaced. This does not pin non-Python assets
or state imported through absolute module names.

The selected provider must still be installed; cloning refuses a missing selected
package rather than silently replacing the memory backend. Legacy config/`.env`-only
providers keep their existing clone behavior. When an
installed/configured provider has recognizable native state but no usable clone
companion, cloning refuses rather than guessing how to sanitize it. Detection is
bounded to known provider names and matching `<name>/` or `<name>.json` paths:
**unknown or retired native files are not comprehensively discovered or sanitized**.
In particular, `--clone-all` is not a generic credential scrubber. Review unusual
provider data yourself, or create a blank profile and configure it separately.

- **Honcho:** prepares a destination AI peer name and session AI-peer prefix in
  local config while preserving the source's user identity, workspace and supported
  tuning. It does not eagerly create a remote peer. Permitted root/`.env` static
  keys can travel; host-only keys and OAuth grants do not. Root/selected-host
  OAuth objects and recognized `hch-at-`/`hch-rt-` tokens trigger reauthentication,
  including a grant whose access token is missing. When effective auth is
  removed, Honcho is disabled in the clone and reported as needing authentication,
  rather than falling back to a different account.
- **OpenViking:** preserves the configured remote account/user and optional peer;
  an absent peer stays absent, so user-memory sharing remains intentional. Local
  `openviking/pending_sessions/` and `openviking/runs/` are excluded. A linked
  profile-private CLI config is materialized privately with its pointer relocated;
  an external/global config link remains shared by intent. Cloning does not create
  a separate remote memory store.

Provider authors: see [offline clone companions](/developer-guide/memory-provider-plugin#offline-clone-companions).


### Every profile owns its credentials

A named profile resolves providers from **its own** `auth.json` and `.env` only. It never inherits the root profile's logins or API keys, and a token refresh inside a profile never writes to the root store. A profile with no provider of its own is asked to set one up (`hermes -p <name> model`, or `hermes -p <name> auth add <provider>`) — it does not silently act as the owner. `hermes update` lists every profile that has no provider of its own so a bot never goes quiet unannounced.

### Clone from a specific profile

```bash
hermes profile create work --clone-from coder
```

`--clone-from <source>` selects the source profile directly and implies `--clone`, including its plugin snapshot and memory-provider preparation. Combine it with `--clone-all` when you want a full copy of that source profile:

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

#### Offline inventory can refuse a clone

Channel ownership is read from manifests and Python declarations **without
importing or activating adapters**, including disabled installations. The reader
supports direct literal `register_platform` / `PlatformEntry` declarations,
imported symbol aliases, unambiguous literal bindings, and inspectable local or
one-level relative-import helpers. It validates explicit calls in registration
scopes and collects declarations from that same helper graph, including all
syntactic branches rather than guessing which will run. This is a bounded
declaration grammar, not an interpreter for arbitrary plugin code.

Unknown or unreadable ownership is a refusal, not permission to keep potentially
shared bot credentials. Malformed manifests, unresolved entry points, opaque or
recursive helpers, callable aliases, reflection, generated code and nonliteral
ownership declarations can therefore block cloning without channels. Repair the
reported installation/metadata or ask its author for supported literal
declarations, then retry; alternatively create a blank profile. Hermes does not
activate the plugin or go online to discover what the missing declaration means.

Enabled external secret sources are checked without fetching secrets. Bulk or
unknown sources whose output names cannot be bounded offline **refuse a clone
without channels**, because they could restore stripped tokens on the next run.
An empty cache, `preserve_existing`, `secrets.sources` ordering or a plugin's
`shape: mapped` declaration does not make such a source safe. Disable that source
in the source profile's `secrets` configuration before retrying, or configure a
blank profile independently.

The supported exception is an enabled 1Password source's explicit
`secrets.onepassword.env` map: channel key mappings are stripped from the destination while non-channel provider/tool
references are preserved. References remain opaque; cloning does not resolve
`op://` values. Shared adapter credentials use the ownership rule below, including
enabled 1Password mappings when checking whether the credential set is complete.

Credentials that a messaging adapter shares with a non-channel capability — `HASS_TOKEN`/`HASS_URL`
(also the Home Assistant tool), `TWILIO_*` (also the telephony skill), `EMAIL_*` (also
mail-sending scripts) — are stripped **only when the source's gateway would run that adapter**
(the platform is enabled in its `config.yaml`, or its credential set is complete and not
explicitly disabled). A source with `platforms.homeassistant.enabled: false` uses `HASS_TOKEN`
as a tool key, so the clone keeps it. Allowlists and ports under those prefixes are always
channel-only and always stripped.

Clones are built in a hidden staging directory inside `profiles/`, beside the final
profile directory, and published with one rename after stripping and preparation,
so a running multiplexer (which rescans `profiles/` on create) can
never start adapters on a half-copied tree. A symlinked source `.env`/`config.yaml` is
materialized as a private copy first — the clone never writes through to the source.
If inventory or preparation fails, staging is removed and no clone is published.

:::tip Local isolation is not remote isolation
Memory providers may intentionally share a remote user or workspace across profiles. Clone preparation separates local run ownership, not every remote resource. Review [memory-provider preparation and limits](#memory-provider-preparation-and-limits) before starting the clone.
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
