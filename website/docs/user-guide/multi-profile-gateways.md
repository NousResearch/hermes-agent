---
sidebar_position: 4
---

# Running Many Gateways at Once

Operate multiple [profiles](./profiles.md) — each with its own bot tokens,
sessions, and memory — as managed services on a single machine. This page
covers the operational concerns: starting them all together, viewing logs
across profiles, preventing the host from sleeping, and recovering from common
launchd/systemd quirks.

If you only run one Hermes agent, you don't need this page — see
[Profiles](./profiles.md) for the basics.

## When to use this

You want this setup when you have two or more Hermes agents that should all
be online at the same time. Common reasons:

- A personal assistant on one Telegram bot and a coding agent on another
- One agent per family member or one per Slack workspace
- Sandbox + production instances of the same configuration
- A research agent + a writing agent + a cron-driven bot — each with isolated
  memory and skills

Every profile already gets its own per-platform LaunchAgent
(`ai.hermes.gateway-<name>.plist`) or systemd user service
(`hermes-gateway-<name>.service`). This guide adds the patterns for managing
them collectively.

## Quick start

```bash
# Create profiles (once)
hermes profile create coder
hermes profile create personal-bot
hermes profile create research

# Configure each
coder setup
personal-bot setup
research setup

# Install each gateway as a managed service
coder gateway install
personal-bot gateway install
research gateway install

# Start them all
coder gateway start
personal-bot gateway start
research gateway start
```

That's it — three independent agents, each on its own process, restarting
automatically on crash and on user login.

## Alternative: one gateway for all profiles (multiplexing)

The model above runs **one process per profile**. That is the default and is
the right choice for most setups. But on a host with many profiles — or a
container deployment where one process per profile is operationally heavy — you
can instead run a **single multiplexing gateway**: the default profile's gateway
becomes the sole inbound process and serves messages for *every* profile on the
box.

This is **opt-in** and **off by default**. When it's off, nothing on this page
changes — every behavior below is inert.

### When to prefer multiplexing

- A container/VPS deployment where N supervisor units, N ports, and N PID files
  are a burden.
- Many low-traffic profiles that don't each justify a full process.
- You want a single thing to start, monitor, and restart.

Stick with one-process-per-profile when you want hard process-level isolation
between profiles (separate memory footprints, independent crash domains, the
ability to restart one profile without touching the others).

### How to opt in

Set the flag on the **default profile** (it owns the multiplexer) and restart
its gateway:

```bash
hermes config set gateway.multiplex_profiles true
hermes gateway restart
```

Equivalently, in the default profile's `~/.hermes/config.yaml`:

```yaml
gateway:
  multiplex_profiles: true
```

(The flag is also accepted as a top-level `multiplex_profiles: true` for
convenience.) On the next start the default gateway enumerates every profile,
brings up each profile's enabled platforms under that profile's own
credentials, and routes each inbound message to the profile it belongs to. Each
turn resolves the routed profile's config, skills, memory, SOUL, **and provider
keys** — credentials are never shared across profiles.

You do **not** run `hermes gateway start` for the secondary profiles — the
default gateway serves them. See the contract changes below.

### What changes when multiplexing is on

Enabling the flag changes how a few things behave. All of these revert the
moment the flag is off.

#### 1. Secondary profiles must not start their own gateway

With a multiplexer running, a named-profile `hermes gateway start` / `run` is a
**hard error**, pointing you back at the multiplexer:

```
The default gateway is running as a profile multiplexer and already serves
profile 'coder'. ...
```

The multiplexer is the single inbound process; a second profile gateway would
double-bind that profile's platforms. Pass `--force` only if you deliberately
want a separate process for that profile (not recommended while the multiplexer
is running). The cross-profile lifecycle wrapper script earlier on this page is
therefore **not** used in multiplex mode — you only manage the default gateway.

#### 2. HTTP-inbound platforms are reached via a `/p/<profile>/` URL prefix

Webhook (and other HTTP-inbound) traffic for a secondary profile arrives on the
default listener under a profile prefix, **not** a second port:

```
# default profile
POST http://host:8644/webhooks/<route>
# the "coder" profile, same listener
POST http://host:8644/p/coder/webhooks/<route>
```

An unknown or unconfigured profile in the prefix returns `404`. Because the one
shared listener already serves every profile this way, a **secondary profile
must not enable a port-binding platform itself** — doing so is a config error
that skips the entire secondary profile while the default and other healthy
profiles continue. The warning names the skipped profile and every conflicting
platform:

```
Skipping secondary profile 'coder' due to port-binding config error: Profile
'coder' enables port-binding platform(s) webhook, but gateway.multiplex_profiles
is on. ... Remove these platform entries from profile 'coder's config.yaml or
configure them only on the default profile.
```

Port-binding platforms covered by this rule: `webhook`, `api_server`,
`msgraph_webhook`, `feishu`, `wecom_callback`, `bluebubbles`, `sms`,
`whatsapp_cloud`, `line`. Configure any of these **only on the default profile**;
every profile is reachable through its `/p/<profile>/` prefix.

Only this shared-listener conflict degrades to a skipped profile. Security
configuration errors remain fatal: for example, an `open` own-policy platform
without `GATEWAY_ALLOW_ALL_USERS` or its platform-specific allow-all opt-in
still aborts gateway startup rather than silently dropping the unsafe profile.

#### 3. Per-credential platforms still need their own token per profile

Polling/connection platforms (Telegram, Discord, Slack, Matrix, Signal, …) work
fine multiplexed, but each profile that enables one must supply its **own** bot
token — the same token cannot be polled by two profiles at once. If two profiles
configure the same `(platform, token)`, startup fails fast naming both profiles
(see [Token-conflict safety](#token-conflict-safety) — the rule is unchanged,
it's just enforced inside the one process now).

#### 4. Session keys are namespaced by profile

Each profile's sessions live under an `agent:<profile>:…` namespace so two
profiles on the same platform/chat never collide in the shared session store.
The **default** profile keeps the historical `agent:main:…` namespace
byte-for-byte, so existing default-profile sessions are unaffected — no
migration, no orphaned history.

#### 5. One PID/lock and one status surface

There is a single process-level PID and lock (the multiplexer, under the default
home). `hermes status` reports the multiplexer and the profiles it serves;
`hermes status -p <name>` slices to one profile. Each profile still writes its
own `runtime_status.json` under its own home, so existing per-profile readers
keep working.

#### What does **not** change

Per-profile `.env` credential isolation is preserved and, if anything,
stricter: a profile's keys are resolved from its own scope and are never unioned
into a shared environment (this also means subprocesses like MCP servers and
Kanban workers only ever see their own profile's secrets). Kanban,
profile-scoped skills/memory/SOUL, and model routing all behave per-profile
exactly as they do with separate gateways.

### Routing shared-bot chats to profiles (`profile_routes`)

Multiplexing selects a profile per **credential** (each profile's own bot
token) or per **URL prefix** (`/p/<profile>/` for HTTP platforms). When several
communities share **one** bot token — for example one Discord bot serving many
guilds — you can additionally route specific guilds/channels/threads to
different profiles with `gateway.profile_routes`:

```yaml
gateway:
  multiplex_profiles: true
  profile_routes:
    # An entire Discord server → one profile
    - name: acme-server
      platform: discord
      guild_id: "1234567890"
      profile: acme

    # One channel in that server → a different profile
    - name: acme-support
      platform: discord
      guild_id: "1234567890"
      chat_id: "9876543210"
      profile: acme-support

    # A Telegram group (no guild concept — chat_id only)
    - name: tg-group
      platform: telegram
      chat_id: "-1001234567890"
      profile: tg-profile
```

Routes are matched most-specific-first (`thread_id` > `chat_id` > `guild_id`),
all declared fields must hold (AND), and a route keyed on a channel also
matches threads/forum posts whose parent is that channel. Messages that match
no route stay on the default/active profile. The routed profile gets the full
per-profile isolation described above (config, skills, memory, credentials,
session namespace). Routing works on every platform adapter, not just Discord.

`profile_routes` requires `gateway.multiplex_profiles: true`; with
multiplexing off the routes are ignored. If a route names a profile that does
not exist on disk, the gateway logs a warning naming the profile and source and
falls back to the default home.

## Start, stop, or restart all gateways at once

The CLI ships with single-profile lifecycle commands. To act across every
profile, wrap them in a shell loop. Put the snippet below in
`~/.local/bin/hermes-gateways` and `chmod +x` it:

```sh
#!/bin/sh
set -eu

# Add or remove profile names here as you create / delete profiles.
profiles="default coder personal-bot research"

usage() {
  echo "Usage: hermes-gateways {start|stop|restart|status|list}"
}

run_for_profile() {
  profile="$1"
  action="$2"
  if [ "$profile" = "default" ]; then
    hermes gateway "$action"
  else
    hermes -p "$profile" gateway "$action"
  fi
}

action="${1:-}"
case "$action" in
  start|stop|restart|status)
    for profile in $profiles; do
      echo "==> $action $profile"
      run_for_profile "$profile" "$action"
    done
    ;;
  list)
    hermes gateway list
    ;;
  *)
    usage
    exit 2
    ;;
esac
```

Then:

```bash
hermes-gateways start      # start every configured profile
hermes-gateways stop       # stop every configured profile
hermes-gateways restart    # restart all
hermes-gateways status     # status across all
hermes-gateways list       # delegates to `hermes gateway list`
```

:::tip
The `default` profile is targeted with `hermes gateway <action>` (no `-p`),
not `hermes -p default gateway <action>`. The wrapper above handles both forms.
:::

## Manage one profile

The shortcut commands every profile installs:

```bash
coder gateway run        # foreground (Ctrl-C to stop)
coder gateway start      # start the managed service
coder gateway stop       # stop the managed service
coder gateway restart    # restart
coder gateway status     # status
coder gateway install    # create the LaunchAgent / systemd unit
coder gateway uninstall  # remove the service file
```

These are equivalent to `hermes -p coder gateway <action>` — useful if a
profile alias is not on `PATH` or if you target profiles dynamically from a
script.

## Service files

Each profile installs its own service with a unique name, so installations
never clash:

| Platform | Path                                                              |
| -------- | ----------------------------------------------------------------- |
| macOS    | `~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist`        |
| Linux    | `~/.config/systemd/user/hermes-gateway-<profile>.service`         |

The default profile keeps the historical names: `ai.hermes.gateway.plist` /
`hermes-gateway.service`.

## Viewing logs

Each profile writes to its own log files:

```bash
# Default profile
tail -f ~/.hermes/logs/gateway.log
tail -f ~/.hermes/logs/gateway.error.log

# Named profile
tail -f ~/.hermes/profiles/<name>/logs/gateway.log
tail -f ~/.hermes/profiles/<name>/logs/gateway.error.log
```

Stream every profile's log simultaneously:

```bash
tail -f ~/.hermes/logs/gateway.log ~/.hermes/profiles/*/logs/gateway.log
```

The CLI also has a structured log viewer:

```bash
hermes logs -f                  # follow default profile
hermes -p coder logs -f         # follow one profile
hermes logs --help              # filters, levels, JSON output
```

## Identify what's actually running

```bash
hermes profile list             # profiles + model + gateway state
hermes-gateways status          # full status across every profile
launchctl list | grep hermes    # macOS — PIDs and labels
systemctl --user list-units 'hermes-gateway-*'   # Linux — units
```

## Editing configuration

Every profile keeps its config inside its own directory:

```
~/.hermes/profiles/<name>/
├── .env              # API keys, bot tokens (chmod 600)
├── config.yaml       # model, provider, toolsets, gateway settings
└── SOUL.md           # personality / system prompt
```

The default profile uses `~/.hermes/` directly with the same three files.

Edit them with any editor or via the CLI:

```bash
hermes config set model.model anthropic/claude-sonnet-4    # default profile
coder config set model.model openai/gpt-5                  # named profile
```

After editing `.env` or `config.yaml`, restart the affected gateway:

```bash
coder gateway restart
# or, for everything:
hermes-gateways restart
```

## Keeping the host awake

The gateway process can run all day, but the operating system will still try
to sleep when idle. Two patterns:

### macOS — `caffeinate`

`caffeinate` is built into macOS and prevents sleep while it runs. No install.

```bash
caffeinate -dis                    # block display, idle, and system sleep
caffeinate -dis -t 28800           # same, auto-exit after 8 hours
caffeinate -i -w $(cat ~/.hermes/gateway.pid) &   # awake while default gateway runs

# Persistent: run in background and forget
nohup caffeinate -dis >/dev/null 2>&1 &
disown

# Inspect / stop
pmset -g assertions | grep -iE 'caffeinate|prevent|user is active'
pkill caffeinate
```

| Flag   | Effect                                            |
| ------ | ------------------------------------------------- |
| `-d`   | block display sleep                               |
| `-i`   | block idle system sleep (default)                 |
| `-m`   | block disk sleep                                  |
| `-s`   | block system sleep (AC-powered Macs only)         |
| `-u`   | simulate user activity (prevents screen lock)     |
| `-t N` | auto-exit after `N` seconds                       |
| `-w P` | exit when PID `P` exits                           |

:::warning Lid-close still sleeps the Mac
`caffeinate` cannot override the hardware-driven lid-close sleep on MacBooks.
For lid-closed operation, change your Energy Saver / Battery preferences or
use a third-party tool.
:::

### Linux — `systemd-inhibit` or `loginctl`

```bash
# Inhibit suspend while a command runs
systemd-inhibit --what=idle:sleep --who=hermes --why="gateways running" \
  sleep infinity &

# Allow user services to keep running after logout (recommended)
sudo loginctl enable-linger "$USER"
```

After enabling lingering, your systemd user units (including
`hermes-gateway-<profile>.service`) continue running across SSH disconnects
and reboots.

## Token-conflict safety

Each profile must use unique bot tokens for each platform. If two profiles
share a Telegram, Discord, Slack, WhatsApp, or Signal token, the second
gateway refuses to start with an error naming the conflicting profile.

To audit:

```bash
grep -H 'TELEGRAM_BOT_TOKEN\|DISCORD_BOT_TOKEN' \
     ~/.hermes/.env ~/.hermes/profiles/*/.env
```

## Updating the code

`hermes update` pulls the latest code once and syncs new bundled skills into
every profile:

```bash
hermes update
hermes-gateways restart
```

User-modified skills are never overwritten.

## Troubleshooting

### "Could not find service in domain for user gui: 501"

You ran `hermes gateway start` after a previous `hermes gateway stop`. The
CLI's `stop` does a full `launchctl unload`, which removes the service from
launchd's registry. The CLI catches this specific error on `start` and
automatically re-loads the plist (`↻ launchd job was unloaded; reloading
service definition`). The service starts normally. Nothing to fix.

### Stale PID after a crash

If a profile's gateway shows `not running` but a process is still alive:

```bash
ps -ef | grep "hermes_cli.*-p <profile>"
cat ~/.hermes/profiles/<profile>/gateway.pid
kill -TERM <pid>          # graceful
kill -KILL <pid>          # if that fails after a few seconds
<profile> gateway start
```

### Forcing a hard reset of one service

```bash
# macOS
launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist
launchctl load   ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist

# Linux
systemctl --user restart hermes-gateway-<profile>.service
```

### Health check

```bash
hermes doctor                  # default profile
hermes -p <profile> doctor     # one profile
```
