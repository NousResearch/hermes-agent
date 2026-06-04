---
sidebar_position: 15
title: "Web Dashboard"
description: "Browser-based administration panel for managing configuration, API keys, MCP servers, messaging pairing, webhooks, the gateway, memory, credentials, sessions, logs, analytics, cron jobs, and skills"
---

# Web Dashboard

The web dashboard is a browser-based UI for managing your Hermes Agent installation. Instead of editing YAML files or running CLI commands, you can configure settings, manage API keys, and monitor sessions from a clean web interface.

:::tip
Hosted-mode auth uses Nous Portal OAuth; if you also want the dashboard to talk to a real backend, `hermes setup --portal` wires up the model and tool gateway too. See [Nous Portal](/integrations/nous-portal).
:::

## Quick Start

```bash
hermes dashboard
```

This starts a local web server and opens `http://127.0.0.1:9119` in your browser. The dashboard runs entirely on your machine — no data leaves localhost.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `9119` | Port to run the web server on |
| `--host` | `127.0.0.1` | Bind address |
| `--no-open` | — | Don't auto-open the browser |
| `--insecure` | off | Allow binding to non-localhost hosts (**DANGEROUS** — exposes API keys on the network; pair with a firewall and strong auth) |

```bash
# Custom port
hermes dashboard --port 8080

# Bind to all interfaces (use with caution on shared networks)
hermes dashboard --host 0.0.0.0

# Start without opening browser
hermes dashboard --no-open
```

## Prerequisites

The default `hermes-agent` install does not ship the HTTP stack or PTY helper — those are optional extras. The **web dashboard** needs FastAPI and Uvicorn (`web` extra). The **Chat** tab also needs `ptyprocess` to spawn the embedded TUI behind a pseudo-terminal (`pty` extra on POSIX). Install both with:

```bash
pip install 'hermes-agent[web,pty]'
```

The `web` extra pulls in FastAPI/Uvicorn; `pty` pulls in `ptyprocess` (POSIX) or `pywinpty` (native Windows — note that the embedded TUI itself still requires WSL). `pip install hermes-agent[all]` includes both extras and is the easiest path if you also want messaging/voice/etc.

When you run `hermes dashboard` without the dependencies, it will tell you what to install. If the frontend hasn't been built yet and `npm` is available, it builds automatically on first launch.

The Chat tab is part of every `hermes dashboard` launch — the embedded browser chat pane (running the TUI over PTY/WebSocket) is always available, with no extra flag required.

## Pages

### Status

The landing page shows a live overview of your installation:

- **Agent version** and release date
- **Gateway status** — running/stopped, PID, connected platforms and their state
- **Active sessions** — count of sessions active in the last 5 minutes
- **Recent sessions** — list of the 20 most recent sessions with model, message count, token usage, and a preview of the conversation

The status page auto-refreshes every 5 seconds.

### Chat

The **Chat** tab embeds the full Hermes TUI (the same interface you get from `hermes --tui`) directly in the browser. Everything you can do in the terminal TUI — slash commands, model picker, tool-call cards, markdown streaming, clarify/sudo/approval prompts, skin theming — works identically here, because the dashboard is running the real TUI binary and rendering its ANSI output through [xterm.js](https://xtermjs.org/) with its WebGL renderer for pixel-perfect cell layout.

**How it works:**

- `/api/pty` opens a WebSocket authenticated with the dashboard's session token
- The server spawns `hermes --tui` behind a POSIX pseudo-terminal
- Keystrokes travel to the PTY; ANSI output streams back to the browser
- xterm.js's WebGL renderer paints each cell to an integer-pixel grid; mouse tracking (SGR 1006), wide characters (Unicode 11), and box-drawing glyphs all render natively
- Resizing the browser window resizes the TUI via the `@xterm/addon-fit` addon

**Resume an existing session:** from the **Sessions** tab, click the play icon (▶) next to any session. That jumps to `/chat?resume=<id>` and launches the TUI with `--resume`, loading the full history.

**Prerequisites:**

- Node.js (same requirement as `hermes --tui`; the TUI bundle is built on first launch)
- `ptyprocess` — installed by the `pty` extra (`pip install 'hermes-agent[web,pty]'`, or `[all]` covers both)
- POSIX kernel (Linux, macOS, or WSL2).  The `/chat` terminal pane specifically needs a POSIX PTY — native Windows Python has no equivalent, so on a native Windows install the rest of the dashboard (sessions, jobs, metrics, config editor) works but the `/chat` tab will show a banner telling you to use WSL2 for that feature.

Close the browser tab and the PTY is reaped cleanly on the server. Re-opening spawns a fresh session.

To point [Hermes Desktop](#connecting-hermes-desktop-to-a-remote-backend) at a dashboard running on another machine instead of its own bundled backend, see the remote-backend section below.

### Connecting Hermes Desktop to a remote backend

Hermes Desktop normally launches its own local backend, but it can also attach to a dashboard running on a remote machine (a VM, a homelab box, etc.) via **Settings → Gateway → Remote gateway**. This is the most common source of "Desktop says the backend is ready but chat never works" reports, because three independent things have to line up and only one of them is what Desktop's readiness check actually verifies.

Desktop's "remote backend is ready" probe only hits `GET /api/status`, which is a public endpoint — it answers as soon as *any* dashboard is running on the host. The live chat connection is a **separate** WebSocket to `/api/ws` (and `/api/pty`), and that socket is gated by two more checks the status probe never touches:

1. **The session token must match.** The socket closes with WS code **4401** if the token Desktop sends doesn't match the dashboard's session token. By default the dashboard generates a **fresh random token on every restart**, so a token you saved in Desktop yesterday is invalid after the service restarts. Pin it by setting `HERMES_DASHBOARD_SESSION_TOKEN` to a stable value.
2. **The bind host must allow the client and match the Host header.** A loopback bind (`127.0.0.1`) only accepts loopback clients, so a remote machine is rejected at the socket layer regardless of token. Bind to a non-loopback address (`--host 0.0.0.0 --insecure` for a trusted LAN) so the peer-IP guard lets the remote client through. The remote URL you enter in Desktop must reach the dashboard by the same host it bound to — the DNS-rebinding guard requires the Host header to match.

#### Remote dashboard setup

Run the remote dashboard with embedded chat **and** a pinned token. For a `systemd` service:

```ini
[Service]
Environment="HERMES_DASHBOARD_SESSION_TOKEN=<long-random-token>"
ExecStart=/path/to/venv/bin/python -m hermes_cli.main dashboard \
    --host 0.0.0.0 --port 9119 --insecure --tui --no-open
```

Generate a token once with `python -c "import secrets; print(secrets.token_urlsafe(32))"`, reload, and restart the service. Then paste that **same** token into Desktop's **Session token** field alongside the **Remote URL** (e.g. `http://VM_IP:9119`).

:::tip Verify before retrying Desktop
From the client machine, hit a protected endpoint with the token before opening Desktop:

```bash
curl -i -H "X-Hermes-Session-Token: <long-random-token>" http://VM_IP:9119/api/config
```

- **200** → the token path Desktop needs is good.
- **401** → Desktop will fail even though `/api/status` reports the backend is ready. Fix the token first.

The REST API reads the token from the `X-Hermes-Session-Token` header; the WebSocket reads the same token from the `?token=` query parameter. Both compare against `HERMES_DASHBOARD_SESSION_TOKEN`, so a 200 here means the WS handshake will authenticate too.
:::

If `/api/config` returns 200 with the token Desktop is using and Desktop *still* fails to connect, the issue is past basic setup — grab a fresh `desktop.log` (Settings → Gateway → Open logs) plus the dashboard's logs from the same retry window and look for the `/api/ws` close code (4403 = chat not enabled, 4401 = token mismatch, 4403 from the request guard = Host/peer rejected).

:::note Public binds use OAuth, not the session token
Everything above describes `--insecure` (loopback or trusted-LAN) mode. If the dashboard is bound to a public address *without* `--insecure`, it engages the [OAuth gate](#oauth-authentication-gated-mode) instead and the legacy session-token path is rejected — Desktop's session-token remote mode is for `--insecure` deployments.
:::

### Config

A form-based editor for `config.yaml`. All 150+ configuration fields are auto-discovered from `DEFAULT_CONFIG` and organized into tabbed categories:

- **model** — default model, provider, base URL, reasoning settings
- **terminal** — backend (local/docker/ssh/modal), timeout, shell preferences
- **display** — skin, tool progress, resume display, spinner settings
- **agent** — max iterations, gateway timeout, service tier
- **delegation** — subagent limits, reasoning effort
- **memory** — provider selection, context injection settings
- **approvals** — dangerous command approval mode (ask/yolo/deny)
- And more — every section of config.yaml has corresponding form fields

Fields with known valid values (terminal backend, skin, approval mode, etc.) render as dropdowns. Booleans render as toggles. Everything else is a text input.

**Actions:**

- **Save** — writes changes to `config.yaml` immediately
- **Reset to defaults** — reverts all fields to their default values (doesn't save until you click Save)
- **Export** — downloads the current config as JSON
- **Import** — uploads a JSON config file to replace the current values

:::tip
Config changes take effect on the next agent session or gateway restart. The web dashboard edits the same `config.yaml` file that `hermes config set` and the gateway read from.
:::

### API Keys

Manage the `.env` file where API keys and credentials are stored. Keys are grouped by category:

- **LLM Providers** — OpenRouter, Anthropic, OpenAI, DeepSeek, etc.
- **Tool API Keys** — Browserbase, Firecrawl, Tavily, ElevenLabs, etc.
- **Messaging Platforms** — Telegram, Discord, Slack bot tokens, etc.
- **Agent Settings** — non-secret env vars like `API_SERVER_ENABLED`

Each key shows:
- Whether it's currently set (with a redacted preview of the value)
- A description of what it's for
- A link to the provider's signup/key page
- An input field to set or update the value
- A delete button to remove it

Advanced/rarely-used keys are hidden by default behind a toggle.

### Sessions

Browse and inspect all agent sessions. Each row shows the session title, source platform icon (CLI, Telegram, Discord, Slack, cron), model name, message count, tool call count, and how long ago it was active. Live sessions are marked with a pulsing badge.

- **Search** — full-text search across all message content using FTS5. Results show highlighted snippets and auto-scroll to the first matching message when expanded.
- **Stats** — a summary bar shows total sessions, how many are active in the store, archived count, total messages, and a per-source breakdown.
- **Expand** — click a session to load its full message history. Messages are color-coded by role (user, assistant, system, tool) and rendered as Markdown with syntax highlighting.
- **Tool calls** — assistant messages with tool calls show collapsible blocks with the function name and JSON arguments.
- **Rename** — set or clear a session's title inline (pencil icon).
- **Export** — download a session (metadata + full message history) as JSON (download icon).
- **Prune** — the header "Prune old sessions" button deletes ended sessions older than N days.
- **Delete** — remove a session and its message history with the trash icon.

![Sessions admin page — stats bar, prune, and per-row rename / export / delete](/img/dashboard/admin-sessions.png)

### Logs

View agent, gateway, and error log files with filtering and live tailing.

- **File** — switch between `agent`, `errors`, and `gateway` log files
- **Level** — filter by log level: ALL, DEBUG, INFO, WARNING, or ERROR
- **Component** — filter by source component: all, gateway, agent, tools, cli, or cron
- **Lines** — choose how many lines to display (50, 100, 200, or 500)
- **Auto-refresh** — toggle live tailing that polls for new log lines every 5 seconds
- **Color-coded** — log lines are colored by severity (red for errors, yellow for warnings, dim for debug)

### Analytics

Usage and cost analytics computed from session history. Select a time period (7, 30, or 90 days) to see:

- **Summary cards** — total tokens (input/output), cache hit percentage, total estimated or actual cost, and total session count with daily average
- **Daily token chart** — stacked bar chart showing input and output token usage per day, with hover tooltips showing breakdowns and cost
- **Daily breakdown table** — date, session count, input tokens, output tokens, cache hit rate, and cost for each day
- **Per-model breakdown** — table showing each model used, its session count, token usage, and estimated cost

### Cron

Create and manage scheduled cron jobs that run agent prompts on a recurring schedule.

- **Create** — fill in a name (optional), prompt, cron expression (e.g. `0 9 * * *`), and delivery target (local, Telegram, Discord, Slack, or email)
- **Job list** — each job shows its name, prompt preview, schedule expression, state badge (enabled/paused/error), delivery target, last run time, and next run time
- **Pause / Resume** — toggle a job between active and paused states
- **Edit** — open a pre-filled modal to change a job's prompt, schedule, name, or delivery target
- **Trigger now** — immediately execute a job outside its normal schedule
- **Delete** — permanently remove a cron job

### Skills

Browse, search, and toggle installed skills and toolsets, and install new ones from the hub. Skills are loaded from `~/.hermes/skills/` and grouped by category.

- **Search** — filter installed skills and toolsets by name, description, or category
- **Category filter** — click category pills to narrow the list (e.g. MLOps, MCP, Red Teaming, AI)
- **Toggle** — enable or disable individual skills with a switch. Changes take effect on the next session.
- **Toolsets** — a separate view shows built-in toolsets (file operations, web browsing, etc.) with their active/inactive status, setup requirements, and list of included tools
- **Browse hub** — a third view searches the skill hub across all sources (the same as `hermes skills search`), installs any result by identifier with a live install log, and offers an "Update all" button to refresh installed skills.

![Skills admin page — the Browse hub view: search, install, and update](/img/dashboard/admin-skills-hub.png)

### MCP

Manage [MCP](/integrations/mcp) servers without the CLI. The same `mcp_servers`
block in `config.yaml` that `hermes mcp` reads from.

**Your MCP servers:**

- **Add** — register an HTTP/SSE server (URL) or a stdio server (command + args), with optional `KEY=VALUE` environment variables for stdio servers
- **Enable / disable** — toggle a server on or off without deleting it. A disabled server stays in config so you can re-enable it later. Takes effect on the next gateway restart.
- **Test** — connect to a server, list its tools, and disconnect — verifies the connection before the agent depends on it
- **Remove** — delete a server from the config
- Secret-shaped env values are redacted in the list view

**Catalog:** browse the Nous-approved MCP servers (the bundled `optional-mcps/`
catalog) and install any of them with one click. Entries that need API keys
prompt for them inline; the values go to `.env`. This is the same catalog
`hermes mcp catalog` / `hermes mcp install` use.

![MCP admin page — your servers with enable/disable toggles, plus the install catalog](/img/dashboard/admin-mcp.png)

### Webhooks

Manage dynamic [webhook subscriptions](/user-guide/messaging/webhooks). The
webhook platform must be enabled in messaging settings first; the page shows a
hint when it isn't.

- **Create** — name, description, event filter, delivery target, optional direct-delivery mode, and an agent prompt. On creation the page surfaces the route URL and the one-time HMAC secret to copy.
- **Enable / disable** — toggle a subscription on or off. Disabled routes stay in the subscriptions file but the gateway rejects their incoming events (403). The gateway hot-reloads the file, so the change takes effect on the next event — no restart needed.
- **List** — each subscription shows its URL, events, and delivery target
- **Delete** — remove a subscription

![Webhooks admin page — subscriptions with enable/disable toggles](/img/dashboard/admin-webhooks.png)

### Pairing

Approve and revoke messaging users without the CLI — how a remote admin
onboards Telegram/Discord/etc. users to a paired gateway. Full parity with
`hermes pairing`.

- **Pending requests** — each shows platform, code, user, and age, with an Approve button
- **Approved users** — each shows platform and user, with a Revoke button
- **Clear pending** — drop all outstanding pairing codes

![Pairing admin page](/img/dashboard/admin-pairing.png)

### Channels

Connect Hermes to any messaging platform from the browser — full parity with
`hermes setup gateway`. The page lists every supported channel (Telegram,
Discord, Slack, Matrix, Mattermost, WhatsApp, Signal, BlueBubbles/iMessage,
Email, SMS/Twilio, DingTalk, Feishu/Lark, WeCom, WeChat, QQ Bot, Yuanbao, plus
the API server and webhook endpoints) with its live connection status.

- **Configure** — open a per-platform form with exactly the fields that channel needs (bot token, app token, server URL, allowlist, etc.). Secrets render as password inputs and are stored redacted; leaving a field blank keeps the existing value. Required fields are marked and validated. A "Setup guide" link points to the platform's credential docs.
- **Enable / disable** — toggle a channel on or off. The credential stays on disk; only the active state changes.
- **Test** — check whether the channel is configured, enabled, and reporting a live connection from the gateway.
- **Restart gateway** — credentials are written to `~/.hermes/.env` and the enabled flag to `config.yaml`; the gateway connects each enabled channel on its next restart, which you can trigger right from the page.

![Channels admin page — every messaging platform with status, enable toggles, and per-platform setup forms](/img/dashboard/admin-channels.png)

### System

A consolidated administration panel for installation-wide operations:

- **Host** — live system stats: OS / kernel, architecture, hostname, Python and Hermes versions, CPU core count + utilization, memory, disk usage of the Hermes home, uptime, and load average. (CPU/memory/disk come from `psutil` when installed; identity fields are always shown.) The Hermes version shows an **update-status badge** (up to date / N commits behind) and a **Check for updates** button. When an update is available on a git or pip install, an **Update now** button opens a confirmation dialog — showing how many commits you'll pull — before running `hermes update` in the background. On Docker/Nix/Homebrew installs the dashboard can't apply the update in place, so it shows the correct out-of-band command instead.
- **Nous Portal** — login status, the active inference provider, and the Tool Gateway routing table (which tools run via the Portal vs. locally), with a link to manage your subscription. Read-only mirror of `hermes portal`.
- **Skill curator** — the background skill-maintenance status (active / paused, interval, last run) with pause/resume and a run-now button. Mirrors `hermes curator`.
- **Gateway** — start, stop, and restart the messaging gateway, with live status (running/stopped, PID, state)
- **Memory** — pick the external memory provider (or built-in only), and reset the built-in `MEMORY.md` / `USER.md` stores
- **Credential pool** — add and remove the rotating API keys the agent round-robins through (per provider). Keys are redacted in the list; the raw value only ever reaches the agent.
- **Operations** — run `doctor`, a security audit, create a backup, restore from a backup archive, update skills, show the system-prompt size breakdown, generate a support dump, or migrate config for retired settings. Each spawns a background action whose live log streams into the page.
- **Checkpoints** — see the `/rollback` shadow store size and prune it
- **Shell hooks** — list configured hooks with their consent + executable status, **create** a hook (event, command, matcher, timeout, with an opt-in consent grant), and remove one. Hooks run arbitrary commands, so the create form carries a security warning and the hook only fires after consent is granted.

![System admin page — host stats and Nous Portal status](/img/dashboard/admin-system-top.png)

![System admin page — skill curator, gateway, memory, and credential pool](/img/dashboard/admin-system-curator.png)

![System admin page — operations, checkpoints, and shell hooks](/img/dashboard/admin-system-ops.png)

Creating a shell hook (note the consent checkbox and the run-arbitrary-commands warning):

![New shell hook modal](/img/dashboard/admin-hook-create.png)

:::warning Security
The web dashboard reads and writes your `.env` file, which contains API keys and secrets. It binds to `127.0.0.1` by default — only accessible from your local machine. If you bind to `0.0.0.0`, anyone on your network can view and modify your credentials. The dashboard has no authentication of its own.
:::

## `/reload` Slash Command

The dashboard PR also adds a `/reload` slash command to the interactive CLI. After changing API keys via the web dashboard (or by editing `.env` directly), use `/reload` in an active CLI session to pick up the changes without restarting:

```
You → /reload
  Reloaded .env (3 var(s) updated)
```

This re-reads `~/.hermes/.env` into the running process's environment. Useful when you've added a new provider key via the dashboard and want to use it immediately.

## REST API

The web dashboard exposes a REST API that the frontend consumes. You can also call these endpoints directly for automation:

### GET /api/status

Returns agent version, gateway status, platform states, and active session count.

### GET /api/sessions

Returns the 20 most recent sessions with metadata (model, token counts, timestamps, preview).

### GET /api/config

Returns the current `config.yaml` contents as JSON.

### GET /api/config/defaults

Returns the default configuration values.

### GET /api/config/schema

Returns a schema describing every config field — type, description, category, and select options where applicable. The frontend uses this to render the correct input widget for each field.

### PUT /api/config

Saves a new configuration. Body: `{"config": {...}}`.

### GET /api/env

Returns all known environment variables with their set/unset status, redacted values, descriptions, and categories.

### PUT /api/env

Sets an environment variable. Body: `{"key": "VAR_NAME", "value": "secret"}`.

### DELETE /api/env

Removes an environment variable. Body: `{"key": "VAR_NAME"}`.

### GET /api/sessions/\{session_id\}

Returns metadata for a single session.

### GET /api/sessions/\{session_id\}/messages

Returns the full message history for a session, including tool calls and timestamps.

### GET /api/sessions/search

Full-text search across message content. Query parameter: `q`. Returns matching session IDs with highlighted snippets.

### DELETE /api/sessions/\{session_id\}

Deletes a session and its message history.

### GET /api/logs

Returns log lines. Query parameters: `file` (agent/errors/gateway), `lines` (count), `level`, `component`.

### GET /api/analytics/usage

Returns token usage, cost, and session analytics. Query parameter: `days` (default 30). Response includes daily breakdowns and per-model aggregates.

### GET /api/cron/jobs

Returns all configured cron jobs with their state, schedule, and run history.

### POST /api/cron/jobs

Creates a new cron job. Body: `{"prompt": "...", "schedule": "0 9 * * *", "name": "...", "deliver": "local"}`.

### POST /api/cron/jobs/\{job_id\}/pause

Pauses a cron job.

### POST /api/cron/jobs/\{job_id\}/resume

Resumes a paused cron job.

### POST /api/cron/jobs/\{job_id\}/trigger

Immediately triggers a cron job outside its schedule.

### DELETE /api/cron/jobs/\{job_id\}

Deletes a cron job.

### GET /api/skills

Returns all skills with their name, description, category, and enabled status.

### PUT /api/skills/toggle

Enables or disables a skill. Body: `{"name": "skill-name", "enabled": true}`.

### GET /api/tools/toolsets

Returns all toolsets with their label, description, tools list, and active/configured status.

### Admin endpoints

These power the MCP, Channels, Webhooks, Pairing, and System pages. All sit behind the
same auth gate as the rest of `/api/`.

| Method & path | Purpose |
|---------------|---------|
| `GET /api/mcp/servers` | List configured MCP servers (env values redacted) |
| `POST /api/mcp/servers` | Add a server. Body: `{name, url?, command?, args?, env?, auth?}` |
| `POST /api/mcp/servers/{name}/test` | Connect, list tools, disconnect |
| `PUT /api/mcp/servers/{name}/enabled` | Enable / disable a server |
| `DELETE /api/mcp/servers/{name}` | Remove a server |
| `GET /api/mcp/catalog` | Browse the Nous-approved MCP catalog |
| `POST /api/mcp/catalog/install` | Install a catalog entry (with required env) |
| `GET /api/messaging/platforms` | List every messaging channel with status + per-platform setup fields |
| `PUT /api/messaging/platforms/{id}` | Configure a channel. Body: `{enabled?, env?, clear_env?}` (env writes to `.env`, enabled to `config.yaml`) |
| `POST /api/messaging/platforms/{id}/test` | Report whether a channel is configured, enabled, and connected |
| `GET /api/pairing` | List pending + approved messaging users |
| `POST /api/pairing/approve` | Approve a code. Body: `{platform, code}` |
| `POST /api/pairing/revoke` | Revoke a user. Body: `{platform, user_id}` |
| `POST /api/pairing/clear-pending` | Drop all pending codes |
| `GET /api/webhooks` | List subscriptions + platform-enabled status |
| `POST /api/webhooks` | Create a subscription (returns one-time secret) |
| `DELETE /api/webhooks/{name}` | Remove a subscription |
| `GET /api/credentials/pool` | List pooled rotation keys (redacted) |
| `POST /api/credentials/pool` | Add a key. Body: `{provider, api_key, label?}` |
| `DELETE /api/credentials/pool/{provider}/{index}` | Remove a key (1-based index) |
| `GET /api/memory` | Active provider + available providers + built-in file sizes |
| `PUT /api/memory/provider` | Select a provider (empty = built-in only) |
| `POST /api/memory/reset` | Reset built-in memory. Body: `{target: all\|memory\|user}` |
| `POST /api/gateway/start` · `/stop` · `/restart` | Gateway lifecycle (backgrounded) |
| `POST /api/ops/doctor` · `/security-audit` · `/backup` · `/import` | Diagnostics & maintenance (backgrounded; tail via `/api/actions/{name}/status`) |
| `GET /api/ops/hooks` | Configured shell hooks + allowlist status |
| `GET /api/ops/checkpoints` · `POST .../prune` | Inspect / prune the `/rollback` store |
| `POST /api/ops/hooks` · `DELETE /api/ops/hooks` | Create / remove a shell hook (consent-gated) |
| `GET /api/system/stats` | Host stats — OS, CPU, memory, disk, uptime |
| `GET /api/hermes/update/check` | Report update availability (commits behind, install method) without applying. `?force=1` busts the 6h cache |
| `GET /api/curator` · `PUT .../paused` · `POST .../run` | Skill-curator status + pause/resume + run |
| `GET /api/portal` | Nous Portal auth + Tool Gateway routing (read-only) |
| `POST /api/ops/prompt-size` · `/dump` · `/config-migrate` | Diagnostics (backgrounded) |
| `PUT /api/webhooks/{name}/enabled` | Enable / disable a webhook route |
| `POST /api/skills/hub/install` · `/uninstall` · `/update` | Skills hub actions (backgrounded) |
| `GET /api/skills/hub/search` | Search the skill hub across all sources |
| `GET /api/sessions/stats` | Session-store statistics |
| `PATCH /api/sessions/{id}` | Rename / archive a session |
| `GET /api/sessions/{id}/export` | Export a session (metadata + messages) as JSON |
| `POST /api/sessions/prune` | Delete ended sessions older than N days |
| `PUT /api/cron/jobs/{id}` | Edit a cron job's prompt / schedule / name / deliver |

## OAuth Authentication (gated mode)

When the dashboard is bound to a public address — anything other than `127.0.0.1` / `localhost` — Hermes Agent engages an OAuth-based auth gate. Every request must carry a verified session cookie or it's bounced through a full OAuth round-trip via the Nous Portal.

This is intended for hosted deployments (typically Fly.io) where the dashboard is reachable over the public internet. Operator-owned dashboards bound to loopback are unaffected.

### When the gate engages

| Flags | Auth gate | Use case |
|-------|-----------|----------|
| `hermes dashboard` (default — binds to `127.0.0.1`) | OFF | Local development |
| `hermes dashboard --host 0.0.0.0` | **ON** | Production / Fly.io deployment |
| `hermes dashboard --host 192.168.1.10 --insecure` | OFF | Trusted LAN; user opts into legacy session-token auth |

The gate is on if and only if:

1. The bind host is not `127.0.0.1`, `::1`, `localhost`, or `0.0.0.0` AND
2. The `--insecure` flag is **not** set.

Setting `--insecure` keeps the existing single-process session-token behaviour — no OAuth dance, no provider plugins required. Use only on networks where you trust every client.

### Fail-closed semantics

If the gate would engage but **no** `DashboardAuthProvider` is registered (no Nous plugin, no custom plugin), `hermes dashboard` refuses to bind with an explicit error message. There is no "default-deny but accept everything" fallback — a misconfigured gated dashboard never starts.

### Default provider: Nous Research

The bundled `plugins/dashboard_auth/nous` plugin is **always installed** and auto-loaded. It auto-registers a `DashboardAuthProvider` named `nous` when a client ID is configured.

#### Configuration

The plugin reads from two surfaces, with the environment variable winning when set non-empty:

**`config.yaml`** — the canonical surface:

```yaml
dashboard:
  oauth:
    client_id: agent:01HXYZ…             # required to engage the gate
    portal_url: https://portal.nousresearch.com  # optional; defaults to production
```

**Environment variables** — operator overrides:

| Env var | Overrides | Format | Provisioned by |
|---------|-----------|--------|----------------|
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | `dashboard.oauth.client_id` | `agent:{instance_id}` | Nous Portal at Fly.io provisioning time |
| `HERMES_DASHBOARD_PORTAL_URL` | `dashboard.oauth.portal_url` | URL (default: `https://portal.nousresearch.com`) | Portal — override only for staging or a custom deployment |

Per the Hermes Agent convention (`~/.hermes/.env` is for API keys / secrets only), **`config.yaml` is the recommended place to set these values** for local dev, on-prem, and any deployment you control directly. The environment-variable path exists so Fly.io's platform-secret injection can push per-deploy `client_id`s without anyone having to edit `config.yaml` inside the image — that's its primary purpose.

Empty environment values are treated as unset, so a provisioned-but-not-populated Fly secret can't accidentally shadow a valid `config.yaml` entry.

If neither source provides a client_id, the plugin reports the specific reason and the dashboard's fail-closed bind error tells you exactly what to fix:

```
Refusing to bind dashboard to 0.0.0.0 — the OAuth auth gate engages on
non-loopback binds, but no auth providers are registered.

Bundled providers reported these issues:
  • nous: HERMES_DASHBOARD_OAUTH_CLIENT_ID is not set (and
    dashboard.oauth.client_id in config.yaml is empty). The Nous Portal
    provisions this env var (shape 'agent:{instance_id}') when it
    deploys a Hermes Agent instance — set it to your provisioned
    client id (either as an env var or under dashboard.oauth.client_id
    in config.yaml), or pass --insecure to skip the OAuth gate entirely.

Or pass --insecure to skip the auth gate (NOT recommended on untrusted
networks).
```

### Username/password provider (no OAuth IDP)

If you don't want to wire up an OAuth identity provider — a self-hosted "just put a password on my dashboard" deployment — the bundled `plugins/dashboard_auth/basic` plugin registers a `DashboardAuthProvider` named `basic` that authenticates with a **username and password** instead of an OAuth redirect.

It plugs into the same gate as the OAuth provider: the gate engages on a non-loopback bind without `--insecure`, the login page renders a credential form for this provider (instead of a "Log in with X" button), and everything downstream of login — session cookies, transparent refresh, WS tickets, logout, the audit log — is identical to the OAuth path. Sessions are stateless HMAC-signed tokens the provider mints itself, so there's **no database and no external IDP**. Password hashing uses stdlib `scrypt` (no third-party dependency).

#### Configuration

Like the Nous provider, it reads from `config.yaml` (canonical) with environment variables winning when set non-empty. It activates only when `username` plus either `password_hash` (preferred) or `password` are configured — otherwise it's a no-op, so OAuth users and loopback/`--insecure` operators are unaffected.

**`config.yaml`:**

```yaml
dashboard:
  basic_auth:
    username: admin
    # Preferred — no plaintext at rest. Compute with:
    #   python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"
    password_hash: "scrypt$16384$8$1$…$…"
    # ...or a plaintext password (hashed in-memory at load; less safe at rest):
    # password: "s3cret"
    secret: "<32+ random bytes, base64 or hex>"  # token-signing key
    session_ttl_seconds: 43200                    # optional; access-token lifetime (default 12h)
```

**Environment overrides:**

| Env var | Overrides | Notes |
|---------|-----------|-------|
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | `dashboard.basic_auth.username` | required to activate |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` | `dashboard.basic_auth.password_hash` | preferred (no plaintext at rest) |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | `dashboard.basic_auth.password` | plaintext; **wins over a config `password_hash`** so you can rotate via env |
| `HERMES_DASHBOARD_BASIC_AUTH_SECRET` | `dashboard.basic_auth.secret` | token-signing key |
| `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS` | `dashboard.basic_auth.session_ttl_seconds` | access-token lifetime |

:::caution Set an explicit `secret` for stable sessions
When `secret` is empty, a random per-process signing key is generated. That's fine for a single process, but it means **every session is invalidated on restart** and sessions **don't span multiple workers**. Set an explicit `secret` for restart-surviving / multi-worker deployments.
:::

The `/auth/password-login` endpoint is rate-limited per client IP (default 10 attempts/minute → HTTP 429) and returns a single generic `401 Invalid credentials` for both unknown users and wrong passwords, so it can't be used as a username-enumeration oracle.

#### Writing your own password provider

`basic` is just one implementation of an extension point. Any plugin can register a password provider: set `supports_password = True` on your `DashboardAuthProvider` subclass and implement `complete_password_login(*, username, password) -> Session` (raise `InvalidCredentialsError` on rejection, `ProviderError` if your backing store is down). The OAuth `start_login` / `complete_login` methods can be left as `NotImplementedError` stubs for a pure-password provider. This is the path for LDAP-bind, a credentials database, or any other non-redirect auth scheme — the framework handles the form, the route, the cookies, and refresh for you.

### Self-hosted OIDC provider

If you run your own identity provider, the bundled `plugins/dashboard_auth/self_hosted` plugin authenticates the dashboard against it using **standard OpenID Connect** — no per-IDP code, no Nous Portal involved. It's verified against and works with any conformant OIDC server:

> **Authentik · Keycloak · Zitadel · Authelia · Auth0 · Okta · Google · …**

Like the Nous provider, it auto-loads and only registers itself once it's configured, so it's a no-op for loopback / `--insecure` dashboards.

#### Configuration

Configure an **issuer** and a **client_id** (a public PKCE client — no client secret). The plugin fetches the IDP's `authorization_endpoint`, `token_endpoint`, and `jwks_uri` from `{issuer}/.well-known/openid-configuration`, so you never hardcode endpoint URLs.

**`config.yaml`** — the canonical surface:

```yaml
dashboard:
  oauth:
    provider: self-hosted
    self_hosted:
      issuer: https://auth.example.com/application/o/hermes/   # required
      client_id: hermes-dashboard                              # required
      scopes: "openid profile email"                           # optional (this is the default)
```

**Environment variables** — operator overrides (env wins over `config.yaml` when set non-empty; an empty value is treated as unset):

| Env var | Overrides | Notes |
|---------|-----------|-------|
| `HERMES_DASHBOARD_OIDC_ISSUER` | `dashboard.oauth.self_hosted.issuer` | OIDC issuer URL — required |
| `HERMES_DASHBOARD_OIDC_CLIENT_ID` | `dashboard.oauth.self_hosted.client_id` | Public client id — required |
| `HERMES_DASHBOARD_OIDC_SCOPES` | `dashboard.oauth.self_hosted.scopes` | Defaults to `openid profile email` |

In your IDP, register a **public** application/client with the authorization-code + PKCE (S256) grant and add the dashboard's callback as an allowed redirect URI. The callback is `<dashboard public URL>/auth/callback` (see [Public URL override](#public-url-override) for how the dashboard derives its public URL behind a proxy).

#### What it verifies

The provider verifies the OpenID Connect **ID token** (RS256/ES256) against the discovered `jwks_uri`, with the `iss` and `aud` claims pinned to your configured `issuer` and `client_id`. Standard OIDC claims map onto the dashboard session:

| Session field | Claim(s) |
|---------------|----------|
| `user_id` | `sub` (required) |
| `email` | `email` |
| `display_name` | `name` → `preferred_username` → `nickname` → `email` |
| `org_id` | `org_id` / `organization`, else joined `groups` |

The ID token is what establishes identity — the access token is treated as opaque (the OIDC spec does not require it to be a JWT). Endpoint URLs are required to be HTTPS (loopback `http://` is allowed for local-dev IDPs), and the discovery document's advertised `issuer` must match your configured one (a trailing-slash difference is tolerated). Refresh tokens, when the IDP issues them, are used for silent re-auth via the standard `refresh_token` grant; logout calls the IDP's RFC 7009 `revocation_endpoint` when advertised.

> **Confidential clients** (those with a `client_secret`) are not supported yet — configure a public + PKCE client, which is the typical choice for a browser-facing dashboard.

#### Worked example: Keycloak

[Keycloak](https://www.keycloak.org/) is one of the easiest self-hosted OIDC servers to stand up for a local test — it runs as a single container in dev mode (in-memory DB) and exposes textbook OIDC discovery. This walkthrough gets you from nothing to a working dashboard login in a few minutes.

**1. Run Keycloak with a pre-configured realm.** Save this realm export as `realm-hermes.json` — it defines a `hermes` realm, a **public PKCE client** (`hermes-dashboard`), and a test user, all imported on boot so there's nothing to click in the admin UI:

```json
{
  "realm": "hermes",
  "enabled": true,
  "clients": [
    {
      "clientId": "hermes-dashboard",
      "name": "Hermes Agent Dashboard",
      "enabled": true,
      "publicClient": true,
      "standardFlowEnabled": true,
      "protocol": "openid-connect",
      "redirectUris": ["http://localhost:9119/auth/callback"],
      "webOrigins": ["http://localhost:9119"],
      "attributes": { "pkce.code.challenge.method": "S256" }
    }
  ],
  "users": [
    {
      "username": "testuser",
      "enabled": true,
      "emailVerified": true,
      "email": "testuser@example.com",
      "firstName": "Test",
      "lastName": "User",
      "credentials": [
        { "type": "password", "value": "testpassword", "temporary": false }
      ]
    }
  ]
}
```

Start it (Keycloak 26+), mounting that file into the import directory:

```bash
docker run --rm -p 8080:8080 \
  -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
  -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin \
  -v "$PWD/realm-hermes.json:/opt/keycloak/data/import/realm-hermes.json:ro" \
  quay.io/keycloak/keycloak:26.0 \
  start-dev --import-realm
```

Once it's up, the realm advertises standard OIDC discovery at
`http://localhost:8080/realms/hermes/.well-known/openid-configuration` (issuer
`http://localhost:8080/realms/hermes`). The admin console is at
`http://localhost:8080/` (`admin` / `admin`).

**2. Point the dashboard at it.** The self-hosted plugin permits a loopback `http://` issuer (HTTPS is required for any non-loopback issuer), so the local Keycloak works as-is:

```bash
export HERMES_DASHBOARD_OIDC_ISSUER="http://localhost:8080/realms/hermes"
export HERMES_DASHBOARD_OIDC_CLIENT_ID="hermes-dashboard"
export HERMES_DASHBOARD_PUBLIC_URL="http://localhost:9119"
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

`HERMES_DASHBOARD_PUBLIC_URL` tells the dashboard its OAuth callback is
`http://localhost:9119/auth/callback` — the redirect URI the realm registered
above. Binding to `0.0.0.0` (a non-loopback bind) without `--insecure` is what
engages the OAuth gate.

**3. Log in.** Open `http://localhost:9119/`, you'll be bounced to `/login`. Click **Sign in with Self-Hosted OIDC** → authenticate at Keycloak as `testuser` / `testpassword` → land back on the authenticated dashboard. The sidebar shows `Logged in as Test User via self-hosted`, and `GET /api/auth/me` returns the verified session (`provider: self-hosted`, `email: testuser@example.com`).

> If you bind or browse on a different host/port, add that origin's
> `…/auth/callback` to the client's **Valid redirect URIs** in the Keycloak
> admin console (Clients → hermes-dashboard → Settings). The same pattern works
> for Authentik, Zitadel, Authelia, and other OIDC servers — only the issuer
> URL and client registration UI differ.

### Public URL override

By default, the dashboard reconstructs the OAuth callback URL from the request — `X-Forwarded-Host` + `X-Forwarded-Proto` + `X-Forwarded-Prefix` (when uvicorn is configured with `proxy_headers=True`, which `start_server` enables under the gate). This works out of the box on Fly.io, which sets all three headers correctly.

For deploys behind reverse proxies that don't reliably forward those headers (manual nginx setups, on-prem ingresses, custom-domain Fly deploys with partial proxy chains), set `dashboard.public_url` (or `HERMES_DASHBOARD_PUBLIC_URL`) to the **complete public URL** the dashboard is reached at:

```yaml
dashboard:
  public_url: "https://dashboard.example.com/hermes"
```

When set, the OAuth callback URL becomes `<public_url>/auth/callback` verbatim — `X-Forwarded-Prefix` is ignored on that code path because the operator has explicitly declared the public URL. This is intentional: stacking the prefix on top would double-prefix the common case where the prefix is already baked into `public_url`.

Same precedence as the other dashboard settings — env wins over `config.yaml`:

| Surface | Override path | When to use |
|---------|---------------|-------------|
| `dashboard.public_url` in `config.yaml` | `HERMES_DASHBOARD_PUBLIC_URL` | Local dev / on-prem (canonical) |
| `HERMES_DASHBOARD_PUBLIC_URL` env var | — | Fly.io platform secrets / CI |
| (unset) | — | Default — reconstruct from `X-Forwarded-*` headers |

Validation rejects values without `http://` / `https://` scheme, without a host, or containing quote / angle / whitespace / control characters. A malformed value silently falls through to header reconstruction so the login flow keeps working rather than dispatching the user to a hostile URL.

> **Note:** `public_url` overrides the OAuth callback URL only. The `Secure` cookie flag is still controlled by `request.url.scheme` (X-Forwarded-Proto under proxy_headers), so an `http://` `public_url` on a TLS-terminated public deploy will produce non-Secure cookies. This is an operator footgun — pair `public_url` with proper TLS termination upstream.

### OAuth flow

The provider implements the [Nous Portal OAuth contract v1](https://github.com/NousResearch/nous-account-service/blob/main/docs/agent-dashboard-oauth-contract.md) — authorization-code grant with PKCE (S256):

1. User hits `/` without a session cookie → gate redirects to `/login`.
2. Login page shows a "Continue with Nous Research" button → `/auth/login?provider=nous`.
3. Server stashes PKCE state in a short-lived cookie, redirects user to `https://portal.nousresearch.com/oauth/authorize?…`.
4. User authenticates with Portal, lands at `/auth/callback?code=…&state=…`.
5. Server exchanges the code for an access token at `POST /api/oauth/token`, verifies the JWT signature against the Portal's JWKS (`/.well-known/jwks.json`), and sets the `hermes_session_at` cookie.
6. User is redirected to `/` (or to the original deep-link path via the `next=` query parameter).

Access tokens have a 15-minute TTL. **There is no refresh token in contract v1** — when the token expires, the SPA's fetch wrapper detects the 401 envelope and full-page-navigates back to `/login` to re-run the flow.

### Cookies set

| Name | Lifetime | Notes |
|------|----------|-------|
| `hermes_session_at` | Token TTL (15 min) | HttpOnly, SameSite=Lax, Secure-when-HTTPS |
| `hermes_session_pkce` | 10 min | HttpOnly; holds the PKCE verifier + provider hint during the round trip |
| `hermes_session_rt` | unused in v1 | Reserved for forward-compat; not written when `refresh_token` is empty |

All three are `Path=/` and `SameSite=Lax`. The `Secure` flag is set when the dashboard is reached over HTTPS (detected via the request URL scheme — honours `X-Forwarded-Proto` from Fly's TLS terminator under `proxy_headers=True`).

### Logout

The sidebar widget shows `Logged in as <user_id…> via nous` with a logout icon. Clicking it POSTs `/auth/logout`, which clears all dashboard-auth cookies and redirects back to `/login`.

### Audit log

Every login start, success, failure, and session-verify failure is written as a JSON line to `$HERMES_HOME/logs/dashboard-auth.log`. Sensitive fields (`access_token`, `refresh_token`, `code`, `code_verifier`, `state`, `Authorization` header) are redacted before logging.

### Custom providers

To plug a non-Nous OAuth provider (e.g. Google, GitHub, custom OIDC), create a plugin that registers a `DashboardAuthProvider`:

```python
# ~/.hermes/plugins/dashboard-auth-myidp/__init__.py
from hermes_cli.dashboard_auth import DashboardAuthProvider, Session, LoginStart

class MyIdPProvider(DashboardAuthProvider):
    name = "myidp"
    display_name = "My Identity Provider"

    def start_login(self, *, redirect_uri): ...
    def complete_login(self, *, code, state, code_verifier, redirect_uri): ...
    def verify_session(self, *, access_token): ...
    def refresh_session(self, *, refresh_token): ...
    def revoke_session(self, *, refresh_token): ...

def register(ctx):
    ctx.register_dashboard_auth_provider(MyIdPProvider())
```

The login page lists all registered providers; multiple providers can be stacked and the user picks one at `/login`.

### Verifying the gate is on

```bash
# Quick env-var path (Fly.io shape). HERMES_DASHBOARD_PORTAL_URL is
# optional — defaults to production.
HERMES_DASHBOARD_OAUTH_CLIENT_ID=agent:test \
  hermes dashboard --host 0.0.0.0

# Or the equivalent via config.yaml (recommended for local dev / on-prem):
#
#   dashboard:
#     oauth:
#       client_id: agent:test
#
# then just:
hermes dashboard --host 0.0.0.0

# Hit /api/status to see the gate state:
curl -s http://127.0.0.1:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["nous"]
```

The dashboard's React StatusPage shows the same fields under "Web server". A sidebar AuthWidget surfaces the current identity once you've signed in.

## Connecting Hermes Desktop to a remote backend

Hermes Desktop can drive a Hermes backend running on another machine (a VPS, a home server, a Mini behind Tailscale). In the app this lives under **Settings → Gateway → Remote gateway**, which asks for a **Remote URL** and a **Session token**. (For the desktop app itself — install, settings, chat — see the [Hermes Desktop](/user-guide/desktop) page.)

The "session token" is the dashboard's session token — the same secret the local web UI uses for `/api` and WebSocket calls. **Hermes does not print it for you to copy.** By default the dashboard mints a fresh random token on every boot and injects it straight into the served HTML, so there is nothing sitting in `config.yaml`, in `/gateway`, or in the logs to grab. For a remote connection you set the token yourself on the backend, then paste that same value into the desktop app.

The desktop app sends the token as an `X-Hermes-Session-Token` header. The backend accepts it only in legacy session-token mode — i.e. when bound non-loopback **with `--insecure`**. A non-loopback bind *without* `--insecure` engages the [OAuth gate](#oauth-authentication-gated-mode) instead, which ignores the session token. So a remote desktop connection means: `--insecure` + a token you control.

### On the backend (the remote machine)

```bash
# 1. Mint a stable token and store it in ~/.hermes/.env (secrets file, 0600).
#    Setting HERMES_DASHBOARD_SESSION_TOKEN pins the token so it survives
#    restarts and is the value the desktop app will use — without this,
#    the token is random per boot and uncopyable.
TOKEN=$(openssl rand -base64 32)
echo "HERMES_DASHBOARD_SESSION_TOKEN=$TOKEN" >> ~/.hermes/.env
chmod 600 ~/.hermes/.env
echo "$TOKEN"   # copy this value into the desktop app

# 2. Run the dashboard bound to a reachable address.
#    --insecure is required for any non-loopback bind and keeps the
#    legacy session-token auth path (instead of the OAuth gate).
hermes dashboard --no-open --insecure --host 0.0.0.0 --port 9119
```

If you run the dashboard as a systemd service, `~/.hermes/.env` is picked up automatically when the unit has `EnvironmentFile=%h/.hermes/.env`, so the token is in the environment at boot.

:::warning
`--insecure` exposes a port that reads and writes your `.env` (API keys, secrets) and can run agent commands. Never expose it to the open internet. Put it behind a VPN. [Tailscale](https://tailscale.com/) is the clean option: bind to the machine's tailscale IP (`--host <tailscale-ip>`) and use `http://<tailscale-ip>:9119` as the Remote URL. Only devices on your tailnet can reach it.
:::

### In Hermes Desktop

**Settings → Gateway → Remote gateway:**

- **Remote URL** — `http://<backend-host>:9119` (path prefixes like `/hermes` are supported if you front it with a reverse proxy)
- **Session token** — paste the `$TOKEN` value from step 1
- **Test remote** — confirms the backend is reachable and the token is accepted
- **Save and reconnect** — switches the desktop shell onto the remote backend

The token is stored encrypted in the desktop app's local config. Leave the token field blank on a later edit to keep the saved one.

### Environment-variable override

Instead of the in-app settings, you can point the desktop at a backend with two env vars before launching it. When `HERMES_DESKTOP_REMOTE_URL` is set, it overrides the saved in-app settings (the Gateway settings panel shows an "env override" badge and disables editing):

| Env var | Value |
|---------|-------|
| `HERMES_DESKTOP_REMOTE_URL` | `http://<backend-host>:9119` |
| `HERMES_DESKTOP_REMOTE_TOKEN` | the same token as `HERMES_DASHBOARD_SESSION_TOKEN` on the backend |

Both must be set together — setting only the URL is an error.

### Troubleshooting

- **"Remote gateway incomplete"** — you haven't entered both a URL and a token. The token only needs re-entering if `remoteTokenSet` is false (no saved token yet).
- **Test remote fails with 401** — the token doesn't match the backend's `HERMES_DASHBOARD_SESSION_TOKEN`, or the backend is running *without* `--insecure` on a non-loopback bind (the OAuth gate is on and ignores the session token). Confirm `--insecure` and that the env var is actually loaded (`curl -s -H "X-Hermes-Session-Token: $TOKEN" http://<host>:9119/api/status` should return JSON, not 401).
- **Connection refused / times out** — the backend bound to `127.0.0.1` (the default) instead of a reachable address, or a firewall/VPN is blocking the port. Bind to `0.0.0.0` or the tailscale IP and open the port to your trusted network.
- **No token anywhere to copy** — expected. You mint it (`HERMES_DASHBOARD_SESSION_TOKEN`); Hermes never auto-surfaces the default ephemeral one.

## CORS

The web server restricts CORS to localhost origins only:

- `http://localhost:9119` / `http://127.0.0.1:9119` (production)
- `http://localhost:3000` / `http://127.0.0.1:3000`
- `http://localhost:5173` / `http://127.0.0.1:5173` (Vite dev server)

If you run the server on a custom port, that origin is added automatically.

## Development

If you're contributing to the web dashboard frontend:

```bash
# Terminal 1: start the backend API
hermes dashboard --no-open

# Terminal 2: start the Vite dev server with HMR
cd web/
npm install
npm run dev
```

The Vite dev server at `http://localhost:5173` proxies `/api` requests to the FastAPI backend at `http://127.0.0.1:9119`.

The frontend is built with React 19, TypeScript, Tailwind CSS v4, and shadcn/ui-style components. Production builds output to `hermes_cli/web_dist/` which the FastAPI server serves as a static SPA.

## Automatic Build on Update

When you run `hermes update`, the web frontend is automatically rebuilt if `npm` is available. This keeps the dashboard in sync with code updates. If `npm` isn't installed, the update skips the frontend build and `hermes dashboard` will build it on first launch.

## Themes & plugins

The dashboard ships with six built-in themes and can be extended with user-defined themes, plugin tabs, and backend API routes — all drop-in, no repo clone needed.

**Switch themes live** from the header bar — click the palette icon next to the language switcher. Selection persists to `config.yaml` under `dashboard.theme` and is restored on page load.

Built-in themes:

| Theme | Character |
|-------|-----------|
| **Hermes Teal** (`default`) | Dark teal + cream, system fonts, comfortable spacing |
| **Hermes Teal (Large)** (`default-large`) | Same as default with 18px text and roomier spacing |
| **Midnight** (`midnight`) | Deep blue-violet, Inter + JetBrains Mono |
| **Ember** (`ember`) | Warm crimson + bronze, Spectral serif + IBM Plex Mono |
| **Mono** (`mono`) | Grayscale, IBM Plex, compact |
| **Cyberpunk** (`cyberpunk`) | Neon green on black, Share Tech Mono |
| **Rosé** (`rose`) | Pink + ivory, Fraunces serif, spacious |

To build your own theme, add a plugin tab, inject into shell slots, or expose plugin-specific REST endpoints, see **[Extending the Dashboard](./extending-the-dashboard)** — the complete guide covers:

- Theme YAML schema — palette, typography, layout, assets, componentStyles, colorOverrides, customCSS
- Layout variants — `standard`, `cockpit`, `tiled`
- Plugin manifest, SDK, shell slots, page-scoped slots (inject widgets into built-in pages without overriding them), backend FastAPI routes
- A full combined theme-plus-plugin walkthrough (Strike Freedom cockpit demo)
- Discovery, reload, and troubleshooting
