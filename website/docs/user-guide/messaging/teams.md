---
sidebar_position: 5
title: "Microsoft Teams"
description: "Set up Hermes Agent as a Microsoft Teams bot"
---

# Microsoft Teams Setup

Connect Hermes Agent to Microsoft Teams as a bot. Unlike Slack's Socket Mode, Teams delivers messages by calling a **public HTTPS webhook**, so your instance needs a publicly reachable endpoint — either a dev tunnel (local dev) or a real domain (production).

Need meeting summaries from Microsoft Graph events rather than normal bot conversations? Use the dedicated setup page: [Teams Meetings](/user-guide/messaging/teams-meetings).

> Run `hermes gateway setup` and pick **Microsoft Teams** for a guided walk-through.

## How the Bot Responds

| Context | Behavior |
|---------|----------|
| **Personal chat (DM)** | Bot responds to every message from an authorized user (`TEAMS_ALLOWED_USERS`, `TEAMS_ALLOW_ALL_USERS`, an approved pairing, or a global `GATEWAY_ALLOWED_USERS`/`GATEWAY_ALLOW_ALL_USERS` grant). No @mention needed. |
| **Group chat** | Teams itself only delivers a group message to the bot once it's @mentioned — Hermes doesn't additionally enforce this. |
| **Channel** | Bot only responds when @mentioned (Hermes enforces this directly — see below), and, if `allowed_channels` is configured, only in a listed channel. |

Teams delivers @mentions as regular messages with `<at>BotName</at>` tags, which Hermes strips automatically before processing.

### Restricting which channels the bot answers in

By default the bot answers in any team channel it has been added to, as long as it's @mentioned and the sender passes normal user authorization (see the matrix below) — this is unchanged from before. Set `allowed_channels` in `~/.hermes/config.yaml` to restrict it to specific channels.

```yaml
platforms:
  teams:
    enabled: true
    extra:
      client_id: "your-client-id"
      client_secret: "your-secret"
      tenant_id: "your-tenant-id"
    # A channel's conversation ID, its channelData.channel.id, or its
    # channelData.team.id (which allows every channel in that team). Copy
    # the id exactly — matching is case-sensitive, not case-folded.
    # "*" instead of a list of ids allows every channel and bypasses normal
    # Hermes user authorization entirely -- see the matrix below; it is not
    # the same as leaving allowed_channels empty/unset, which still requires
    # some applicable authorization grant (TEAMS_ALLOWED_USERS,
    # TEAMS_ALLOW_ALL_USERS, an approved pairing, or a global grant).
    allowed_channels:
      - "19:abc123@thread.tacv2"
      - "19:def456@thread.tacv2"
```

The exact authorization matrix for a channel message:

| `allowed_channels` | Channel matches | Channel does NOT match |
|---|---|---|
| Empty / unset (default) | Sender still needs normal Hermes user authorization (`TEAMS_ALLOWED_USERS`, `TEAMS_ALLOW_ALL_USERS`, an approved pairing, or a global grant) — same as before this setting existed | n/a — no channel is excluded |
| Specific channel/team ids | Anyone may interact by @mentioning the bot — being in the channel is the access signal, no further user authorization is consulted | Dropped silently before the message ever reaches the agent or any user authorization check |
| `"*"` (every channel) | Anyone may interact by @mentioning the bot in ANY channel — bypasses normal Hermes user authorization entirely | n/a — every channel matches |

So once a channel is listed (or `allowed_channels` is `"*"`), you don't need to also add every teammate in that channel to `TEAMS_ALLOWED_USERS`. Channel authorization only ever widens access to channels — it never affects DMs, which are still governed by the normal Hermes user-authorization chain regardless. When `allowed_channels` names specific ids, a channel that isn't one of them is dropped outright — no user-authorization check is ever consulted for it, so adding a sender to `TEAMS_ALLOWED_USERS` cannot grant access from an unlisted channel.

Independently of `allowed_channels`, every channel message needs an explicit @mention by default (`require_mention`, default `true`) — this applies whether or not a channel is listed, and is what makes the **Channel** row above true even without `allowed_channels` configured. Set `require_mention: false` to disable it. This has no effect on group chats — see the table above.

Channel authorization covers ordinary messages only. Clicking an [Adaptive Card approval button](#interactive-approval-cards) is a separate check that requires `TEAMS_ALLOWED_USERS` or `TEAMS_ALLOW_ALL_USERS` regardless of `allowed_channels` — someone who can message the bot in a listed channel but is neither in `TEAMS_ALLOWED_USERS` nor covered by `TEAMS_ALLOW_ALL_USERS` can ask the agent to do something, but can't click Allow/Deny on the resulting approval card.

---

For source or local installs, include the Teams extra so the bundled adapter can
import the Microsoft Teams SDK:

```bash
uv sync --extra teams
# or, for editable installs:
uv pip install -e ".[teams]"
```

## Step 1: Install the Teams CLI

The `@microsoft/teams.cli` automates bot registration — no Azure portal needed.

```bash
npm install -g @microsoft/teams.cli@preview
teams login
```

To verify your login and find your own AAD object ID (needed for `TEAMS_ALLOWED_USERS`):

```bash
teams status --verbose
```

---

## Step 2: Expose the Webhook Port

Teams cannot deliver messages to `localhost`. For local development, use any tunnel tool to get a public HTTPS URL. The default port is `3978` — change it with `TEAMS_PORT` if needed.

```bash
# devtunnel (Microsoft)
devtunnel create hermes-bot --allow-anonymous
devtunnel port create hermes-bot -p 3978 --protocol http  # replace 3978 with TEAMS_PORT if changed
devtunnel host hermes-bot

# ngrok
ngrok http 3978  # replace 3978 with TEAMS_PORT if changed

# cloudflared
cloudflared tunnel --url http://localhost:3978  # replace 3978 with TEAMS_PORT if changed
```

Copy the `https://` URL from the output — you'll use it in the next step. Leave the tunnel running while developing.

The public tunnel URL uses HTTPS, but Hermes' local webhook listener uses plain HTTP. The tunnel terminates TLS and forwards HTTP to port `3978`; do not configure the local tunnel port as HTTPS.

For production, point your bot's endpoint at your server's public domain instead (see [Production Deployment](#production-deployment)).

---

## Step 3: Create the Bot

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://<your-tunnel-url>/api/messages"
```

The CLI outputs your `CLIENT_ID`, `CLIENT_SECRET`, and `TENANT_ID`, plus an install link for Step 6. Save the client secret — it won't be shown again.

---

## Step 4: Configure Environment Variables

Add to `~/.hermes/.env`:

```bash
# Required
TEAMS_CLIENT_ID=<your-client-id>
TEAMS_CLIENT_SECRET=<your-client-secret>
TEAMS_TENANT_ID=<your-tenant-id>

# Restrict access to specific users (recommended)
# Use AAD object IDs from `teams status --verbose`
TEAMS_ALLOWED_USERS=<your-aad-object-id>
```

---

## Step 5: Start the Gateway

**Docker** (must run from the directory that contains `docker-compose.yml` — usually your cloned `hermes-agent` repo, not `~`):

```bash
cd /path/to/hermes-agent
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d gateway
```

**Native / systemd install** (typical `hermes` one-liner installer under `~/.hermes/hermes-agent`):

```bash
hermes gateway restart
# or foreground: hermes gateway run
```

The Teams SDK is optional; when Teams is enabled, the gateway lazy-installs it into Hermes' own venv on first start (do **not** use system `pip install` on Ubuntu 24.04 — that hits PEP 668 `externally-managed-environment`). To install manually into the Hermes venv:

```bash
~/.hermes/hermes-agent/venv/bin/pip install microsoft-teams-apps aiohttp
# or from a clone of the agent: uv sync --extra teams
```

The default webhook port is `3978` (override with `TEAMS_PORT`). Check that it's running:

```bash
curl http://localhost:3978/health   # should return: ok
# Docker:
docker logs -f hermes
# Native:
hermes gateway status -l
```

Look for:
```
[teams] Webhook server listening on * (all interfaces, IPv4+IPv6):3978/api/messages
```

---

## Step 6: Install the App in Teams

```bash
teams app get <teamsAppId> --install-link
```

Open the printed link in your browser — it opens directly in the Teams client. After installing, send a direct message to your bot — it's ready.

---

## Configuration Reference

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TEAMS_CLIENT_ID` | Azure AD App (client) ID |
| `TEAMS_CLIENT_SECRET` | Azure AD client secret |
| `TEAMS_TENANT_ID` | Azure AD tenant ID |
| `TEAMS_ALLOWED_USERS` | Comma-separated AAD object IDs allowed to use the bot |
| `TEAMS_ALLOW_ALL_USERS` | Set `true` to skip the allowlist and allow anyone |
| `TEAMS_HOME_CHANNEL` | Conversation ID for cron/proactive message delivery |
| `TEAMS_HOME_CHANNEL_NAME` | Display name for the home channel |
| `TEAMS_PORT` | Webhook port (default: `3978`) |

### config.yaml

Alternatively, configure credentials via `~/.hermes/config.yaml`:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      client_id: "your-client-id"
      client_secret: "your-secret"
      tenant_id: "your-tenant-id"
      port: 3978
```

`config.yaml` is the documented, primary way to set the channel allowlist and mention requirement — see [Restricting which channels the bot answers in](#restricting-which-channels-the-bot-answers-in). `TEAMS_ALLOWED_CHANNELS` (comma-separated) and `TEAMS_REQUIRE_MENTION` also work as an internal fallback, but `config.yaml` wins whenever both are set for the same key:

```yaml
platforms:
  teams:
    allowed_channels:
      - "19:abc123@thread.tacv2"   # empty/unset: unrestricted (default)
    require_mention: true          # default: true
```

---

## Features

### Interactive Approval Cards

When the agent needs to run a potentially dangerous command, it sends an Adaptive Card with four buttons instead of asking you to type `/approve`:

- **Allow Once** — approve this specific command
- **Allow Session** — approve this pattern for the rest of the session
- **Always Allow** — permanently approve this pattern
- **Deny** — reject the command

Clicking a button resolves the approval inline and replaces the card with the decision.

Clicking a button requires `TEAMS_ALLOWED_USERS` or `TEAMS_ALLOW_ALL_USERS` — a channel allowlist does not grant this. See [the note above](#restricting-which-channels-the-bot-answers-in) if you're using `allowed_channels`. (This check reads those two env vars directly rather than through the profile-scoped resolution the rest of Hermes uses, so it is not multiplex-profile-aware — a pre-existing adapter limitation, not something introduced by `allowed_channels`.)

### Meeting Summary Delivery (Teams Meeting Pipeline)

When the [Teams meeting pipeline plugin](/user-guide/messaging/msgraph-webhook) is enabled, this adapter also handles outbound delivery of meeting summaries — one Teams integration surface, not two. After a meeting's transcript is summarized, the writer posts the summary into your chosen Teams target.

Pipeline summary delivery is configured under the `teams` platform entry alongside the bot config:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      # existing bot config (client_id, client_secret, tenant_id, port) ...

      # Meeting summary delivery (only used when the teams_pipeline plugin is enabled)
      delivery_mode: "graph"       # or "incoming_webhook"
      # For delivery_mode: graph — pick ONE of:
      chat_id: "19:meeting_..."    # post into a Teams chat
      # team_id: "..."             # OR post into a channel
      # channel_id: "..."
      # access_token: "..."        # optional; falls back to MSGRAPH_* app credentials
      # For delivery_mode: incoming_webhook:
      # incoming_webhook_url: "https://outlook.office.com/webhook/..."
```

| Mode | Use when | Trade-off |
|------|----------|-----------|
| `incoming_webhook` | Simple "post a summary into this channel" with a static Teams-generated URL. | No reply threading, no reactions, shows as the webhook's configured identity. |
| `graph` | Threaded channel posts or 1:1/group chat posts under the bot's identity via Microsoft Graph. | Requires the [Graph app registration](/guides/microsoft-graph-app-registration) with `ChannelMessage.Send` (channel) or `Chat.ReadWrite.All` (chat) application permissions. |

If the `teams_pipeline` plugin is **not** enabled, these settings are inert — they only wire up when the pipeline runtime binds to the Graph webhook ingress.

---

## Production Deployment

For a permanent server, terminate TLS at a reverse proxy and forward requests to the plain HTTP Hermes listener, normally `http://127.0.0.1:3978`. Register the proxy's public HTTPS endpoint with Teams:

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://your-domain.com/api/messages"
```

If you've already created the bot and just need to update the endpoint:

```bash
teams app update --id <teamsAppId> --endpoint "https://your-domain.com/api/messages"
```

Make sure the public HTTPS endpoint is reachable from the internet and uses a valid TLS certificate. Teams rejects self-signed certificates. Keep the Hermes listener behind the proxy; port `3978` does not serve HTTPS itself.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Can't find a suitable configuration file` from `docker compose` | You are not in the repo that has `docker-compose.yml`, or you are on a native install — use `hermes gateway restart` instead, or `cd` into the clone first |
| `requirements not met` / `Teams SDK missing` / `No adapter available for teams` | Restart gateway so lazy-install can run, or install into the **Hermes venv**: `~/.hermes/hermes-agent/venv/bin/pip install microsoft-teams-apps aiohttp`. System `pip` fails on Ubuntu 24.04 (PEP 668) and would not affect the service anyway |
| `health` endpoint works but bot doesn't respond | Check that your tunnel is still running and the bot's messaging endpoint matches the tunnel URL |
| Logs show `"UNKNOWN / HTTP/1.0" 400` when Teams sends a message | The tunnel or reverse proxy is forwarding HTTPS to Hermes' plain HTTP listener. Terminate TLS at the proxy and forward HTTP to port `3978` |
| `KeyError: 'teams'` in logs | Restart the container — this is fixed in the current version |
| Bot responds with auth errors | Verify `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET`, and `TEAMS_TENANT_ID` are all set correctly |
| `No inference provider configured` | Check that `ANTHROPIC_API_KEY` (or another provider key) is set in `~/.hermes/.env` |
| Bot receives messages but ignores them | Your AAD object ID may not be in `TEAMS_ALLOWED_USERS`. Run `teams status --verbose` to find it |
| Tunnel URL changes on restart | devtunnel URLs are persistent if you use a named tunnel (`devtunnel create hermes-bot`). ngrok and cloudflared generate a new URL each run unless you have a paid plan — update the bot endpoint with `teams app update` when it changes |
| Teams shows "This bot is not responding" | The webhook returned an error. Check `docker logs hermes` / `hermes gateway status -l` for tracebacks |
| `[teams] Failed to connect` in logs | The SDK failed to authenticate. Double-check your credentials and that the tenant ID matches the account you used in `teams login` |

---

## Security

:::warning
**Set `TEAMS_ALLOWED_USERS`** with the AAD object IDs of authorized users, for a fixed, known list of who can DM the bot, rather than relying on `TEAMS_ALLOW_ALL_USERS`.

Treat `TEAMS_CLIENT_SECRET` like a password — rotate it periodically via the Azure portal or Teams CLI.
:::

- Store credentials in `~/.hermes/.env` with permissions `600` (`chmod 600 ~/.hermes/.env`)
- DMs are governed by the normal Hermes user-authorization chain (`TEAMS_ALLOWED_USERS`, `TEAMS_ALLOW_ALL_USERS`, an approved pairing, or a global `GATEWAY_ALLOWED_USERS`/`GATEWAY_ALLOW_ALL_USERS` grant). An unrecognized sender is offered a pairing code by default rather than being silently ignored, even with `TEAMS_ALLOWED_USERS` set — Teams isn't wired into the automatic switch-to-silent-drop some other platforms get from having an allowlist configured. Set `platforms.teams.unauthorized_dm_behavior: ignore` in config.yaml if you want unapproved DMs dropped with no reply at all.
- When `allowed_channels` names specific channel/team ids, a channel message from any other channel is dropped before the agent ever sees it, regardless of who sent it — but `allowed_channels: ["*"]` authorizes every channel sender outright, bypassing user authorization entirely (see the matrix above)
- Your public endpoint (`/api/messages`) is authenticated by the Teams Bot Framework — requests without valid JWTs are rejected

## Related Docs

- [Teams Meetings](/user-guide/messaging/teams-meetings)
- [Operate the Teams Meeting Pipeline](/guides/operate-teams-meeting-pipeline)
