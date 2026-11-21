---
sidebar_position: 5
title: "Microsoft Teams"
description: "Set up Hermes Agent as a Microsoft Teams bot"
---

# Microsoft Teams Setup

Python dependency commands on this page use a
[PM-prepared source checkout](../../reference/package-management.md#developer-workflow).
After a dependency change, reactivate the checkout and restart Hermes.

Connect Hermes Agent to Microsoft Teams as a bot. By default Teams delivers messages by calling a **public HTTPS webhook**, so your instance needs a publicly reachable endpoint — either a dev tunnel (local dev) or a real domain (production). A host without a public endpoint can use [socket mode](#socket-mode-no-public-endpoint) instead (preview).

Need meeting summaries from Microsoft Graph events rather than normal bot conversations? Use the dedicated setup page: [Teams Meetings](./teams-meetings.md).

> Run `hermes gateway setup` and pick **Microsoft Teams** for a guided walk-through.

## How the Bot Responds

| Context | Behavior |
|---------|----------|
| **Personal chat (DM)** | Bot responds to every message. No @mention needed. |
| **Group chat** | Bot only responds when @mentioned. |
| **Channel** | Bot only responds when @mentioned. |

Teams delivers @mentions as regular messages with `<at>BotName</at>` tags, which Hermes strips automatically before processing.

Without resource-specific consent (RSC) Teams only delivers messages that @mention the bot, so no filtering is needed. Once the app manifest grants `ChannelMessage.Read.Group` or `ChatMessage.Read.Chat`, Teams delivers **every** message in the conversation — set `require_mention: true` (or `TEAMS_REQUIRE_MENTION=true`) so the bot only answers channel/group-chat messages that @mention it or reply to one of its own messages. Personal chats are never gated, and a gated message is dropped before its attachments are downloaded.

---

For source or local installs, include the Teams extra so the bundled adapter can
import the Microsoft Teams SDK:

```bash
python -c "import pm; pm.sync_venv(['teams'], explicit=True)"
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

The Teams SDK is optional. When policy permits, the gateway requests the
`teams` extra through PM on first start. PM prepares a complete environment
instead of modifying the running interpreter. To request the extra explicitly
from the prepared source checkout:

```bash
python -c "import pm; pm.sync_venv(['teams'], explicit=True)"
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
| `TEAMS_SOCKET_MODE` | `true` = receive over outbound WebSockets instead of the webhook ([socket mode](#socket-mode-no-public-endpoint), preview; default `false`) |
| `TEAMS_SOCKET_GEOS` | Comma-separated socket-mode regions (default: `amer,emea,apac`) |
| `TEAMS_REQUIRE_MENTION` | Set `true` to answer only @mentions / replies to the bot in channels and group chats (default: `false`; for apps with RSC message-read consent) |

### config.yaml

Alternatively, configure via `~/.hermes/config.yaml`:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      client_id: "your-client-id"
      client_secret: "your-secret"
      tenant_id: "your-tenant-id"
      port: 3978
    require_mention: false   # true once the app has RSC message-read consent
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

### Meeting Summary Delivery (Teams Meeting Pipeline)

When the [Teams meeting pipeline plugin](./msgraph-webhook.md) is enabled, this adapter also handles outbound delivery of meeting summaries — one Teams integration surface, not two. After a meeting's transcript is summarized, the writer posts the summary into your chosen Teams target.

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
| `graph` | Threaded channel posts or 1:1/group chat posts under the bot's identity via Microsoft Graph. | Requires the [Graph app registration](../../guides/microsoft-graph-app-registration.md) with `ChannelMessage.Send` (channel) or `Chat.ReadWrite.All` (chat) application permissions. |

If the `teams_pipeline` plugin is **not** enabled, these settings are inert — they only wire up when the pipeline runtime binds to the Graph webhook ingress.

---

## Socket Mode (no public endpoint)

Teams' socket mode (public preview, new bots only) delivers activities over **outbound** WebSockets that Hermes opens to Microsoft's Agent Communication Service, so the host needs no public URL, tunnel or inbound port. Replies and proactive messages still go out over HTTPS as before. It needs `microsoft-teams-apps` 2.2.1 or newer (the `teams` extra pins it).

1. Switch the bot to socket mode with the Teams CLI (3.1.0 or newer): `teams app update <appId> --messaging-mode socket`. A 403 "only available for new bots" means the registration predates socket mode; use the HTTPS endpoint.
2. Set `TEAMS_SOCKET_MODE=true` (or `socket_mode: true` under `platforms.teams.extra`) and restart the gateway. The log shows `Socket mode connected (geos: amer, emea, apac)`; no webhook listener starts, so `TEAMS_PORT`/`TEAMS_HOST` are ignored.
3. Optional: `TEAMS_SOCKET_GEOS=emea` limits the regions. The SDK must hold a socket in every region your conversations live in, or their activities are not delivered, so keep the default unless you know your tenant's region.

Limits from Microsoft: preview status, no fallback while no socket is connected (Hermes reports the platform as failed and the gateway reconnects it), at most 4 sockets per bot per region, and replies must arrive within 25 s (Hermes acknowledges at once and answers asynchronously). Outbound egress must allow WebSocket upgrades to `*.service.signalr.net` and `*.botapi.skype.com`. See [Microsoft's connectivity guide](https://learn.microsoft.com/en-us/microsoftteams/platform/agents-in-teams/agent-connectivity).

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
| `requirements not met` / `Teams SDK missing` / `No adapter available for teams` | Request the `teams` extra through PM as shown above, then restart the gateway. Do not install into a system Python or mutate the Hermes environment. |
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
**Always set `TEAMS_ALLOWED_USERS`** with the AAD object IDs of authorized users. Without this, anyone who can find or install your bot can interact with it.

Treat `TEAMS_CLIENT_SECRET` like a password — rotate it periodically via the Azure portal or Teams CLI.
:::

- Store credentials in `~/.hermes/.env` with permissions `600` (`chmod 600 ~/.hermes/.env`)
- The bot only accepts messages from users in `TEAMS_ALLOWED_USERS`; unauthorized messages are silently dropped
- Your public endpoint (`/api/messages`) is authenticated by the Teams Bot Framework — requests without valid JWTs are rejected

## Related Docs

- [Teams Meetings](./teams-meetings.md)
- [Operate the Teams Meeting Pipeline](../../guides/operate-teams-meeting-pipeline.md)
