---
sidebar_position: 5
title: "Microsoft Teams"
description: "Set up Hermes Agent as a Microsoft Teams bot"
---

# Microsoft Teams Setup

Connect Hermes Agent to Microsoft Teams as a bot. Unlike Slack's Socket Mode, Teams delivers messages by calling a **public HTTPS webhook**, so your instance needs a publicly reachable endpoint — either a dev tunnel (local dev) or a real domain (production).

## How the Bot Responds

| Context | Behavior |
|---------|----------|
| **Personal chat (DM)** | Bot responds to every message. No @mention needed. |
| **Group chat** | Bot only responds when @mentioned. |
| **Channel** | Bot only responds when @mentioned. |

Teams delivers @mentions as regular messages with `<at>BotName</at>` tags, which Hermes strips automatically before processing.

---

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
devtunnel port create hermes-bot -p 3978 --protocol https
devtunnel host hermes-bot

# ngrok
ngrok http 3978

# cloudflared
cloudflared tunnel --url http://localhost:3978
```

Copy the `https://` URL from the output — you'll use it in the next step. Leave the tunnel running while developing.

For production, point your bot's endpoint at your server's public domain instead (see [Production Deployment](#production-deployment)).

---

## Step 3: Create the Bot

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://<your-tunnel-url>/api/messages"
```

The CLI outputs your `CLIENT_ID`, `CLIENT_SECRET`, and `TENANT_ID`. Save them — you'll need all three.

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

```bash
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d gateway
```

This starts the gateway. The default webhook port is `3978` (override with `TEAMS_PORT`). Check that it's running:

```bash
curl http://localhost:3978/health   # should return: ok
docker logs -f hermes
```

Look for:
```
[teams] Webhook server listening on 0.0.0.0:3978/api/messages
```

---

## Step 6: Install the App in Teams

```bash
teams app install --id <teamsAppId>
```

The `teamsAppId` was printed by `teams app create` in Step 3. After installing, open Microsoft Teams and send a direct message to your bot — it's ready.

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

---

## Production Deployment

For a permanent server, skip devtunnel and register your bot with your server's public HTTPS endpoint:

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://your-domain.com/api/messages"
```

If you've already created the bot and just need to update the endpoint:

```bash
teams app update --id <teamsAppId> --endpoint "https://your-domain.com/api/messages"
```

Make sure your configured port (`TEAMS_PORT`, default `3978`) is reachable from the internet and that your TLS certificate is valid — Teams rejects self-signed certificates.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `health` endpoint works but bot doesn't respond | Check that your tunnel is still running and the bot's messaging endpoint matches the tunnel URL |
| `KeyError: 'teams'` in logs | Restart the container — this is fixed in the current version |
| Bot responds with auth errors | Verify `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET`, and `TEAMS_TENANT_ID` are all set correctly |
| `No inference provider configured` | Check that `ANTHROPIC_API_KEY` (or another provider key) is set in `~/.hermes/.env` |
| Bot receives messages but ignores them | Your AAD object ID may not be in `TEAMS_ALLOWED_USERS`. Run `teams status --verbose` to find it |
| Tunnel URL changes on restart | devtunnel URLs are persistent if you use a named tunnel (`devtunnel create hermes-bot`). ngrok and cloudflared generate a new URL each run unless you have a paid plan — update the bot endpoint with `teams app update` when it changes |
| Teams shows "This bot is not responding" | The webhook returned an error. Check `docker logs hermes` for tracebacks |
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
