---
sidebar_position: 5
title: "Microsoft Teams"
description: "Set up Hermes Agent as a Microsoft Teams bot"
---

# Microsoft Teams Setup

Connect Hermes Agent to Microsoft Teams as a bot. Unlike Slack's Socket Mode, Teams delivers messages by calling a **public HTTPS webhook**, so your instance needs a publicly reachable endpoint — either a dev tunnel (local dev) or a real domain (production).

Need meeting summaries from Microsoft Graph events rather than normal bot conversations? Use the dedicated setup page: [Teams Meetings](/docs/user-guide/messaging/teams-meetings).

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
devtunnel port create hermes-bot -p 3978 --protocol https  # replace 3978 with TEAMS_PORT if changed
devtunnel host hermes-bot

# ngrok
ngrok http 3978  # replace 3978 with TEAMS_PORT if changed

# cloudflared
cloudflared tunnel --url http://localhost:3978  # replace 3978 with TEAMS_PORT if changed
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
| `TEAMS_SHAREPOINT_SITE_ID` | SharePoint site ID for outbound channel/group file uploads (e.g. `tenant.sharepoint.com,GUID,GUID`). Optional — leave unset for DM-only file sends. |
| `TEAMS_SHAREPOINT_FOLDER` | Folder path under the SharePoint site's default drive where uploads land. Default: `hermes`. Created on first upload. |

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

### File Attachments

The Teams adapter handles file attachments in both directions, so the agent can read what users send it and reply with images, audio, video, and documents.

#### Inbound (user → Hermes)

When you attach a file in a Teams chat, the adapter downloads the bytes and hands them to the agent as a regular file the model can see (images and PDFs go through vision; audio is transcribed; video and other documents arrive as cached files the agent can `read_file`).

| Type | Supported |
|------|-----------|
| Images | PNG, JPEG, GIF, WebP (rendered via vision) |
| Audio | MP3, WAV, M4A, OGG (transcribed) |
| Video | MP4, MOV, WebM (cached as a file reference) |
| Documents | PDF, DOCX, XLSX, TXT (cached as a file reference) |

The adapter first tries the Bot Framework attachment URL, then falls back to **Microsoft Graph hosted-content download** if the Bot Framework returns 401/403. The Graph fallback uses the same Azure AD app credentials (`TEAMS_CLIENT_ID` / `TEAMS_CLIENT_SECRET` / `TEAMS_TENANT_ID`) — no extra setup is needed when the bot already has the standard Teams permissions, but tenants with stricter scopes may need to grant the bot additional Graph application permissions to download hosted content.

#### Outbound (Hermes → user)

When the agent sends a file (via `send_message` with a `MEDIA:<path>` body, or any tool that produces a file artifact), the path the adapter takes depends on the conversation type:

| Conversation | Delivery mechanism |
|--------------|--------------------|
| **Personal chat (DM)** | **FileConsent card flow.** The bot posts a `FileConsentCard`, the user clicks **Accept**, and the adapter uploads bytes to the Teams-provided OneDrive upload URL. No SharePoint required. |
| **Channel or group chat** | **SharePoint upload.** The adapter uploads the file to the configured SharePoint document library, then posts a `FileInfoCard` linking to it. Requires `TEAMS_SHAREPOINT_SITE_ID` to be set. |

To enable channel/group file sends, set the SharePoint site and (optionally) folder:

```bash
TEAMS_SHAREPOINT_SITE_ID=tenant.sharepoint.com,<site-guid>,<web-guid>
TEAMS_SHAREPOINT_FOLDER=hermes   # default; created on first upload
```

Find the site ID via Microsoft Graph:

```
GET https://graph.microsoft.com/v1.0/sites/{hostname}:/{server-relative-path}
```

The bot's Azure AD app needs `Files.ReadWrite.All` and `Sites.ReadWrite.All` application permissions for SharePoint uploads. DM file sends via FileConsent need no extra Graph permissions.

If `TEAMS_SHAREPOINT_SITE_ID` is unset, the adapter still handles DM file sends; channel/group file sends will be skipped with a warning in the logs.

### Meeting Summary Delivery (Teams Meeting Pipeline)

When the [Teams meeting pipeline plugin](/docs/user-guide/messaging/msgraph-webhook) is enabled, this adapter also handles outbound delivery of meeting summaries — one Teams integration surface, not two. After a meeting's transcript is summarized, the writer posts the summary into your chosen Teams target.

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
| `graph` | Threaded channel posts or 1:1/group chat posts under the bot's identity via Microsoft Graph. | Requires the [Graph app registration](/docs/guides/microsoft-graph-app-registration) with `ChannelMessage.Send` (channel) or `Chat.ReadWrite.All` (chat) application permissions. |

If the `teams_pipeline` plugin is **not** enabled, these settings are inert — they only wire up when the pipeline runtime binds to the Graph webhook ingress.

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

## Related Docs

- [Teams Meetings](/docs/user-guide/messaging/teams-meetings)
- [Operate the Teams Meeting Pipeline](/docs/guides/operate-teams-meeting-pipeline)
