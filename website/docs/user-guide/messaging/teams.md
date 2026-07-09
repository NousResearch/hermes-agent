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
| **Personal chat (DM)** | Bot responds to every message. No @mention needed. |
| **Group chat** | Bot only responds when @mentioned. |
| **Channel** | Bot only responds when @mentioned. |

Teams delivers @mentions as regular messages with `<at>BotName</at>` tags, which Hermes strips automatically before processing.

If your Teams app has resource-specific consent (RSC) permissions that make Teams deliver every group-chat or channel message to the bot, Hermes still ignores unmentioned group/channel messages by default. Set `TEAMS_RESPOND_TO_ALL_MESSAGES=true` or `platforms.teams.extra.respond_to_all_messages: true` only when you intentionally want the bot to process every message in shared spaces by default.

Mode admins can also change the behavior for the current Teams conversation from inside Teams:

```text
@Hermes /teams-mode status
@Hermes /teams-mode all
@Hermes /teams-mode mentions
@Hermes /teams-mode default
```

`all` makes Hermes process every delivered message in that conversation, `mentions` requires @mentions again, and `default` clears the conversation override so it follows `TEAMS_RESPOND_TO_ALL_MESSAGES` or `respond_to_all_messages`. Group-chat and channel mode commands must mention the bot; unmentioned command-like messages are consumed in shared spaces so they do not reach the agent as normal prompts. Hermes stores these overrides under `HERMES_HOME/state/teams_response_modes.json`.

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
| `TEAMS_RESPOND_TO_ALL_MESSAGES` | Set `true` to process every group-chat/channel message delivered by Teams; default requires @mention outside DMs |
| `TEAMS_MODE_ALLOWED_USERS` | Comma-separated AAD object IDs allowed to run `/teams-mode`; defaults to `TEAMS_ALLOWED_USERS` |
| `TEAMS_RESPONSE_MODE_STATE_FILE` | Optional path for persisted per-conversation `/teams-mode` overrides |

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
      respond_to_all_messages: false
      mode_allowed_users: "aad-object-id-1,aad-object-id-2"
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

### Channel and Group Chat Media

Teams delivers personal-chat attachments through the normal Bot Framework attachment payload. Channel and group-chat messages can be more fragmented: inline images may be exposed as Graph hosted content, while file attachments are often SharePoint or OneDrive references.

Hermes handles this in layers:

1. Cache the Bot Framework attachment payload first.
2. If a channel or group-chat message looked like it had media but no media was cached, read the message through Microsoft Graph.
3. Download inline images from the message's `hostedContents`.
4. Download reference attachments through the Graph Drive `/shares/{shareId}/driveItem/content` endpoint and cache them as images, documents, video, or audio based on the returned bytes and filename.

Graph media fallback is enabled by default. Disable it if you only want Bot Framework attachment handling:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      graph_ingest_attachments: false
```

To use channel or group-chat Graph media fallback, configure the Teams app manifest with `webApplicationInfo` and `authorization.permissions.resourceSpecific` entries for `ChannelMessage.Read.Group` and/or `ChatMessage.Read.Chat`. The Entra app ID can be the same as the bot ID. Microsoft documents this manifest shape in [Receive all messages for bots and agents](https://learn.microsoft.com/en-us/microsoftteams/platform/bots/how-to/conversations/channel-messages-for-bots-and-agents). Those permissions can make Teams deliver every message to the bot; Hermes still requires an @mention outside DMs unless you set `respond_to_all_messages` or enable `respond-all` with `/teams-mode all` in that conversation.

For Graph authentication, Hermes uses `MSGRAPH_TENANT_ID`, `MSGRAPH_CLIENT_ID`, and `MSGRAPH_CLIENT_SECRET` when present. If those are not set, it falls back to the Teams bot credentials. Reading SharePoint or OneDrive file bytes through Graph Drive content can require app-level file permissions such as `Files.Read.All` or `Sites.Read.All`, depending on your tenant policy; see Microsoft's [Download driveItem content](https://learn.microsoft.com/en-us/graph/api/driveitem-get-content) permissions table.

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
| Image attachments do not reach the agent and logs show `[teams] Failed to cache image attachment` | Update Hermes to a version with authenticated Teams image fetches. Teams image `contentUrl` values can require the bot's Bot Framework token, so also verify `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET`, and `TEAMS_TENANT_ID` are correct |
| Channel or group-chat images/files still do not reach the agent | Verify the manifest has `ChannelMessage.Read.Group` and/or `ChatMessage.Read.Chat` under `authorization.permissions.resourceSpecific`, reinstall or update the app in that conversation, and make sure the Graph app can read hosted content and SharePoint/OneDrive file bytes |
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

- [Teams Meetings](/user-guide/messaging/teams-meetings)
- [Operate the Teams Meeting Pipeline](/guides/operate-teams-meeting-pipeline)
