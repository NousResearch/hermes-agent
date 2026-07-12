---
sidebar_position: 5
title: "Webex"
description: "Set up Hermes Agent as a Webex bot using websocket or webhook delivery"
---

# Webex Setup

Hermes Agent integrates with Webex as a first-class messaging platform. The
default setup uses WebSockets, so Hermes can receive events without exposing a
public webhook. Webhook mode is also supported when you want Webex to push
events into a public HTTPS endpoint.

## Overview

| Component | Value |
|-----------|-------|
| **Library** | `aiohttp` + official Webex JavaScript SDK listener |
| **Default connection** | WebSocket |
| **Optional connection** | Webhook |
| **Primary credential** | Webex bot token |
| **User identification** | Email address or Webex person ID |
| **Streaming edits** | Supported for text/markdown replies |

## Step 1: Create a Webex Bot

1. Go to [developer.webex.com/my-apps/new/bot](https://developer.webex.com/my-apps/new/bot)
2. Create a new bot
3. Copy the **bot access token**
4. Invite the bot to the Webex spaces where you want Hermes to respond

:::warning Keep the bot token secret
Anyone with the token can act as your bot. If it leaks, rotate it in the
Webex developer portal and update `WEBEX_BOT_TOKEN`.
:::

## Step 2: Install the Listener Dependencies

WebSocket mode uses the bundled Webex JavaScript listener. Hermes installs its
locked packages on first use. Contributors can preinstall them from the repo
root:

```bash
npm ci --omit=optional --prefix plugins/platforms/webex
```

## Step 3: Configure Hermes

### Option A: Interactive Setup

```bash
hermes gateway setup
```

Select **Webex** when prompted.

### Option B: Manual Configuration

Store the bot token in `~/.hermes/.env`:

```bash
WEBEX_BOT_TOKEN=your-webex-bot-token
```

Put behavior and routing settings in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    webex:
      enabled: true
      allow_from:
        - you@example.com
      home_channel:
        platform: webex
        chat_id: Y2lzY29zcGFyazovL3VzL1JPT00v...
        name: Ops
      extra:
        connection_mode: websocket
        require_mention: true
```

Then start the gateway:

```bash
hermes gateway
```

## WebSocket vs Webhook

| Mode | Best for | Requirements |
|------|----------|--------------|
| `websocket` | Local development, laptops, private servers | `WEBEX_BOT_TOKEN` |
| `webhook` | Public deployments where you want inbound HTTPS callbacks | `WEBEX_BOT_TOKEN` + signing secret + public HTTPS URL |

### Webhook Mode

If you prefer webhooks, store the signing secret in `~/.hermes/.env`:

```bash
WEBEX_WEBHOOK_SECRET=change-me
```

Then configure the transport in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    webex:
      enabled: true
      allow_from:
        - you@example.com
      extra:
        connection_mode: webhook
        public_url: https://example.com
        host: 0.0.0.0
        port: 8646
        path: /webex/webhook
```

## How Hermes Behaves in Webex

| Context | Behavior |
|---------|----------|
| **Direct space** | Hermes replies normally to messages and slash commands |
| **Group space** | Free-form chat requires directly mentioning the bot |
| **Group space slash command** | Mention the bot before the command so Webex delivers the message to Hermes |
| **Thread replies** | Hermes replies in the same Webex thread when `parentId` is present and loads recent thread context when first tagged in an existing thread |

Examples in a group space:

```text
@Hermes summarize this thread
@Hermes /sethome
@Hermes /status
```

Hermes does **not** auto-tag the user in outbound replies.

## Home Room

Use `/sethome` in a Webex chat to make it the default destination for cron
results and cross-platform delivery.

You can also set it directly in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    webex:
      home_channel:
        platform: webex
        chat_id: Y2lzY29zcGFyazovL3VzL1JPT00v...
        name: My Webex Room
```

## Troubleshooting

### Hermes responds in DMs but not in group spaces

That usually means the bot was not directly mentioned. In group spaces, free-form
messages are mention-gated by default.

### `/sethome` or `/status` seems ignored in a group space

Mention the bot before the command. Webex bots receive group-space messages
only when directly mentioned. Also verify that the Webex adapter loaded at
gateway startup.

### Webhook mode does not connect

Check that:

- `gateway.platforms.webex.extra.public_url` is an HTTPS URL
- the configured public URL is reachable from Webex
- `WEBEX_WEBHOOK_SECRET` is set and matches the configured webhook
- your Python trust store can validate outbound TLS connections
