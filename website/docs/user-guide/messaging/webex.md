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

WebSocket mode uses the bundled Webex JavaScript listener. Run this once from
the repo root:

```bash
npm install --prefix plugins/platforms/webex
```

## Step 3: Configure Hermes

### Option A: Interactive Setup

```bash
hermes gateway setup
```

Select **Webex** when prompted.

### Option B: Manual Configuration

Add the following to `~/.hermes/.env`:

```bash
WEBEX_BOT_TOKEN=your-webex-bot-token
WEBEX_ALLOWED_USERS=you@example.com

# Optional
WEBEX_CONNECTION_MODE=websocket
WEBEX_HOME_CHANNEL=Y2lzY29zcGFyazovL3VzL1JPT00v...
WEBEX_HOME_CHANNEL_NAME=Ops
```

Then start the gateway:

```bash
hermes gateway
```

## WebSocket vs Webhook

| Mode | Best for | Requirements |
|------|----------|--------------|
| `websocket` | Local development, laptops, private servers | `WEBEX_BOT_TOKEN`, `npm install --prefix plugins/platforms/webex` |
| `webhook` | Public deployments where you want inbound HTTPS callbacks | `WEBEX_BOT_TOKEN` + public HTTPS URL |

### Webhook Mode

If you prefer webhooks, add:

```bash
WEBEX_CONNECTION_MODE=webhook
WEBEX_WEBHOOK_PUBLIC_URL=https://example.com
WEBEX_WEBHOOK_SECRET=change-me
```

Optional local bind settings:

```bash
WEBEX_WEBHOOK_HOST=0.0.0.0
WEBEX_WEBHOOK_PORT=8646
WEBEX_WEBHOOK_PATH=/webex/webhook
```

## How Hermes Behaves in Webex

| Context | Behavior |
|---------|----------|
| **Direct space** | Hermes replies normally to messages and slash commands |
| **Group space** | Free-form chat requires directly mentioning the bot |
| **Group space slash command** | Recognized Hermes slash commands work directly, with or without a leading bot mention |
| **Thread replies** | Hermes replies in the same Webex thread when `parentId` is present and loads recent thread context when first tagged in an existing thread |

Examples in a group space:

```text
@mrshubot-hermes summarize this thread
/sethome
@mrshubot-hermes /status
```

Hermes does **not** auto-tag the user in outbound replies.

## Home Room

Use `/sethome` in a Webex chat to make it the default destination for cron
results and cross-platform delivery.

You can also set it directly:

```bash
WEBEX_HOME_CHANNEL=Y2lzY29zcGFyazovL3VzL1JPT00v...
WEBEX_HOME_CHANNEL_NAME=My Webex Room
```

## Troubleshooting

### Hermes responds in DMs but not in group spaces

That usually means the bot was not directly mentioned. In group spaces, free-form
messages are mention-gated by default.

### `/sethome` or `/status` seems ignored in a group space

Recognized Hermes slash commands should work directly. If they do not, verify
that you are running an up-to-date Hermes build and that the Webex adapter
loaded successfully at gateway startup.

### Webhook mode does not connect

Check that:

- `WEBEX_WEBHOOK_PUBLIC_URL` is an HTTPS URL
- the public URL is reachable from Webex
- `WEBEX_WEBHOOK_SECRET` matches your configured webhook
- your Python trust store can validate outbound TLS connections
