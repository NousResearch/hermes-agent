---
sidebar_position: 10
title: "DingTalk"
description: "Set up Hermes Agent as a DingTalk bot using Stream Mode for real-time messaging"
---

# DingTalk Setup

Hermes connects to DingTalk through the [dingtalk-stream](https://pypi.org/project/dingtalk-stream/) SDK using Stream Mode — a WebSocket-based protocol that requires no public webhook URL. Messages arrive in real-time and responses are sent via the session webhook in markdown format.

DingTalk (钉钉) is Alibaba's enterprise communication platform with over 700 million registered users, making it the #1 business application in China. It combines messaging, video conferencing, task management, and workflow automation into a single platform used by millions of organizations.

:::info Dependencies
The DingTalk adapter requires additional Python packages:

```bash
pip install dingtalk-stream httpx
```

`httpx` is already a core Hermes dependency, so in practice you only need `dingtalk-stream`.
:::

---

## Prerequisites

- **DingTalk developer account** — register at [open-dev.dingtalk.com](https://open-dev.dingtalk.com)
- **An application created** on the DingTalk Open Platform with Robot (机器人) capability enabled

---

## Step 1: Create a DingTalk Application

1. Go to [open-dev.dingtalk.com](https://open-dev.dingtalk.com) and log in
2. Click **Create Application** (创建应用)
3. Fill in the application name and description
4. Under **Capabilities** (添加能力), enable **Robot** (机器人)
5. In the Robot configuration:
   - Enable **Stream Mode** (Stream 模式) — this is critical, as it eliminates the need for a public webhook URL
   - Set the bot name and avatar
6. Navigate to **Credentials & Basic Info** (凭证与基本信息) to find:
   - **AppKey** — this is your `DINGTALK_CLIENT_ID`
   - **AppSecret** — this is your `DINGTALK_CLIENT_SECRET`
7. Publish the application (发布)

:::tip
Stream Mode is strongly recommended over the legacy HTTP webhook approach. It works behind firewalls, NATs, and requires no public IP or domain — the SDK maintains a persistent WebSocket connection to DingTalk's servers.
:::

---

## Step 2: Configure Hermes

The easiest way:

```bash
hermes gateway setup
```

Select **DingTalk** from the platform menu. The wizard will:

1. Check if `dingtalk-stream` is installed
2. Prompt for your AppKey (Client ID)
3. Prompt for your AppSecret (Client Secret)
4. Configure allowed users and access policies

### Manual Configuration

Add to `~/.hermes/.env`:

```bash
# Required
DINGTALK_CLIENT_ID=your-app-key
DINGTALK_CLIENT_SECRET=your-app-secret

# Security (recommended)
DINGTALK_ALLOWED_USERS=user1_staff_id,user2_staff_id    # Comma-separated DingTalk staff IDs

# Optional
DINGTALK_HOME_CHANNEL=user1_staff_id                     # Default delivery target for cron jobs
```

Then start the gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

---

## Access Control

### DM Access

DM access follows the same pattern as all other Hermes platforms:

1. **`DINGTALK_ALLOWED_USERS` set** → only those users can message
2. **No allowlist set** → unknown users get a DM pairing code (approve via `hermes pairing approve dingtalk CODE`)
3. **`DINGTALK_ALLOW_ALL_USERS=true`** → anyone can message (use with caution)

### Group Access

In group chats, the bot responds when @mentioned. Group access follows the same rules — only allowed users can trigger the bot, even in groups.

---

## Features

### Stream Mode (No Webhook URL)

Unlike traditional bot platforms that require a publicly accessible webhook endpoint, DingTalk's Stream Mode uses a persistent WebSocket connection initiated from your side. This means:

- **No public IP required** — works behind firewalls and NATs
- **No domain or SSL certificate needed** — the SDK handles the connection
- **Automatic reconnection** — if the connection drops, the adapter reconnects with exponential backoff (2s → 5s → 10s → 30s → 60s)

### Markdown Replies

Responses are sent in DingTalk's markdown format, which supports rich text formatting including headers, bold, italic, links, and code blocks.

### DM and Group Chat

The adapter supports both:

- **Direct Messages (1:1)** — private conversations with the bot
- **Group Chat** — the bot responds when @mentioned in a group

### Message Deduplication

The adapter tracks recently processed message IDs (up to 1,000 messages within a 5-minute window) to prevent duplicate processing if DingTalk redelivers a message.

### Auto-Reconnection

If the WebSocket connection drops, the adapter automatically reconnects using exponential backoff:

- Retry intervals: 2s, 5s, 10s, 30s, 60s
- Reconnection is transparent — no manual intervention needed

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"dingtalk-stream not installed"** | Run `pip install dingtalk-stream httpx` in the Hermes environment |
| **"DINGTALK_CLIENT_ID not set"** | Set `DINGTALK_CLIENT_ID` and `DINGTALK_CLIENT_SECRET` in `~/.hermes/.env` |
| **Bot not responding** | Verify the application is published on open-dev.dingtalk.com and Stream Mode is enabled |
| **Connection keeps dropping** | Check network connectivity. The adapter will auto-reconnect with backoff. Check logs for specific error messages. |
| **Messages processed twice** | This is rare — the deduplication window handles most cases. If persistent, check that only one gateway instance is running. |
| **Bot responds to no one** | Configure `DINGTALK_ALLOWED_USERS`, use DM pairing, or explicitly allow all users through gateway policy if you want broader access. |
| **Group messages ignored** | Ensure the bot is @mentioned in group chats. Only @mentions trigger the bot in groups. |

---

## Security

:::warning
**Always configure access controls.** The bot has terminal access by default. Without `DINGTALK_ALLOWED_USERS` or DM pairing, the gateway denies all incoming messages as a safety measure.
:::

- Use DM pairing or explicit allowlists for safe onboarding of new users
- Keep your AppSecret confidential — treat it like a password
- The `DINGTALK_CLIENT_SECRET` in `~/.hermes/.env` should be readable only by the user running Hermes
- DingTalk's Stream Mode connection is encrypted via TLS

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DINGTALK_CLIENT_ID` | Yes | — | DingTalk application AppKey |
| `DINGTALK_CLIENT_SECRET` | Yes | — | DingTalk application AppSecret |
| `DINGTALK_ALLOWED_USERS` | No | — | Comma-separated DingTalk staff IDs |
| `DINGTALK_ALLOW_ALL_USERS` | No | `false` | Allow all users (not recommended) |
| `DINGTALK_HOME_CHANNEL` | No | — | Default delivery target for cron jobs |
