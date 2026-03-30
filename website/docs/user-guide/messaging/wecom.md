---
sidebar_position: 14
title: "WeCom (Enterprise WeChat)"
description: "Connect Hermes Agent to WeCom via the AI Bot WebSocket gateway"
---

# WeCom (Enterprise WeChat)

Connect Hermes to [WeCom](https://work.weixin.qq.com/) (企业微信), Tencent's enterprise messaging platform. The adapter uses WeCom's AI Bot WebSocket gateway for real-time bidirectional communication — no public endpoint or webhook needed.

## Prerequisites

- A WeCom organization account
- An AI Bot created in the WeCom Admin Console
- The Bot ID and Secret from the bot's credentials page

## Setup

### 1. Create an AI Bot

1. Log in to the [WeCom Admin Console](https://work.weixin.qq.com/wework_admin/frame)
2. Navigate to **Applications** → **Create Application** → **AI Bot**
3. Configure the bot name and description
4. Copy the **Bot ID** and **Secret** from the credentials page

### 2. Configure Hermes

Run the interactive setup:

```bash
hermes gateway setup
```

Select **WeCom** and enter your Bot ID and Secret.

Or set environment variables in `~/.hermes/.env`:

```bash
WECOM_BOT_ID=your-bot-id
WECOM_SECRET=your-secret

# Optional: restrict access
WECOM_ALLOWED_USERS=user_id_1,user_id_2

# Optional: home channel for cron/notifications
WECOM_HOME_CHANNEL=chat_id
```

### 3. Start the gateway

```bash
hermes gateway start
```

## Features

- **WebSocket transport** — persistent connection, no public endpoint needed
- **DM and group messaging** — configurable access policies
- **Media support** — images, files, voice, video upload and download
- **AES-encrypted media** — automatic decryption for inbound attachments
- **Quote context** — preserves reply threading
- **Markdown rendering** — rich text responses
- **Auto-reconnect** — exponential backoff on connection drops

## Configuration Options

Set these in `config.yaml` under `platforms.wecom.extra`:

| Key | Default | Description |
|-----|---------|-------------|
| `bot_id` | — | WeCom AI Bot ID (required) |
| `secret` | — | WeCom AI Bot Secret (required) |
| `websocket_url` | `wss://openws.work.weixin.qq.com` | WebSocket gateway URL |
| `dm_policy` | `open` | DM access: `open`, `allowlist`, `disabled`, `pairing` |
| `group_policy` | `open` | Group access: `open`, `allowlist`, `disabled` |
| `allow_from` | `[]` | User IDs allowed for DMs (when dm_policy=allowlist) |
| `group_allow_from` | `[]` | Group IDs allowed (when group_policy=allowlist) |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "WECOM_BOT_ID and WECOM_SECRET are required" | Set both env vars or configure in setup wizard |
| "invalid secret (errcode=40013)" | Verify the secret matches your bot's credentials |
| "Timed out waiting for subscribe acknowledgement" | Check network connectivity to `openws.work.weixin.qq.com` |
| Bot doesn't respond in groups | Check `group_policy` setting and group allowlist |
