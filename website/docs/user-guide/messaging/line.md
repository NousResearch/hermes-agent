# LINE

Connect Hermes to [LINE](https://line.me/) via the LINE Messaging API.

## Prerequisites

- A LINE account
- A [LINE Developers Console](https://developers.line.biz/console/) account

## Setup

### 1. Create a LINE Bot

1. Go to [LINE Developers Console](https://developers.line.biz/console/)
2. Create a new **Provider** (or use an existing one)
3. Create a new **Messaging API** channel under the provider

### 2. Get Credentials

In the channel settings:

- **Basic settings** tab: copy the **Channel secret**
- **Messaging API** tab: issue a **Channel access token** (long-lived)

### 3. Configure Hermes

Using the setup wizard:

```bash
hermes setup
# Select "LINE" from the platform list
```

Or manually set environment variables in `~/.hermes/.env`:

```bash
LINE_CHANNEL_ACCESS_TOKEN=your_channel_access_token
LINE_CHANNEL_SECRET=your_channel_secret
LINE_ALLOWED_USERS=U1234567890abcdef   # comma-separated LINE user IDs
LINE_HOME_CHANNEL=U1234567890abcdef     # for cron/notification delivery
```

### 4. Configure Webhook

Set the webhook URL in LINE Developers Console → Messaging API tab:

```
https://your-server.com:8443/line/webhook
```

- Enable **Use webhook** toggle
- Disable **Auto-reply messages** in the LINE Official Account Manager
- Disable **Greeting messages** in the LINE Official Account Manager

### 5. Start Hermes

```bash
hermes --gateway
```

## Configuration Options

| Environment Variable | Description | Default |
|---|---|---|
| `LINE_CHANNEL_ACCESS_TOKEN` | Channel access token (required) | - |
| `LINE_CHANNEL_SECRET` | Channel secret (required) | - |
| `LINE_ALLOWED_USERS` | Comma-separated allowed user IDs | - |
| `LINE_HOME_CHANNEL` | Default delivery destination | - |
| `LINE_HOME_CHANNEL_NAME` | Display name for home channel | `Home` |
| `LINE_WEBHOOK_HOST` | Webhook server bind address | `0.0.0.0` |
| `LINE_WEBHOOK_PORT` | Webhook server port | `8443` |
| `LINE_WEBHOOK_PATH` | Webhook URL path | `/line/webhook` |
| `LINE_ALLOW_ALL_USERS` | Allow all users (skip allowlist) | `false` |

## Features

- Text messaging (send and receive)
- Image, video, audio, and file attachments
- Sticker recognition (keywords extracted as text)
- Location messages
- Group and multi-person chat support
- Typing indicators (loading animation)
- DM pairing for user authorization

## Message Limits

- Maximum message length: 5,000 characters
- Maximum 5 messages per API request
- Reply tokens expire after ~30 seconds

## Troubleshooting

### Bot doesn't respond

1. Verify webhook URL is accessible from the internet
2. Check that **Use webhook** is enabled in LINE Developers Console
3. Confirm `LINE_CHANNEL_ACCESS_TOKEN` and `LINE_CHANNEL_SECRET` are correct
4. Check Hermes logs for webhook signature verification errors

### "Auto-reply messages" interfere

Disable auto-reply and greeting messages in the [LINE Official Account Manager](https://manager.line.biz/):
Settings → Response settings → set all to disabled.
