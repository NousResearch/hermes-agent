---
sidebar_position: 4
title: "Slack"
description: "Set up Hermes Agent as a Slack bot"
---

# Slack Setup

Connect Hermes Agent to Slack using Socket Mode for real-time communication.

## Setup Steps

1. **Create an app:** Go to [Slack API](https://api.slack.com/apps), create a new app
2. **Enable Socket Mode:** In app settings → Socket Mode → Enable
3. **Get tokens:**
   - Bot Token (`xoxb-...`): OAuth & Permissions → Install to Workspace
   - App Token (`xapp-...`): Basic Information → App-Level Tokens → Generate (with `connections:write` scope)
4. **Configure:** Run `hermes gateway setup` and select Slack, or add to `~/.hermes/.env` manually:

```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_ALLOWED_USERS=U01234ABCDE    # Comma-separated Slack user IDs
```

5. **Start the gateway:**

```bash
hermes gateway
```

## Optional: Home Channel

Set a default channel for cron job delivery:

```bash
SLACK_HOME_CHANNEL=C01234567890
```

## Required Bot Scopes

Make sure your Slack app has these OAuth scopes:

- `chat:write` — Send messages
- `channels:history` — Read channel messages
- `im:history` — Read DM messages
- `files:write` — Upload files (audio, images)

## Voice Messages

Voice messages on Slack are automatically transcribed (requires `VOICE_TOOLS_OPENAI_KEY`). TTS audio is sent as file attachments.

## Security

:::warning
Always set `SLACK_ALLOWED_USERS` to restrict who can use the bot. Without it, the gateway denies all users by default.
:::
