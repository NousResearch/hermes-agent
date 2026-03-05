---
sidebar_position: 3
title: "Discord"
description: "Set up Hermes Agent as a Discord bot"
---

# Discord Setup

Connect Hermes Agent to Discord to chat with it in DMs or server channels.

## Setup Steps

1. **Create a bot:** Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. **Enable intents:** Bot → Privileged Gateway Intents → enable **Message Content Intent**
3. **Get your user ID:** Enable Developer Mode in Discord settings, right-click your name → Copy ID
4. **Invite to your server:** OAuth2 → URL Generator → scopes: `bot`, `applications.commands` → permissions: Send Messages, Read Message History, Attach Files
5. **Configure:** Run `hermes gateway setup` and select Discord, or add to `~/.hermes/.env` manually:

```bash
DISCORD_BOT_TOKEN=MTIz...
DISCORD_ALLOWED_USERS=YOUR_USER_ID
```

6. **Start the gateway:**

```bash
hermes gateway
```

## Optional: Home Channel

Set a default channel for cron job delivery:

```bash
DISCORD_HOME_CHANNEL=123456789012345678
DISCORD_HOME_CHANNEL_NAME="#bot-updates"
```

Or use `/sethome` in any Discord channel.

## Required Bot Permissions

When generating the invite URL, make sure to include:

- **Send Messages** — bot needs to reply
- **Read Message History** — for context
- **Attach Files** — for audio, images, and file outputs

## Voice Messages

Voice messages on Discord are automatically transcribed (requires `VOICE_TOOLS_OPENAI_KEY`). TTS audio is sent as MP3 file attachments.

## Security

:::warning
Always set `DISCORD_ALLOWED_USERS` to restrict who can use the bot. Without it, the gateway denies all users by default.
:::
