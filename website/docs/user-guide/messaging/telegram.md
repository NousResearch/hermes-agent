---
sidebar_position: 2
title: "Telegram"
description: "Set up Hermes Agent as a Telegram bot"
---

# Telegram Setup

Connect Hermes Agent to Telegram so you can chat from your phone, send voice memos, and receive scheduled task results.

## Setup Steps

1. **Create a bot:** Message [@BotFather](https://t.me/BotFather) on Telegram, use `/newbot`
2. **Get your user ID:** Message [@userinfobot](https://t.me/userinfobot) — it replies with your numeric ID
3. **Configure:** Run `hermes gateway setup` and select Telegram, or add to `~/.hermes/.env` manually:

```bash
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_ALLOWED_USERS=YOUR_USER_ID    # Comma-separated for multiple users
```

4. **Start the gateway:**

```bash
hermes gateway
```

## Optional: Home Channel

Set a home channel for cron job delivery:

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"
```

Or use the `/sethome` command in any Telegram chat to set it dynamically.

## Voice Messages

Voice messages sent on Telegram are automatically transcribed using OpenAI's Whisper API and injected as text into the conversation. Requires `VOICE_TOOLS_OPENAI_KEY` in `~/.hermes/.env`.

### Voice Bubbles (TTS)

When the agent generates audio via text-to-speech, it's delivered as native Telegram voice bubbles (the round, inline-playable kind).

- **OpenAI and ElevenLabs** produce Opus natively — no extra setup needed
- **Edge TTS** (the default free provider) outputs MP3 and needs **ffmpeg** to convert to Opus:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

Without ffmpeg, Edge TTS audio is sent as a regular audio file (still playable, but rectangular player instead of voice bubble).

## Exec Approval

When the agent tries to run a potentially dangerous command, it asks you for approval in the chat:

> ⚠️ This command is potentially dangerous (recursive delete). Reply "yes" to approve.

Reply "yes"/"y" to approve or "no"/"n" to deny.

## Security

:::warning
Always set `TELEGRAM_ALLOWED_USERS` to restrict who can use the bot. Without it, the gateway denies all users by default.
:::

You can also use [DM pairing](/user-guide/messaging#dm-pairing-alternative-to-allowlists) for a more dynamic approach.
