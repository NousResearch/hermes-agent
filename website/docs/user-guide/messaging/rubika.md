---
sidebar_position: 20
title: "Rubika"
description: "Set up Hermes Agent as a Rubika bot"
---

# Rubika Setup

Hermes Agent integrates with Rubika, connecting via Rubika's Bot API using
polling. Supports text, images, documents, chat/inline keypads (buttons),
and group chats with mention-gating.

## Step 1: Create a Bot via BotFather

1. Open Rubika and message `BotFather@`
2. Follow its prompts to create a bot and receive an API token
3. Keep this token secret — anyone with it can control your bot

## Step 2: Configure Hermes

### Option A: Interactive Setup

```bash
hermes gateway setup
```

Select **Rubika** when prompted; it asks for your bot token.

### Option B: Manual Configuration

Add to `~/.hermes/.env`:

```bash
RUBIKA_BOT_TOKEN=your-token-here
RUBIKA_ALLOWED_USERS=123456   # comma-separated sender ids, or * for any
```

### Start the Gateway

```bash
hermes gateway
```

## Group Chats

By default the bot responds to every group message it can see. To require an
explicit `@mention` first:

```bash
RUBIKA_REQUIRE_MENTION=true
RUBIKA_BOT_USERNAME=your_bot_username
```

## Home Channel (cron delivery)

```bash
RUBIKA_HOME_CHANNEL=your-chat-id
```

## Buttons

`send()`'s `metadata` dict accepts `chat_keypad` (persistent) and
`inline_keypad` (per-message) as lists of `(button_id, button_text)` pairs.
v1 only emits Rubika's `"Simple"` button type — payment, calendar, location,
and media-picker button types are not yet supported.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Bot never responds | Check `RUBIKA_BOT_TOKEN` is set and `hermes gateway status` shows Rubika connected |
| Group messages ignored | `RUBIKA_REQUIRE_MENTION=true` needs a matching `@RUBIKA_BOT_USERNAME` prefix |
| Media send fails | Rubika's upload flow is two-step (request URL, then upload) — check network access to `botapi.rubika.ir` *and* the returned per-upload URL |
