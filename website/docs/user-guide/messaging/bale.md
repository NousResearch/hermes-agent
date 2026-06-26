---
sidebar_position: 10
title: "Bale"
description: "Set up Hermes Agent as a Bale messenger bot"
---

# Bale Setup

Hermes Agent integrates with Bale (بله) messenger as a full-featured conversational bot. Because Bale's Bot API is highly compatible with the Telegram Bot API, the Hermes Bale integration supports almost all the same features as Telegram, including text, media, inline keyboards, and slash commands.

## Step 1: Create a Bot via BotFather in Bale

Every Bale bot requires an API token issued by [@BotFather](https://ble.ir/botfather), Bale's official bot management tool.

1. Open Bale and search for **@BotFather**, or visit [ble.ir/botfather](https://ble.ir/botfather)
2. Send `/newbot`
3. Choose a **display name** (e.g., "Hermes Agent") — this can be anything
4. Choose a **username** — this must be unique and end in `bot` (e.g., `my_hermes_bot`)
5. BotFather replies with your **API token**. It looks like this:

```
123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

:::warning
Keep your bot token secret. Anyone with this token can control your bot.
:::

## Step 2: Configure Hermes

Add your Bale token to the Hermes environment or `.env` file:

```bash
BALE_BOT_TOKEN="your-bale-token"
BALE_ALLOWED_USERS="123456789,987654321" # Optional: restrict access
```

If you prefer to configure it via `config.yaml`, add the `bale` platform under `platforms`:

```yaml
platforms:
  bale:
    allow_from:
      - 123456789
```

## Step 3: Run the Gateway

Start the Hermes gateway. It will automatically load the Bale plugin and connect to `tapi.bale.ai`:

```bash
hermes gateway start
```

Your agent is now live on Bale! You can search for its username and send it a message.

## Differences from Telegram

Since Bale uses a separate infrastructure from Telegram, there are a few minor differences:
- Voice/Audio and Document uploads are handled through Bale's specific file servers.
- Some advanced Telegram features (like forum topics/threads or specific interactive markdown rendering) might behave slightly differently depending on the Bale client.
- The base API url for Bale is automatically handled by the plugin (`https://tapi.bale.ai/bot`).
