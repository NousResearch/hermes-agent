---
sidebar_position: 9
title: "Zulip"
description: "Set up Hermes Agent as a Zulip bot"
---

# Zulip Setup

Hermes Agent integrates with Zulip as a bot, letting you chat with your AI assistant through direct messages or stream topics. Zulip is an open-source team chat platform with a unique **topic-based threading model** — every stream message belongs to a topic, and topics act as independent conversation threads. Hermes maps each topic to its own session, so conversations in different topics stay cleanly isolated. The bot connects via Zulip's REST API (v1) and long-polls the event queue for real-time messages, processes them through the Hermes Agent pipeline (including tool use, memory, and reasoning), and responds in the same topic.

No external Zulip library is required — the adapter uses `aiohttp`, which is already a Hermes dependency.

Before setup, here's the part most people want to know: how Hermes behaves once it's in your Zulip instance.

## How Hermes Behaves

| Context | Behavior |
|---------|----------|
| **Direct messages** | Hermes responds to every message. No `@mention` needed. Each DM thread has its own session. |
| **Stream topics** | Hermes responds when you `@**Bot Name**` mention it in the topic. Without a mention, Hermes stays silent. |
| **Free-response streams** | Streams listed in `ZULIP_FREE_RESPONSE_STREAMS` respond to every message, no mention required. |
| **Topics** | Each topic gets its own session. Context is isolated per `stream_id + topic` pair. |

:::tip
Zulip's topic model means you can have many parallel conversations with Hermes inside the same stream — each topic is a separate thread of context. This is particularly useful for keeping long-running project discussions organized.
:::

### Session Model in Zulip

Hermes uses these session keys:

- **DMs** → sorted list of user IDs (so a 1:1 DM is always the same session regardless of who sends first).
- **Streams** → `stream_id:topic` pair. Opening a new topic in the same stream starts a fresh session.

This maps naturally to Zulip's conversation model. Multi-party DMs (group PMs) share a single session across all participants.

## Step 1: Create a Bot Account

1. In Zulip, click your **avatar** (top-right) → **Personal Settings** → **Bots**.
2. Click **Add a new bot**.
3. Fill in the details:
   - **Bot type**: **Generic bot**
   - **Full name**: e.g. `Hermes`
   - **Username**: e.g. `hermes-bot` (becomes the bot's email address)
4. Click **Add**.
5. Zulip creates the bot and shows its **email** and **API key**. Copy both — you'll need them in Step 3.

:::warning
Anyone with the bot's API key has full control of the bot. Store it somewhere safe (a password manager, for example). Never commit it to Git or share it publicly.
:::

## Step 2: Subscribe the Bot to Streams

The bot only receives messages from streams it is subscribed to:

1. Go to the stream you want Hermes to respond in.
2. Click the stream's settings (gear icon) → **Subscribers**.
3. Add your bot's email (e.g. `hermes-bot@zulip.example.com`).

For DMs, no subscription is needed — anyone can DM the bot once it exists.

## Step 3: Find Your Zulip User ID

Hermes uses your Zulip User ID to control who can interact with the bot:

1. Click your **avatar** (top-right) → **View your profile**.
2. Click on the three dots → **Copy link to profile**, or look at the URL — your user ID appears as `#narrow/dm/123-...` in links.

Alternatively, use the API:

```bash
curl -u your-email@example.com:your-api-key \
  https://zulip.example.com/api/v1/users/me
```

The response includes `"user_id": 123` — that number is your user ID.

## Step 4: Configure Hermes Agent

### Option A: Interactive Setup (Recommended)

Run the guided setup command:

```bash
hermes gateway setup
```

Select **Zulip** when prompted, then paste your server URL, bot email, API key, and user ID when asked.

### Option B: Manual Configuration

Add the following to your `~/.hermes/.env` file:

```bash
# Required
ZULIP_SITE=https://zulip.example.com
ZULIP_EMAIL=hermes-bot@zulip.example.com
ZULIP_API_KEY=***
ZULIP_ALLOWED_USERS=123

# Multiple allowed users (comma-separated)
# ZULIP_ALLOWED_USERS=123,456,789

# Optional: respond to all stream messages, not just mentions (default: true = require mention)
# ZULIP_REQUIRE_MENTION=false

# Optional: streams where bot responds without @mention (comma-separated stream IDs)
# ZULIP_FREE_RESPONSE_STREAMS=42,57

# Optional: home stream + topic for cron / notification delivery
# ZULIP_HOME_STREAM=42
# ZULIP_HOME_TOPIC=notifications
```

### Start the Gateway

Once configured, start the Zulip gateway:

```bash
hermes gateway
```

The bot should connect within a few seconds. Test it by sending a DM or by `@**Hermes**`-mentioning it in a subscribed stream.

:::tip
You can run `hermes gateway` in the background or as a systemd service for persistent operation. See the deployment docs for details.
:::

## Home Channel

Hermes delivers proactive messages (cron job output, reminders, notifications) to a designated "home channel". In Zulip, a home channel is a **stream + topic** pair:

### Using the Slash Command

In any topic where the bot is present, type `/sethome`. That `stream_id:topic` becomes the home channel.

### Manual Configuration

Add this to your `~/.hermes/.env`:

```bash
ZULIP_HOME_STREAM=42
ZULIP_HOME_TOPIC=notifications
```

To find a stream ID: click the stream name → settings (gear icon); the stream ID appears in the URL and stream details.

## Mention Behavior

By default, the bot only responds in streams when `@mentioned`. You can change this:

| Variable | Default | Description |
|----------|---------|-------------|
| `ZULIP_REQUIRE_MENTION` | `true` | Set to `false` to respond to every message in subscribed streams (DMs always work). |
| `ZULIP_FREE_RESPONSE_STREAMS` | _(none)_ | Comma-separated stream IDs where the bot responds without `@mention`, even when `ZULIP_REQUIRE_MENTION` is `true`. |

Zulip mentions use the `@**Full Name**` syntax (double-asterisks around the bot's full name, not its username). Silent mentions — `@_**Full Name**_` — also count and are also stripped from the message text before Hermes processes it.

## Troubleshooting

### Bot is not responding to messages

**Cause**: The bot isn't subscribed to the stream, or your user ID isn't in `ZULIP_ALLOWED_USERS`, or you forgot to `@**Hermes**`-mention it.

**Fix**: Subscribe the bot to the stream (stream settings → Subscribers). Verify your user ID is in `ZULIP_ALLOWED_USERS`. Either `@mention` the bot or add the stream to `ZULIP_FREE_RESPONSE_STREAMS`. Restart the gateway.

### 401 / 403 errors during startup

**Cause**: The bot email or API key is invalid.

**Fix**: Check `ZULIP_EMAIL` and `ZULIP_API_KEY` in your `.env`. Test them with curl:

```bash
curl -u hermes-bot@zulip.example.com:YOUR_API_KEY \
  https://zulip.example.com/api/v1/users/me
```

If this returns the bot's profile, the credentials are valid. If it returns an authentication error, regenerate the API key from **Personal Settings → Bots**.

### `BAD_EVENT_QUEUE_ID` in logs

**Cause**: Normal — Zulip periodically expires idle event queues. The adapter re-registers automatically.

**Fix**: No action needed. If you see repeated re-registration failures, check network connectivity and that the server is reachable.

### Bot responds in the wrong topic

**Cause**: Each topic is its own session. If you create a new topic, Hermes starts a fresh conversation without any prior context.

**Fix**: This is intentional. To continue a conversation, stay in the same topic. To share context across topics, merge them using Zulip's topic-move feature (requires admin permissions).

## Per-Channel Prompts

Assign ephemeral system prompts to specific Zulip topics. The prompt is injected at runtime on every turn — never persisted to transcript history — so changes take effect immediately.

```yaml
zulip:
  channel_prompts:
    "42:research":
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "42:code-review":
      Code review mode. Be precise about edge cases and
      performance implications.
```

Keys are `stream_id:topic` strings. All messages in the matching topic get the prompt injected as an ephemeral system instruction.

## Security

:::warning
Always set `ZULIP_ALLOWED_USERS` to restrict who can interact with the bot. Without it, the gateway denies all users by default as a safety measure. Only add user IDs of people you trust — authorized users have full access to the agent's capabilities, including tool use and system access.
:::

For more information on securing your Hermes Agent deployment, see the [Security Guide](../security.md).

## Notes

- **Self-hosted friendly**: Works with any self-hosted Zulip instance or Zulip Cloud.
- **No extra dependencies**: The adapter uses `aiohttp` for HTTP and long-polling, which is already included with Hermes Agent.
- **One bot per agent**: Each Hermes instance runs as a single bot user. Multi-agent deployments should create separate bot accounts.
- **10,000-character message limit**: Messages longer than 10k chars are split into multiple posts automatically.

## Reference

- [Zulip REST API](https://zulip.com/api/rest)
- [Real-time events](https://zulip.com/api/real-time-events)
- [Register event queue](https://zulip.com/api/register-queue)
