---
sidebar_position: 4
title: "Slack"
description: "Set up Hermes Agent as a Slack bot using Socket Mode"
---

# Slack Setup

Connect Hermes Agent to Slack as a bot using Socket Mode. Socket Mode uses WebSockets instead of
public HTTP endpoints, so your Hermes instance doesn't need to be publicly accessible — it works
behind firewalls, on your laptop, or on a private server.

:::warning Classic Slack Apps Deprecated
Classic Slack apps (using RTM API) were **fully deprecated in March 2025**. Hermes uses the modern
Bolt SDK with Socket Mode. If you have an old classic app, you must create a new one following
the steps below.
:::

## Overview

| Component | Value |
|-----------|-------|
| **Library** | `slack-bolt` / `slack_sdk` for Python (Socket Mode) |
| **Connection** | WebSocket — no public URL required |
| **Auth tokens needed** | Bot Token (`xoxb-`) + App-Level Token (`xapp-`) |
| **User identification** | Slack Member IDs (e.g., `U01ABC2DEF3`) |

---

## Step 1: Create a Slack App

The fastest path is to paste a manifest Hermes generates for you. It
declares every built-in slash command (`/btw`, `/stop`, `/model`, …),
every required OAuth scope, every event subscription, and enables Socket
Mode — all at once.

### Option A: From a Hermes-generated manifest (recommended)

1. Generate the manifest:
   ```bash
   hermes slack manifest --write
   ```
   This writes `~/.hermes/slack-manifest.json` and prints paste-in
   instructions.
2. Go to [https://api.slack.com/apps](https://api.slack.com/apps) →
   **Create New App** → **From an app manifest**
3. Pick your workspace, paste the JSON contents, review, click **Next**
   → **Create**
4. Skip ahead to **Step 6: Install App to Workspace**. The manifest
   handled scopes, events, and slash commands for you.

### Option B: From scratch (manual)

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App**
3. Choose **From scratch**
4. Enter an app name (e.g., "Hermes Agent") and select your workspace
5. Click **Create App**

You'll land on the app's **Basic Information** page. Continue with
Steps 2–6 below.

---

## Step 2: Configure Bot Token Scopes

Navigate to **Features → OAuth & Permissions** in the sidebar. Scroll to **Scopes → Bot Token Scopes** and add the following:

| Scope | Purpose |
|-------|---------|
| `chat:write` | Send messages as the bot |
| `app_mentions:read` | Detect when @mentioned in channels |
| `channels:history` | Read messages in public channels the bot is in |
| `channels:read` | List and get info about public channels |
| `groups:history` | Read messages in private channels the bot is invited to |
| `im:history` | Read direct message history |
| `im:read` | View basic DM info |
| `im:write` | Open and manage DMs |
| `users:read` | Look up user information |
| `files:read` | Read and download attached files, including voice notes/audio |
| `files:write` | Upload files (images, audio, documents) |

:::caution Missing scopes = missing features
Without `channels:history` and `groups:history`, the bot **will not receive messages in channels** —
it will only work in DMs. Without `files:read`, Hermes can chat but **cannot reliably read user-uploaded attachments**.
These are the most commonly missed scopes.
:::

**Optional scopes:**

| Scope | Purpose |
|-------|---------|
| `groups:read` | List and get info about private channels |

---

## Step 3: Enable Socket Mode

Socket Mode lets the bot connect via WebSocket instead of requiring a public URL.

1. In the sidebar, go to **Settings → Socket Mode**
2. Toggle **Enable Socket Mode** to ON
3. You'll be prompted to create an **App-Level Token**:
   - Name it something like `hermes-socket` (the name doesn't matter)
   - Add the **`connections:write`** scope
   - Click **Generate**
4. **Copy the token** — it starts with `xapp-`. This is your `SLACK_APP_TOKEN`

:::tip
You can always find or regenerate app-level tokens under **Settings → Basic Information → App-Level Tokens**.
:::

---

## Step 4: Subscribe to Events

This step is critical — it controls what messages the bot can see.


1. In the sidebar, go to **Features → Event Subscriptions**
2. Toggle **Enable Events** to ON
3. Expand **Subscribe to bot events** and add:

| Event | Required? | Purpose |
|-------|-----------|---------|
| `message.im` | **Yes** | Bot receives direct messages |
| `message.channels` | **Yes** | Bot receives messages in **public** channels it's added to |
| `message.groups` | **Recommended** | Bot receives messages in **private** channels it's invited to |
| `app_mention` | **Yes** | Prevents Bolt SDK errors when bot is @mentioned |

4. Click **Save Changes** at the bottom of the page

:::danger Missing event subscriptions is the #1 setup issue
If the bot works in DMs but **not in channels**, you almost certainly forgot to add
`message.channels` (for public channels) and/or `message.groups` (for private channels).
Without these events, Slack simply never delivers channel messages to the bot.
:::


---

## Step 5: Enable the Messages Tab

This step enables direct messages to the bot. Without it, users see **"Sending messages to this app has been turned off"** when trying to DM the bot.

1. In the sidebar, go to **Features → App Home**
2. Scroll to **Show Tabs**
3. Toggle **Messages Tab** to ON
4. Check **"Allow users to send Slash commands and messages from the messages tab"**

:::danger Without this step, DMs are completely blocked
Even with all the correct scopes and event subscriptions, Slack will not allow users to send direct messages to the bot unless the Messages Tab is enabled. This is a Slack platform requirement, not a Hermes configuration issue.
:::

---

## Step 6: Install App to Workspace

1. In the sidebar, go to **Settings → Install App**
2. Click **Install to Workspace**
3. Review the permissions and click **Allow**
4. After authorization, you'll see a **Bot User OAuth Token** starting with `xoxb-`
5. **Copy this token** — this is your `SLACK_BOT_TOKEN`

:::tip
If you change scopes or event subscriptions later, you **must reinstall the app** for the changes
to take effect. The Install App page will show a banner prompting you to do so.
:::

---

## Step 7: Find User IDs for the Allowlist

Hermes uses Slack **Member IDs** (not usernames or display names) for the allowlist.

To find a Member ID:

1. In Slack, click on the user's name or avatar
2. Click **View full profile**
3. Click the **⋮** (more) button
4. Select **Copy member ID**

Member IDs look like `U01ABC2DEF3`. You need your own Member ID at minimum.

---

## Step 8: Configure Hermes

Add the following to your `~/.hermes/.env` file:

```bash
# Required
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SLACK_ALLOWED_USERS=U01ABC2DEF3              # Comma-separated Member IDs

# Optional
SLACK_HOME_CHANNEL=C01234567890              # Default channel for cron/scheduled messages
SLACK_HOME_CHANNEL_NAME=general              # Human-readable name for the home channel (optional)
```

Or run the interactive setup:

```bash
hermes gateway setup    # Select Slack when prompted
```

Then start the gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

---

## Step 9: Invite the Bot to Channels

After starting the gateway, you need to **invite the bot** to any channel where you want it to respond:

```
/invite @Hermes Agent
```

The bot will **not** automatically join channels. You must invite it to each channel individually.

---

## Slash Commands

Every Hermes command (`/btw`, `/stop`, `/new`, `/model`, `/help`, ...)
is a native Slack slash command — exactly the way they work on Telegram
and Discord. Type `/` in Slack and the autocomplete picker lists every
Hermes command with its description.

Under the hood: Hermes ships with a generated Slack app manifest (see
Step 1, Option A) that declares every command in
[`COMMAND_REGISTRY`](https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/commands.py)
as a slash command. In Socket Mode, Slack routes the command event
through the WebSocket regardless of the manifest's `url` field.

### Refreshing slash commands after updates

When Hermes adds new commands (e.g. after `hermes update`), regenerate
the manifest and update your Slack app:

```bash
hermes slack manifest --write
```

Then in Slack:
1. Open [https://api.slack.com/apps](https://api.slack.com/apps) →
   your Hermes app
2. **Features → App Manifest → Edit**
3. Paste the new contents of `~/.hermes/slack-manifest.json`
4. **Save**. Slack will prompt to reinstall the app if scopes or slash
   commands changed.

### Legacy `/hermes <subcommand>` still works

For backward compatibility with older manifests, you can still type
`/hermes btw run the tests` — Hermes routes it the same way as `/btw
run the tests`. Free-form questions also work: `/hermes what's the
weather?` is treated as a regular message.

### Using commands inside threads (the `!cmd` prefix)

Slack itself blocks native slash commands inside thread replies — try
`/queue` in a thread and Slack responds with *"/queue is not supported
in threads. Sorry!"* There is no app-side setting that re-enables them;
Slack never delivers them to Hermes.

As a workaround, Hermes recognises a leading `!` as an alternate
command prefix that works in threads (and anywhere else). Type
`!queue`, `!stop`, `!model gpt-5.4`, etc. as a regular thread reply —
Hermes treats it identically to the slash form and replies in the same
thread.

Only the first token is checked against the known command list, so
casual messages like `!nice work` pass through to the agent unchanged.

### Advanced: emit only the slash-commands array

If you maintain your Slack manifest by hand and just want the slash
command list:

```bash
hermes slack manifest --slashes-only > /tmp/slashes.json
```

Paste that array into the `features.slash_commands` key of your
existing manifest.

---

## How the Bot Responds

Understanding how Hermes behaves in different contexts:

| Context | Behavior |
|---------|----------|
| **DMs** | Bot responds to every message — no @mention needed |
| **Channels** | Bot **only responds when @mentioned** (e.g., `@Hermes Agent what time is it?`). In channels, Hermes replies in a thread attached to that message. |
| **Threads** | If you @mention Hermes inside an existing thread, it replies in that same thread. Once the bot has an active session in a thread, **subsequent replies in that thread do not require @mention** — the bot follows the conversation naturally. |

:::tip
In channels, always @mention the bot to start a conversation. Once the bot is active in a thread, you can reply in that thread without mentioning it. Outside of threads, messages without @mention are ignored to prevent noise in busy channels.
:::

---

## Configuration Options

Beyond the required environment variables from Step 8, you can customize Slack bot behavior through `~/.hermes/config.yaml`.

### Thread & Reply Behavior

```yaml
platforms:
  slack:
    # Controls how multi-part responses are threaded
    # "off"   — never thread replies to the original message
    # "first" — first chunk threads to user's message (default)
    # "all"   — all chunks thread to user's message
    reply_to_mode: "first"

    extra:
      # Whether to reply in a thread (default: true).
      # When false, channel messages get direct channel replies instead
      # of threads. Messages inside existing threads still reply in-thread.
      reply_in_thread: true

      # Also post thread replies to the main channel
      # (Slack's "Also send to channel" feature).
      # Only the first chunk of the first reply is broadcast.
      reply_broadcast: false
```

| Key | Default | Description |
|-----|---------|-------------|
| `platforms.slack.reply_to_mode` | `"first"` | Threading mode for multi-part messages: `"off"`, `"first"`, or `"all"` |
| `platforms.slack.extra.reply_in_thread` | `true` | When `false`, channel messages get direct replies instead of threads. Messages inside existing threads still reply in-thread. |
| `platforms.slack.extra.reply_broadcast` | `false` | When `true`, thread replies are also posted to the main channel. Only the first chunk is broadcast. |

### Session Isolation

```yaml
# Global setting — applies to Slack and all other platforms
group_sessions_per_user: true
```

When `true` (the default), each user in a shared channel gets their own isolated conversation session. Two people talking to Hermes in `#general` will have separate histories and contexts.

Set to `false` if you want a collaborative mode where the entire channel shares one conversation session. Be aware this means users share context growth and token costs, and one user's `/reset` clears the session for everyone.

### Mention & Trigger Behavior

```yaml
slack:
  # Require @mention in channels (this is the default behavior;
  # the Slack adapter enforces @mention gating in channels regardless,
  # but you can set this explicitly for consistency with other platforms)
  require_mention: true

  # Prevent thread auto-engagement: only reply to channel messages that
  # contain an explicit @mention. With this OFF (default), Slack can
  # "auto-engage" — remembering past mentions in a thread and following
  # up on bot-message replies, and resuming active sessions without a
  # fresh mention. With strict_mention ON, every new channel message
  # must @mention the bot before Hermes will respond.
  strict_mention: false

  # Custom mention patterns that trigger the bot
  # (in addition to the default @mention detection)
  mention_patterns:
    - "hey hermes"
    - "hermes,"

  # Text prepended to every outgoing message
  reply_prefix: ""
```

:::tip When to use `strict_mention`
Set this to `true` in busy workspaces where Slack's default "the bot remembers this thread" behavior surprises users — for example, a long tech-support thread where the bot helped at the start and you'd rather it stay silent unless explicitly pinged again. DMs and active interactive sessions are unaffected.
:::

:::info
Slack supports both patterns: `@mention` required to start a conversation by default, but you can opt specific channels out via `SLACK_FREE_RESPONSE_CHANNELS` (comma-separated channel IDs) or `slack.free_response_channels` in `config.yaml`. Once the bot has an active session in a thread, subsequent thread replies do not require a mention. In DMs the bot always responds without needing a mention.
:::

### Unauthorized User Handling

```yaml
slack:
  # What happens when an unauthorized user (not in SLACK_ALLOWED_USERS) DMs the bot
  # "pair"   — prompt them for a pairing code (default)
  # "ignore" — silently drop the message
  unauthorized_dm_behavior: "pair"
```

You can also set this globally for all platforms:

```yaml
unauthorized_dm_behavior: "pair"
```

The platform-specific setting under `slack:` takes precedence over the global setting.

### Voice Transcription

```yaml
# Global setting — enable/disable automatic transcription of incoming voice messages
stt_enabled: true
```

When `true` (the default), incoming audio messages are automatically transcribed using the configured STT provider before being processed by the agent.

### Full Example

```yaml
# Global gateway settings
group_sessions_per_user: true
unauthorized_dm_behavior: "pair"
stt_enabled: true

# Slack-specific settings
slack:
  require_mention: true
  unauthorized_dm_behavior: "pair"

# Platform config
platforms:
  slack:
    reply_to_mode: "first"
    extra:
      reply_in_thread: true
      reply_broadcast: false
```

---


## Home Channel

Set `SLACK_HOME_CHANNEL` to a channel ID where Hermes will deliver scheduled messages,
cron job results, and other proactive notifications. To find a channel ID:

1. Right-click the channel name in Slack
2. Click **View channel details**
3. Scroll to the bottom — the Channel ID is shown there

```bash
SLACK_HOME_CHANNEL=C01234567890
```

Make sure the bot has been **invited to the channel** (`/invite @Hermes Agent`).

---

## Multi-Workspace Support

Hermes can connect to **multiple Slack workspaces** simultaneously using a single gateway instance. Each workspace is authenticated independently with its own bot user ID.

### Configuration

Provide multiple bot tokens as a **comma-separated list** in `SLACK_BOT_TOKEN`:

```bash
# Multiple bot tokens — one per workspace
SLACK_BOT_TOKEN=xoxb-workspace1-token,xoxb-workspace2-token,xoxb-workspace3-token

# A single app-level token is still used for Socket Mode
SLACK_APP_TOKEN=xapp-your-app-token
```

Or in `~/.hermes/config.yaml`:

```yaml
platforms:
  slack:
    token: "xoxb-workspace1-token,xoxb-workspace2-token"
```

### OAuth Token File

In addition to tokens in the environment or config, Hermes also loads tokens from an **OAuth token file** at:

```
~/.hermes/slack_tokens.json
```

This file is a JSON object mapping team IDs to token entries:

```json
{
  "T01ABC2DEF3": {
    "token": "xoxb-workspace-token-here",
    "team_name": "My Workspace"
  }
}
```

Tokens from this file are merged with any tokens specified via `SLACK_BOT_TOKEN`. Duplicate tokens are automatically deduplicated.

### How it works

- The **first token** in the list is the primary token, used for the Socket Mode connection (AsyncApp).
- Each token is authenticated via `auth.test` on startup. The gateway maps each `team_id` to its own `WebClient` and `bot_user_id`.
- When a message arrives, Hermes uses the correct workspace-specific client to respond.
- The primary `bot_user_id` (from the first token) is used for backward compatibility with features that expect a single bot identity.

---

## Voice Messages

Hermes supports voice on Slack:

- **Incoming:** Voice/audio messages are automatically transcribed using the configured STT provider: local `faster-whisper`, Groq Whisper (`GROQ_API_KEY`), or OpenAI Whisper (`VOICE_TOOLS_OPENAI_KEY`)
- **Outgoing:** TTS responses are sent as audio file attachments

---

## Per-Channel Prompts

Assign ephemeral system prompts to specific Slack channels. The prompt is injected at runtime on every turn — never persisted to transcript history — so changes take effect immediately.

```yaml
slack:
  channel_prompts:
    "C01RESEARCH": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "C02ENGINEERING": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

Keys are Slack channel IDs (find them via channel details → "About" → scroll to bottom). All messages in the matching channel get the prompt injected as an ephemeral system instruction.

## Per-Channel Skill Bindings

Auto-load a skill whenever a new session starts in a specific channel or DM. Unlike per-channel prompts (which are injected on every turn), skill bindings inject the skill content as a user message at **session start** — it becomes part of the conversation history and does not need to be reloaded on subsequent turns.

This is ideal for DMs or channels with a dedicated purpose (flashcards, a domain-specific Q&A bot, a support triage channel, etc.) where you don't want the model's own skill selector to decide whether to load on every short reply.

```yaml
slack:
  channel_skill_bindings:
    # DM channel — always runs in "german-flashcards" mode
    - id: "D0ATH9TQ0G6"
      skills:
        - german-flashcards
    # Research channel — preload multiple skills in order
    - id: "C01RESEARCH"
      skills:
        - arxiv
        - writing-plans
    # Short form: single skill as a string
    - id: "C02SUPPORT"
      skill: hubspot-on-demand
```

Notes:
- The binding matches by channel ID. For threaded messages in a bound channel, the thread inherits the parent channel's binding.
- The skill is loaded only at session start (new session or after auto-reset). If you change the binding, run `/new` or wait for the session to auto-reset for it to take effect.
- Combine with `channel_prompts` for per-channel tone/constraints on top of the skill's instructions.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Bot doesn't respond to DMs | Verify `message.im` is in your event subscriptions and the app is reinstalled |
| Bot works in DMs but not in channels | **Most common issue.** Add `message.channels` and `message.groups` to event subscriptions, reinstall the app, and invite the bot to the channel with `/invite @Hermes Agent` |
| Bot doesn't respond to @mentions in channels | 1) Check `message.channels` event is subscribed. 2) Bot must be invited to the channel. 3) Ensure `channels:history` scope is added. 4) Reinstall the app after scope/event changes |
| Bot ignores messages in private channels | Add both the `message.groups` event subscription and `groups:history` scope, then reinstall the app and `/invite` the bot |
| "Sending messages to this app has been turned off" in DMs | Enable the **Messages Tab** in App Home settings (see Step 5) |
| "not_authed" or "invalid_auth" errors | Regenerate your Bot Token and App Token, update `.env` |
| Bot responds but can't post in a channel | Invite the bot to the channel with `/invite @Hermes Agent` |
| Bot can chat but can't read uploaded images/files | Add `files:read`, then **reinstall** the app. Hermes now surfaces attachment access diagnostics in-chat when Slack returns scope/auth/permission failures. |
| `missing_scope` error | Add the required scope in OAuth & Permissions, then **reinstall** the app |
| Socket disconnects frequently | Check your network; Bolt auto-reconnects but unstable connections cause lag |
| Changed scopes/events but nothing changed | You **must reinstall** the app to your workspace after any scope or event subscription change |

### Quick Checklist

If the bot isn't working in channels, verify **all** of the following:

1. ✅ `message.channels` event is subscribed (for public channels)
2. ✅ `message.groups` event is subscribed (for private channels)
3. ✅ `app_mention` event is subscribed
4. ✅ `channels:history` scope is added (for public channels)
5. ✅ `groups:history` scope is added (for private channels)
6. ✅ App was **reinstalled** after adding scopes/events
7. ✅ Bot was **invited** to the channel (`/invite @Hermes Agent`)
8. ✅ You are **@mentioning** the bot in your message

---

## Security

:::warning
**Always set `SLACK_ALLOWED_USERS`** with the Member IDs of authorized users. Without this setting,
the gateway will **deny all messages** by default as a safety measure. Never share your bot tokens —
treat them like passwords.
:::

- Tokens should be stored in `~/.hermes/.env` (file permissions `600`)
- Rotate tokens periodically via the Slack app settings
- Audit who has access to your Hermes config directory
- Socket Mode means no public endpoint is exposed — one less attack surface
