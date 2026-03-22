---
sidebar_position: 3
title: "Discord"
description: "Set up Hermes Agent as a Discord bot"
---

# Discord Setup

Hermes Agent integrates with Discord as a bot, letting you chat with your AI assistant through direct messages or server channels. The bot receives your messages, processes them through the Hermes Agent pipeline (including tool use, memory, and reasoning), and responds in real time. It supports text, voice messages, file attachments, and slash commands.

Before setup, here's the part most people want to know: how Hermes behaves once it's in your server.

## How Hermes Behaves

| Context | Behavior |
|---------|----------|
| **DMs** | Hermes responds to every message. No `@mention` needed. Each DM has its own session. |
| **Server channels** | By default, Hermes only responds when you `@mention` it. If you post in a channel without mentioning it, Hermes ignores the message. |
| **Free-response channels** | You can make specific channels mention-free with `DISCORD_FREE_RESPONSE_CHANNELS`, or disable mentions globally with `DISCORD_REQUIRE_MENTION=false`. |
| **Threads** | Hermes replies in the same thread. Mention rules still apply unless that thread or its parent channel is configured as free-response. Threads stay isolated from the parent channel for session history. |
| **Shared channels with multiple users** | By default, Hermes isolates session history per user inside the channel for safety and clarity. Two people talking in the same channel do not share one transcript unless you explicitly disable that. |

:::tip
If you want a normal bot-help channel where people can talk to Hermes without tagging it every time, add that channel to `DISCORD_FREE_RESPONSE_CHANNELS`.
:::

### Discord Gateway Model

Hermes on Discord is not a webhook that replies statelessly. It runs through the full messaging gateway, which means each incoming message goes through:

1. authorization (`DISCORD_ALLOWED_USERS`)
2. mention / free-response checks
3. session lookup
4. session transcript loading
5. normal Hermes agent execution, including tools, memory, and slash commands
6. response delivery back to Discord

That matters because behavior in a busy server depends on both Discord routing and Hermes session policy.

### Session Model in Discord

By default:

- each DM gets its own session
- each server thread gets its own session namespace
- each user in a shared channel gets their own session inside that channel

So if Alice and Bob both talk to Hermes in `#research`, Hermes treats those as separate conversations by default even though they are using the same visible Discord channel.

This is controlled by `config.yaml`:

```yaml
group_sessions_per_user: true
```

Set it to `false` only if you explicitly want one shared conversation for the entire room:

```yaml
group_sessions_per_user: false
```

Shared sessions can be useful for a collaborative room, but they also mean:

- users share context growth and token costs
- one person's long tool-heavy task can bloat everyone else's context
- one person's in-flight run can interrupt another person's follow-up in the same room

### Interrupts and Concurrency

Hermes tracks running agents by session key.

With the default `group_sessions_per_user: true`:

- Alice interrupting her own in-flight request only affects Alice's session in that channel
- Bob can keep talking in the same channel without inheriting Alice's history or interrupting Alice's run

With `group_sessions_per_user: false`:

- the whole room shares one running-agent slot for that channel/thread
- follow-up messages from different people can interrupt or queue behind each other

This guide walks you through the full setup process — from creating your bot on Discord's Developer Portal to sending your first message.

## Step 1: Create a Discord Application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and sign in with your Discord account.
2. Click **New Application** in the top-right corner.
3. Enter a name for your application (e.g., "Hermes Agent") and accept the Developer Terms of Service.
4. Click **Create**.

You'll land on the **General Information** page. Note the **Application ID** — you'll need it later to build the invite URL.

## Step 2: Create the Bot

1. In the left sidebar, click **Bot**.
2. Discord automatically creates a bot user for your application. You'll see the bot's username, which you can customize.
3. Under **Authorization Flow**:
   - Set **Public Bot** to **OFF** — this prevents other people from inviting your bot to their servers.
   - Leave **Require OAuth2 Code Grant** set to **OFF**.

:::tip
You can set a custom avatar and banner for your bot on this page. This is what users will see in Discord.
:::

## Step 3: Enable Privileged Gateway Intents

This is the most critical step in the entire setup. Without the correct intents enabled, your bot will connect to Discord but **will not be able to read message content**.

On the **Bot** page, scroll down to **Privileged Gateway Intents**. You'll see three toggles:

| Intent | Purpose | Required? |
|--------|---------|-----------| 
| **Presence Intent** | See user online/offline status | Optional |
| **Server Members Intent** | Access the member list, resolve usernames | **Required** |
| **Message Content Intent** | Read the text content of messages | **Required** |

**Enable both Server Members Intent and Message Content Intent** by toggling them **ON**.

- Without **Message Content Intent**, your bot receives message events but the message text is empty — the bot literally cannot see what you typed.
- Without **Server Members Intent**, the bot cannot resolve usernames for the allowed users list and may fail to identify who is messaging it.

:::warning[This is the #1 reason Discord bots don't work]
If your bot is online but never responds to messages, the **Message Content Intent** is almost certainly disabled. Go back to the [Developer Portal](https://discord.com/developers/applications), select your application → Bot → Privileged Gateway Intents, and make sure **Message Content Intent** is toggled ON. Click **Save Changes**.
:::

**Regarding server count:**
- If your bot is in **fewer than 100 servers**, you can simply toggle intents on and off freely.
- If your bot is in **100 or more servers**, Discord requires you to submit a verification application to use privileged intents. For personal use, this is not a concern.

Click **Save Changes** at the bottom of the page.

## Step 4: Get the Bot Token

The bot token is the credential Hermes Agent uses to log in as your bot. Still on the **Bot** page:

1. Under the **Token** section, click **Reset Token**.
2. If you have two-factor authentication enabled on your Discord account, enter your 2FA code.
3. Discord will display your new token. **Copy it immediately.**

:::warning[Token shown only once]
The token is only displayed once. If you lose it, you'll need to reset it and generate a new one. Never share your token publicly or commit it to Git — anyone with this token has full control of your bot.
:::

Store the token somewhere safe (a password manager, for example). You'll need it in Step 8.

## Step 5: Generate the Invite URL

You need an OAuth2 URL to invite the bot to your server. There are two ways to do this:

### Option A: Using the Installation Tab (Recommended)

1. In the left sidebar, click **Installation**.
2. Under **Installation Contexts**, enable **Guild Install**.
3. For **Install Link**, select **Discord Provided Link**.
4. Under **Default Install Settings** for Guild Install:
   - **Scopes**: select `bot` and `applications.commands`
   - **Permissions**: select the permissions listed below.

### Option B: Manual URL

You can construct the invite URL directly using this format:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=274878286912
```

Replace `YOUR_APP_ID` with the Application ID from Step 1.

### Required Permissions

These are the minimum permissions your bot needs:

- **View Channels** — see the channels it has access to
- **Send Messages** — respond to your messages
- **Embed Links** — format rich responses
- **Attach Files** — send images, audio, and file outputs
- **Read Message History** — maintain conversation context

### Recommended Additional Permissions

- **Send Messages in Threads** — respond in thread conversations
- **Add Reactions** — react to messages for acknowledgment

### Permission Integers

| Level | Permissions Integer | What's Included |
|-------|-------------------|-----------------|
| Minimal | `117760` | View Channels, Send Messages, Read Message History, Attach Files |
| Recommended | `274878286912` | All of the above plus Embed Links, Send Messages in Threads, Add Reactions |

## Step 6: Invite to Your Server

1. Open the invite URL in your browser (from the Installation tab or the manual URL you constructed).
2. In the **Add to Server** dropdown, select your server.
3. Click **Continue**, then **Authorize**.
4. Complete the CAPTCHA if prompted.

:::info
You need the **Manage Server** permission on the Discord server to invite a bot. If you don't see your server in the dropdown, ask a server admin to use the invite link instead.
:::

After authorizing, the bot will appear in your server's member list (it will show as offline until you start the Hermes gateway).

## Step 7: Find Your Discord User ID

Hermes Agent uses your Discord User ID to control who can interact with the bot. To find it:

1. Open Discord (desktop or web app).
2. Go to **Settings** → **Advanced** → toggle **Developer Mode** to **ON**.
3. Close settings.
4. Right-click your own username (in a message, the member list, or your profile) → **Copy User ID**.

Your User ID is a long number like `284102345871466496`.

:::tip
Developer Mode also lets you copy **Channel IDs** and **Server IDs** the same way — right-click the channel or server name and select Copy ID. You'll need a Channel ID if you want to set a home channel manually.
:::

## Step 8: Configure Hermes Agent

### Option A: Interactive Setup (Recommended)

Run the guided setup command:

```bash
hermes gateway setup
```

Select **Discord** when prompted, then paste your bot token and user ID when asked.

### Option B: Manual Configuration

Add the following to your `~/.hermes/.env` file:

```bash
# Required
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=284102345871466496

# Multiple allowed users (comma-separated)
# DISCORD_ALLOWED_USERS=284102345871466496,198765432109876543
```

Optional behavior settings in `~/.hermes/config.yaml`:

```yaml
discord:
  require_mention: true

group_sessions_per_user: true
```

- `discord.require_mention: true` keeps Hermes quiet in normal server traffic unless mentioned
- `group_sessions_per_user: true` keeps each participant's context isolated inside shared channels and threads

### Start the Gateway

Once configured, start the Discord gateway:

```bash
hermes gateway
```

The bot should come online in Discord within a few seconds. Send it a message — either a DM or in a channel it can see — to test.

:::tip
You can run `hermes gateway` in the background or as a systemd service for persistent operation. See the deployment docs for details.
:::

## Home Channel

You can designate a "home channel" where the bot sends proactive messages (such as cron job output, reminders, and notifications). There are two ways to set it:

### Using the Slash Command

Type `/sethome` in any Discord channel where the bot is present. That channel becomes the home channel.

### Manual Configuration

Add these to your `~/.hermes/.env`:

```bash
DISCORD_HOME_CHANNEL=123456789012345678
DISCORD_HOME_CHANNEL_NAME="#bot-updates"
```

Replace the ID with the actual channel ID (right-click → Copy Channel ID with Developer Mode on).

## Voice Messages

Hermes Agent supports Discord voice messages:

- **Incoming voice messages** are automatically transcribed using the configured STT provider: local `faster-whisper` (no key), Groq Whisper (`GROQ_API_KEY`), or OpenAI Whisper (`VOICE_TOOLS_OPENAI_KEY`).
- **Text-to-speech**: Use `/voice tts` to have the bot send spoken audio responses alongside text replies.
- **Discord voice channels**: Hermes can also join a voice channel, listen to users speaking, and talk back in the channel.

For the full setup and operational guide, see:
- [Voice Mode](/docs/user-guide/features/voice-mode)
- [Use Voice Mode with Hermes](/docs/guides/use-voice-mode-with-hermes)

## Troubleshooting

### Bot is online but not responding to messages

**Cause**: Message Content Intent is disabled.

**Fix**: Go to [Developer Portal](https://discord.com/developers/applications) → your app → Bot → Privileged Gateway Intents → enable **Message Content Intent** → Save Changes. Restart the gateway.

### "Disallowed Intents" error on startup

**Cause**: Your code requests intents that aren't enabled in the Developer Portal.

**Fix**: Enable all three Privileged Gateway Intents (Presence, Server Members, Message Content) in the Bot settings, then restart.

### Bot can't see messages in a specific channel

**Cause**: The bot's role doesn't have permission to view that channel.

**Fix**: In Discord, go to the channel's settings → Permissions → add the bot's role with **View Channel** and **Read Message History** enabled.

### 403 Forbidden errors

**Cause**: The bot is missing required permissions.

**Fix**: Re-invite the bot with the correct permissions using the URL from Step 5, or manually adjust the bot's role permissions in Server Settings → Roles.

### Bot is offline

**Cause**: The Hermes gateway isn't running, or the token is incorrect.

**Fix**: Check that `hermes gateway` is running. Verify `DISCORD_BOT_TOKEN` in your `.env` file. If you recently reset the token, update it.

### "User not allowed" / Bot ignores you

**Cause**: Your User ID isn't in `DISCORD_ALLOWED_USERS`.

**Fix**: Add your User ID to `DISCORD_ALLOWED_USERS` in `~/.hermes/.env` and restart the gateway.

### People in the same channel are sharing context unexpectedly

**Cause**: `group_sessions_per_user` is disabled, or the platform cannot provide a user ID for the messages in that context.

**Fix**: Set this in `~/.hermes/config.yaml` and restart the gateway:

```yaml
group_sessions_per_user: true
```

If you intentionally want a shared room conversation, leave it off — just expect shared transcript history and shared interrupt behavior.

## Security

:::warning
Always set `DISCORD_ALLOWED_USERS` to restrict who can interact with the bot. Without it, the gateway denies all users by default as a safety measure. Only add User IDs of people you trust — authorized users have full access to the agent's capabilities, including tool use and system access.
:::

For more information on securing your Hermes Agent deployment, see the [Security Guide](../security.md).
