---
sidebar_position: 10
title: "DingTalk"
description: "Set up Hermes Agent as a DingTalk chatbot"
---

# DingTalk Setup

Hermes Agent integrates with DingTalk (钉钉) as a chatbot, letting you chat with your AI assistant through direct messages or group chats. The bot connects via DingTalk's Stream Mode — a long-lived WebSocket connection that requires no public URL or webhook server — and replies using markdown-formatted messages through DingTalk's session webhook API.

Before setup, here's the part most people want to know: how Hermes behaves once it's in your DingTalk workspace.

## How Hermes Behaves

| Context | Behavior |
|---------|----------|
| **DMs (1:1 chat)** | Hermes responds to every message. No `@mention` needed. Each DM has its own session. |
| **Group chats** | Hermes responds when you `@mention` it. Without a mention, Hermes ignores the message. |
| **Shared groups with multiple users** | By default, Hermes isolates session history per user inside the group. Two people talking in the same group do not share one transcript unless you explicitly disable that. |

### Session Model in DingTalk

By default:

- each DM gets its own session
- each user in a shared group chat gets their own session inside that group

This is controlled by `config.yaml`:

```yaml
group_sessions_per_user: true
```

Set it to `false` only if you explicitly want one shared conversation for the entire group:

```yaml
group_sessions_per_user: false
```

This guide walks you through the full setup process — from creating your DingTalk bot to sending your first message.

## Prerequisites

Install the required Python packages:

```bash
pip install "hermes-agent[dingtalk]"
```

Or individually:

```bash
pip install dingtalk-stream httpx alibabacloud-dingtalk
```

- `dingtalk-stream` — DingTalk's official SDK for Stream Mode (WebSocket-based real-time messaging)
- `httpx` — async HTTP client used for sending replies via session webhooks
- `alibabacloud-dingtalk` — DingTalk OpenAPI SDK for AI Cards, emoji reactions, and media downloads

## Step 1: Create a DingTalk App

1. Go to the [DingTalk Developer Console](https://open-dev.dingtalk.com/).
2. Log in with your DingTalk admin account.
3. Click **Application Development** → **Custom Apps** → **Create App via H5 Micro-App** (or **Robot** depending on your console version).
4. Fill in:
   - **App Name**: e.g., `Hermes Agent`
   - **Description**: optional
5. After creating, navigate to **Credentials & Basic Info** to find your **Client ID** (AppKey) and **Client Secret** (AppSecret). Copy both.

:::warning[Credentials shown only once]
The Client Secret is only displayed once when you create the app. If you lose it, you'll need to regenerate it. Never share these credentials publicly or commit them to Git.
:::

## Step 2: Enable the Robot Capability

1. In your app's settings page, go to **Add Capability** → **Robot**.
2. Enable the robot capability.
3. Under **Message Reception Mode**, select **Stream Mode** (recommended — no public URL needed).

:::tip
Stream Mode is the recommended setup. It uses a long-lived WebSocket connection initiated from your machine, so you don't need a public IP, domain name, or webhook endpoint. This works behind NAT, firewalls, and on local machines.
:::

## Step 3: Find Your DingTalk User ID

Hermes Agent uses your DingTalk User ID to control who can interact with the bot. DingTalk User IDs are alphanumeric strings set by your organization's admin.

To find yours:

1. Ask your DingTalk organization admin — User IDs are configured in the DingTalk admin console under **Contacts** → **Members**.
2. Alternatively, the bot logs the `sender_id` for each incoming message. Start the gateway, send the bot a message, then check the logs for your ID.

## Step 4: Configure Hermes Agent

### Option A: Interactive Setup (Recommended)

Run the guided setup command:

```bash
hermes gateway setup
```

Select **DingTalk** when prompted. The setup wizard can authorize via one of two paths:

- **QR-code device flow (recommended).** Scan the QR that prints in your terminal with the DingTalk mobile app — your Client ID and Client Secret are returned automatically and written to `~/.hermes/.env`. No developer-console trip needed.
- **Manual paste.** If you already have credentials (or QR scanning isn't convenient), paste your Client ID, Client Secret, and allowed user IDs when prompted.

:::note openClaw branding disclosure
Because DingTalk's `verification_uri_complete` is hardcoded to the openClaw identity at the API layer, the QR currently authorizes under an `openClaw` source string until Alibaba / DingTalk-Real-AI registers a Hermes-specific template server-side. This is purely how DingTalk presents the consent screen — the bot you create is fully yours and private to your tenant.
:::

### Option B: Manual Configuration

Add the following to your `~/.hermes/.env` file:

```bash
# Required
DINGTALK_CLIENT_ID=your-app-key
DINGTALK_CLIENT_SECRET=your-app-secret

# Security: restrict who can interact with the bot
DINGTALK_ALLOWED_USERS=user-id-1

# Multiple allowed users (comma-separated)
# DINGTALK_ALLOWED_USERS=user-id-1,user-id-2
```

Optional behavior settings in `~/.hermes/config.yaml`:

```yaml
group_sessions_per_user: true

gateway:
  platforms:
    dingtalk:
      extra:
        # Require @mention in groups before the bot replies (parity with Slack/Telegram/Discord).
        # DMs ignore this — the bot always replies in 1:1 chats.
        require_mention: true

        # Per-platform allowlist. When set, only these DingTalk user IDs can interact with the bot
        # (same semantics as DINGTALK_ALLOWED_USERS, but scoped here instead of in .env).
        allowed_users:
          - user-id-1
          - user-id-2
```

- `group_sessions_per_user: true` keeps each participant's context isolated inside shared group chats
- `require_mention: true` prevents the bot from responding to every group message — it only answers when someone @-mentions it
- `allowed_users` under `dingtalk.extra` is an alternative to `DINGTALK_ALLOWED_USERS`; if both are set, they're merged

### Start the Gateway

Once configured, start the DingTalk gateway:

```bash
hermes gateway
```

The bot should connect to DingTalk's Stream Mode within a few seconds. Send it a message — either a DM or in a group where it's been added — to test.

:::tip
You can run `hermes gateway` in the background or as a systemd service for persistent operation. See the deployment docs for details.
:::

## Features

### AI Cards

Hermes can reply using DingTalk AI Cards instead of plain markdown messages. Cards provide a richer, more structured display and support streaming updates as the agent generates its response.

To enable AI Cards, configure a card template ID in `config.yaml`:

```yaml
platforms:
  dingtalk:
    enabled: true
    extra:
      card_template_id: "your-card-template-id"
```

You can find your card template ID in the DingTalk Developer Console under your app's AI Card settings. When AI Cards are enabled, all replies are sent as cards with streaming text updates.

### Emoji Reactions

Hermes automatically adds emoji reactions to your messages to show processing status:

- 🤔Thinking — added when the bot starts processing your message
- 🥳Done — added when the response is complete (replaces the Thinking reaction)

These reactions work in both DMs and group chats.

### Display Settings

You can customize DingTalk's display behavior independently from other platforms:

```yaml
display:
  platforms:
    dingtalk:
      show_reasoning: false   # Show model reasoning/thinking in replies
      streaming: true         # Enable streaming responses (works with AI Cards)
      tool_progress: all      # Show tool execution progress (all/new/off)
      interim_assistant_messages: true  # Show intermediate commentary messages
```

To disable tool progress and intermediate messages for a cleaner experience:

```yaml
display:
  platforms:
    dingtalk:
      tool_progress: off
      interim_assistant_messages: false
```

## Troubleshooting

### Bot is not responding to messages

**Cause**: The robot capability isn't enabled, or `DINGTALK_ALLOWED_USERS` doesn't include your User ID.

**Fix**: Verify the robot capability is enabled in your app settings and that Stream Mode is selected. Check that your User ID is in `DINGTALK_ALLOWED_USERS`. Restart the gateway.

### "dingtalk-stream not installed" error

**Cause**: The `dingtalk-stream` Python package is not installed.

**Fix**: Install it:

```bash
pip install dingtalk-stream httpx
```

### "DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required"

**Cause**: The credentials aren't set in your environment or `.env` file.

**Fix**: Verify `DINGTALK_CLIENT_ID` and `DINGTALK_CLIENT_SECRET` are set correctly in `~/.hermes/.env`. The Client ID is your AppKey, and the Client Secret is your AppSecret from the DingTalk Developer Console.

### Stream disconnects / reconnection loops

**Cause**: Network instability, DingTalk platform maintenance, or credential issues.

**Fix**: The adapter automatically reconnects with exponential backoff (2s → 5s → 10s → 30s → 60s). Check that your credentials are valid and your app hasn't been deactivated. Verify your network allows outbound WebSocket connections.

### Bot is offline

**Cause**: The Hermes gateway isn't running, or it failed to connect.

**Fix**: Check that `hermes gateway` is running. Look at the terminal output for error messages. Common issues: wrong credentials, app deactivated, `dingtalk-stream` or `httpx` not installed.

### "No session_webhook available"

**Cause**: The bot tried to reply but doesn't have a session webhook URL. This typically happens if the webhook expired or the bot was restarted between receiving the message and sending the reply.

**Fix**: Send a new message to the bot — each incoming message provides a fresh session webhook for replies. This is a normal DingTalk limitation; the bot can only reply to messages it has received recently.

## Security

:::warning
Always set `DINGTALK_ALLOWED_USERS` to restrict who can interact with the bot. Without it, the gateway denies all users by default as a safety measure. Only add User IDs of people you trust — authorized users have full access to the agent's capabilities, including tool use and system access.
:::

For more information on securing your Hermes Agent deployment, see the [Security Guide](../security.md).

## Notes

- **Stream Mode**: No public URL, domain name, or webhook server needed. The connection is initiated from your machine via WebSocket, so it works behind NAT and firewalls.
- **AI Cards**: Optionally reply with rich AI Cards instead of plain markdown. Configure via `card_template_id`.
- **Emoji Reactions**: Automatic 🤔Thinking/🥳Done reactions for processing status.
- **Markdown responses**: Replies are formatted in DingTalk's markdown format for rich text display.
- **Media support**: Images and files in incoming messages are automatically resolved and can be processed by vision tools.
- **Message deduplication**: The adapter deduplicates messages with a 5-minute window to prevent processing the same message twice.
- **Auto-reconnection**: If the stream connection drops, the adapter automatically reconnects with exponential backoff.
- **Message length limit**: Responses are capped at 20,000 characters per message. Longer responses are truncated.
