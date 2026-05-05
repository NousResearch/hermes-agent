---
sidebar_position: 16
title: "Yuanbao"
description: "Connect Hermes Agent to the Yuanbao enterprise messaging platform via WebSocket gateway"
---

# Yuanbao

Connect Hermes to [Yuanbao](https://yuanbao.tencent.com/), Tencent's enterprise messaging platform. The adapter uses a WebSocket gateway for real-time message delivery and supports both direct (C2C) and group conversations.

:::info
Yuanbao is an enterprise messaging platform primarily used within Tencent and enterprise environments. It uses WebSocket for real-time communication, HMAC-based authentication, and supports rich media including images, files, and voice messages.
:::

## Prerequisites

- A Yuanbao account with bot creation permissions
- Yuanbao APP_ID and APP_SECRET (from platform admin)
- Python packages: `websockets` and `httpx`
- For media support: `aiofiles`

Install the required dependencies:

```bash
pip install websockets httpx aiofiles
```

## Setup

### 1. Create a Bot in Yuanbao

1. Download the Yuanbao app from [https://yuanbao.tencent.com/](https://yuanbao.tencent.com/)
2. In the app, go to **PAI → My Bot** and create a new bot
3. After the bot is created, copy the **APP_ID** and **APP_SECRET**

### 2. Run the Setup Wizard

The easiest way to configure Yuanbao is through the interactive setup:

```bash
hermes gateway setup
```

Select **Yuanbao** when prompted. The wizard will:

1. Ask for your APP_ID
2. Ask for your APP_SECRET
3. Save the configuration automatically

:::tip
The WebSocket URL and API Domain have sensible defaults built in. You only need to provide APP_ID and APP_SECRET to get started.
:::

### 3. Configure Environment Variables

After initial setup, verify these variables in `~/.hermes/.env`:

```bash
# Required
YUANBAO_APP_ID=your-app-id
YUANBAO_APP_SECRET=your-app-secret
YUANBAO_WS_URL=wss://api.yuanbao.example.com/ws
YUANBAO_API_DOMAIN=https://api.yuanbao.example.com

# Optional: bot account ID (normally obtained automatically from sign-token)
# YUANBAO_BOT_ID=your-bot-id

# Optional: internal routing environment (e.g. test/staging/production)
# YUANBAO_ROUTE_ENV=production

# Optional: home channel for cron/notifications (format: direct:<account> or group:<group_code>)
YUANBAO_HOME_CHANNEL=direct:bot_account_id
YUANBAO_HOME_CHANNEL_NAME="Bot Notifications"

# Optional: restrict access (legacy, see Access Control below for fine-grained policies)
YUANBAO_ALLOWED_USERS=user_account_1,user_account_2
```

### 4. Start the Gateway

```bash
hermes gateway
```

The adapter will connect to the Yuanbao WebSocket gateway, authenticate using HMAC signatures, and begin processing messages.

## Features

- **WebSocket gateway** — real-time bidirectional communication
- **HMAC authentication** — secure request signing with APP_ID/APP_SECRET
- **C2C messaging** — direct user-to-bot conversations
- **Group messaging** — conversations in group chats
- **Media support** — images, files, and voice messages via COS (Cloud Object Storage)
- **Markdown formatting** — messages are automatically chunked for Yuanbao's size limits
- **Message deduplication** — prevents duplicate processing of the same message
- **Heartbeat/keep-alive** — maintains WebSocket connection stability
- **Typing indicators** — shows "typing…" status while the agent processes
- **Automatic reconnection** — handles WebSocket disconnections with exponential backoff
- **Group information queries** — retrieve group details and member lists
- **Sticker/Emoji support** — send TIMFaceElem stickers and emoji in conversations
- **Auto-sethome** — first user to message the bot is automatically set as the home channel owner
- **Slow-response notification** — sends a waiting message when the agent takes longer than expected

## Configuration Options

### Chat ID Formats

Yuanbao uses prefixed identifiers depending on conversation type:

| Chat Type | Format | Example |
|-----------|--------|---------|
| Direct message (C2C) | `direct:<account>` | `direct:user123` |
| Group message | `group:<group_code>` | `group:grp456` |

### Media Uploads

The Yuanbao adapter automatically handles media uploads via COS (Tencent Cloud Object Storage):

- **Images**: Supports JPEG, PNG, GIF, WebP
- **Files**: Supports all common document types
- **Voice**: Supports WAV, MP3, OGG

Media URLs are automatically validated and downloaded before upload to prevent SSRF attacks.

## Home Channel

Use the `/sethome` command in any Yuanbao chat (DM or group) to designate it as the **home channel**. Scheduled tasks (cron jobs) deliver their results to this channel.

:::tip Auto-sethome
If no home channel is configured, the first user to message the bot will be automatically set as the home channel owner. If the current home channel is a group chat, the first DM will upgrade it to a direct channel.
:::

You can also set it manually in `~/.hermes/.env`:

```bash
YUANBAO_HOME_CHANNEL=direct:user_account_id
# or for a group:
# YUANBAO_HOME_CHANNEL=group:group_code
YUANBAO_HOME_CHANNEL_NAME="My Bot Updates"
```

### Example: Set Home Channel

1. Start a conversation with the bot in Yuanbao
2. Send the command: `/sethome`
3. The bot responds: "Home channel set to [chat_name] with ID [chat_id]. Cron jobs will deliver to this location."
4. Future cron jobs and notifications will be sent to this channel

### Example: Cron Job Delivery

Create a cron job:

```bash
/cron "0 9 * * *" Check server status
```

The scheduled output will be delivered to your Yuanbao home channel every day at 9 AM.

## Usage Tips

### Starting a Conversation

Send any message to the bot in Yuanbao:

```
hello
```

The bot responds in the same conversation thread.

### Available Commands

All standard Hermes commands work on Yuanbao:

| Command | Description |
|---------|-------------|
| `/new` | Start a fresh conversation |
| `/model [provider:model]` | Show or change the model |
| `/sethome` | Set this chat as the home channel |
| `/status` | Show session info |
| `/help` | Show available commands |

### Sending Files

To send a file to the bot, simply attach it directly in the Yuanbao chat. The bot will automatically download and process the file attachment.

You can also include a message with the attachment:

```
Please analyze this document
```

### Receiving Files

When you ask the bot to create or export a file, it sends the file directly to your Yuanbao chat.

## Troubleshooting

### Bot is online but not responding to messages

**Cause**: Authentication failed during WebSocket handshake.

**Fix**:
1. Verify APP_ID and APP_SECRET are correct
2. Check that the WebSocket URL is accessible
3. Ensure the bot account has proper permissions
4. Review gateway logs: `tail -f ~/.hermes/logs/gateway.log`

### "Connection refused" error

**Cause**: WebSocket URL is unreachable or incorrect.

**Fix**:
1. Verify the WebSocket URL format (should start with `wss://`)
2. Check network connectivity to the Yuanbao API domain
3. Confirm firewall allows WebSocket connections
4. Test URL with: `curl -I https://[YUANBAO_API_DOMAIN]`

### Media uploads fail

**Cause**: COS credentials are invalid or media server is unreachable.

**Fix**:
1. Verify API_DOMAIN is correct
2. Check that media upload permissions are enabled for your bot
3. Ensure the media file is accessible and not corrupted
4. Check COS bucket configuration with platform admin

### Messages not delivered to home channel

**Cause**: Home channel ID format is incorrect or cron job hasn't triggered.

**Fix**:
1. Verify YUANBAO_HOME_CHANNEL is in correct format
2. Test with `/sethome` command to auto-detect correct format
3. Check cron job schedule with `/status`
4. Verify bot has send permissions in the target chat

### Frequent disconnections

**Cause**: WebSocket connection is unstable or network is unreliable.

**Fix**:
1. Check gateway logs for error patterns
2. Increase heartbeat timeout in connection settings
3. Ensure stable network connection to Yuanbao API
4. Consider enabling verbose logging: `HERMES_LOG_LEVEL=debug`

## Access Control

Yuanbao supports fine-grained access control for both DM and group conversations:

```bash
# DM policy: open (default) | allowlist | disabled
YUANBAO_DM_POLICY=open
# Comma-separated user IDs allowed to DM the bot (only used when DM_POLICY=allowlist)
YUANBAO_DM_ALLOW_FROM=user_id_1,user_id_2

# Group policy: open (default) | allowlist | disabled
YUANBAO_GROUP_POLICY=open
# Comma-separated group codes allowed (only used when GROUP_POLICY=allowlist)
YUANBAO_GROUP_ALLOW_FROM=group_code_1,group_code_2
```

These can also be set in `config.yaml`:

```yaml
platforms:
  yuanbao:
    extra:
      dm_policy: allowlist
      dm_allow_from: "user1,user2"
      group_policy: open
      group_allow_from: ""
```

## Advanced Configuration

### Message Chunking

Yuanbao has a maximum message size. Hermes automatically chunks large responses with Markdown-aware splitting (respects code fences, tables, and paragraph boundaries).

### Connection Parameters

The following connection parameters are built into the adapter with sensible defaults:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| WebSocket connect timeout | 15 seconds | Time to wait for WS handshake |
| Heartbeat interval | 30 seconds | Ping frequency to keep connection alive |
| Max reconnect attempts | 100 | Maximum number of reconnection tries |
| Reconnect backoff | 1s → 60s (exponential) | Wait time between reconnect attempts |
| Reply heartbeat interval | 2 seconds | RUNNING status send frequency |
| Send timeout | 30 seconds | Timeout for outbound WS messages |

:::note
These values are currently not configurable via environment variables. They are optimized for typical Yuanbao deployments.
:::

### Verbose Logging

Enable debug logging to troubleshoot connection issues:

```bash
HERMES_LOG_LEVEL=debug hermes gateway
```

## Integration with Other Features

### Cron Jobs

Schedule tasks that run on Yuanbao:

```
/cron "0 */4 * * *" Report system health
```

Results are delivered to your home channel.

### Background Tasks

Run long operations without blocking the conversation:

```
/background Analyze all files in the archive
```

### Cross-Platform Messages

Send a message from CLI to Yuanbao:

```bash
hermes chat -q "Send 'Hello from CLI' to yuanbao:group:group_code"
```

## Related Documentation

- [Messaging Gateway Overview](./index.md)
- [Slash Commands Reference](/docs/reference/slash-commands.md)
- [Cron Jobs](/docs/user-guide/features/cron.md)
- [Background Sessions](/docs/user-guide/cli#background-sessions)