---
sidebar_position: 14
title: "WeCom (Enterprise WeChat)"
description: "Connect Hermes Agent to WeCom via the AI Bot WebSocket gateway"
---

# WeCom (Enterprise WeChat)

Connect Hermes to [WeCom](https://work.weixin.qq.com/) (企业微信), Tencent's enterprise messaging platform. The adapter uses WeCom's AI Bot WebSocket gateway for real-time bidirectional communication — no public endpoint or webhook needed.

## Prerequisites

- A WeCom organization account
- An AI Bot created in the WeCom Admin Console
- The Bot ID and Secret from the bot's credentials page
- Python packages: `aiohttp` and `httpx`

## Setup

### 1. Create an AI Bot

1. Log in to the [WeCom Admin Console](https://work.weixin.qq.com/wework_admin/frame)
2. Navigate to **Applications** → **Create Application** → **AI Bot**
3. Configure the bot name and description
4. Copy the **Bot ID** and **Secret** from the credentials page

### 2. Configure Hermes

Run the interactive setup:

```bash
hermes gateway setup
```

Select **WeCom** and enter your Bot ID and Secret.

Or set environment variables in `~/.hermes/.env`:

```bash
WECOM_BOT_ID=your-bot-id
WECOM_SECRET=your-secret

# Optional: restrict access
WECOM_ALLOWED_USERS=user_id_1,user_id_2

# Optional: home channel for cron/notifications
WECOM_HOME_CHANNEL=chat_id
```

### 3. Start the gateway

```bash
hermes gateway
```

## Features

- **WebSocket transport** — persistent connection, no public endpoint needed
- **DM and group messaging** — configurable access policies
- **Per-group sender allowlists** — fine-grained control over who can interact in each group
- **Media support** — images, files, voice, video upload and download
- **AES-encrypted media** — automatic decryption for inbound attachments
- **Quote context** — preserves reply threading
- **Markdown rendering** — rich text responses
- **Reply-mode streaming** — correlates responses to inbound message context
- **Auto-reconnect** — exponential backoff on connection drops

## Configuration Options

Set these in `config.yaml` under `platforms.wecom.extra`:

| Key | Default | Description |
|-----|---------|-------------|
| `bot_id` | — | WeCom AI Bot ID (required) |
| `secret` | — | WeCom AI Bot Secret (required) |
| `websocket_url` | `wss://openws.work.weixin.qq.com` | WebSocket gateway URL |
| `dm_policy` | `open` | DM access: `open`, `allowlist`, `disabled`, `pairing` |
| `group_policy` | `open` | Group access: `open`, `allowlist`, `disabled` |
| `allow_from` | `[]` | User IDs allowed for DMs (when dm_policy=allowlist) |
| `group_allow_from` | `[]` | Group IDs allowed (when group_policy=allowlist) |
| `groups` | `{}` | Per-group configuration (see below) |

## Access Policies

### DM Policy

Controls who can send direct messages to the bot:

| Value | Behavior |
|-------|----------|
| `open` | Anyone can DM the bot (default) |
| `allowlist` | Only user IDs in `allow_from` can DM |
| `disabled` | All DMs are ignored |
| `pairing` | Pairing mode (for initial setup) |

```bash
WECOM_DM_POLICY=allowlist
```

### Group Policy

Controls which groups the bot responds in:

| Value | Behavior |
|-------|----------|
| `open` | Bot responds in all groups (default) |
| `allowlist` | Bot only responds in group IDs listed in `group_allow_from` |
| `disabled` | All group messages are ignored |

```bash
WECOM_GROUP_POLICY=allowlist
```

### Per-Group Sender Allowlists

For fine-grained control, you can restrict which users are allowed to interact with the bot within specific groups. This is configured in `config.yaml`:

```yaml
platforms:
  wecom:
    enabled: true
    extra:
      bot_id: "your-bot-id"
      secret: "your-secret"
      group_policy: "allowlist"
      group_allow_from:
        - "group_id_1"
        - "group_id_2"
      groups:
        group_id_1:
          allow_from:
            - "user_alice"
            - "user_bob"
        group_id_2:
          allow_from:
            - "user_charlie"
        "*":
          allow_from:
            - "user_admin"
```

**How it works:**

1. The `group_policy` and `group_allow_from` controls determine whether a group is allowed at all.
2. If a group passes the top-level check, the `groups.<group_id>.allow_from` list (if present) further restricts which senders within that group can interact with the bot.
3. A wildcard `"*"` group entry serves as a default for groups not explicitly listed.
4. Allowlist entries support the `*` wildcard to allow all users, and entries are case-insensitive.
5. Entries can optionally use the `wecom:user:` or `wecom:group:` prefix format — the prefix is stripped automatically.

If no `allow_from` is configured for a group, all users in that group are allowed (assuming the group itself passes the top-level policy check).

## Media Support

### Inbound (receiving)

The adapter receives media attachments from users and caches them locally for agent processing:

| Type | How it's handled |
|------|-----------------|
| **Images** | Downloaded and cached locally. Supports both URL-based and base64-encoded images. |
| **Files** | Downloaded and cached. Filename is preserved from the original message. |
| **Voice** | Voice message text transcription is extracted if available. |
| **Mixed messages** | WeCom mixed-type messages (text + images) are parsed and all components extracted. |

**Quoted messages:** Media from quoted (replied-to) messages is also extracted, so the agent has context about what the user is replying to.

### AES-Encrypted Media Decryption

WeCom encrypts some inbound media attachments with AES-256-CBC. The adapter handles this automatically:

- When an inbound media item includes an `aeskey` field, the adapter downloads the encrypted bytes and decrypts them using AES-256-CBC with PKCS#7 padding.
- The AES key is the base64-decoded value of the `aeskey` field (must be exactly 32 bytes).
- The IV is derived from the first 16 bytes of the key.
- This requires the `cryptography` Python package (`pip install cryptography`).

No configuration is needed — decryption happens transparently when encrypted media is received.

### Outbound (sending)

| Method | What it sends | Size limit |
|--------|--------------|------------|
| `send` | Markdown text messages | 4000 chars |
| `send_image` / `send_image_file` | Native image messages | 10 MB |
| `send_document` | File attachments | 20 MB |
| `send_voice` | Voice messages (AMR format only for native voice) | 2 MB |
| `send_video` | Video messages | 10 MB |

**Chunked upload:** Files are uploaded in 512 KB chunks through a three-step protocol (init → chunks → finish). The adapter handles this automatically.

**Automatic downgrade:** When media exceeds the native type's size limit but is under the absolute 20 MB file limit, it is automatically sent as a generic file attachment instead:

- Images > 10 MB → sent as file
- Videos > 10 MB → sent as file
- Voice > 2 MB → sent as file
- Non-AMR audio → sent as file (WeCom only supports AMR for native voice)

Files exceeding the absolute 20 MB limit are rejected with an informational message sent to the chat.

## Reply-Mode Stream Responses

When the bot receives a message via the WeCom callback, the adapter remembers the inbound request ID. If a response is sent while the request context is still active, the adapter uses WeCom's reply-mode (`aibot_respond_msg`) with streaming to correlate the response directly to the inbound message. This provides a more natural conversation experience in the WeCom client.

If the inbound request context has expired or is unavailable, the adapter falls back to proactive message sending via `aibot_send_msg`.

Reply-mode also works for media: uploaded media can be sent as a reply to the originating message.

## Connection and Reconnection

The adapter maintains a persistent WebSocket connection to WeCom's gateway at `wss://openws.work.weixin.qq.com`.

### Connection Lifecycle

1. **Connect:** Opens a WebSocket connection and sends an `aibot_subscribe` authentication frame with the bot_id and secret.
2. **Heartbeat:** Sends application-level ping frames every 30 seconds to keep the connection alive.
3. **Listen:** Continuously reads inbound frames and dispatches message callbacks.

### Reconnection Behavior

On connection loss, the adapter uses exponential backoff to reconnect:

| Attempt | Delay |
|---------|-------|
| 1st retry | 2 seconds |
| 2nd retry | 5 seconds |
| 3rd retry | 10 seconds |
| 4th retry | 30 seconds |
| 5th+ retry | 60 seconds |

After each successful reconnection, the backoff counter resets to zero. All pending request futures are failed on disconnect so callers don't hang indefinitely.

### Deduplication

Inbound messages are deduplicated using message IDs with a 5-minute window and a maximum cache of 1000 entries. This prevents double-processing of messages during reconnection or network hiccups.

## All Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WECOM_BOT_ID` | ✅ | — | WeCom AI Bot ID |
| `WECOM_SECRET` | ✅ | — | WeCom AI Bot Secret |
| `WECOM_ALLOWED_USERS` | — | _(empty)_ | Comma-separated user IDs for the gateway-level allowlist |
| `WECOM_HOME_CHANNEL` | — | — | Chat ID for cron/notification output |
| `WECOM_WEBSOCKET_URL` | — | `wss://openws.work.weixin.qq.com` | WebSocket gateway URL |
| `WECOM_DM_POLICY` | — | `open` | DM access policy |
| `WECOM_GROUP_POLICY` | — | `open` | Group access policy |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `WECOM_BOT_ID and WECOM_SECRET are required` | Set both env vars or configure in setup wizard |
| `WeCom startup failed: aiohttp not installed` | Install aiohttp: `pip install aiohttp` |
| `WeCom startup failed: httpx not installed` | Install httpx: `pip install httpx` |
| `invalid secret (errcode=40013)` | Verify the secret matches your bot's credentials |
| `Timed out waiting for subscribe acknowledgement` | Check network connectivity to `openws.work.weixin.qq.com` |
| Bot doesn't respond in groups | Check `group_policy` setting and ensure the group ID is in `group_allow_from` |
| Bot ignores certain users in a group | Check per-group `allow_from` lists in the `groups` config section |
| Media decryption fails | Install `cryptography`: `pip install cryptography` |
| `cryptography is required for WeCom media decryption` | The inbound media is AES-encrypted. Install: `pip install cryptography` |
| Voice messages sent as files | WeCom only supports AMR format for native voice. Other formats are auto-downgraded to file. |
| `File too large` error | WeCom has a 20 MB absolute limit on all file uploads. Compress or split the file. |
| Images sent as files | Images > 10 MB exceed the native image limit and are auto-downgraded to file attachments. |
| `Timeout sending message to WeCom` | The WebSocket may have disconnected. Check logs for reconnection messages. |
| `WeCom websocket closed during authentication` | Network issue or incorrect credentials. Verify bot_id and secret. |
