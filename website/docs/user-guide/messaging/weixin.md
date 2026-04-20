---
sidebar_position: 15
title: "Weixin (WeChat)"
description: "Connect Hermes Agent to personal WeChat accounts via the iLink Bot API"
---

# Weixin (WeChat)

Connect Hermes to [WeChat](https://weixin.qq.com/) (微信), Tencent's personal messaging platform. The adapter uses Tencent's **iLink Bot API** for personal WeChat accounts — this is distinct from WeCom (Enterprise WeChat). Messages are delivered via long-polling, so no public endpoint or webhook is required.

:::info
This adapter is for **personal WeChat accounts** (微信). If you need enterprise/corporate WeChat, see the [WeCom adapter](./wecom.md) instead.
:::

## Prerequisites

- A personal WeChat account
- Python packages: `aiohttp` and `cryptography`
- Terminal QR rendering is included when Hermes is installed with the `messaging` extra

Install the required dependencies:

```bash
pip install aiohttp cryptography
# Optional: for terminal QR code display
pip install hermes-agent[messaging]
```

## Setup

### 1. Run the Setup Wizard

The easiest way to connect your WeChat account is through the interactive setup:

```bash
hermes gateway setup
```

Select **Weixin** when prompted. The wizard will:

1. Request a QR code from the iLink Bot API
2. Display the QR code in your terminal (or provide a URL)
3. Wait for you to scan the QR code with the WeChat mobile app
4. Prompt you to confirm the login on your phone
5. Save the account credentials automatically to `~/.hermes/weixin/accounts/`

Once confirmed, you'll see a message like:

```
微信连接成功，account_id=your-account-id
```

The wizard stores the `account_id`, `token`, and `base_url` so you don't need to configure them manually.

### 2. Configure Environment Variables

After initial QR login, set at minimum the account ID in `~/.hermes/.env`:

```bash
WEIXIN_ACCOUNT_ID=your-account-id

# Optional: override the token (normally auto-saved from QR login)
# WEIXIN_TOKEN=your-bot-token

# Optional: restrict access
WEIXIN_DM_POLICY=open
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2

# Optional: restore legacy multiline splitting behavior
# WEIXIN_SPLIT_MULTILINE_MESSAGES=true

# Optional: home channel for cron/notifications
WEIXIN_HOME_CHANNEL=chat_id
WEIXIN_HOME_CHANNEL_NAME=Home
```

### 3. Start the Gateway

```bash
hermes gateway
```

The adapter will restore saved credentials, connect to the iLink API, and begin long-polling for messages.

## Features

- **Long-poll transport** — no public endpoint, webhook, or WebSocket needed
- **QR code login** — scan-to-connect setup via `hermes gateway setup`
- **DM and group messaging** — configurable access policies
- **Media support** — images, video, files, and voice messages
- **AES-128-ECB encrypted CDN** — automatic encryption/decryption for all media transfers
- **Context token persistence** — disk-backed reply continuity across restarts
- **Markdown formatting** — preserves Markdown, including headers, tables, and code blocks, so WeChat clients that support Markdown can render it natively
- **Smart message chunking** — messages stay as a single bubble when under the limit; only oversized payloads split at logical boundaries
- **Typing indicators** — shows "typing…" status in the WeChat client while the agent processes
- **SSRF protection** — outbound media URLs are validated before download
- **Message deduplication** — 5-minute sliding window prevents double-processing
- **Automatic retry with backoff** — recovers from transient API errors

## Configuration Options

Set these in `config.yaml` under `platforms.weixin.extra`:

| Key | Default | Description |
|-----|---------|-------------|
| `account_id` | — | iLink Bot account ID (required) |
| `token` | — | iLink Bot token (required, auto-saved from QR login) |
| `base_url` | `https://ilinkai.weixin.qq.com` | iLink API base URL |
| `cdn_base_url` | `https://novac2c.cdn.weixin.qq.com/c2c` | CDN base URL for media transfer |
| `dm_policy` | `open` | DM access: `open`, `allowlist`, `disabled`, `pairing` |
| `group_policy` | `disabled` | Group access: `open`, `allowlist`, `disabled` |
| `allow_from` | `[]` | User IDs allowed for DMs (when dm_policy=allowlist) |
| `group_allow_from` | `[]` | Group IDs allowed (when group_policy=allowlist) |
| `split_multiline_messages` | `false` | When `true`, split multi-line replies into multiple chat messages (legacy behavior). When `false`, keep multi-line replies as one message unless they exceed the length limit. |

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
WEIXIN_DM_POLICY=allowlist
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2
```

### Group Policy

Controls which groups the bot responds in:

| Value | Behavior |
|-------|----------|
| `open` | Bot responds in all groups |
| `allowlist` | Bot only responds in group IDs listed in `group_allow_from` |
| `disabled` | All group messages are ignored (default) |

```bash
WEIXIN_GROUP_POLICY=allowlist
WEIXIN_GROUP_ALLOWED_USERS=group_id_1,group_id_2
```

:::note
The default group policy is `disabled` for Weixin (unlike WeCom where it defaults to `open`). This is intentional since personal WeChat accounts may be in many groups.
:::

## Media Support

### Inbound (receiving)

The adapter receives media attachments from users, downloads them from the WeChat CDN, decrypts them, and caches them locally for agent processing:

| Type | How it's handled |
|------|-----------------| 
| **Images** | Downloaded, AES-decrypted, and cached as JPEG. |
| **Video** | Downloaded, AES-decrypted, and cached as MP4. |
| **Files** | Downloaded, AES-decrypted, and cached. Original filename is preserved. |
| **Voice** | If a text transcription is available, it's extracted as text. Otherwise the audio (SILK format) is downloaded and cached. |

**Quoted messages:** Media from quoted (replied-to) messages is also extracted, so the agent has context about what the user is replying to.

### AES-128-ECB Encrypted CDN

WeChat media files are transferred through an encrypted CDN. The adapter handles this transparently:

- **Inbound:** Encrypted media is downloaded from the CDN using `encrypted_query_param` URLs, then decrypted with AES-128-ECB using the per-file key provided in the message payload.
- **Outbound:** Files are encrypted locally with a random AES-128-ECB key, uploaded to the CDN, and the encrypted reference is included in the outbound message.
- The AES key is 16 bytes (128-bit). Keys may arrive as raw base64 or hex-encoded — the adapter handles both formats.
- This requires the `cryptography` Python package.

No configuration is needed — encryption and decryption happen automatically.

### Outbound (sending)

| Method | What it sends |
|--------|--------------|
| `send` | Text messages with Markdown formatting | 
| `send_image` / `send_image_file` | Native image messages (via CDN upload) |
| `send_document` | File attachments (via CDN upload) |
| `send_video` | Video messages (via CDN upload) |

All outbound media goes through the encrypted CDN upload flow:

1. Generate a random AES-128 key
2. Encrypt the file with AES-128-ECB + PKCS#7 padding
3. Request an upload URL from the iLink API (`getuploadurl`)
4. Upload the ciphertext to the CDN
5. Send the message with the encrypted media reference

## Context Token Persistence

The iLink Bot API requires a `context_token` to be echoed back with each outbound message for a given peer. The adapter maintains a disk-backed context token store:

- Tokens are saved per account+peer to `~/.hermes/weixin/accounts/<account_id>.context-tokens.json`
- On startup, previously saved tokens are restored
- Every inbound message updates the stored token for that sender
- Outbound messages automatically include the latest context token

This ensures reply continuity even after gateway restarts.

## Markdown Formatting

WeChat clients connected through the iLink Bot API can render Markdown directly, so the adapter preserves Markdown instead of rewriting it:

- **Headers** stay as Markdown headings (`#`, `##`, ...)
- **Tables** stay as Markdown tables
- **Code fences** stay as fenced code blocks
- **Excessive blank lines** are collapsed to double newlines outside fenced code blocks

## Message Chunking

Messages are delivered as a single chat message whenever they fit within the platform limit. Only oversized payloads are split for delivery:

- Maximum message length: **4000 characters**
- Messages under the limit stay intact even when they contain multiple paragraphs or line breaks
- Oversized messages split at logical boundaries (paragraphs, blank lines, code fences)
- Code fences are kept intact whenever possible (never split mid-block unless the fence itself exceeds the limit)
- Oversized individual blocks fall back to the base adapter's truncation logic
- A 0.3 s inter-chunk delay prevents WeChat rate-limit drops when multiple chunks are sent

## Typing Indicators

The adapter shows typing status in the WeChat client:

1. When a message arrives, the adapter fetches a `typing_ticket` via the `getconfig` API
2. Typing tickets are cached for 10 minutes per user
3. `send_typing` sends a typing-start signal; `stop_typing` sends a typing-stop signal
4. The gateway automatically triggers typing indicators while the agent processes a message

## Long-Poll Connection

The adapter uses HTTP long-polling (not WebSocket) to receive messages:

### How It Works

1. **Connect:** Validates credentials and starts the poll loop
2. **Poll:** Calls `getupdates` with a 35-second timeout; the server holds the request until messages arrive or the timeout expires
3. **Dispatch:** Inbound messages are dispatched concurrently via `asyncio.create_task`
4. **Sync buffer:** A persistent sync cursor (`get_updates_buf`) is saved to disk so the adapter resumes from the correct position after restarts

### Retry Behavior

On API errors, the adapter uses a simple retry strategy:

| Condition | Behavior |
|-----------|----------|
| Transient error (1st–2nd) | Retry after 2 seconds |
| Repeated errors (3+) | Back off for 30 seconds, then reset counter |
| Session expired (`errcode=-14`) | Pause for 10 minutes (re-login may be needed) |
| Timeout | Immediately re-poll (normal long-poll behavior) |

### Deduplication

Inbound messages are deduplicated using message IDs with a 5-minute window. This prevents double-processing during network hiccups or overlapping poll responses.

### Token Lock

Only one Weixin gateway instance can use a given token at a time. The adapter acquires a scoped lock on startup and releases it on shutdown. If another gateway is already using the same token, startup fails with an informative error message.

## All Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WEIXIN_ACCOUNT_ID` | ✅ | — | iLink Bot account ID (from QR login) |
| `WEIXIN_TOKEN` | ✅ | — | iLink Bot token (auto-saved from QR login) |
| `WEIXIN_BASE_URL` | — | `https://ilinkai.weixin.qq.com` | iLink API base URL |
| `WEIXIN_CDN_BASE_URL` | — | `https://novac2c.cdn.weixin.qq.com/c2c` | CDN base URL for media transfer |
| `WEIXIN_DM_POLICY` | — | `open` | DM access policy: `open`, `allowlist`, `disabled`, `pairing` |
| `WEIXIN_GROUP_POLICY` | — | `disabled` | Group access policy: `open`, `allowlist`, `disabled` |
| `WEIXIN_ALLOWED_USERS` | — | _(empty)_ | Comma-separated user IDs for DM allowlist |
| `WEIXIN_GROUP_ALLOWED_USERS` | — | _(empty)_ | Comma-separated group IDs for group allowlist |
| `WEIXIN_HOME_CHANNEL` | — | — | Chat ID for cron/notification output |
| `WEIXIN_HOME_CHANNEL_NAME` | — | `Home` | Display name for the home channel |
| `WEIXIN_ALLOW_ALL_USERS` | — | — | Gateway-level flag to allow all users (used by setup wizard) |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Weixin startup failed: aiohttp and cryptography are required` | Install both: `pip install aiohttp cryptography` |
| `Weixin startup failed: WEIXIN_TOKEN is required` | Run `hermes gateway setup` to complete QR login, or set `WEIXIN_TOKEN` manually |
| `Weixin startup failed: WEIXIN_ACCOUNT_ID is required` | Set `WEIXIN_ACCOUNT_ID` in your `.env` or run `hermes gateway setup` |
| `Another local Hermes gateway is already using this Weixin token` | Stop the other gateway instance first — only one poller per token is allowed |
| Session expired (`errcode=-14`) | Your login session has expired. Re-run `hermes gateway setup` to scan a new QR code |
| QR code expired during setup | The QR auto-refreshes up to 3 times. If it keeps expiring, check your network connection |
| Bot doesn't respond to DMs | Check `WEIXIN_DM_POLICY` — if set to `allowlist`, the sender must be in `WEIXIN_ALLOWED_USERS` |
| Bot ignores group messages | Group policy defaults to `disabled`. Set `WEIXIN_GROUP_POLICY=open` or `allowlist` |
| Media download/upload fails | Ensure `cryptography` is installed. Check network access to `novac2c.cdn.weixin.qq.com` |
| `Blocked unsafe URL (SSRF protection)` | The outbound media URL points to a private/internal address. Only public URLs are allowed |
| Voice messages show as text | If WeChat provides a transcription, the adapter uses the text. This is expected behavior |
| Messages appear duplicated | The adapter deduplicates by message ID. If you see duplicates, check if multiple gateway instances are running |
| `iLink POST ... HTTP 4xx/5xx` | API error from the iLink service. Check your token validity and network connectivity |
| Terminal QR code doesn't render | Reinstall with the messaging extra: `pip install hermes-agent[messaging]`. Alternatively, open the URL printed above the QR |
