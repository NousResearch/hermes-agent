---
sidebar_position: 11
title: "WeChat (Weixin)"
description: "Set up Hermes Agent as a WeChat (Weixin) bot via the iLink Bot API"
---

# WeChat (Weixin) Setup

Hermes Agent can connect to WeChat (Weixin) through the iLink Bot API. The Weixin adapter runs as part of the normal Hermes gateway process: it long-polls for new direct messages, downloads and decrypts CDN media, and sends replies back through the same API using the required `context_token` from the inbound message.

This integration is best for 1:1 bot conversations on WeChat where you want Hermes available from a phone-friendly messaging client with native support for images, videos, files, and typing indicators.

## Overview

The Weixin integration provides:

- plain-text chat replies
- native image, video, and file delivery
- inbound voice handling with speech-to-text fallback
- typing indicators
- context-aware replies using WeChat `context_token` values
- CDN media upload/download with AES-128-ECB encryption

## Prerequisites

Install the required Python packages:

```bash
pip install httpx cryptography
```

Optional but useful:

```bash
pip install qrcode
```

- `httpx` is used for the WeChat HTTP API and CDN requests
- `cryptography` is required for AES-128-ECB media encryption/decryption
- `qrcode` lets the QR login script render the login QR directly in your terminal

## Setup

### Option A: QR Login Script

Run the login helper:

```bash
python3 scripts/weixin_login.py
```

The script will:

1. Request a QR code from the iLink Bot API
2. Show the QR code URL and, if `qrcode` is installed, render it in your terminal
3. Poll until you scan and confirm the login in WeChat
4. Save the credentials under `~/.hermes/weixin/accounts/`
5. Print the environment variables to add to `~/.hermes/.env`

### Option B: Hermes Gateway Setup Wizard

You can also use the interactive setup flow:

```bash
hermes gateway setup
```

Choose **WeChat (Weixin)** and paste the token and account ID returned by the QR login step.

### Manual Configuration

Add the required settings to `~/.hermes/.env`:

```bash
WEIXIN_TOKEN=your-bot-token
WEIXIN_ACCOUNT_ID=your-ilink-bot-id

# Optional overrides
# WEIXIN_BASE_URL=https://ilinkai.weixin.qq.com
# WEIXIN_CDN_BASE_URL=https://novac2c.cdn.weixin.qq.com/c2c

# Security
# WEIXIN_ALLOWED_USERS=user-id-1,user-id-2
# WEIXIN_ALLOW_ALL_USERS=true

# Optional home channel for cron / send_message
# WEIXIN_HOME_CHANNEL=user-id-1
```

Start the gateway once configured:

```bash
hermes gateway
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `WEIXIN_TOKEN` | Yes | Bot token returned by the QR login flow |
| `WEIXIN_ACCOUNT_ID` | Yes | iLink bot account ID returned by the QR login flow |
| `WEIXIN_BASE_URL` | No | Override the Weixin API base URL |
| `WEIXIN_CDN_BASE_URL` | No | Override the Weixin CDN base URL |
| `WEIXIN_ALLOWED_USERS` | No | Comma-separated Weixin user IDs allowed to message the bot |
| `WEIXIN_ALLOW_ALL_USERS` | No | Allow all Weixin users without an allowlist |
| `WEIXIN_HOME_CHANNEL` | No | Default Weixin user/chat ID for cron delivery and `send_message` |

## Supported Features

- Text replies
- Native image sending and receiving
- Native file attachment sending (TXT, PDF, etc.)
- Native video sending (MP4)
- Audio file delivery (WAV, MP3, etc. — sent as file attachment)
- Voice-message ingestion with STT fallback
- Typing indicators via `getconfig` + `sendtyping`
- Session resume from saved `get_updates_buf`
- CDN media upload with AES-128-ECB encryption

## Known Limitations

- **Streaming**: WeChat does not support message editing, so streaming mode sends raw `MEDIA:` tags as text instead of delivering files. If you use streaming with WeChat, disable it globally or accept text-only streaming. This is an upstream limitation affecting all platforms without `edit_message` support.
- **Audio delivery**: Audio is delivered as a file attachment rather than a native voice bubble. Bot-originated voice playback is not reliably supported by the WeChat client. This matches the official SDK behavior, which also sends audio as file attachments.
- Outbound replies require a valid `context_token`. A user must message the bot first before Hermes can reply or proactively send into that conversation.
- The adapter currently targets direct-message style conversations only. Group chat behavior is not supported.
- Inbound WeChat voice media uses SILK. Hermes will try to transcode it to WAV when possible, but may fall back to storing raw SILK.
- If the WeChat session expires (`errcode -14`), the adapter pauses requests for one hour before retrying.
- WeChat does not render markdown. Hermes strips markdown formatting before delivery.

## Troubleshooting

### "No token configured"

The gateway cannot see `WEIXIN_TOKEN`.

Fix:

- run `python3 scripts/weixin_login.py` again
- add the printed token to `~/.hermes/.env`
- restart the gateway

### The bot connects but cannot reply

Most outbound Weixin sends require a fresh `context_token`.

Fix:

- send the bot a new inbound message from WeChat first
- then retry the reply, cron delivery, or `send_message` action

### Media upload fails

Common causes:

- `cryptography` is missing
- the file is larger than the 100 MB adapter limit
- the CDN request returned no `x-encrypted-param` header

Fix:

- verify `pip install httpx cryptography`
- retry with a smaller attachment
- inspect the gateway log for the exact CDN error

### Voice messages are not transcribed

Hermes can store WeChat voice media even if local SILK transcoding tools are unavailable.

Fix:

- install a compatible `silk-decoder` or `ffmpeg` build if you want WAV conversion
- otherwise rely on WeChat's built-in voice-to-text when it is present in the inbound payload

### Session pauses for one hour

This happens when WeChat returns session-expiry `errcode -14`.

Fix:

- wait for the cooldown to expire, or
- re-run `python3 scripts/weixin_login.py` to refresh the bot token if the session is no longer valid
