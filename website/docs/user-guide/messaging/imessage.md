---
sidebar_position: 16
title: "iMessage"
description: "Set up Hermes Agent as a native iMessage bot via the imsg CLI on macOS"
---

# iMessage Setup

Hermes connects to iMessage natively through the [imsg](https://github.com/steipete/imsg) CLI tool, which provides direct access to macOS Messages.app via AppleScript and the chat.db database. No third-party server or bridge required — just your Mac.

:::info macOS Only
The iMessage adapter requires macOS. It uses `imsg` to interface with Messages.app, which is only available on Apple platforms.
:::

---

## Prerequisites

- **macOS** — any recent version with Messages.app
- **imsg CLI** — lightweight CLI for Apple Messages ([GitHub](https://github.com/steipete/imsg))
- **Full Disk Access** — your terminal app needs permission to read `~/Library/Messages/chat.db`

### Installing imsg

```bash
brew install steipete/tap/imsg
```

Verify it works:

```bash
imsg chats --json
```

If you see a JSON list of your chats, you're ready. If you get a permissions error, grant **Full Disk Access** to your terminal app (see Troubleshooting below).

---

## Step 1: Grant Permissions

1. Open **System Settings → Privacy & Security → Full Disk Access**
2. Add your terminal app (Terminal.app, iTerm2, Alacritty, etc.)
3. Restart your terminal

---

## Step 2: Configure Hermes

The easiest way:

```bash
hermes gateway setup
```

Select **iMessage** from the platform menu. The wizard will:

1. Check if `imsg` is installed
2. Verify Full Disk Access permissions
3. Ask for allowed users (phone numbers or Apple IDs)
4. Configure the home channel for notifications

### Manual Configuration

Add to `~/.hermes/.env`:

```bash
# Required
IMESSAGE_ENABLED=true

# Security (recommended)
IMESSAGE_ALLOWED_USERS=+15551234567,user@icloud.com    # Comma-separated phone numbers or Apple IDs

# Optional
IMESSAGE_HOME_CHANNEL=+15551234567                     # Default delivery target for cron jobs
IMESSAGE_HOME_CHANNEL_NAME=My iPhone                   # Display name for the home channel
IMESSAGE_WATCH_CHAT_IDS=1,5,12                         # Only watch specific chats (omit for all)
IMESSAGE_WATCH_MODE=auto                               # auto | fsevents | poll
IMESSAGE_POLL_INTERVAL=3.0                             # Seconds between polls (poll mode only)
```

Then start the gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a launchd service (macOS)
```

---

## Watch Modes

The adapter supports three modes for detecting new messages:

| Mode | Description | When to Use |
|------|-------------|-------------|
| `auto` (default) | Tries FSEvents first, falls back to poll if no events detected after 30s | Recommended for most setups |
| `fsevents` | Watches `chat.db` via macOS filesystem events | Best for real-time; may not work in all environments |
| `poll` | Periodically queries for new messages via `imsg watch --since-rowid` | Most reliable; slightly higher latency |

Set the mode with `IMESSAGE_WATCH_MODE`. If you're experiencing missed messages with the default `auto` mode, try `poll`.

---

## Access Control

DM access follows the standard Hermes pattern:

1. **`IMESSAGE_ALLOWED_USERS` set** → only those users can message
2. **No allowlist set** → unknown users get a DM pairing code (approve via `hermes pairing approve imessage CODE`)
3. **`IMESSAGE_ALLOW_ALL_USERS=true`** → anyone can message (use with caution)

---

## Features

### Attachments

**Incoming** (user → agent):
- **Images** — PNG, JPEG, GIF, WebP
- **Audio** — MP3, M4A, WAV, OGG
- **Video** — MP4, MOV
- **Documents** — PDF, ZIP, and other file types

**Outgoing** (agent → user):
- **Files** — sent via `imsg send --file` as native iMessage attachments

### Phone Number & Apple ID Redaction

All identifiers are automatically redacted in logs:
- `+15551234567` → `+155****4567`
- `user@example.com` → `us****@example.com`

### Health Monitoring

The adapter monitors the watch process and automatically reconnects if:
- The process exits unexpectedly (with exponential backoff: 2s → 60s)
- No activity is detected for 120 seconds (logs a debug message)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"imsg CLI not found"** | Install via `brew install steipete/tap/imsg` and ensure it's in your PATH |
| **"imsg chats failed"** | Grant Full Disk Access to your terminal: System Settings → Privacy & Security → Full Disk Access |
| **Messages not received** | Try `IMESSAGE_WATCH_MODE=poll` — FSEvents may not fire in all environments |
| **Messages replayed on restart** | Normal for the first poll cycle if chat history is large; dedup prevents double-processing |
| **Bot responds to no one** | Set `IMESSAGE_ALLOWED_USERS` with phone numbers or Apple IDs, or use DM pairing |
| **Slow message detection** | Decrease `IMESSAGE_POLL_INTERVAL` (default: 3.0 seconds) |

---

## Limitations

- **macOS only** — requires Messages.app and the `imsg` CLI
- **No typing indicators** — iMessage doesn't expose typing status via AppleScript
- **No reactions** — emoji reactions are not supported
- **No threaded conversations** — iMessage threads are not mapped

---

## Security

:::warning
**Always configure access controls.** The bot has terminal access by default. Without `IMESSAGE_ALLOWED_USERS` or DM pairing, the gateway denies all incoming messages as a safety measure.
:::

- Phone numbers and Apple IDs are redacted in all log output
- Use DM pairing or explicit allowlists for safe onboarding of new users
- The `chat.db` file contains your full message history — Full Disk Access grants read access to it

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `IMESSAGE_ENABLED` | Yes | `false` | Enable the iMessage adapter |
| `IMESSAGE_ALLOWED_USERS` | No | — | Comma-separated phone numbers or Apple IDs |
| `IMESSAGE_ALLOW_ALL_USERS` | No | `false` | Allow any user to interact (skip allowlist) |
| `IMESSAGE_HOME_CHANNEL` | No | — | Default delivery target for cron jobs |
| `IMESSAGE_HOME_CHANNEL_NAME` | No | — | Display name for the home channel |
| `IMESSAGE_WATCH_CHAT_IDS` | No | — | Comma-separated chat IDs to watch (omit for all) |
| `IMESSAGE_WATCH_MODE` | No | `auto` | Watch mode: `auto`, `fsevents`, or `poll` |
| `IMESSAGE_POLL_INTERVAL` | No | `3.0` | Seconds between polls (poll mode only) |
