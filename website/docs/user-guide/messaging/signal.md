---
sidebar_position: 6
title: "Signal"
description: "Set up Hermes Agent as a Signal messenger bot via signal-cli daemon"
---

# Signal Setup

Hermes connects to Signal through the [signal-cli](https://github.com/AsamK/signal-cli) daemon running in HTTP mode. The adapter streams messages in real-time via SSE (Server-Sent Events) and sends responses via JSON-RPC.

Signal is the most privacy-focused mainstream messenger — end-to-end encrypted by default, open-source protocol, minimal metadata collection. This makes it ideal for security-sensitive agent workflows.

:::info No New Python Dependencies
The Signal adapter uses `httpx` (already a core Hermes dependency) for all communication. No additional Python packages are required. You just need signal-cli installed externally.
:::

---

## Prerequisites

- **signal-cli** — Java-based Signal client ([GitHub](https://github.com/AsamK/signal-cli))
- **Java 17+** runtime — required by signal-cli
- **A phone number** with Signal installed (for linking as a secondary device)

### Installing signal-cli

```bash
# Linux (Debian/Ubuntu)
sudo apt install signal-cli

# macOS
brew install signal-cli

# Manual install (any platform)
# Download from https://github.com/AsamK/signal-cli/releases
# Extract and add to PATH
```

### Alternative: Docker (signal-cli-rest-api)

If you prefer Docker, use the [signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) container:

```bash
docker run -d --name signal-cli \
  -p 8080:8080 \
  -v $HOME/.local/share/signal-cli:/home/.local/share/signal-cli \
  -e MODE=json-rpc \
  bbernhard/signal-cli-rest-api
```

:::tip
Use `MODE=json-rpc` for best performance. The `normal` mode spawns a JVM per request and is much slower.
:::

---

## Step 1: Link Your Signal Account

Signal-cli works as a **linked device** — like WhatsApp Web, but for Signal. Your phone stays the primary device.

```bash
# Generate a linking URI (displays a QR code or link)
signal-cli link -n "HermesAgent"
```

1. Open **Signal** on your phone
2. Go to **Settings → Linked Devices**
3. Tap **Link New Device**
4. Scan the QR code or enter the URI

---

## Step 2: Start the signal-cli Daemon

```bash
# Replace +1234567890 with your Signal phone number (E.164 format)
signal-cli --account +1234567890 daemon --http 127.0.0.1:8080
```

:::tip
Keep this running in the background. You can use `systemd`, `tmux`, `screen`, or run it as a service.
:::

Verify it's running:

```bash
curl http://127.0.0.1:8080/api/v1/check
# Should return: {"versions":{"signal-cli":...}}
```

---

## Step 3: Configure Hermes

The easiest way:

```bash
hermes gateway setup
```

Select **Signal** from the platform menu. The wizard will:

1. Check if signal-cli is installed
2. Prompt for the HTTP URL (default: `http://127.0.0.1:8080`)
3. Test connectivity to the daemon
4. Ask for your account phone number
5. Configure allowed users and access policies

### Manual Configuration

Add to `~/.hermes/.env`:

```bash
# Required
SIGNAL_HTTP_URL=http://127.0.0.1:8080
SIGNAL_ACCOUNT=+1234567890

# Security (at least one is recommended)
SIGNAL_ALLOWED_USERS=+1234567890,+0987654321    # Comma-separated E.164 numbers or UUIDs
SIGNAL_DM_POLICY=pairing                          # pairing | allowlist | open
SIGNAL_GROUP_POLICY=disabled                      # disabled | allowlist | open

# Optional
SIGNAL_GROUP_ALLOWED_USERS=groupId1,groupId2     # For group_policy=allowlist
SIGNAL_HOME_CHANNEL=+1234567890                  # Default delivery target for cron jobs
SIGNAL_IGNORE_STORIES=true                       # Ignore Signal story messages
SIGNAL_DEBUG=false                                # Enable verbose Signal debug logging
```

Then start the gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a system service
```

---

## Access Policies

### DM Policy

| Policy | Behavior |
|--------|----------|
| `pairing` (default) | Unknown users get a one-time pairing code. You approve via `hermes pairing approve signal CODE`. |
| `allowlist` | Only users in `SIGNAL_ALLOWED_USERS` can message. Others are silently ignored. |
| `open` | Anyone can message the bot. Use with caution — the bot has terminal access. |

### Group Policy

| Policy | Behavior |
|--------|----------|
| `disabled` (default) | All group messages are ignored. The bot only responds to DMs. |
| `allowlist` | Only groups in `SIGNAL_GROUP_ALLOWED_USERS` are monitored. |
| `open` | The bot responds in any group it's a member of. |

---

## Features

### Attachments

The adapter supports sending and receiving:

- **Images** — PNG, JPEG, GIF, WebP (auto-detected via magic bytes)
- **Audio** — MP3, OGG, WAV, M4A (voice messages transcribed if Whisper is configured)
- **Documents** — PDF, ZIP, and other file types

Attachment size limit: **100 MB**.

### Typing Indicators

The bot sends typing indicators while processing messages, refreshing every 8 seconds.

### Phone Number Redaction

All phone numbers are automatically redacted in logs:
- `+15551234567` → `+155****4567`
- This applies to both Hermes gateway logs and the global redaction system

### Health Monitoring

The adapter monitors the SSE connection and automatically reconnects if:
- The connection drops (with exponential backoff: 2s → 60s)
- No activity is detected for 120 seconds (pings signal-cli to verify)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Cannot reach signal-cli"** during setup | Ensure signal-cli daemon is running: `signal-cli --account +YOUR_NUMBER daemon --http 127.0.0.1:8080` |
| **Messages not received** | Check that `SIGNAL_ALLOWED_USERS` includes the sender's number in E.164 format (with `+` prefix) |
| **"signal-cli not found on PATH"** | Install signal-cli and ensure it's in your PATH, or use Docker |
| **Connection keeps dropping** | Check signal-cli logs for errors. Ensure Java 17+ is installed. |
| **Group messages ignored** | `SIGNAL_GROUP_POLICY` defaults to `disabled`. Set to `allowlist` or `open`. |
| **Bot responds to everyone** | Set `SIGNAL_DM_POLICY=pairing` or `allowlist` and configure `SIGNAL_ALLOWED_USERS` |
| **Duplicate messages** | Ensure only one signal-cli instance is listening on your phone number |

---

## Security

:::warning
**Always configure access controls.** The bot has terminal access by default. Without `SIGNAL_ALLOWED_USERS` or DM pairing, the gateway denies all incoming messages as a safety measure.
:::

- Phone numbers are redacted in all log output
- Use `SIGNAL_DM_POLICY=pairing` (default) for safe onboarding of new users
- Keep groups disabled unless you specifically need group support
- Signal's end-to-end encryption protects message content in transit
- The signal-cli session data in `~/.local/share/signal-cli/` contains account credentials — protect it like a password

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SIGNAL_HTTP_URL` | Yes | — | signal-cli HTTP endpoint |
| `SIGNAL_ACCOUNT` | Yes | — | Bot phone number (E.164) |
| `SIGNAL_ALLOWED_USERS` | No | — | Comma-separated phone numbers/UUIDs |
| `SIGNAL_ALLOW_ALL_USERS` | No | `false` | Allow all users (dangerous) |
| `SIGNAL_DM_POLICY` | No | `pairing` | DM access policy |
| `SIGNAL_GROUP_POLICY` | No | `disabled` | Group message policy |
| `SIGNAL_GROUP_ALLOWED_USERS` | No | — | Allowed group IDs |
| `SIGNAL_HOME_CHANNEL` | No | — | Default delivery target |
| `SIGNAL_IGNORE_STORIES` | No | `true` | Ignore story messages |
| `SIGNAL_DEBUG` | No | `false` | Debug logging (Signal module only) |
