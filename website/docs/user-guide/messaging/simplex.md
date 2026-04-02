---
sidebar_position: 17
title: "SimpleX"
description: "Set up Hermes Agent as a SimpleX Chat bot via simplex-chat WebSocket API"
---

# SimpleX Setup

Hermes connects to SimpleX Chat through the [simplex-chat](https://simplex.chat/) terminal app running with its WebSocket API enabled. The adapter streams messages in real-time via WebSocket JSON events and sends responses via WebSocket JSON commands.

SimpleX Chat is the first messenger with no user identifiers — not even random numbers. It uses pairwise per-queue identifiers and routes messages through relay servers, providing strong metadata privacy. This makes it an excellent choice for privacy-sensitive agent deployments.

:::info Dependency
The SimpleX adapter requires the `websockets` Python package. Install it with: `pip install websockets` (or `uv pip install websockets`).
:::

---

## Prerequisites

- **simplex-chat** — SimpleX Chat terminal client ([GitHub](https://github.com/simplex-chat/simplex-chat), [Downloads](https://simplex.chat/downloads/))
- **websockets** Python package — `pip install websockets`

### Installing simplex-chat

```bash
# Linux (static binary)
curl -L https://github.com/simplex-chat/simplex-chat/releases/latest/download/simplex-chat-ubuntu-22_04-x86-64 -o simplex-chat
chmod +x simplex-chat
sudo mv simplex-chat /usr/local/bin/

# Arch Linux (AUR)
yay -S simplex-chat-bin

# macOS
brew install simplex-chat
```

---

## Step 1: Create a SimpleX Identity

SimpleX doesn't use phone numbers, emails, or usernames. Each instance has its own identity, created on first run.

```bash
# Start simplex-chat interactively to create an identity
simplex-chat

# At the prompt, set a display name:
> /profile HermesAgent

# Generate a one-time invitation link for connecting:
> /connect
# This prints a simplex:// URI — share it with users who should be able to talk to the bot

# Exit interactive mode
> /quit
```

---

## Step 2: Start simplex-chat with WebSocket API

```bash
# Start with WebSocket API on port 5225
simplex-chat -p 5225
```

:::tip
Keep this running in the background. Use `systemd`, `tmux`, or `screen`. For production, create a systemd service:

```ini
[Unit]
Description=SimpleX Chat WebSocket API
After=network.target

[Service]
Type=simple
User=hermes
ExecStart=/usr/local/bin/simplex-chat -p 5225
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
:::

---

## Step 3: Configure Hermes

The easiest way:

```bash
hermes gateway setup
```

Select **SimpleX** from the platform menu. The wizard will prompt for the WebSocket URL and configure access settings.

### Manual Configuration

Add to `~/.hermes/.env`:

```bash
# Required
SIMPLEX_WS_URL=ws://127.0.0.1:5225

# Security (recommended)
SIMPLEX_ALLOWED_USERS=42,108                     # Comma-separated contact IDs
# Or allow all contacts:
# SIMPLEX_ALLOW_ALL_USERS=true

# Optional
SIMPLEX_AUTO_ACCEPT=true                          # Auto-accept incoming contact requests (default: true)
SIMPLEX_GROUP_ALLOWED=99,201                      # Group IDs to monitor, or * for all (omit to disable)
SIMPLEX_HOME_CHANNEL=42                           # Default delivery target for cron jobs
SIMPLEX_HOME_CHANNEL_NAME=Home                    # Display name for the home channel
```

Then start the gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

---

## Access Control

### DM Access

DM access follows the same pattern as all other Hermes platforms:

1. **`SIMPLEX_ALLOWED_USERS` set** — only those contact IDs can message
2. **No allowlist set** — unknown users get a DM pairing code (approve via `hermes pairing approve simplex CODE`)
3. **`SIMPLEX_ALLOW_ALL_USERS=true`** — anyone can message (use with caution)

### Contact Requests

When `SIMPLEX_AUTO_ACCEPT=true` (the default), the bot automatically accepts incoming contact requests. Set to `false` to require manual approval.

### Group Access

Group access is controlled by the `SIMPLEX_GROUP_ALLOWED` env var:

| Configuration | Behavior |
|---------------|----------|
| Not set (default) | All group messages are ignored. The bot only responds to DMs. |
| Set with group IDs | Only listed groups are monitored (e.g., `99,201`). |
| Set to `*` | The bot responds in any group it's a member of. |

---

## Features

### Attachments

The adapter supports sending and receiving:

- **Images** — PNG, JPEG, GIF, WebP (auto-detected via magic bytes)
- **Audio** — MP3, OGG, WAV, M4A (voice messages transcribed if Whisper is configured)
- **Documents** — PDF and other file types

### Health Monitoring

The adapter monitors the WebSocket connection and automatically reconnects if:
- The connection drops (with exponential backoff: 2s → 60s, with jitter)
- No activity is detected for 120 seconds (forces reconnect)

### Contact ID Redaction

Contact IDs are partially redacted in logs for privacy:
- `12345678` → `12**78`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Cannot reach simplex-chat"** during setup | Ensure simplex-chat is running with WebSocket API: `simplex-chat -p 5225` |
| **Messages not received** | Check that `SIMPLEX_ALLOWED_USERS` includes the sender's contact ID, or enable `SIMPLEX_ALLOW_ALL_USERS` |
| **"websockets not installed"** | Install the dependency: `pip install websockets` |
| **Connection keeps dropping** | Check simplex-chat logs. Ensure the WebSocket port isn't blocked by a firewall. |
| **Group messages ignored** | Configure `SIMPLEX_GROUP_ALLOWED` with specific group IDs, or `*` to allow all groups. |
| **Bot responds to no one** | Configure `SIMPLEX_ALLOWED_USERS`, use DM pairing, or set `SIMPLEX_ALLOW_ALL_USERS=true`. |
| **Contact requests not accepted** | Ensure `SIMPLEX_AUTO_ACCEPT=true` (default) or manually accept via simplex-chat CLI. |

---

## Security

:::warning
**Always configure access controls.** The bot has terminal access by default. Without `SIMPLEX_ALLOWED_USERS` or DM pairing, the gateway denies all incoming messages as a safety measure.
:::

- Contact IDs are redacted in all log output
- Use DM pairing or explicit allowlists for safe onboarding of new users
- Keep groups disabled unless you specifically need group support
- SimpleX provides strong metadata privacy — no user identifiers, pairwise connections, onion routing available
- The simplex-chat data directory (`~/.simplex/`) contains identity keys — protect it like a password

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SIMPLEX_WS_URL` | Yes | — | simplex-chat WebSocket endpoint (e.g., `ws://127.0.0.1:5225`) |
| `SIMPLEX_ALLOWED_USERS` | No | — | Comma-separated contact IDs allowed to interact |
| `SIMPLEX_ALLOW_ALL_USERS` | No | `false` | Allow any contact to interact (skip allowlist) |
| `SIMPLEX_AUTO_ACCEPT` | No | `true` | Auto-accept incoming contact requests |
| `SIMPLEX_GROUP_ALLOWED` | No | — | Group IDs to monitor, or `*` for all (omit to disable groups) |
| `SIMPLEX_HOME_CHANNEL` | No | — | Default delivery target for cron jobs |
| `SIMPLEX_HOME_CHANNEL_NAME` | No | `Home` | Display name for the home channel |
