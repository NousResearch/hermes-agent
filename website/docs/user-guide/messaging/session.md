---
sidebar_position: 11
title: "Session"
description: "Set up Hermes Agent as a Session Protocol bot via the built-in Node.js bridge"
---

# Session Setup

[Session](https://getsession.org/) is an open source private messaging app that protects your metadata, encrypts your communications, and makes sure your messaging activities leave no digital trail behind. It is a fully decentralised, end-to-end encrypted messenger built on the Oxen network with onion-like routing. There are no phone numbers, no central servers to trust, and no account sign-up — your identity is a cryptographic key pair.

Hermes connects to Session through a built-in Node.js bridge that depends on the `@bonesgit/session-desktop-library`. 



## Prerequisites

- **Node.js 24.12.0 or newer** — the Session bridge runs as a Node.js child process managed by the gateway
- **npm** — the setup wizard runs `npm install` automatically in `scripts/session-bridge/` the first time you configure Session

### Install Node.js

```bash
# Linux (via nvm — recommended)
nvm install 24
nvm use 24

# macOS
brew install node

# Check version
node --version   # must be >= 24.12.0
```


## Step 1: Run the Setup Wizard

```bash
hermes gateway setup
```

Select **Session** from the platform menu. The wizard:

1. Installs bridge npm dependencies (`npm install`)
2. Probes the data directory with `--check` mode to detect any pre-existing registered account (without creating one)
3. Asks whether you want to restore an existing mnemonic or create a new account
4. Asks for your Session ID (to set as home channel and initial allowed user)
5. Runs `--setup` mode to create or restore the account

### New account (no existing mnemonic)

The wizard runs the bridge in setup mode, creates a fresh account, and saves the mnemonic to `~/.hermes/.env`. On success a note that the gateway does not need the mnemonic at runtime — you may delete it from `.env` after backing it up securely.

### Restore existing account

If you already have a Session mnemonic (e.g. from a previous install), enter it when prompted. Input is masked. The mnemonic is saved to `.env` and the account is restored on the next gateway start — you may delete it from `.env` after backing it up securely.

### Pre-existing data directory

If `SESSION_DATA_PATH` already contains a registered account, the wizard detects it via `--check` and warns you. If the bot ID in the data dir differs from `SESSION_BOT_ID` in `.env`, the wizard corrects the `.env` value automatically. To restore a *different* account you must wipe the data directory first.

In all cases the wizard also sets:
- `SESSION_HOME_CHANNEL` — your own Session ID (where unprompted cron messages go)
- `SESSION_ALLOWED_USERS` — seeded to your own Session ID so only you can DM the bot initially
- `SESSION_BOT_ID` — the bot's own Session ID, resolved during setup and stored for reference


## Step 2: Configure Hermes

The wizard handles the required settings automatically. For manual configuration, add to `~/.hermes/.env`:

```bash
# Required
SESSION_MNEMONIC="word1 word2 ... word13"   # 13-word mnemonic — never share this
                                             # (only needed during --setup; gateway does not require it at runtime)

# Security
SESSION_ALLOWED_USERS=05abc...,05def...     # Comma-separated Session IDs allowed to DM the bot
# OR allow all users (use with caution):
SESSION_ALLOW_ALL_USERS=false

# Optional
SESSION_HOME_CHANNEL=05abc...               # Default delivery target for cron jobs
SESSION_HOME_CHANNEL_NAME=Home
SESSION_BOT_NAME=Hermes                     # Display name used for @mention detection in groups
SESSION_DATA_PATH=/home/you/.hermes/session-data   # Where Session DB and keys are stored
SESSION_BRIDGE_PORT=8095                    # HTTP port for the Node.js bridge (default: 8095)
SESSION_LOG_LEVEL=warn                      # Bridge log verbosity
SESSION_STARTUP_TIMEOUT=15                  # Seconds to wait for bridge ready on startup
```

Then start the gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

The gateway spawns the Session bridge automatically on startup.


## Access Control

### DM Access

| Configuration | Behavior |
|---------------|----------|
| `SESSION_ALLOWED_USERS` set | Only listed Session IDs can DM the bot |
| Not set | Unknown DM senders receive a pairing code (approve via `hermes pairing approve session CODE`) |
| `SESSION_ALLOW_ALL_USERS=true` | Anyone can DM (use with caution) |

### Group Access

The bot only responds in groups when **@mentioned** by name or by its Session ID:

```
@Hermes what's the weather today?
@05abc1234... summarise this thread
```

Messages without an @mention are silently ignored.


## Features

### Messaging

- **DMs** — full bidirectional text messaging
- **Groups** — responds to @mentions only (by display name or Session ID)
- **Replies/quotes** — the bot quotes the message it's responding to when `reply_to` is set

### Attachments

The adapter supports sending and receiving:

- **Images** — PNG, JPEG, GIF, WebP
- **Audio** — OGG, MP3, WAV, M4A (voice messages transcribed if Whisper is configured)
- **Video** — MP4 and other video formats
- **Documents** — PDF, ZIP, and other file types

### Typing Indicators

The bot sends a typing indicator while processing each message and clears it when the reply is sent. Only for DM, groups ignore this.

### Contact Requests

When a Session ID sends its first message, the bridge emits an `isIncomingRequest` event. The adapter automatically accepts the contact request so the conversation can proceed — access control (whether the user can actually interact with the bot) is enforced by the gateway's `SESSION_ALLOWED_USERS` / `SESSION_ALLOW_ALL_USERS` check.


## Runtime Commands (Tier 2)

The bridge exposes additional endpoints beyond the core messaging pipeline. The agent can invoke these at runtime via the `terminal` tool and `curl`:

:::note
`SESSION_BOT_NAME` is applied automatically as the display name each time the gateway connects. The `set-display-name` curl command is only needed if you want to change the name at runtime without restarting the gateway.
:::

```bash
# Set bot avatar image (path must be accessible to the bridge process)
curl -s -X POST http://127.0.0.1:8095/set-display-image \
  -H 'Content-Type: application/json' \
  -d '{"imagePath": "/home/you/.hermes/avatar.png"}'

# Set bot display name at runtime (normally set automatically from SESSION_BOT_NAME)
curl -s -X POST http://127.0.0.1:8095/set-display-name \
  -H 'Content-Type: application/json' \
  -d '{"name": "Hermes"}'

# Send a reaction
curl -s -X POST http://127.0.0.1:8095/react \
  -H 'Content-Type: application/json' \
  -d '{"conversationId": "05abc...", "messageDbId": 42, "emoji": "👍"}'

# Create a group
curl -s -X POST http://127.0.0.1:8095/create-group \
  -H 'Content-Type: application/json' \
  -d '{"name": "My Group", "members": ["05abc...", "05def..."]}'

# Add members to a group
curl -s -X POST http://127.0.0.1:8095/add-group-members \
  -H 'Content-Type: application/json' \
  -d '{"groupId": "03abc...", "sessionIds": ["05abc..."], "withHistory": false}'

# Remove members from a group
curl -s -X POST http://127.0.0.1:8095/remove-group-members \
  -H 'Content-Type: application/json' \
  -d '{"groupId": "03abc...", "sessionIds": ["05abc..."]}'

# Promote members to admin
curl -s -X POST http://127.0.0.1:8095/promote-group-members \
  -H 'Content-Type: application/json' \
  -d '{"groupId": "03abc...", "memberIds": ["05abc..."]}'

# Leave a group
curl -s -X POST http://127.0.0.1:8095/leave-group \
  -H 'Content-Type: application/json' \
  -d '{"groupId": "03abc..."}'

# Block a contact
curl -s -X POST http://127.0.0.1:8095/block-contact \
  -H 'Content-Type: application/json' \
  -d '{"sessionId": "05abc..."}'

# Unblock a contact
curl -s -X POST http://127.0.0.1:8095/unblock-contact \
  -H 'Content-Type: application/json' \
  -d '{"sessionId": "05abc..."}'
```

The default bridge port is `8095`, configurable via `SESSION_BRIDGE_PORT`.

:::tip Session ID format
DM Session IDs start with `05` (e.g. `05abc...`). Group IDs start with `03` (e.g. `03abc...`). The bridge uses this prefix to determine conversation type automatically.
:::


## Session Data & Security

Session data (account keys, message DB) is stored at `SESSION_DATA_PATH` (default: `~/.hermes/session-data`). Bridge process logs go to `~/.hermes/logs/session-bridge.log`.

:::warning
**Protect your mnemonic.** It is the only way to restore the Session account. It is stored in `~/.hermes/.env` — ensure this file has restricted permissions (`chmod 600 ~/.hermes/.env`). The mnemonic is never written to any log file by design.
:::

- **Always set `SESSION_ALLOWED_USERS`** — without it, the DM pairing flow applies to everyone who messages the bot
- Session IDs are public hex strings — safe to share, safe to log
- End-to-end encryption is handled by the Session Protocol itself; Hermes never sees plaintext on the wire


## Cron Delivery

Use `deliver: session` in a cron job to send to `SESSION_HOME_CHANNEL`, or specify a target directly:

```yaml
- cron: "0 8 * * *"
  prompt: "Good morning briefing"
  deliver: session                    # → SESSION_HOME_CHANNEL
  # deliver: session:05abc...         # → specific Session ID
```


## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Bridge won't start** | Check `~/.hermes/logs/session-bridge.log`. Ensure Node.js >= 24.12.0 is installed and in PATH. |
| **"did not become ready in Xs"** | Bridge is taking too long to sync with the Session swarm. Increase `SESSION_STARTUP_TIMEOUT` or check network connectivity. |
| **Messages not received** | Ensure `SESSION_ALLOWED_USERS` contains the sender's Session ID (starting with `05`). |
| **Bot doesn't respond in groups** | Send `@Hermes` or `@<bot Session ID>` — the bot only triggers on @mentions in groups. |
| **Bridge exits unexpectedly** | Check `session-bridge.log` for Node.js errors. The gateway marks this as a retryable fatal error and restarts automatically. |
| **Avatar not updating** | Image must be a local file accessible to the bridge process. Supported formats: JPEG, PNG, GIF, WebP. |
| **Bridge deps outdated** | Run `hermes doctor --fix` to update bridge npm packages automatically. |



## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SESSION_MNEMONIC` | Setup only | — | 13-word account mnemonic; needed for `--setup` but not required at gateway runtime, backup and remove |
| `SESSION_BOT_ID` | Yes (auto-set) | — | Bot's own Session ID; set automatically by the setup wizard; used by `check_session_requirements()` to detect if Session is configured |
| `SESSION_ALLOWED_USERS` | No | — | Comma-separated Session IDs allowed to DM the bot |
| `SESSION_ALLOW_ALL_USERS` | No | `false` | Allow any Session ID to DM the bot |
| `SESSION_HOME_CHANNEL` | No | — | Default cron delivery target (your Session ID, starts with `05`) |
| `SESSION_HOME_CHANNEL_NAME` | No | `Home` | Display name for the home channel |
| `SESSION_BOT_NAME` | No | `Hermes` | Bot display name; applied automatically on connect and used for @mention detection |
| `SESSION_DATA_PATH` | No | `~/.hermes/session-data` | Where Session DB and keys are persisted |
| `SESSION_BRIDGE_PORT` | No | `8095` | HTTP port for the Node.js bridge |
| `SESSION_LOG_LEVEL` | No | `warn` | Bridge log verbosity (`error`, `warn`, `info`, `debug`) |
| `SESSION_STARTUP_TIMEOUT` | No | `15` | Seconds to wait for bridge ready on gateway start |
