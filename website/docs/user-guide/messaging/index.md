---
sidebar_position: 1
title: "Messaging Gateway"
description: "Chat with Hermes from Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant, or your browser — architecture and setup overview"
---

# Messaging Gateway

Chat with Hermes from Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant, or your browser. The gateway is a single background process that connects to all your configured platforms, handles sessions, runs cron jobs, and delivers voice messages.

For the full voice feature set — including CLI microphone mode, spoken replies in messaging, and Discord voice-channel conversations — see [Voice Mode](/docs/user-guide/features/voice-mode) and [Use Voice Mode with Hermes](/docs/guides/use-voice-mode-with-hermes).

## Architecture

```mermaid
flowchart TB
    subgraph Gateway["Hermes Gateway"]
        subgraph Adapters["Platform adapters"]
            tg[Telegram]
            dc[Discord]
            wa[WhatsApp]
            sl[Slack]
            sig[Signal]
            em[Email]
            ha[Home Assistant]
        end

        store["Session store<br/>per chat"]
        agent["AIAgent<br/>run_agent.py"]
        cron["Cron scheduler<br/>ticks every 60s"]
    end

    tg --> store
    dc --> store
    wa --> store
    sl --> store
    sig --> store
    em --> store
    ha --> store
    store --> agent
    cron --> store
```

Each platform adapter receives messages, routes them through a per-chat session store, and dispatches them to the AIAgent for processing. The gateway also runs the cron scheduler, ticking every 60 seconds to execute any due jobs.

## Quick Setup

The easiest way to configure messaging platforms is the interactive wizard:

```bash
hermes gateway setup        # Interactive setup for all messaging platforms
```

This walks you through configuring each platform with arrow-key selection, shows which platforms are already configured, and offers to start/restart the gateway when done.

## Gateway Commands

```bash
hermes gateway              # Run in foreground
hermes gateway setup        # Configure messaging platforms interactively
hermes gateway install      # Install as a user service (Linux) / launchd service (macOS)
sudo hermes gateway install --system   # Linux only: install a boot-time system service
hermes gateway start        # Start the default service
hermes gateway stop         # Stop the default service
hermes gateway status       # Check default service status
hermes gateway status --system         # Linux only: inspect the system service explicitly
```

## Chat Commands (Inside Messaging)

| Command | Description |
|---------|-------------|
| `/new` or `/reset` | Start a fresh conversation |
| `/model [provider:model]` | Show or change the model (supports `provider:model` syntax) |
| `/provider` | Show available providers with auth status |
| `/personality [name]` | Set a personality |
| `/retry` | Retry the last message |
| `/undo` | Remove the last exchange |
| `/status` | Show session info |
| `/stop` | Stop the running agent |
| `/sethome` | Set this chat as the home channel |
| `/compress` | Manually compress conversation context |
| `/title [name]` | Set or show the session title |
| `/resume [name]` | Resume a previously named session |
| `/usage` | Show token usage for this session |
| `/insights [days]` | Show usage insights and analytics |
| `/reasoning [level\|show\|hide]` | Change reasoning effort or toggle reasoning display |
| `/voice [on\|off\|tts\|join\|leave\|status]` | Control messaging voice replies and Discord voice-channel behavior |
| `/rollback [number]` | List or restore filesystem checkpoints |
| `/background <prompt>` | Run a prompt in a separate background session |
| `/reload-mcp` | Reload MCP servers from config |
| `/update` | Update Hermes Agent to the latest version |
| `/help` | Show available commands |
| `/<skill-name>` | Invoke any installed skill |

## Session Management

### Session Persistence

Sessions persist across messages until they reset. The agent remembers your conversation context.

### Reset Policies

Sessions reset based on configurable policies:

| Policy | Default | Description |
|--------|---------|-------------|
| Daily | 4:00 AM | Reset at a specific hour each day |
| Idle | 1440 min | Reset after N minutes of inactivity |
| Both | (combined) | Whichever triggers first |

Configure per-platform overrides in `~/.hermes/gateway.json`:

```json
{
  "reset_by_platform": {
    "telegram": { "mode": "idle", "idle_minutes": 240 },
    "discord": { "mode": "idle", "idle_minutes": 60 }
  }
}
```

## Security

**By default, the gateway denies all users who are not in an allowlist or paired via DM.** This is the safe default for a bot with terminal access.

```bash
# Restrict to specific users (recommended):
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=123456789012345678
SIGNAL_ALLOWED_USERS=+155****4567,+155****6543
EMAIL_ALLOWED_USERS=trusted@example.com,colleague@work.com

# Or allow
GATEWAY_ALLOWED_USERS=123456789,987654321

# Or explicitly allow all users (NOT recommended for bots with terminal access):
GATEWAY_ALLOW_ALL_USERS=true
```

### DM Pairing (Alternative to Allowlists)

Instead of manually configuring user IDs, unknown users receive a one-time pairing code when they DM the bot:

```bash
# The user sees: "Pairing code: XKGH5N7P"
# You approve them with:
hermes pairing approve telegram XKGH5N7P

# Other pairing commands:
hermes pairing list          # View pending + approved users
hermes pairing revoke telegram 123456789  # Remove access
```

Pairing codes expire after 1 hour, are rate-limited, and use cryptographic randomness.

## Interrupting the Agent

Send any message while the agent is working to interrupt it. Key behaviors:

- **In-progress terminal commands are killed immediately** (SIGTERM, then SIGKILL after 1s)
- **Tool calls are cancelled** — only the currently-executing one runs, the rest are skipped
- **Multiple messages are combined** — messages sent during interruption are joined into one prompt
- **`/stop` command** — interrupts without queuing a follow-up message

## Tool Progress Notifications

Control how much tool activity is displayed in `~/.hermes/config.yaml`:

```yaml
display:
  tool_progress: all    # off | new | all | verbose
```

When enabled, the bot sends status messages as it works:

```text
💻 `ls -la`...
🔍 web_search...
📄 web_extract...
🐍 execute_code...
```

## Service Management

### Linux (systemd)

```bash
hermes gateway install               # Install as user service
systemctl --user start hermes-gateway
systemctl --user stop hermes-gateway
systemctl --user status hermes-gateway
journalctl --user -u hermes-gateway -f

# Enable lingering (keeps running after logout)
sudo loginctl enable-linger $USER

# Or install a boot-time system service that still runs as your user
sudo hermes gateway install --system
sudo hermes gateway start --system
sudo hermes gateway status --system
journalctl -u hermes-gateway -f
```

Use the user service on laptops and dev boxes. Use the system service on VPS or headless hosts that should come back at boot without relying on systemd linger.

Avoid keeping both the user and system gateway units installed at once unless you really mean to. Hermes will warn if it detects both because start/stop/status behavior gets ambiguous.

### macOS (launchd)

```bash
hermes gateway install
launchctl start ai.hermes.gateway
launchctl stop ai.hermes.gateway
tail -f ~/.hermes/logs/gateway.log
```

## Platform-Specific Toolsets

Each platform has its own toolset:

| Platform | Toolset | Capabilities |
|----------|---------|--------------|
| CLI | `hermes-cli` | Full access |
| Telegram | `hermes-telegram` | Full tools including terminal |
| Discord | `hermes-discord` | Full tools including terminal |
| WhatsApp | `hermes-whatsapp` | Full tools including terminal |
| Slack | `hermes-slack` | Full tools including terminal |
| Signal | `hermes-signal` | Full tools including terminal |
| Email | `hermes-email` | Full tools including terminal |
| Home Assistant | `hermes-homeassistant` | Full tools + HA device control (ha_list_entities, ha_get_state, ha_call_service, ha_list_services) |

## Next Steps

- [Telegram Setup](telegram.md)
- [Discord Setup](discord.md)
- [Slack Setup](slack.md)
- [WhatsApp Setup](whatsapp.md)
- [Signal Setup](signal.md)
- [Email Setup](email.md)
- [Home Assistant Integration](homeassistant.md)
