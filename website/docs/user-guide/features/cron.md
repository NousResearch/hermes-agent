---
sidebar_position: 5
title: "Scheduled Tasks (Cron)"
description: "Schedule automated tasks with natural language — cron jobs, delivery options, and the gateway scheduler"
---

# Scheduled Tasks (Cron)

Schedule tasks to run automatically with natural language or cron expressions. The agent can self-schedule using the `schedule_cronjob` tool from any platform.

## Creating Scheduled Tasks

### In the CLI

Use the `/cron` slash command:

```
/cron add 30m "Remind me to check the build"
/cron add "every 2h" "Check server status"
/cron add "0 9 * * *" "Morning briefing"
/cron list
/cron remove <job_id>
```

### Through Natural Conversation

Simply ask the agent on any platform:

```
Every morning at 9am, check Hacker News for AI news and send me a summary on Telegram.
```

The agent will use the `schedule_cronjob` tool to set it up.

## How It Works

**Cron execution is handled by the gateway daemon.** The gateway ticks the scheduler every 60 seconds, running any due jobs in isolated agent sessions:

```bash
hermes gateway install     # Install as system service (recommended)
hermes gateway             # Or run in foreground

hermes cron list           # View scheduled jobs
hermes cron status         # Check if gateway is running
```

:::info
Even if no messaging platforms are configured, the gateway stays running for cron. A file lock prevents duplicate execution if multiple processes overlap.
:::

## Delivery Options

When scheduling jobs, you specify where the output goes:

| Option | Description |
|--------|-------------|
| `"origin"` | Back to where the job was created |
| `"local"` | Save to local files only |
| `"telegram"` | Telegram home channel |
| `"discord"` | Discord home channel |
| `"telegram:123456"` | Specific Telegram chat |

The agent knows your connected platforms and home channels — it'll choose sensible defaults.

## Schedule Formats

- **Relative:** `30m`, `2h`, `1d`
- **Human-readable:** `"every 2 hours"`, `"daily at 9am"`
- **Cron expressions:** `"0 9 * * *"` (standard 5-field cron syntax)

## Managing Jobs

```bash
# CLI commands
hermes cron list           # View all scheduled jobs
hermes cron status         # Check if the scheduler is running

# Slash commands (inside chat)
/cron list
/cron remove <job_id>
```

## Security

:::warning
Scheduled task prompts are scanned for instruction-override patterns (prompt injection). Jobs with suspicious content are blocked.
:::
