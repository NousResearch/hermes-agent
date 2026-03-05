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

### The Gateway Scheduler

The scheduler runs as a background thread inside the gateway process. On each tick (every 60 seconds):

1. It loads all jobs from `~/.hermes/cron/jobs.json`
2. Checks each enabled job's `next_run_at` against the current time
3. For each due job, spawns a fresh `AIAgent` session with the job's prompt
4. The agent runs to completion with full tool access
5. The final response is delivered to the configured target
6. The job's run count is incremented and next run time computed
7. Jobs that hit their repeat limit are auto-removed

A **file-based lock** (`~/.hermes/cron/.tick.lock`) prevents duplicate execution if multiple processes overlap (e.g., gateway + manual tick).

:::info
Even if no messaging platforms are configured, the gateway stays running for cron. A file lock prevents duplicate execution if multiple processes overlap.
:::

## Delivery Options

When scheduling jobs, you specify where the output goes:

| Option | Description | Example |
|--------|-------------|---------|
| `"origin"` | Back to where the job was created | Default on messaging platforms |
| `"local"` | Save to local files only (`~/.hermes/cron/output/`) | Default on CLI |
| `"telegram"` | Telegram home channel | Uses `TELEGRAM_HOME_CHANNEL` env var |
| `"discord"` | Discord home channel | Uses `DISCORD_HOME_CHANNEL` env var |
| `"telegram:123456"` | Specific Telegram chat by ID | For directing output to a specific chat |
| `"discord:987654"` | Specific Discord channel by ID | For directing output to a specific channel |

**How `"origin"` works:** When a job is created from a messaging platform, Hermes records the source platform and chat ID. When the job runs and deliver is `"origin"`, the output is sent back to that exact platform and chat. If origin info isn't available (e.g., job created from CLI), delivery falls back to local.

**How platform names work:** When you specify a bare platform name like `"telegram"`, Hermes first checks if the job's origin matches that platform and uses the origin chat ID. Otherwise, it falls back to the platform's home channel configured via environment variable (e.g., `TELEGRAM_HOME_CHANNEL`).

The agent's final response is automatically delivered — you do **not** need to include `send_message` in the cron prompt.

The agent knows your connected platforms and home channels — it'll choose sensible defaults.

## Schedule Formats

### Relative Delays (One-Shot)

Run once after a delay:

```
30m     → Run once in 30 minutes
2h      → Run once in 2 hours
1d      → Run once in 1 day
```

Supported units: `m`/`min`/`minutes`, `h`/`hr`/`hours`, `d`/`day`/`days`.

### Intervals (Recurring)

Run repeatedly at fixed intervals:

```
every 30m    → Every 30 minutes
every 2h     → Every 2 hours
every 1d     → Every day
```

### Cron Expressions

Standard 5-field cron syntax for precise scheduling:

```
0 9 * * *       → Daily at 9:00 AM
0 9 * * 1-5     → Weekdays at 9:00 AM
0 */6 * * *     → Every 6 hours
30 8 1 * *      → First of every month at 8:30 AM
0 0 * * 0       → Every Sunday at midnight
```

#### Cron Expression Cheat Sheet

```
┌───── minute (0-59)
│ ┌───── hour (0-23)
│ │ ┌───── day of month (1-31)
│ │ │ ┌───── month (1-12)
│ │ │ │ ┌───── day of week (0-7, 0 and 7 = Sunday)
│ │ │ │ │
* * * * *

Special characters:
  *     Any value
  ,     List separator (1,3,5)
  -     Range (1-5)
  /     Step values (*/15 = every 15)
```

:::note
Cron expressions require the `croniter` Python package. Install with `pip install croniter` if not already available.
:::

### ISO Timestamps

Run once at a specific date/time:

```
2026-03-15T09:00:00    → One-time at March 15, 2026 9:00 AM
```

## Repeat Behavior

The `repeat` parameter controls how many times a job runs:

| Schedule Type | Default Repeat | Behavior |
|--------------|----------------|----------|
| One-shot (`30m`, timestamp) | 1 (run once) | Runs once, then auto-deleted |
| Interval (`every 2h`) | Forever (`null`) | Runs indefinitely until removed |
| Cron expression | Forever (`null`) | Runs indefinitely until removed |

You can override the default:

```python
schedule_cronjob(
    prompt="...",
    schedule="every 2h",
    repeat=5  # Run exactly 5 times, then auto-delete
)
```

When a job hits its repeat limit, it is automatically removed from the job list.

## Real-World Examples

### Daily Standup Report

```
Schedule a daily standup report: Every weekday at 9am, check the GitHub
repository at github.com/myorg/myproject for:
1. Pull requests opened/merged in the last 24 hours
2. Issues created or closed
3. Any CI/CD failures on the main branch
Format as a brief standup-style summary. Deliver to telegram.
```

The agent creates:
```python
schedule_cronjob(
    prompt="Check github.com/myorg/myproject for PRs, issues, and CI status from the last 24 hours. Format as a standup report.",
    schedule="0 9 * * 1-5",
    name="Daily Standup Report",
    deliver="telegram"
)
```

### Weekly Backup Verification

```
Every Sunday at 2am, verify that backups exist in /data/backups/ for
each day of the past week. Check file sizes are > 1MB. Report any
gaps or suspiciously small files.
```

### Monitoring Alerts

```
Every 15 minutes, curl https://api.myservice.com/health and verify
it returns HTTP 200 with {"status": "ok"}. If it fails, include the
error details and response code. Deliver to telegram:123456789.
```

```python
schedule_cronjob(
    prompt="Run 'curl -s -o /dev/null -w \"%{http_code}\" https://api.myservice.com/health' and verify it returns 200. Also fetch the full response with 'curl -s https://api.myservice.com/health' and check for {\"status\": \"ok\"}. Report the result.",
    schedule="every 15m",
    name="API Health Check",
    deliver="telegram:123456789"
)
```

### Periodic Disk Usage Check

```python
schedule_cronjob(
    prompt="Check disk usage with 'df -h' and report any partitions above 80% usage. Also check Docker disk usage with 'docker system df' if Docker is installed.",
    schedule="0 8 * * *",
    name="Disk Usage Report",
    deliver="origin"
)
```

## Managing Jobs

```bash
# CLI commands
hermes cron list           # View all scheduled jobs
hermes cron status         # Check if the scheduler is running

# Slash commands (inside chat)
/cron list
/cron remove <job_id>
```

The agent can also manage jobs conversationally:
- `list_cronjobs` — Shows all jobs with IDs, schedules, repeat status, and next run times
- `remove_cronjob` — Removes a job by ID (use `list_cronjobs` to find the ID)

## Job Storage

Jobs are stored as JSON in `~/.hermes/cron/jobs.json`. Output from job runs is saved to `~/.hermes/cron/output/{job_id}/{timestamp}.md`.

The storage uses atomic file writes (temp file + rename) to prevent corruption from concurrent access.

## Self-Contained Prompts

:::warning Important
Cron job prompts run in a **completely fresh agent session** with zero memory of any prior conversation. The prompt must contain **everything** the agent needs:

- Full context and background
- Specific file paths, URLs, server addresses
- Clear instructions and success criteria
- Any credentials or configuration details

**BAD:** `"Check on that server issue"`
**GOOD:** `"SSH into server 192.168.1.100 as user 'deploy', check if nginx is running with 'systemctl status nginx', and verify https://example.com returns HTTP 200."`
:::

## Security

:::warning
Scheduled task prompts are scanned for instruction-override patterns (prompt injection). Jobs matching threat patterns like credential exfiltration, SSH backdoor attempts, or prompt injection are blocked at creation time. Content with invisible Unicode characters (zero-width spaces, directional overrides) is also rejected.
:::
