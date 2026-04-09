---
sidebar_position: 12
title: "Cron Troubleshooting"
description: "Diagnose and fix common Hermes cron issues — jobs not firing, delivery failures, skill loading errors, and performance problems"
---

# Cron Troubleshooting

When a cron job isn't behaving as expected, work through these checks in order. Most issues fall into one of four categories: timing, delivery, permissions, or skill loading.

---

## Jobs Not Firing

### Check 1: Verify the job exists and is active

```bash
hermes cron list
```

Look for the job and confirm its state is `scheduled` (not `paused` or `completed`). If it shows `completed`, the repeat count may be exhausted — edit the job to reset it.

### Check 2: Confirm the schedule is correct

A misformatted schedule silently defaults to one-shot or is rejected entirely. Test your expression:

| Your expression | Should evaluate to |
|----------------|-------------------|
| `0 9 * * *` | 9:00 AM every day |
| `0 9 * * 1` | 9:00 AM every Monday |
| `every 2h` | Every 2 hours from now |
| `30m` | 30 minutes from now |
| `2025-06-01T09:00:00` | June 1, 2025 at 9:00 AM UTC |

If the job fires once and then disappears from the list, it's a one-shot schedule (`30m`, `1d`, or an ISO timestamp) — expected behavior.

### Check 3: Is the gateway or CLI actually running?

Cron ticks are delivered by:
- **Gateway mode**: the long-running gateway process ticking every 60 seconds
- **CLI mode**: only when you run `hermes cron` commands or have an active CLI session

If you're expecting jobs to fire automatically, use gateway mode (`hermes gateway` or `hermes serve`). A CLI session that exits will stop cron scheduling.

### Check 4: Check the system clock and timezone

Jobs use the local timezone. If your machine's clock is wrong or in a different timezone than expected, jobs will fire at the wrong times. Verify:

```bash
date
hermes cron list   # Compare next_run times with local time
```

---

## Delivery Failures

### Check 1: Verify the deliver target is correct

Delivery targets are case-sensitive and require the correct platform to be configured. A misconfigured target silently drops the response.

| Target | Requires |
|--------|----------|
| `telegram` | `TELEGRAM_BOT_TOKEN` in `~/.hermes/.env` |
| `discord` | `DISCORD_BOT_TOKEN` in `~/.hermes/.env` |
| `slack` | `SLACK_BOT_TOKEN` in `~/.hermes/.env` |
| `email` | SMTP configured in `config.yaml` |
| `local` | Write access to `~/.hermes/cron/output/` |

If delivery fails, the job still runs — it just won't send anywhere. Check `hermes cron list` for updated `last_error` field (if available).

### Check 2: Check `[SILENT]` usage

If your cron job produces no output or the agent responds with `[SILENT]`, delivery is suppressed. This is intentional for monitoring jobs — but make sure your prompt isn't accidentally suppressing everything.

A prompt that says "respond with [SILENT] if nothing changed" will silently swallow non-empty responses too. Check your conditional logic.

### Check 3: Platform token permissions

Each messaging platform bot needs specific permissions to receive messages. If delivery silently fails:

- **Telegram**: Bot must be an admin in the target group/channel
- **Discord**: Bot must have permission to send in the target channel
- **Slack**: Bot must be added to the workspace and have `chat:write` scope

### Check 4: Response wrapping

By default, cron responses are wrapped with a header and footer (`cron.wrap_response: true` in `config.yaml`). Some platforms or integrations may not handle this well. To disable:

```yaml
cron:
  wrap_response: false
```

---

## Skill Loading Failures

### Check 1: Verify skills are installed

```bash
hermes skills list
```

Skills must be installed before they can be attached to cron jobs. If a skill is missing, install it first with `hermes skills install <skill-name>` or via `/skills` in the CLI.

### Check 2: Check skill name vs. skill folder name

Skill names are case-sensitive and must match the installed skill's folder name. If your job specifies `ai-funding-daily-report` but the skill folder is `ai-funding-daily-report`, confirm the exact name from `hermes skills list`.

### Check 3: Skills that require interactive tools

Cron jobs run with the `cronjob` toolset disabled (recursion guard). If a skill requires browser automation, code execution, or other interactive tools, the job will fail at execution time.

Check the skill's documentation to confirm it works in non-interactive (headless) mode.

### Check 4: Multi-skill ordering

When using multiple skills, they load in order. If Skill A depends on context from Skill B, make sure B loads first:

```bash
/cron add "0 9 * * *" "..." --skill context-skill --skill target-skill
```

In this example, `context-skill` loads before `target-skill`.

---

## Job Errors and Failures

### Check 1: Review recent job output

If a job ran and failed, you may see error context in:

1. The chat where the job delivers (if delivery succeeded)
2. `~/.hermes/logs/` for scheduler logs
3. The job's `last_run` metadata via `hermes cron list`

### Check 2: Common error patterns

**"No such file or directory" for scripts**
The `script` path must be an absolute path (or relative to the Hermes config directory). Verify:
```bash
ls ~/.hermes/scripts/your-script.py   # Must exist
hermes cron edit <job_id> --script ~/.hermes/scripts/your-script.py
```

**"Skill not found" at job execution**
The skill must be installed on the machine running the scheduler. If you move between machines, skills don't automatically sync. Run `hermes skills sync` or reinstall.

**Job runs but delivers nothing**
Likely a delivery target issue (see Delivery Failures above) or a silently suppressed response (`[SILENT]`).

**Job hangs or times out**
The scheduler has a default execution timeout. Long-running jobs should use scripts to handle collection and deliver only the result — don't let the agent run unbounded loops.

### Check 3: Lock contention

The scheduler uses file-based locking to prevent overlapping ticks. If two gateway instances are running (or a CLI session conflicts with a gateway), jobs may be delayed or skipped.

Kill duplicate gateway processes:
```bash
ps aux | grep hermes
# Kill duplicate processes, keep only one
```

### Check 4: Permissions on jobs.json

Jobs are stored in `~/.hermes/cron/jobs.json`. If this file is not readable/writable by your user, the scheduler will fail silently:

```bash
ls -la ~/.hermes/cron/jobs.json
chmod 600 ~/.hermes/cron/jobs.json   # Your user should own it
```

---

## Performance Issues

### Slow job startup

Each cron job creates a fresh AIAgent session, which may involve provider authentication and model loading. For time-sensitive schedules, add buffer time (e.g., `0 8 * * *` instead of `0 9 * * *`).

### Too many concurrent jobs

The default thread pool allows limited concurrent job execution. If you have many overlapping jobs, they queue up. Consider staggering schedules or splitting high-frequency jobs across different time windows.

### Large script output

Scripts that dump megabytes of output will slow down the agent and may hit token limits. Filter/summarize at the script level — emit only what the agent needs to reason about.

---

## Diagnostic Commands

```bash
hermes cron list                    # Show all jobs, states, next_run times
hermes cron run <job_id>            # Trigger immediate execution (for testing)
hermes cron edit <job_id>           # Fix configuration issues
hermes logs                         # View recent Hermes logs
hermes skills list                  # Verify installed skills
```

---

## Getting More Help

If you've worked through this guide and the issue persists:

1. Run the job immediately with `hermes cron run <job_id>` and watch for errors in the chat output
2. Check `~/.hermes/logs/scheduler.log` (if logging is enabled)
3. Open an issue at [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) with:
   - The job ID and schedule
   - The delivery target
   - What you expected vs. what happened
   - Relevant error messages from the logs

---

*For the complete cron reference, see [Automate Anything with Cron](/docs/guides/automate-with-cron) and [Scheduled Tasks (Cron)](/docs/user-guide/features/cron).*
