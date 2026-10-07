---
sidebar_position: 12
title: "Cron Troubleshooting"
description: "Diagnose and fix common Rabbit cron issues — jobs not firing, delivery failures, skill loading errors, and performance problems"
---

# Cron Troubleshooting

When a cron job isn't behaving as expected, work through these checks in order. Most issues fall into one of four categories: timing, delivery, permissions, or skill loading.

---

## Jobs Not Firing

### Check 1: Verify the job exists and is active

```bash
rabbit cron list
```

Look for the job and confirm its state is `[active]` (not `[paused]` or `[completed]`). If it shows `[completed]`, the repeat count may be exhausted — edit the job to reset it.

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

### Check 3: Is the gateway running?

Cron jobs are fired by the gateway's background ticker thread, which ticks every 60 seconds. A regular CLI chat session does **not** automatically fire cron jobs.

If you're expecting jobs to fire automatically, you need a running gateway (`rabbit gateway` for foreground, or `rabbit gateway start` for the installed service). For one-off debugging, you can manually trigger a tick with `rabbit cron tick`.

**Desktop app:** the desktop's primary backend runs its own ticker, and it ticks **every local profile's** cron store — so jobs on a secondary profile keep firing even while that profile's backend is asleep (the desktop puts idle profile backends to sleep after ~10 minutes). You do not need to keep a profile open for its scheduled jobs to run.

### Check 4: Check the system clock and timezone

Jobs use the local timezone. If your machine's clock is wrong or in a different timezone than expected, jobs will fire at the wrong times. Verify:

```bash
date
rabbit cron list   # Compare next_run times with local time
```

---

## Delivery Failures

### Check 1: Verify the deliver target is correct

Delivery targets are case-sensitive and require the correct platform to be configured. A misconfigured target silently drops the response.

| Target | Requires |
|--------|----------|
| `telegram` | `TELEGRAM_BOT_TOKEN` in `~/.rabbit/.env` |
| `discord` | `DISCORD_BOT_TOKEN` in `~/.rabbit/.env` |
| `slack` | `SLACK_BOT_TOKEN` in `~/.rabbit/.env` |
| `whatsapp` | WhatsApp gateway configured |
| `signal` | Signal gateway configured |
| `matrix` | Matrix homeserver configured |
| `email` | SMTP configured in `config.yaml` |
| `sms` | SMS provider configured |
| `local` | Write access to `~/.rabbit/cron/output/` |
| `origin` | Delivers to the chat where the job was created |

Other supported platforms include `mattermost`, `homeassistant` (plugin), `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`, and `webhook`. You can also target a specific chat with `platform:chat_id` syntax (e.g., `telegram:-1001234567890`).

If delivery fails, the job still runs — it just won't send anywhere. Check `rabbit cron list` for updated `last_error` field (if available).

### Check 2: Check `[SILENT]` usage

If your cron job produces no output, delivery is suppressed. If the agent response includes the cron quiet marker `[SILENT]`, delivery is also suppressed. This is intentional for monitoring jobs — but make sure your prompt is not accidentally suppressing everything.

Use prompts like "respond with only [SILENT] if nothing changed." Avoid asking the agent to include `[SILENT]` inside a longer explanation, because cron treats that marker as a suppression signal.

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

### Check 5: Relay-fronted platforms (Rabbit Cloud / Team Gateway)

When a platform's credential lives in the relay connector (e.g. Slack or Discord fronted by a Team Gateway) rather than in your local `.env`, the **running gateway's live relay adapter is the only sender** — there is no standalone delivery path.

- Scheduled fires work as long as the gateway is running: its ticker owns relay-fronted delivery.
- A standalone `rabbit cron run <id>` automatically **forwards the run to the gateway** over the api_server (`POST /api/jobs/{id}/run`). This requires the `api_server` platform to be enabled with an `API_SERVER_KEY` (16+ characters). A `--prompt` / `cronjob(action='run', prompt=...)` context is forwarded with it and applies to that single fire only.
- If the gateway is not reachable, the run fails with a "relay-fronted … start the gateway" error instead of the misleading `platform 'slack' not configured/enabled`. Start the gateway and retry.

---

## Skill Loading Failures

### Check 1: Verify skills are installed

```bash
rabbit skills list
```

Skills must be installed before they can be attached to cron jobs. If a skill is missing, install it first with `rabbit skills install <skill-name>` or via `/skills` in the CLI.

### Check 2: Check skill name vs. skill folder name

Skill names are case-sensitive and must match the installed skill's folder name. If your job specifies `ai-funding-report` but the skill folder is `ai-funding-daily-report`, confirm the exact name from `rabbit skills list`.

### Check 3: Skills that require interactive tools

Cron jobs run with the `cronjob`, `messaging`, and `clarify` toolsets disabled. This prevents recursive cron creation, direct message sending (delivery is handled by the scheduler), and interactive prompts. If a skill relies on these toolsets, it won't work in a cron context.

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
2. `~/.rabbit/logs/agent.log` for scheduler messages (or `errors.log` for warnings)
3. The job's `last_run` metadata via `rabbit cron list`

### Check 2: Common error patterns

**"No such file or directory" for scripts**
The `script` path must be an absolute path (or relative to the Rabbit config directory). Verify:
```bash
ls ~/.rabbit/scripts/your-script.py   # Must exist
rabbit cron edit <job_id> --script ~/.rabbit/scripts/your-script.py
```

**"Skill not found" at job execution**
The skill must be installed on the machine running the scheduler. If you move between machines, skills don't automatically sync — reinstall them with `rabbit skills install <skill-name>`.

**Job runs but delivers nothing**
Likely a delivery target issue (see Delivery Failures above), no output, or a response containing the cron quiet marker `[SILENT]`.

**Job hangs or times out**
The scheduler uses an inactivity-based timeout (default 600s, configurable via `RABBIT_CRON_TIMEOUT` env var, `0` for unlimited). The agent can run as long as it's actively calling tools — the timer only fires after sustained inactivity. Long-running jobs should use scripts to handle data collection and deliver only the result.

### Check 3: Lock contention

The scheduler uses file-based locking to prevent overlapping ticks. If two gateway instances are running (or a CLI session conflicts with a gateway), jobs may be delayed or skipped.

Kill duplicate gateway processes:
```bash
ps aux | grep rabbit
# Kill duplicate processes, keep only one
```

### Check 4: Permissions on jobs.json

Jobs are stored in `~/.rabbit/cron/jobs.json`. If this file is not readable/writable by your user, the scheduler will fail silently:

```bash
ls -la ~/.rabbit/cron/jobs.json
chmod 600 ~/.rabbit/cron/jobs.json   # Your user should own it
```

---

## Performance Issues

### Slow job startup

Each cron job creates a fresh AIAgent session, which may involve provider authentication and model loading. For time-sensitive schedules, add buffer time (e.g., `0 8 * * *` instead of `0 9 * * *`).

### Too many overlapping jobs

The scheduler executes jobs sequentially within each tick. If multiple jobs are due at the same time, they run one after another. Consider staggering schedules (e.g., `0 9 * * *` and `5 9 * * *` instead of both at `0 9 * * *`) to avoid delays.

### Large script output

Scripts that dump megabytes of output will slow down the agent and may hit token limits. Filter/summarize at the script level — emit only what the agent needs to reason about.

---

## Diagnostic Commands

```bash
rabbit cron list                    # Show all jobs, states, next_run times
rabbit cron run <job_id>            # Schedule for next tick (for testing)
rabbit cron edit <job_id>           # Fix configuration issues
rabbit logs                         # View recent Rabbit logs
rabbit skills list                  # Verify installed skills
```

---

## Getting More Help

If you've worked through this guide and the issue persists:

1. Run the job with `rabbit cron run <job_id>` (fires on next gateway tick) and watch for errors in the chat output
2. Check `~/.rabbit/logs/agent.log` for scheduler messages and `~/.rabbit/logs/errors.log` for warnings
3. Open an issue at [github.com/seven0070/Rabbit-](https://github.com/seven0070/Rabbit-) with:
   - The job ID and schedule
   - The delivery target
   - What you expected vs. what happened
   - Relevant error messages from the logs

---

*For the complete cron reference, see [Automate Anything with Cron](./automate-with-cron.md) and [Scheduled Tasks (Cron)](../user-guide/features/cron.md).*
