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

If you're expecting jobs to fire automatically, you need a running gateway (`hermes gateway` for foreground, or `hermes gateway start` for the installed service). For one-off debugging, you can manually trigger a tick with `hermes cron tick`.

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
| `whatsapp` | WhatsApp gateway configured |
| `signal` | Signal gateway configured |
| `matrix` | Matrix homeserver configured |
| `email` | SMTP configured in `config.yaml` |
| `sms` | SMS provider configured |
| `local` | Write access to `~/.hermes/cron/output/` |
| `origin` | Delivers to the chat where the job was created |

Other supported platforms include `mattermost`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`, and `webhook`. You can also target a specific chat with `platform:chat_id` syntax (e.g., `telegram:-1001234567890`).

If delivery fails, the job still runs — it just won't send anywhere. Check `hermes cron list` for updated `last_error` field (if available).

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

Cron jobs run with the `cronjob`, `messaging`, and `clarify` toolsets disabled. This prevents recursive cron creation, direct message sending (delivery is handled by the scheduler), and interactive prompts. If a skill relies on these toolsets, it won't work in a cron context.

Check the skill's documentation to confirm it works in non-interactive (headless) mode.

### Check 4: Multi-skill ordering

When using multiple skills, they load in order. If Skill A depends on context from Skill B, make sure B loads first:

```bash
/cron add "0 9 * * *" "..." --skill context-skill --skill target-skill
```

In this example, `context-skill` loads before `target-skill`.

---

## Memory and Memory Providers

Cron deliberately isolates memory. If a job cannot use `memory`, Hindsight, Honcho, mem0, etc., check this section before treating it as a bug.

### Built-in MEMORY.md / USER.md are always off for cron

Every cron agent is constructed with `skip_memory=True`. That protects the built-in local memory files from being polluted by cron system prompts (identity lines, model/tool banners, etc.). There is **no** global config switch to turn built-in memory back on for all cron jobs.

If the built-in `memory` tool reports that memory is unavailable, that is expected under cron.

### External providers are opt-in per job

External memory providers (Hindsight, Honcho, mem0, …) are controlled separately via the per-job field `memory_provider`:

| Job `memory_provider` | Built-in MEMORY.md | Provider tools | Auto prompt / prefetch / sync / session retain |
|-----------------------|--------------------|----------------|--------------------------------------------------|
| *(unset)* / `off`     | skipped            | no             | no                                               |
| `tools`               | skipped            | yes            | **no** (explicit tool calls only)                |
| `full`                | skipped            | yes            | yes (same lifecycle as interactive sessions)     |

Default is **`off`**. Prefer **`tools`** for scheduled maintenance jobs that need to search/store facts without auto-writing the whole cron prompt into the user representation. Use **`full`** only when you explicitly want automatic lifecycle behavior and understand the risk from [#4052](https://github.com/NousResearch/hermes-agent/issues/4052) (cron prompts misattributed as user messages).

Example (agent tool):

```python
cronjob(
    action="create",
    schedule="every day 3am",
    name="nightly-memory-curator",
    memory_provider="tools",
    prompt=(
        "Use hindsight_recall / hindsight_retain (or your active provider tools) "
        "to dedupe and store durable facts from the last day. "
        "Do not treat this prompt as user speech."
    ),
)
```

CLI form (when the flag is available on your build):

```bash
hermes cron create "every day 3am" "Curate long-term memory via provider tools..." \
  --memory-provider tools \
  --name nightly-memory-curator
```

If your installed CLI has no `--memory-provider` flag yet, create/update the job via the `cronjob` tool (or set `"memory_provider": "tools"` on the job record) — the field is stored in `~/.hermes/cron/jobs.json` and honored by the scheduler.

### Common failure modes

**"Memory is not available" on the built-in `memory` tool**
Expected. Built-in memory is disabled for cron. Use the active provider's tools after setting `memory_provider=tools` (or `full`), or run the work interactively outside cron.

**Provider tools missing even with `memory.provider` set in config.yaml**
Confirm the job itself has `memory_provider: tools` (or `full`). A configured global provider does **not** auto-enable in cron.

**Provider tools present but recall/context is empty every run**
You are probably on `tools` mode. Automatic prefetch and system-prompt injection are disabled on purpose. Call the provider search/recall tools explicitly in the job prompt.

**User representation polluted with "I am Hermes…" / model identity**
Classic #4052 symptom. Ensure the job is not on `memory_provider=full` (or legacy paths that enabled full auto-sync). Prefer `tools` and never auto-sync raw cron system prompts into user-modeling backends.

**Want full interactive memory behavior in a scheduled job**
Set `memory_provider=full` on that one job only. Still never enables built-in MEMORY.md/USER.md writes under cron.

### Related code / design pointers

- Mode resolution: `agent/memory_provider_mode.py` (`off` | `tools` | `full`)
- Init + flags: `agent/agent_init.py` (`_memory_provider_auto_sync`, `_memory_provider_prefetch`, `_memory_provider_prompt_context`)
- Cron construction: `cron/scheduler.py` (`skip_memory=True` + per-job `memory_provider_mode`)
- Tracking / history: [#9763](https://github.com/NousResearch/hermes-agent/issues/9763), [#4052](https://github.com/NousResearch/hermes-agent/issues/4052), PR [#18565](https://github.com/NousResearch/hermes-agent/pull/18565)

---

## Job Errors and Failures

### Check 1: Review recent job output

If a job ran and failed, you may see error context in:

1. The chat where the job delivers (if delivery succeeded)
2. `~/.hermes/logs/agent.log` for scheduler messages (or `errors.log` for warnings)
3. The job's `last_run` metadata via `hermes cron list`

### Check 2: Common error patterns

**"No such file or directory" for scripts**
The `script` path must be an absolute path (or relative to the Hermes config directory). Verify:
```bash
ls ~/.hermes/scripts/your-script.py   # Must exist
hermes cron edit <job_id> --script ~/.hermes/scripts/your-script.py
```

**"Skill not found" at job execution**
The skill must be installed on the machine running the scheduler. If you move between machines, skills don't automatically sync — reinstall them with `hermes skills install <skill-name>`.

**Job runs but delivers nothing**
Likely a delivery target issue (see Delivery Failures above), no output, or a response containing the cron quiet marker `[SILENT]`.

**Job hangs or times out**
The scheduler uses an inactivity-based timeout (default 600s, configurable via `HERMES_CRON_TIMEOUT` env var, `0` for unlimited). The agent can run as long as it's actively calling tools — the timer only fires after sustained inactivity. Long-running jobs should use scripts to handle data collection and deliver only the result.

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

### Too many overlapping jobs

The scheduler executes jobs sequentially within each tick. If multiple jobs are due at the same time, they run one after another. Consider staggering schedules (e.g., `0 9 * * *` and `5 9 * * *` instead of both at `0 9 * * *`) to avoid delays.

### Large script output

Scripts that dump megabytes of output will slow down the agent and may hit token limits. Filter/summarize at the script level — emit only what the agent needs to reason about.

---

## Diagnostic Commands

```bash
hermes cron list                    # Show all jobs, states, next_run times
hermes cron run <job_id>            # Schedule for next tick (for testing)
hermes cron edit <job_id>           # Fix configuration issues
hermes logs                         # View recent Hermes logs
hermes skills list                  # Verify installed skills
```

---

## Getting More Help

If you've worked through this guide and the issue persists:

1. Run the job with `hermes cron run <job_id>` (fires on next gateway tick) and watch for errors in the chat output
2. Check `~/.hermes/logs/agent.log` for scheduler messages and `~/.hermes/logs/errors.log` for warnings
3. Open an issue at [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) with:
   - The job ID and schedule
   - The delivery target
   - What you expected vs. what happened
   - Relevant error messages from the logs

---

*For the complete cron reference, see [Automate Anything with Cron](/guides/automate-with-cron) and [Scheduled Tasks (Cron)](/user-guide/features/cron).*
