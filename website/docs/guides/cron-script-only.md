---
sidebar_position: 13
title: "Script-Only Cron Jobs (No LLM)"
description: "Classic watchdog cron jobs that skip the LLM entirely — a script runs on schedule and its stdout gets delivered to your messaging platform. Memory alerts, disk alerts, CI pings, periodic health checks."
---

# Script-Only Cron Jobs

Sometimes you already know exactly what message you want to send. You don't need an agent to reason about it — you just need a script to run on a timer, and its output (if any) to land in Telegram / Discord / Slack / Signal.

Hermes calls this **no-agent mode**. It's the cron system minus the LLM.

```
   ┌──────────────────┐          ┌──────────────────┐
   │ scheduler tick   │  every   │ run script       │
   │ (every N minutes)│ ──────▶ │ (bash or python) │
   └──────────────────┘          └──────────────────┘
                                          │
                                          │ stdout
                                          ▼
                                 ┌──────────────────┐
                                 │ delivery router  │
                                 │ (telegram/disc…) │
                                 └──────────────────┘
```

- **No LLM call.** Zero tokens, zero agent loop, zero model spend.
- **Script is the job.** The script decides whether to alert. Emit output → message gets sent. Emit nothing → silent tick.
- **Bash or Python.** `.sh` / `.bash` files run under `/bin/bash`; any other extension runs under the current Python interpreter. Anything in `~/.hermes/scripts/` is accepted.
- **Same scheduler.** Lives in `cronjob` alongside LLM jobs — pausing, resuming, listing, logs, and delivery targeting all work the same way.

## When to Use It

Use no-agent mode for:

- **Memory / disk / GPU watchdogs.** Run every 5 minutes, alert only when a threshold is breached.
- **CI hooks.** Deploy finished → post the commit SHA. Build failed → send the last 100 lines of the log.
- **Periodic metrics.** "Daily Stripe revenue at 9am" as a simple API call + pretty-print.
- **External event pollers.** Check an API, alert on state change.
- **Heartbeats.** Ping a dashboard every N minutes to prove the host is alive.

Use a normal (LLM-driven) cron job when you need the agent to **decide** what to say — summarize a long document, pick interesting items from a feed, draft a human-friendly message. The no-agent path is for cases where the script's stdout already IS the message.

## Create One from Chat

The real win of no-agent mode is that the agent itself can set up the watchdog for you — no editor, no shell, no remembering CLI flags. You describe what you want, Hermes writes the script, schedules it, and tells you when it'll fire.

### Example transcript

> **You:** ping me on telegram if RAM is over 85% every 5 minutes
>
> **Hermes:** *(writes `~/.hermes/scripts/memory-watchdog.sh`, then calls `cronjob(...)` with `no_agent=true`)*
>
> Set up. Runs every 5 min, alerts Telegram only when RAM is over 85%. Script: `memory-watchdog.sh`. Job ID: `abc123`.

Under the hood, the agent makes two tool calls:

```python
# 1. Write the check script
write_file(
    path="~/.hermes/scripts/memory-watchdog.sh",
    content='''#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  echo "RAM ${ram_pct}% on $(hostname)"
fi
# Empty stdout = silent tick; no message sent.
''',
)

# 2. Schedule it — no_agent=True skips the LLM on every tick
cronjob(
    action="create",
    schedule="every 5m",
    script="memory-watchdog.sh",
    no_agent=True,
    deliver="telegram",
    name="memory-watchdog",
)
```

From that point on every tick is free: the scheduler runs the script, pipes its stdout to Telegram if non-empty, and never touches a model.

### What the agent decides for you

When you phrase a request like "alert me when X" or "every N minutes check Y and tell me if Z", Hermes' `cronjob` tool description tells it to reach for `no_agent=True` whenever the message content is fully determined by the script. It falls back to the normal LLM-driven path when the request needs reasoning (*"summarize the new issues"*, *"pick the most interesting headlines"*, *"draft a friendly reminder"*).

You don't have to specify `--no-agent` yourself. Just describe the behavior.

### Managing watchdogs from chat

The agent can pause, resume, edit, and remove jobs the same way it creates them:

> **You:** stop the memory watchdog for tonight
>
> **Hermes:** *(calls `cronjob(action='pause', job_id='abc123')`)*
>
> Paused. Resume with "turn it back on" or via `hermes cron resume abc123`.

> **You:** change it to every 15 minutes
>
> **Hermes:** *(calls `cronjob(action='update', job_id='abc123', schedule='every 15m')`)*

The full lifecycle (create / list / update / pause / resume / run-now / remove) is available to the agent without you learning any CLI commands.

## Create One from the CLI

Prefer the shell? The CLI path gives you the same result with three commands:

```bash
# 1. Write your script
cat > ~/.hermes/scripts/memory-watchdog.sh <<'EOF'
#!/usr/bin/env bash
# Alert when RAM usage is over 85%. Silent otherwise.
RAM_PCT=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$RAM_PCT" -ge 85 ]; then
  echo "⚠ RAM ${RAM_PCT}% on $(hostname)"
fi
# Empty stdout = silent run; no message sent.
EOF
chmod +x ~/.hermes/scripts/memory-watchdog.sh

# 2. Schedule it
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"

# 3. Verify
hermes cron list
hermes cron run <job_id>    # fire it once to test
```

That's the whole thing. No prompt, no skill, no model.


## How Script Output Maps to Delivery

| Script behavior | Result |
|-----------------|--------|
| Exit 0, non-empty stdout | stdout is delivered verbatim |
| Exit 0, empty stdout | Silent tick — no delivery |
| Exit 0, stdout contains `{"wakeAgent": false}` on the last line | Silent tick (shared gate with LLM jobs) |
| Non-zero exit code | Error alert is delivered (so a broken watchdog doesn't fail silently) |
| Script timeout | Error alert is delivered |

The "silent when empty" behavior is the key to the classic watchdog pattern: the script is free to run every minute, but the channel only sees a message when something actually needs attention.

## Script Rules

Scripts must live in `~/.hermes/scripts/`. This is enforced at both job-creation time and run time — absolute paths, `~/` expansion, and path-traversal patterns (`../`) are rejected. The same directory is shared with the pre-check script gate used by LLM jobs.

Interpreter choice is by file extension:

| Extension | Interpreter |
|-----------|-------------|
| `.sh`, `.bash` | `/bin/bash` |
| anything else | `sys.executable` (current Python) |

We intentionally do NOT honour `#!/...` shebangs — keeping the interpreter set explicit and small reduces the surface the scheduler trusts.

## Schedule Syntax

Same as all other cron jobs:

```bash
hermes cron create "every 5m"        # interval
hermes cron create "every 2h"
hermes cron create "0 9 * * *"       # standard cron: 9am daily
hermes cron create "30m"             # one-shot: run once in 30 minutes
```

See the [cron feature reference](/docs/user-guide/features/cron) for the full syntax.

## Delivery Targets

`--deliver` accepts everything the gateway knows about. Some common shapes:

```bash
--deliver telegram                       # platform home channel
--deliver telegram:-1001234567890        # specific chat
--deliver telegram:-1001234567890:17585  # specific Telegram forum topic
--deliver discord:#ops
--deliver slack:#engineering
--deliver signal:+15551234567
--deliver local                          # just save to ~/.hermes/cron/output/
```

No running gateway is required at script-run time for bot-token platforms (Telegram, Discord, Slack, Signal, SMS, WhatsApp) — the tool calls each platform's REST endpoint directly using the credentials already in `~/.hermes/.env` / `~/.hermes/config.yaml`.

## Editing and Lifecycle

```bash
hermes cron list                                    # see all jobs
hermes cron pause <job_id>                          # stop firing, keep definition
hermes cron resume <job_id>
hermes cron edit <job_id> --schedule "every 10m"    # adjust cadence
hermes cron edit <job_id> --agent                   # flip to LLM mode
hermes cron edit <job_id> --no-agent --script …     # flip back
hermes cron remove <job_id>                         # delete it
```

Everything that works on LLM jobs (pause, resume, manual trigger, delivery target changes) works on no-agent jobs too.

## Worked Example: Disk Space Alert

```bash
cat > ~/.hermes/scripts/disk-alert.sh <<'EOF'
#!/usr/bin/env bash
# Alert when / or /home is over 90% full.
THRESHOLD=90
df -h / /home 2>/dev/null | awk -v t="$THRESHOLD" '
  NR > 1 && $5+0 >= t {
    printf "⚠ Disk %s full on %s\n", $5, $6
  }
'
EOF
chmod +x ~/.hermes/scripts/disk-alert.sh

hermes cron create "*/15 * * * *" \
  --no-agent \
  --script disk-alert.sh \
  --deliver telegram \
  --name "disk-alert"
```

Silent when both filesystems are under 90%; fires exactly one line per over-threshold filesystem when one fills up.

## Comparison with Other Patterns

| Approach | What runs | When to use |
|----------|-----------|-------------|
| `hermes send` (one-shot) | Any shell command piping into it | Ad-hoc delivery or as the action of an external scheduler (systemd, launchd) |
| `cronjob --no-agent` (this page) | Your script on Hermes' schedule | Recurring watchdogs / alerts / metrics that don't need reasoning |
| `cronjob` (default, LLM) | Agent with optional pre-check script | When the message content requires reasoning over data |
| OS cron + `hermes send` | Your script on the OS schedule | When Hermes might be unhealthy (the thing you're monitoring) |

For critical system-health watchdogs that must fire *even when the gateway is down*, keep using OS-level cron + a plain `curl` or `hermes send` call — those run as independent OS processes and don't depend on Hermes being up. The in-gateway scheduler is the right choice when the thing being monitored is external.

## Related

- [Automate Anything with Cron](/docs/guides/automate-with-cron) — LLM-driven cron patterns.
- [Scheduled Tasks (Cron) reference](/docs/user-guide/features/cron) — full schedule syntax, lifecycle, delivery routing.
- [Pipe Script Output with `hermes send`](/docs/guides/pipe-script-output) — the one-shot counterpart for ad-hoc scripts.
- [Gateway Internals](/docs/developer-guide/gateway-internals) — delivery-router internals.
