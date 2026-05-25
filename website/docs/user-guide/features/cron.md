---
sidebar_position: 5
title: "Scheduled Tasks (Cron)"
description: "Schedule automated tasks with natural language, manage them with one cron tool, and attach one or more skills"
---

# Scheduled Tasks (Cron)

Schedule tasks to run automatically with natural language or cron expressions. Hermes exposes cron management through a single `cronjob` tool with action-style operations instead of separate schedule/list/remove tools.

## What cron can do now

Cron jobs can:

- schedule one-shot or recurring tasks
- pause, resume, edit, trigger, and remove jobs
- attach zero, one, or multiple skills to a job
- deliver results back to the origin chat, local files, or configured platform targets
- run in fresh agent sessions with the normal static tool list
- run in **no-agent mode** — a script on a schedule, its stdout delivered verbatim, zero LLM involvement (see the [no-agent mode](#no-agent-mode-script-only-jobs) section below)

All of this is available to Hermes itself through the `cronjob` tool, so you can create, pause, edit, and remove jobs by asking in plain language — no CLI required.

:::warning
Cron-run sessions cannot recursively create more cron jobs. Hermes disables cron management tools inside cron executions to prevent runaway scheduling loops.
:::

## Creating scheduled tasks

### In chat with `/cron`

```bash
/cron add 30m "Remind me to check the build"
/cron add "every 2h" "Check server status"
/cron add "every 1h" "Summarize new feed items" --skill blogwatcher
/cron add "every 1h" "Use both skills and combine the result" --skill blogwatcher --skill maps
```

### From the standalone CLI

```bash
hermes cron create "every 2h" "Check server status"
hermes cron create "every 1h" "Summarize new feed items" --skill blogwatcher
hermes cron create "every 1h" "Use both skills and combine the result" \
  --skill blogwatcher \
  --skill maps \
  --name "Skill combo"
```

### Through natural conversation

Ask Hermes normally:

```text
Every morning at 9am, check Hacker News for AI news and send me a summary on Telegram.
```

Hermes will use the unified `cronjob` tool internally.

## Skill-backed cron jobs

A cron job can load one or more skills before it runs the prompt.

### Single skill

```python
cronjob(
    action="create",
    skill="blogwatcher",
    prompt="Check the configured feeds and summarize anything new.",
    schedule="0 9 * * *",
    name="Morning feeds",
)
```

### Multiple skills

Skills are loaded in order. The prompt becomes the task instruction layered on top of those skills.

```python
cronjob(
    action="create",
    skills=["blogwatcher", "maps"],
    prompt="Look for new local events and interesting nearby places, then combine them into one short brief.",
    schedule="every 6h",
    name="Local brief",
)
```

This is useful when you want a scheduled agent to inherit reusable workflows without stuffing the full skill text into the cron prompt itself.

## Running a job inside a project directory

Cron jobs default to running detached from any repo — no `AGENTS.md`, `CLAUDE.md`, or `.cursorrules` is loaded, and the terminal / file / code-exec tools run from whatever working directory the gateway started in. Pass `--workdir` (CLI) or `workdir=` (tool call) to change that:

```bash
# Standalone CLI (schedule and prompt are positional)
hermes cron create "every 1d at 09:00" \
  "Audit open PRs, summarize CI health, and post to #eng" \
  --workdir /home/me/projects/acme
```

```python
# From a chat, via the cronjob tool
cronjob(
    action="create",
    schedule="every 1d at 09:00",
    workdir="/home/me/projects/acme",
    prompt="Audit open PRs, summarize CI health, and post to #eng",
)
```

When `workdir` is set:

- `AGENTS.md`, `CLAUDE.md`, and `.cursorrules` from that directory are injected into the system prompt (same discovery order as the interactive CLI)
- `terminal`, `read_file`, `write_file`, `patch`, `search_files`, and `execute_code` all use that directory as their working directory (via `TERMINAL_CWD`)
- The path must be an absolute directory that exists — relative paths and missing directories are rejected at create / update time
- Pass `--workdir ""` (or `workdir=""` via the tool) on edit to clear it and restore the old behaviour

:::note Serialization
Jobs with a `workdir` run sequentially on the scheduler tick, not in the parallel pool. This is deliberate — `TERMINAL_CWD` is process-global, so two workdir jobs running at the same time would corrupt each other's cwd. Workdir-less jobs still run in parallel as before.
:::

## Editing jobs

You do not need to delete and recreate jobs just to change them.

### Chat

```bash
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Use the revised task"
/cron edit <job_id> --skill blogwatcher --skill maps
/cron edit <job_id> --remove-skill blogwatcher
/cron edit <job_id> --clear-skills
```

### Standalone CLI

```bash
hermes cron edit <job_id> --schedule "every 4h"
hermes cron edit <job_id> --prompt "Use the revised task"
hermes cron edit <job_id> --skill blogwatcher --skill maps
hermes cron edit <job_id> --add-skill maps
hermes cron edit <job_id> --remove-skill blogwatcher
hermes cron edit <job_id> --clear-skills
```

Notes:

- repeated `--skill` replaces the job's attached skill list
- `--add-skill` appends to the existing list without replacing it
- `--remove-skill` removes specific attached skills
- `--clear-skills` removes all attached skills

## Lifecycle actions

Cron jobs now have a fuller lifecycle than just create/remove.

### Chat

```bash
/cron list
/cron pause <job_id>
/cron resume <job_id>
/cron run <job_id>
/cron remove <job_id>
```

### Standalone CLI

```bash
hermes cron list
hermes cron pause <job_id>
hermes cron resume <job_id>
hermes cron run <job_id>
hermes cron remove <job_id>
hermes cron status
hermes cron tick
```

What they do:

- `pause` — keep the job but stop scheduling it
- `resume` — re-enable the job and compute the next future run
- `run` — trigger the job on the next scheduler tick
- `remove` — delete it entirely

## How it works

**Cron execution is handled by the gateway daemon.** The gateway ticks the scheduler every 60 seconds, running any due jobs in isolated agent sessions.

```bash
hermes gateway install     # Install as a user service
sudo hermes gateway install --system   # Linux: boot-time system service for servers
hermes gateway             # Or run in foreground

hermes cron list
hermes cron status
```

### Gateway scheduler behavior

On each tick Hermes:

1. loads jobs from `~/.hermes/cron/jobs.json`
2. checks `next_run_at` against the current time
3. starts a fresh `AIAgent` session for each due job
4. optionally injects one or more attached skills into that fresh session
5. runs the prompt to completion
6. delivers the final response
7. updates run metadata and the next scheduled time

A file lock at `~/.hermes/cron/.tick.lock` prevents overlapping scheduler ticks from double-running the same job batch.

## Delivery options

When scheduling jobs, you specify where the output goes:

| Option | Description | Example |
|--------|-------------|---------|
| `"origin"` | Back to where the job was created | Default on messaging platforms |
| `"local"` | Save to local files only (`~/.hermes/cron/output/`) | Default on CLI |
| `"telegram"` | Telegram home channel | Uses `TELEGRAM_HOME_CHANNEL` |
| `"telegram:123456"` | Specific Telegram chat by ID | Direct delivery |
| `"telegram:-100123:17585"` | Specific Telegram topic | `chat_id:thread_id` format |
| `"discord"` | Discord home channel | Uses `DISCORD_HOME_CHANNEL` |
| `"discord:#engineering"` | Specific Discord channel | By channel name |
| `"slack"` | Slack home channel | |
| `"whatsapp"` | WhatsApp home | |
| `"signal"` | Signal | |
| `"matrix"` | Matrix home room | |
| `"mattermost"` | Mattermost home channel | |
| `"email"` | Email | |
| `"sms"` | SMS via Twilio | |
| `"homeassistant"` | Home Assistant | |
| `"dingtalk"` | DingTalk | |
| `"feishu"` | Feishu/Lark | |
| `"wecom"` | WeCom | |
| `"weixin"` | Weixin (WeChat) | |
| `"bluebubbles"` | BlueBubbles (iMessage) | |
| `"qqbot"` | QQ Bot (Tencent QQ) | |
| `"all"` | Fan out to every connected home channel | Resolved at fire time |
| `"telegram,discord"` | Fan out to a specific set of channels | Comma-separated list |
| `"origin,all"` | Deliver to the origin **plus** every other connected channel | Combine any tokens |

The agent's final response is automatically delivered. You do not need to call `send_message` in the cron prompt.

### Routing intent (`all`)

`all` lets you ship one cron job to every messaging channel you have configured, without having to enumerate them by name. It is **resolved at fire time**, so a job created before you wired up Telegram will pick up Telegram on the next tick after you set `TELEGRAM_HOME_CHANNEL`.

Semantics: `all` expands to every platform with a configured home channel. Zero is fine; the job simply produces no delivery targets and is recorded as a delivery failure upstream.

`all` composes with explicit targets. `origin,all` delivers to the origin chat *plus* every other connected home channel, de-duplicating by `(platform, chat_id, thread_id)`.

### Response wrapping

By default, delivered cron output is wrapped with a header and footer so the recipient knows it came from a scheduled task:

```
Cronjob Response: Morning feeds
-------------

<agent output here>

Note: The agent cannot see this message, and therefore cannot respond to it.
```

To deliver the raw agent output without the wrapper, set `cron.wrap_response` to `false`:

```yaml
# ~/.hermes/config.yaml
cron:
  wrap_response: false
```

### Silent suppression

If the agent's final response starts with `[SILENT]`, delivery is suppressed entirely. The output is still saved locally for audit (in `~/.hermes/cron/output/`), but no message is sent to the delivery target.

This is useful for monitoring jobs that should only report when something is wrong:

```text
Check if nginx is running. If everything is healthy, respond with only [SILENT].
Otherwise, report the issue.
```

Failed jobs always deliver regardless of the `[SILENT]` marker — only successful runs can be silenced.

## Script timeout

Pre-run scripts (attached via the `script` parameter) have a default timeout of 120 seconds. If your scripts need longer — for example, to include randomized delays that avoid bot-like timing patterns — you can increase this:

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 300   # 5 minutes
```

Or set the `HERMES_CRON_SCRIPT_TIMEOUT` environment variable. The resolution order is: env var → config.yaml → 120s default.

## No-agent mode (script-only jobs)

For recurring jobs that don't need LLM reasoning — classic watchdogs, disk/memory alerts, heartbeats, CI pings — pass `no_agent=True` at creation time. The scheduler runs your script on schedule and delivers its stdout directly, skipping the agent entirely:

```bash
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"
```

Semantics:

- Script stdout (trimmed) → delivered verbatim as the message.
- **Empty stdout → silent tick**, no delivery. This is the watchdog pattern: "only say something when something is wrong".
- Non-zero exit or timeout → an error alert is delivered, so a broken watchdog can't fail silently.
- `{"wakeAgent": false}` on the last line → silent tick (same gate LLM jobs use).
- No tokens, no model, no provider fallback — the job never touches the inference layer.

`.sh` / `.bash` files run under `/bin/bash`; anything else under the current Python interpreter (`sys.executable`). Scripts must live in `~/.hermes/scripts/` (same sandboxing rule as the pre-run script gate).

### The agent sets these up for you

The `cronjob` tool's schema exposes `no_agent` to Hermes directly, so you can describe a watchdog in chat and let the agent wire it up:

```text
Ping me on Telegram if RAM is over 85%, every 5 minutes.
```

Hermes will write the check script to `~/.hermes/scripts/` via `write_file`, then call:

```python
cronjob(action="create", schedule="every 5m",
        script="memory-watchdog.sh", no_agent=True,
        deliver="telegram", name="memory-watchdog")
```

It picks `no_agent=True` automatically when the message content is fully determined by the script (watchdogs, threshold alerts, heartbeats). The same tool also lets the agent pause, resume, edit, and remove jobs — so the whole lifecycle is chat-driven without anyone touching the CLI.

See the [Script-Only Cron Jobs guide](/docs/guides/cron-script-only) for worked examples.

## Chaining jobs with `context_from`

Cron jobs run in isolated sessions with no memory of previous runs. But sometimes one job's output is exactly what the next job needs. The `context_from` parameter wires that connection automatically — Job B's prompt gets Job A's most recent output prepended as context at runtime.

```python
# Job 1: Collect raw data
cronjob(
    action="create",
    prompt="Fetch the top 10 AI/ML stories from Hacker News. Save them to ~/.hermes/data/briefs/raw.md in markdown format with title, URL, and score.",
    schedule="0 7 * * *",
    name="AI News Collector",
)

# Job 2: Triage — receives Job 1's output as context
# Get Job 1's ID from: cronjob(action="list")
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/raw.md. Score each story 1–10 for engagement potential and novelty. Output the top 5 to ~/.hermes/data/briefs/ranked.md.",
    schedule="30 7 * * *",
    context_from="<job1_id>",
    name="AI News Triage",
)

# Job 3: Ship — receives Job 2's output as context
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/ranked.md. Write 3 tweet drafts (hook + body + hashtags). Deliver to telegram:7976161601.",
    schedule="0 8 * * *",
    context_from="<job2_id>",
    name="AI News Brief",
)
```

**How it works:**

- When Job 2 fires, Hermes reads Job 1's most recent output from `~/.hermes/cron/output/{job1_id}/*.md`
- That output is prepended to Job 2's prompt automatically
- Job 2 doesn't need to hardcode "read this file" — it receives the content as context
- The chain can be any length: Job 1 → Job 2 → Job 3 → ...

**What `context_from` accepts:**

| Format | Example |
|--------|---------|
| Single job ID (string) | `context_from="a1b2c3d4"` |
| Multiple job IDs (list) | `context_from=["job_a", "job_b"]` |

Outputs are concatenated in the order listed.

**When to use it:**

- Multi-stage pipelines (collect → filter → format → deliver)
- Dependent tasks where step N's work depends on step N−1's output
- Fan-out/fan-in patterns where one job aggregates results from several others

## Provider recovery

Cron jobs inherit your configured fallback providers and credential pool rotation. If the primary API key is rate-limited or the provider returns an error, the cron agent can:

- **Fall back to an alternate provider** if you have `fallback_providers` (or the legacy `fallback_model`) configured in `config.yaml`
- **Rotate to the next credential** in your [credential pool](/docs/user-guide/configuration#credential-pool-strategies) for the same provider

This means cron jobs that run at high frequency or during peak hours are more resilient — a single rate-limited key won't fail the entire run.

## Schedule formats

The agent's final response is automatically delivered — you do **not** need to include `send_message` in the cron prompt for that same destination. If a cron run calls `send_message` to the exact target the scheduler will already deliver to, Hermes skips that duplicate send and tells the model to put the user-facing content in the final response instead. Use `send_message` only for additional or different targets.

### Relative delays (one-shot)

```text
30m     → Run once in 30 minutes
2h      → Run once in 2 hours
1d      → Run once in 1 day
```

### Intervals (recurring)

```text
every 30m    → Every 30 minutes
every 2h     → Every 2 hours
every 1d     → Every day
```

### Cron expressions

```text
0 9 * * *       → Daily at 9:00 AM
0 9 * * 1-5     → Weekdays at 9:00 AM
0 */6 * * *     → Every 6 hours
30 8 1 * *      → First of every month at 8:30 AM
0 0 * * 0       → Every Sunday at midnight
```

### ISO timestamps

```text
2026-03-15T09:00:00    → One-time at March 15, 2026 9:00 AM
```

## Repeat behavior

| Schedule type | Default repeat | Behavior |
|--------------|----------------|----------|
| One-shot (`30m`, timestamp) | 1 | Runs once |
| Interval (`every 2h`) | forever | Runs until removed |
| Cron expression | forever | Runs until removed |

You can override it:

```python
cronjob(
    action="create",
    prompt="...",
    schedule="every 2h",
    repeat=5,
)
```

## Managing jobs programmatically

The agent-facing API is one tool:

```python
cronjob(action="create", ...)
cronjob(action="list")
cronjob(action="update", job_id="...")
cronjob(action="pause", job_id="...")
cronjob(action="resume", job_id="...")
cronjob(action="run", job_id="...")
cronjob(action="remove", job_id="...")
```

For `update`, pass `skills=[]` to remove all attached skills.

## Toolsets available to cron jobs

Cron runs each job in a fresh agent session with no chat platform attached. By default the cron agent gets **the toolset you configured for the `cron` platform in `hermes tools`** — not the CLI default, not everything under the sun.

```bash
hermes tools
# → pick the "cron" platform in the curses UI
# → toggle toolsets on/off just like you would for Telegram/Discord/etc.
```

Tighter per-job control is available via the `enabled_toolsets` field on `cronjob.create` (or on an existing job via `cronjob.update`):

```text
cronjob(action="create", name="weekly-news-summary",
        schedule="every sunday 9am",
        enabled_toolsets=["web", "file"],      # just web + file, no terminal/browser/etc.
        prompt="Summarize this week's AI news: ...")
```

When `enabled_toolsets` is set on a job it wins; otherwise the `hermes tools` cron-platform config wins; otherwise Hermes falls back to the built-in defaults. This matters for cost control: carrying `moa`, `browser`, `delegation` into every tiny "fetch news" job bloats the tool-schema prompt on every LLM call.

### Skipping the agent entirely: `wakeAgent`

If your cron job attaches a pre-check script (via `script=`), the script can decide at runtime whether Hermes should even invoke the agent. Emit a final stdout line of the form:

```text
{"wakeAgent": false}
```

…and cron skips the agent run entirely for this tick. Useful for frequent polls (every 1–5 min) that only need to wake the LLM when state actually changed — otherwise you pay for zero-content agent turns over and over.

```python
# pre-check script
import json, sys
latest = fetch_latest_issue_count()
prev = read_state("issue_count")
if latest == prev:
    print(json.dumps({"wakeAgent": False}))   # skip this tick
    sys.exit(0)
write_state("issue_count", latest)
print(json.dumps({"wakeAgent": True, "context": {"new_issues": latest - prev}}))
```

When `wakeAgent` is omitted, the default is `true` (wake the agent as usual).

#### Recipes: cheap pre-run gates

The `wakeAgent` gate gives you a $0 way to decide whether a scheduled job should spend any LLM tokens at all. Three patterns cover most use cases.

**File-change gate** — only run when a watched file has new content since the last successful tick. The scheduler records each job's `last_run_at`; compare it against the file's mtime.

```bash
#!/bin/bash
# ~/.hermes/scripts/feed-changed.sh
FEED="$HOME/data/feed.json"
STATE="$HOME/.hermes/scripts/.feed-changed.last"
test -f "$FEED" || { echo '{"wakeAgent": false}'; exit 0; }
mtime=$(stat -c %Y "$FEED")
last=$(cat "$STATE" 2>/dev/null || echo 0)
if [ "$mtime" -le "$last" ]; then
  echo '{"wakeAgent": false}'
else
  echo "$mtime" > "$STATE"
  echo '{"wakeAgent": true}'
fi
```

```text
cronjob(action="create", name="process-feed",
        schedule="every 30m",
        script="feed-changed.sh",
        prompt="A new ~/data/feed.json has landed. Summarize what changed.")
```

**External-flag gate** — only run when some other process has signalled readiness (e.g. a deploy hook drops a file, a CI job sets a value in your state store).

```bash
#!/bin/bash
# ~/.hermes/scripts/flag-ready.sh
if test -f /tmp/new-data-ready; then
  rm -f /tmp/new-data-ready
  echo '{"wakeAgent": true}'
else
  echo '{"wakeAgent": false}'
fi
```

```text
cronjob(action="create", name="nightly-analysis",
        schedule="0 9 * * *",
        script="flag-ready.sh",
        prompt="Run the nightly analysis over today's batch.")
```

**SQL-count gate** — only run when there are new rows to process in your own database. The script can also pass the count through to the agent via `context`, so the agent knows how much it's looking at without re-querying.

```python
#!/usr/bin/env python
# ~/.hermes/scripts/new-rows.py
import json, sqlite3
conn = sqlite3.connect("/home/me/data/app.db")
n = conn.execute(
    "SELECT COUNT(*) FROM messages WHERE ts > strftime('%s','now','-2 hours')"
).fetchone()[0]
if n < 1:
    print(json.dumps({"wakeAgent": False}))
else:
    print(json.dumps({"wakeAgent": True, "context": {"new_rows": n}}))
```

```text
cronjob(action="create", name="summarize-new-msgs",
        schedule="every 2h",
        script="new-rows.py",
        prompt="Summarize the new messages from the last 2 hours.")
```

The same pattern works for any data source you can query from a script — Postgres, an HTTP API, your own state store — without baking a SQL evaluator into the cron subsystem.

:::tip
Hermes's own `~/.hermes/state.db` is an internal schema that changes between releases. Don't query it from a pre-run gate — point at your own database or feed instead.
:::

Credit: this recipe set was prompted by @iankar8's exploration in [#2654](https://github.com/NousResearch/hermes-agent/pull/2654), which proposed adding sql/file/command triggers as a parallel mechanism. The `script` + `wakeAgent` gate already covers all three cases at $0, so the work landed as documentation instead.

### Chaining jobs: `context_from`

A cron job can consume the most recent successful output of one or more other jobs by listing their names (or IDs) in `context_from`:

```text
cronjob(action="create", name="daily-digest",
        schedule="every day 7am",
        context_from=["ai-news-fetch", "github-prs-fetch"],
        prompt="Write the daily digest using the outputs above.")
```

The referenced jobs' most recent completed outputs are injected above the prompt as context for this run. Each upstream entry must be a valid job ID or name (see `cronjob action="list"`). Note: chaining reads the *most recent completed* output — it does not wait for upstream jobs that are running in the same tick.

## Job storage

Jobs are stored in `~/.hermes/cron/jobs.json`. Output from job runs is saved to `~/.hermes/cron/output/{job_id}/{timestamp}.md`.

Jobs may store `model` and `provider` as `null`. When those fields are omitted, Hermes resolves them at execution time from the global configuration. They only appear in the job record when a per-job override is set.

The storage uses atomic file writes so interrupted writes do not leave a partially written job file behind.

## Self-contained prompts still matter

:::warning Important
Cron jobs run in a completely fresh agent session. The prompt must contain everything the agent needs that is not already provided by attached skills.
:::

**BAD:** `"Check on that server issue"`

**GOOD:** `"SSH into server 192.168.1.100 as user 'deploy', check if nginx is running with 'systemctl status nginx', and verify https://example.com returns HTTP 200."`

## Security

Scheduled task prompts are scanned for prompt-injection and credential-exfiltration patterns at creation and update time. Prompts containing invisible Unicode tricks, SSH backdoor attempts, or obvious secret-exfiltration payloads are blocked.
