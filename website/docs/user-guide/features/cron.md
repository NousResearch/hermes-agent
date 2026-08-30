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

:::tip
**Which model does a cron job run on?** Resolution at fire time is: per-job pin → `cron.model` in `config.yaml` → the global default from `hermes model`.

- **Per-job pin** — set by *you* via the dashboard, `hermes cron create/edit --model … --provider …`, or by editing `~/.hermes/cron/jobs.json`. Once set, it sticks until you change it. The agent's `cronjob` tool cannot set or change per-job models — inference pins are user-owned.
- **`cron.model` / `cron.model_provider`** — a cron-fleet default: every unpinned job runs on this model, independent of your chat model. Set it once (`hermes config set cron.model <name>`) and switching your chat model with `hermes model` or `/model` never touches your cron fleet.
- **Global default** — only when neither of the above is set does a job follow `hermes model`. In this case Hermes **snapshots** the provider and model at creation, and if the global default later changes the job **fails closed**: it skips the run, makes no inference call, and alerts you **once** — the job stays skipped (and silent) on subsequent ticks until you act or the config is restored (#44585). For recurring or otherwise repeatable jobs, pin the provider/model explicitly (`hermes cron edit <job_id> --provider <provider> --model <model>`) to proceed. A consumed finite one-shot cannot be updated; create a new future one-shot with an explicit provider and model instead. This prevents an unattended job from silently inheriting a switch to a paid provider/model. Setting `cron.model` (or a per-job pin) is the deliberate way to route cron spend, and the drift guard does not engage for an axis covered by it. Operators who instead want unpinned jobs to track the changing global default can [disable the drift guard](#letting-unpinned-jobs-track-global-defaults).

Whichever provider a job resolves to, its provider-specific request settings (e.g. `request_overrides` such as `extra_body`/`extra_headers` for custom providers) carry into the scheduled run just like an interactive session.

`hermes setup --portal` is the lowest-friction option for unattended runs since OAuth refresh is automatic. See [Nous Portal](/integrations/nous-portal).
:::

:::tip
**Per-job reasoning effort.** A job can pin its own thinking level, independent of the model pin: one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra`. When set, it overrides both the global `agent.reasoning_effort` and per-model `agent.reasoning_overrides` for that job's runs (`none` disables thinking). Set it via `hermes cron create/edit --reasoning-effort high`; pass an empty string on edit to clear the pin and follow config again. (It is deliberately not exposed on the agent's `cronjob` tool — model configuration stays a user decision.) Levels a model doesn't support are clamped or omitted by the provider at request time — pinning `xhigh` on a model that caps at `high` runs at `high`. The pin has no effect on `no_agent` jobs (there is no LLM call to tune). Use it to run heavy scheduled analyses at `high` while cheap recurring jobs run at `minimal`, without touching your global default.
:::

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

## Pre-dispatch configuration validation

Before constructing any agent machinery for a scheduled run, the scheduler
validates that the job's configuration can actually produce a successful run:

- the provider API key resolves (skipped when a `fallback_providers` chain is
  configured, since the fallback path may rescue a missing primary key),
- attached skills are ready (no missing required environment variables,
  commands, or credential files),
- delivery platform targets are known and have gateway credentials configured
  (`local`/`origin` targets are never checked).

When validation fails, the job's `last_status` becomes `blocked_config`, ONE
alert is delivered (it is not repeated every tick), and **no LLM call is
made** — a misconfigured job never spends tokens. The next healthy run clears
the blocked state so a future configuration break alerts again.

To disable the validation and restore the old behavior (the run proceeds and
fails during execution):

```yaml
cron:
  preflight: false
```

Or: `hermes config set cron.preflight false`

## Letting unpinned jobs track global defaults

The model/provider drift guard is enabled by default. If your unpinned cron
jobs should deliberately follow every global model or provider change, disable
it in `config.yaml`:

```yaml
cron:
  model_drift_guard: false
```

Or use the config command:

```bash
hermes config set cron.model_drift_guard false
```

This disables both the runtime block and the warning shown when global
inference settings change. Existing snapshots remain stored, so setting the
option back to `true` re-enables protection without recreating jobs.

:::warning
With the guard disabled, unattended unpinned jobs immediately inherit changed
global defaults. A switch to a paid provider or model can therefore spend money
on every scheduled run.
:::

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
- `terminal`, `read_file`, `write_file`, `patch`, `search_files`, and `execute_code` all use that directory as their working directory
- The path must be an absolute directory that exists — relative paths and missing directories are rejected at create / update time
- Pass `--workdir ""` (or `workdir=""` via the tool) on edit to clear it and restore the old behaviour

:::note Serialization
Jobs with a `workdir` run sequentially on the scheduler tick, not in the parallel pool. This is deliberate: the cron worker applies the job workdir through process-global terminal state, so two workdir jobs running at the same time would corrupt each other's cwd. Workdir-less jobs still run in parallel as before.
:::

## Editing jobs

You do not need to delete and recreate jobs just to change them.

:::tip Job reference
The `<job_id>` placeholder below (and in [Lifecycle actions](#lifecycle-actions)) also accepts the job's name (case-insensitive) — handy when you remember `morning-digest` but not the hex ID. An exact job ID takes precedence over name matches; if the reference is not an ID and a name matches more than one job, the command refuses and prints the candidate IDs so you can disambiguate.
:::

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
hermes cron pause <job_id_or_name>
hermes cron resume <job_id_or_name>
hermes cron run <job_id_or_name>
hermes cron remove <job_id_or_name>
hermes cron edit <job_id_or_name> [...flags]
hermes cron status
hermes cron tick
```

What they do:

- `pause` — keep the job but stop scheduling it
- `resume` — re-enable the job and compute the next future run
- `run` — trigger the job on the next scheduler tick
- `remove` — delete it entirely
- `edit` — modify schedule, prompt, delivery, etc.

**Name-based lookup.** All four mutating verbs (`pause`, `resume`, `run`, `remove`, `edit`) plus the agent's `cronjob` tool now accept a job **name** (case-insensitive) in place of the hex ID. The agent and CLI both prefer an exact ID match if one exists; ambiguous name matches (multiple jobs sharing the same name) are refused with the full list of candidate IDs so you can pick one explicitly. Names are not unique, so this guard is load-bearing — it prevents silently mutating the wrong job when two share a name.

## Agent-managed scheduling (cron jobs that manage cron jobs)

By default, agents launched *by* the scheduler cannot use the `cronjob` tool —
a scheduled job cannot create, edit, or remove other jobs. Opt in via
`config.yaml`:

```yaml
cron:
  allow_agent_scheduling: true   # default: false
```

When enabled, a scheduled agent can manage the cron table like any chat
session: schedule follow-up one-shots from within scheduled work, tune its own
cadence, or run a "cron librarian" job that reconciles the whole table
(list, then update/remove/create as needed). Two properties keep this sane:

- **One flat, user-owned table.** Jobs created from a cron run land in the
  same `jobs.json` as every other job with no special ownership — you can
  list, edit, or remove them exactly as if you had created them yourself.
- **No dangling delivery.** A cron run is ephemeral, so `deliver: origin`
  from inside one is resolved **at create time** to the creating job's own
  concrete target (`platform:chat_id[:thread_id]`, or `local` if the creating
  job delivers nowhere). A job created by a scheduled agent can never point
  its output at a session that no longer exists. Explicit targets
  (`local`, `all`, `telegram:<chat_id>`) are honored verbatim.

Prefer prompts that update existing jobs (list first, then update by ID)
over ones that create new jobs each run.

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

### Execution history

Hermes records each claimed cron attempt in the profile-local
`~/.hermes/cron/executions.db` before executor or provider dispatch. Attempts
move through `claimed`, `running`, and one immutable terminal state:
`completed`, `failed`, or `unknown`. After restart, Hermes marks an abandoned
attempt `unknown` only when the original PID and process-start fingerprint prove
that its owner is gone. Unknown attempts are audit records and are never
automatically rerun.

Inspect recent attempts with `hermes cron runs [job-id] --limit 20` (alias:
`history`). Terminal history is bounded; active attempts are never pruned. The
ledger is included in quick backups.

### Repeated-failure review nudge

Each job tracks a `failure_streak` — consecutive failed runs (delivery
failures don't count). A run that fails before the agent is reached at all —
a bad import after a half-applied update, a provider client that cannot be
constructed — counts and alerts the same as one the agent itself failed. When
a *recurring* job's streak reaches the threshold, the failure message
delivered to chat gains a review nudge telling you the job has failed N runs
in a row and suggesting you fix, pause (`hermes cron pause <job>`), or remove
it. Any successful run resets the streak, and `hermes cron list` shows the
streak alongside a failing job's last run. One-shot jobs never nudge.

```yaml
cron:
  failure_nudge_threshold: 3   # default; 0 disables the nudge
```

### Failure incidents: acknowledge a known failure

A recurring job that keeps failing with the *same* error pings you on every
run. Each failure is also recorded as a durable **incident**, keyed by the
job plus a normalized signature of the error text, in the same per-profile
ledger database as the execution history.

```bash
hermes cron incidents                 # list incidents (newest activity first)
hermes cron incidents --state alerted # filter: detected | alerted | closed
hermes cron incidents ack <id>        # acknowledge — stop re-pinging
```

Acknowledging an incident silences the per-run failure ping for that exact
signature only. Nothing else changes: the run history still records every
failure, the failure streak keeps counting, and the moment the job starts
failing with a *different* error a new incident is minted and alerts fire
again. A successful run doesn't touch incidents — they are per-signature, not
per-job.

Incident lifecycle: `detected` (failure recorded) → `alerted` (at least one
failure ping reached delivery) → `closed` (acknowledged; terminal for that
signature). Stored error text is secret-redacted and truncated before it is
written.

Recording is always on and costs nothing to ignore — no ping is ever
suppressed until you explicitly `ack`.

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
| `"bot-chat"` | This profile's canonical Bot Chat — the bot reads the output and responds | Machine-local |
| `"bot-chat:research"` | Another local profile's Bot Chat | Validated at create time |
| `"all"` | Fan out to every connected home channel | Resolved at fire time |
| `"telegram,discord"` | Fan out to a specific set of channels | Comma-separated list |
| `"origin,all"` | Deliver to the origin **plus** every other connected channel | Combine any tokens |

The agent's final response is automatically delivered to the configured `deliver:` target — the agent does not send messages itself, so there is nothing to call in the cron prompt.

### Bot Chat delivery (`bot-chat`)

`bot-chat` delivers the output **into a profile's canonical "Bot Chat" session as a real message**. Unlike every other target — where the recipient is a human reading a channel — the recipient here is the bot itself: it receives the output as an incoming message, acts on anything that needs action, and responds in its chat. Use it when scheduled output should be *processed*, not just posted.

- `bot-chat` (bare) targets the job's own profile.
- `bot-chat:<profile>` targets another profile **on the same machine**. Names are validated against `hermes profile list` when the job is created; profiles on other gateways or machines can never be targeted, so same-named profiles across machines are unambiguous.
- Each delivery costs the target bot one full agent turn — mind the schedule frequency.
- Composes with other targets (`bot-chat,telegram`) but is never included in `all`.

### Routing intent (`all`)

`all` lets you ship one cron job to every messaging channel you have configured, without having to enumerate them by name. It is **resolved at fire time**, so a job created before you wired up Telegram will pick up Telegram on the next tick after you set `TELEGRAM_HOME_CHANNEL`.

Semantics: `all` expands to every platform with a configured home channel. Zero is fine; the job simply produces no delivery targets and is recorded as a delivery failure upstream.

`all` composes with explicit targets. `origin,all` delivers to the origin chat *plus* every other connected home channel, de-duplicating by `(platform, chat_id, thread_id)`.

### Telegram cron topic (`TELEGRAM_CRON_THREAD_ID`)

When Telegram topic mode is enabled, the root DM is reserved as a system lobby — replies sent there are rebuffed with a lobby reminder and `reply_to_message_id` is dropped, so you cannot reply to a cron message that landed in the main chat.

Point cron at a dedicated forum topic instead:

1. In Telegram, open the bot DM and create a topic named e.g. `Cron`. Long-press the topic header → **Copy link**; the trailing integer is the topic's `message_thread_id`.
2. Set `TELEGRAM_CRON_THREAD_ID=<that id>` in your `.env`.

This applies only to cron deliveries. `TELEGRAM_HOME_CHANNEL_THREAD_ID` (used elsewhere, e.g. restart notifications) is unchanged. Explicit `deliver="telegram:chat_id:thread_id"` targets continue to win over the env var. Replies to cron messages now arrive in the existing topic session, so you can act on them directly.

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

### Continuable jobs (reply to a cron delivery)

By default a cron delivery is fire-and-forget: the message is sent, but it does
not live in the chat's conversation history, so if you reply to it the agent
has no record of what it said. Set a job **continuable** and the delivered brief
becomes a conversation you can reply into — the agent has the brief in context
instead of asking "what is Task #2?".

Opt-in, **default off**. Enable globally in config, or per-job via the `cronjob`
tool's `attach_to_session` (which overrides the global setting for that one job):

```yaml
# ~/.hermes/config.yaml
cron:
  mirror_delivery: false   # set true to make cron deliveries continuable
```

Behaviour is **thread-preferred**, scoped to the job's own conversation:

- **Thread-capable platforms** (Telegram topics, Discord/Slack threads): each
  delivery opens its own dedicated thread and the brief is seeded into that
  thread's session, so a reply in-thread continues with full context. A
  recurring job (e.g. a daily brief) opens a fresh thread per run, keeping each
  delivery's follow-up discussion isolated.
- **DM-only platforms** (WhatsApp, Signal, SMS): no threads exist, so the brief
  is mirrored into the origin DM session instead — the DM itself is the
  continuation surface.

Only the job's **own conversation** is ever touched:

- the **origin chat** the job was created in;
- the **home-channel fallback** when `deliver: origin` captured no origin (jobs
  created by scripts or the API rather than from a live gateway chat) — the
  user's primary conversation standing in for the origin;
- a job's **single explicit `platform:chat` target**, but only when the job
  itself opts in with `attach_to_session: true` — the job author declares that
  target a conversation. The global `mirror_delivery` flag alone never makes an
  explicitly-addressed chat continuable.

Broadcast / fan-out targets (`all`, bare-platform home channels) are never made
continuable. The mirror is
written as a labelled user turn (`[Cron delivery: <task name>]`), which keeps
the conversation history alternation-safe across all model providers.

#### Flat, in-channel continuation (Slack)

The thread-preferred behaviour above mints a dedicated thread on every
delivery. If you'd rather have a continuable job land **flat in the channel
timeline** — no thread — set the Slack **continuable surface** to `in_channel`:

```yaml
# ~/.hermes/config.yaml
slack:
  cron_continuable_surface: in_channel   # default: thread
  reply_in_thread: false                 # required pairing (see below)
  require_mention: false                 # so a plain reply continues the job
```

In `in_channel` mode the brief is delivered as an ordinary top-level channel
message (no thread is opened), and your reply continues the job via the
channel's shared session. Three settings work together:

- **`cron_continuable_surface: in_channel`** — skips thread creation on delivery.
- **`reply_in_thread: false`** (required) — makes the bot answer your reply
  *flat* in the channel and key it to the same whole-channel session the brief
  was seeded into. Without it the continuation still works but arrives in a
  thread (it falls back safely to thread-style continuation, never a dropped
  reply — the gateway logs a warning at startup so you can spot the mismatch).
- **`require_mention: false`** (or add the channel to `free_response_channels`)
  — so you can reply with a plain message; otherwise the bot only wakes when you
  `@`-mention it on each reply.

Because the continuation is the **whole-channel** session, it is shared: other
chatter in the channel — and a second continuable in-channel job — join the same
rolling conversation. That is inherent to "flat in a channel" and is the same
tradeoff `reply_in_thread: false` users already accept; use the default
`thread` surface when you want each delivery's follow-up isolated.

This is a Slack capability today. Other platforms accept the key but fall back
to the `thread` surface (their continuation primitives differ); the choice is
per-platform, set under each platform's config. It's a gateway-side config flag
— a `/restart` picks it up; no Slack app reinstall is needed.

:::note 1:1 DMs
`cron_continuable_surface` is a **channel** setting — a 1:1 DM has no
thread-vs-timeline split to choose between (the DM is already flat), so the key
has no effect there. What governs whether a DM cron delivery is continuable is
the separate, pre-existing knob **`slack.dm_top_level_threads_as_sessions`**:

- **`false`** — all top-level DMs share one rolling DM session, so a continuable
  cron brief and your reply land in the **same** session and the job continues in
  context. This is what you want for continuable cron in a DM.
- **`true`** (default) — each top-level DM message is its own session, so a reply
  to a delivered brief starts a *fresh* session that has no record of the brief.
  Continuation does not work in this mode (for cron or any other flat delivery).

So for a continuable cron job delivered to a 1:1 DM, set
`slack.dm_top_level_threads_as_sessions: false`. `cron_continuable_surface` is
not required (and is ignored) for DMs.
:::

### Silent suppression

If the agent's final response contains `[SILENT]`, delivery is suppressed entirely. The output is still saved locally for audit (in `~/.hermes/cron/output/`), but no message is sent to the delivery target.

This is useful for monitoring jobs that should only report when something is wrong:

```text
Check if nginx is running. If everything is healthy, respond with only [SILENT].
Otherwise, report the issue.
```

Failed jobs always deliver regardless of the `[SILENT]` marker — only successful runs can be silenced. For quiet monitoring jobs, prompt the agent to reply with only `[SILENT]` when there is nothing to report.

## Script timeout

Pre-run scripts (attached via the `script` parameter) have a default timeout of 3600 seconds (1 hour). This bounds the **script only** — skill-based / LLM-driven jobs run on a separate inactivity budget and are not capped by this value. If your scripts need a different limit, you can change it:

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 1800   # 30 minutes
```

Or set the `HERMES_CRON_SCRIPT_TIMEOUT` environment variable. The resolution order is: env var → config.yaml → 3600s default.

Cron also bounds post-run session and agent-resource cleanup. This happens after the LLM turn returns, so it is separate from the inactivity timeout. The default is 10 seconds per cleanup operation. If a storage or client finalizer stops returning, the scheduler logs an error, releases the job's in-flight guard, and allows later runs to dispatch instead of skipping that job forever.

```yaml
# ~/.hermes/config.yaml
cron:
  cleanup_timeout_seconds: 10
```

Set `cleanup_timeout_seconds: 0` only to restore the legacy unbounded cleanup behavior.

## Media send timeout

When a cron delivery includes media attachments (a generated PDF, TTS audio, an exported report) sent through a live gateway adapter, each attachment upload is bounded by a timeout — 300 seconds by default. Large files on slow uplinks can need more:

```yaml
# ~/.hermes/config.yaml
cron:
  media_send_timeout_seconds: 600   # 10 minutes per attachment
```

Or set the `HERMES_CRON_MEDIA_SEND_TIMEOUT` environment variable. The resolution order is: env var → config.yaml → 300s default. A timed-out attachment is recorded in the job's run status as a partial delivery failure (the text still delivers).

## Bot Chat delivery timeout

A `bot-chat` delivery runs a full agent turn in the target bot's chat, so its bound is minutes, not seconds — 600s by default:

```yaml
# ~/.hermes/config.yaml
cron:
  bot_chat_delivery_timeout_seconds: 900
```

A timed-out delivery is recorded in `last_delivery_error`; the bot's turn may still complete on its own.

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

`.sh` / `.bash` files run under `bash` from `PATH` when available, otherwise `/bin/bash` (important on Windows Git Bash). Anything else runs under the current Python interpreter (`sys.executable`). Scripts must resolve inside `$HERMES_HOME/scripts/` — relative names, absolute paths, and `~`-prefixed paths are accepted when the resolved target stays in that directory; paths that escape it are rejected. Subprocess env is sanitized (`_sanitize_subprocess_env`): provider API credentials and other Hermes-managed secrets are **not** inherited by cron scripts.

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

See the [Script-Only Cron Jobs guide](/guides/cron-script-only) for worked examples.

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

**Continuity: carry the previous run's output**

Set `continuity=true` and the job injects its *own* most recent output into each run. Recurring jobs normally start every run with amnesia — a news scout re-reports the same stories, a monitor re-alerts on the same condition. With continuity on, the job wakes up seeing what it reported last time and can dedupe and continue where it left off:

```python
cronjob(
    action="create",
    prompt="Scan HN and arXiv for new agent-tooling papers. Report only items NOT already covered in your previous run's output.",
    schedule="every 6h",
    continuity=True,
    name="Agent Tooling Scout",
)
```

The first run has no previous output, so the prompt runs as-is. On later runs the previous output is prepended with continuity framing ("avoid repeating what was already reported"). It combines freely with upstream jobs (`context_from=["<other_job_id>"]` plus `continuity=true`), and `continuity=false` on update turns it off while preserving other `context_from` entries. Internally the flag is stored as the reserved `self` entry in `context_from`.

From the CLI: `hermes cron create "every 6h" "Scan for news" --continuity`, and `hermes cron edit <job_id> --continuity` / `--no-continuity` to toggle it on an existing job. The same toggle appears in the dashboard's cron editor and the desktop Bot Mode routine dialog.

**When to use it:**

- Multi-stage pipelines (collect → filter → format → deliver)
- Dependent tasks where step N's work depends on step N−1's output
- Fan-out/fan-in patterns where one job aggregates results from several others
- Recurring scouts/monitors that should dedupe against their own previous report (`continuity=true`)

## Provider recovery

Cron jobs inherit your configured fallback providers and credential pool rotation. If the primary API key is rate-limited or the provider returns an error, the cron agent can:

- **Fall back to an alternate provider** if you have `fallback_providers` (or the legacy `fallback_model`) configured in `config.yaml`
- **Rotate to the next credential** in your [credential pool](/user-guide/configuration#credential-pool-strategies) for the same provider

This means cron jobs that run at high frequency or during peak hours are more resilient — a single rate-limited key won't fail the entire run.

## Missed scheduled fires (`last_fire_error`)

On hosted (managed-cron) deployments, a scheduled fire travels from the platform scheduler through the dashboard to the gateway's internal API server. If that final hand-off fails — the gateway process is down, or its API-server listener never started — the run never begins, so there is no execution record and no `last_status` to inspect. The tell-tale shape: the job works every time you trigger it manually, but never auto-fires.

These misses are stamped on the job record as `last_fire_error` (timestamp + reason) and surfaced by:

- `cronjob` tool → `action: "list"` — the `last_fire_error` field
- `hermes cron list` — a red `⚠ Missed scheduled fire:` line under the job
- The dashboard job view

The stamp always reflects **current** auto-fire health: it is overwritten by newer misses and cleared automatically by the next successful run. If you see it, the job and its schedule are fine — the gateway side of the fire path needs attention (most commonly, restart the gateway through its supervisor so it loads the full profile environment: `hermes gateway restart`).

### Misfire catch-up

When an external scheduler provider is active (managed cron on hosted deployments), the gateway also runs a catch-up sweep: a job whose scheduled time passed with no fire delivered — and whose grace window has elapsed — is claimed and run locally, so an outage in the fire hand-off costs minutes instead of the whole day. The sweep is de-duplicated against late scheduler retries by the same store claim used for normal fires.

```yaml
cron:
  misfire_grace_minutes: 10   # wait this long for the scheduler's own retries
                              # before catching up locally; 0 disables catch-up
```

Local (built-in ticker) deployments don't need this — the ticker already picks up past-due jobs on its next tick.

## Schedule formats

The agent's final response is automatically delivered to the job's `deliver:` target — the agent no longer fires messages itself, so the user-facing content simply goes in the final response. To deliver to **additional or different** targets, list multiple `deliver:` targets on the cron job (comma-separated, e.g. `deliver: "telegram,discord"`) rather than having the agent send them.

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
every hour   → Every hour (bare unit = 1)
```

### Natural day/time schedules (recurring)

```text
every monday 9am         → Weekly, Mondays at 9:00 AM
every day at 9am         → Daily at 9:00 AM
weekdays at 9am          → Weekdays at 9:00 AM
weekends at 10am         → Saturdays and Sundays at 10:00 AM
daily at 7am             → Daily at 7:00 AM
monday, wednesday at 9am → Mondays and Wednesdays at 9:00 AM
```

Times accept `9am`, `9:30pm`, `14:00`, bare 24-hour hours (`at 7`), `noon`, and `midnight`. These forms compile to cron expressions internally (they require the `croniter` package, installed by default).

### Cron expressions

```text
0 9 * * *       → Daily at 9:00 AM
0 9 * * 1-5     → Weekdays at 9:00 AM
0 9 * * MON-FRI → Weekdays at 9:00 AM (named weekdays/months accepted)
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

### Manual runs are asynchronous

`cronjob(action="run")` fires the job immediately **in the background** (like
`delegate_task`): the tool call returns at once with a handle, and the job's
outcome — success/failure, delivery target, next scheduled run, and an output
excerpt — re-enters the conversation as a new message when the run finishes.
The agent (and you) can keep working in the meantime, and a job that is
already mid-run is refused with "already running" instead of double-firing.

You can also pass `prompt` with `action="run"` to inject transient per-run
context:

```python
cronjob(action="run", job_id="...", prompt="CONTEXT: focus on the EU region today")
```

The context is appended to the job's stored prompt under a `## Run Context`
header for that single fire only — it is never persisted to the job
definition, and it passes the same prompt-injection scan as stored prompts.

Runtimes that can't receive detached results (one-shot `hermes -z`, `hermes
cron run` from the CLI, cron child sessions, Kanban workers) fall back to
synchronous execution automatically.

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

When `enabled_toolsets` is set on a job it wins; otherwise the `hermes tools` cron-platform config wins; otherwise Hermes falls back to the built-in defaults. This matters for cost control: carrying `browser`, `delegation` into every tiny "fetch news" job bloats the tool-schema prompt on every LLM call.

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

Job definitions are plain JSON on disk: they survive `hermes update`, gateway restarts, and machine reboots. A job that was mid-run during a restart is marked `unknown` in the execution ledger — it is not automatically retried, but the job's next scheduled tick fires normally. See [Execution history](#execution-history) for details.

:::tip
Ask the agent to manage jobs through the `cronjob` tool, `hermes cron edit`, or `/cron` — not by patching `jobs.json` directly. Direct edits can fail silently when [file write safety](../security.md#file-write-safety) blocks the path (for example when `HERMES_WRITE_SAFE_ROOT` is set), and the [file-mutation verifier](../configuration.md#file-mutation-verifier) footer is the authoritative signal that nothing was saved.
:::

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
