---
sidebar_position: 11
title: "Cron Internals"
description: "How Hermes stores, schedules, edits, pauses, skill-loads, and delivers cron jobs"
---

# Cron Internals

The cron subsystem provides scheduled task execution — from simple one-shot delays to recurring cron-expression jobs with skill injection and cross-platform delivery.

## Key Files

| File | Purpose |
|------|---------|
| `cron/jobs.py` | Job model, storage, atomic read/write to `jobs.json` |
| `cron/scheduler.py` | Scheduler loop — due-job detection, execution, repeat tracking |
| `tools/cronjob_tools.py` | Model-facing `cronjob` tool registration and handler |
| `gateway/run.py` | Gateway integration — cron ticking in the long-running loop |
| `hermes_cli/cron.py` | CLI `hermes cron` subcommands |

## Scheduling Model

Four schedule formats are supported:

| Format | Example | Behavior |
|--------|---------|----------|
| **Relative delay** | `30m`, `2h`, `1d` | One-shot, fires after the specified duration |
| **Interval** | `every 2h`, `every 30m` | Recurring, fires at regular intervals |
| **Cron expression** | `0 9 * * *` | Standard 5-field cron syntax (minute, hour, day, month, weekday) |
| **ISO timestamp** | `2025-01-15T09:00:00` | One-shot, fires at the exact time |

The model-facing surface is a single `cronjob` tool with action-style operations: `create`, `list`, `update`, `pause`, `resume`, `run`, `remove`.

## Job Storage

Jobs are stored in `~/.hermes/cron/jobs.json` with atomic write semantics (write to temp file, then rename). Each job record contains:

```json
{
  "id": "a1b2c3d4e5f6",
  "name": "Daily briefing",
  "prompt": "Summarize today's AI news and funding rounds",
  "schedule": {
    "kind": "cron",
    "expr": "0 9 * * *",
    "display": "0 9 * * *"
  },
  "skills": ["ai-funding-daily-report"],
  "deliver": "telegram:-1001234567890",
  "repeat": {
    "times": null,
    "completed": 42
  },
  "state": "scheduled",
  "enabled": true,
  "next_run_at": "2025-01-16T09:00:00Z",
  "last_run_at": "2025-01-15T09:00:00Z",
  "last_status": "ok",
  "created_at": "2025-01-01T00:00:00Z",
  "model": null,
  "provider": null,
  "script": null
}
```

### Job Lifecycle States

| State | Meaning |
|-------|---------|
| `scheduled` | Active, will fire at next scheduled time |
| `paused` | Suspended — won't fire until resumed |
| `completed` | Repeat count exhausted or one-shot that has fired |
| `running` | Currently executing (transient state) |

### Backward Compatibility

Older jobs may have a single `skill` field instead of the `skills` array. The scheduler normalizes this at load time — single `skill` is promoted to `skills: [skill]`.

## Scheduler Runtime

### Tick Cycle

The scheduler runs on a periodic tick (default: every 60 seconds):

```text
tick()
  1. Acquire scheduler lock (prevents overlapping ticks)
  2. Load all jobs from jobs.json
  3. Filter to due jobs (next_run <= now AND state == "scheduled")
  4. For each due job:
     a. Set state to "running"
     b. Create fresh AIAgent session (no conversation history)
     c. Load attached skills in order (injected as user messages)
     d. Run the job prompt through the agent
     e. Deliver the response to the configured target
     f. Update run_count, compute next_run
     g. If repeat count exhausted → state = "completed"
     h. Otherwise → state = "scheduled"
  5. Write updated jobs back to jobs.json
  6. Release scheduler lock
```

### Gateway Integration

In gateway mode, the cron **trigger** (the part that decides *when* a due job
fires — "Axis B") is selected through a pluggable `CronScheduler` provider. The
gateway calls `resolve_cron_scheduler()` (`cron/scheduler_provider.py`) and runs
the resolved provider's `start()` in a dedicated background thread, alongside a
separate gateway-housekeeping thread.

The active provider is chosen by the `cron.provider` config key:

- **empty (default)** → the built-in `InProcessCronScheduler`, which runs the
  historical in-process loop calling `scheduler.tick()` every 60 seconds. This
  is byte-identical to the pre-provider behavior.
- **a named provider** (e.g. `chronos`, a managed-cron provider for
  scale-to-zero deployments) → discovered from `plugins/cron/<name>/` or
  `$HERMES_HOME/plugins/<name>/`.

If a named provider is missing, fails to load, or reports `is_available() ==
False`, the resolver falls back to the built-in with a warning — **cron is
never left without a trigger.** The built-in provider lives in core
(`cron/scheduler_provider.py`), not in `plugins/`, so the fallback can't be
accidentally removed.

What "firing" *means* (job execution + delivery) is unchanged and shared by all
providers — it stays in `scheduler.run_job()` / `scheduler._deliver_result()`.
A provider only controls the trigger, never execution.

In CLI mode, cron jobs only fire when `hermes cron` commands are run or during active CLI sessions.

### Managed cron (Chronos) for scale-to-zero

Hosted gateways can run the **Chronos** provider (`cron.provider: chronos`)
instead of the built-in ticker. Chronos lets an idle gateway **scale to zero**
and still fire cron jobs: rather than a 60-second in-process loop (which would
keep the process awake), it asks Nous infrastructure to arm exactly **one
managed one-shot per job at that job's real next-fire time**. At fire time Nous
calls the gateway back over an authenticated webhook (`POST /api/cron/fire`);
the gateway runs the job through the same `run_one_job` path as the built-in,
then re-arms the next one-shot. Between fires the process can be fully stopped —
it wakes only on a genuine fire, never on a periodic timer.

The flow (the managed scheduler is provided by Nous; the agent holds no
scheduler credentials):

```
create/update a cron job
  → Chronos asks Nous to arm a one-shot at the job's next_run_at
      (authenticated with the agent's existing Nous token)
  → at fire time Nous calls the gateway: POST {callback_url}/api/cron/fire
      (authenticated with a short-lived, purpose-scoped Nous-minted JWT)
  → the gateway verifies the token, claims the job (store compare-and-set
    de-duplicates normal retries during the bounded claim window), runs it, and
    re-arms the next one-shot. Full-run exclusion is only guaranteed between
    same-host processes sharing a lock-capable local cron store; the claim is
    not a renewable distributed lease
```

Config (all non-secret; on hosted agents Nous sets these at provision time):

| key | meaning |
|---|---|
| `cron.provider` | `chronos` to activate (empty = built-in ticker) |
| `cron.chronos.portal_url` | Nous base URL (arming + the fire-token issuer) |
| `cron.chronos.callback_url` | the gateway's own public base URL for inbound fires |
| `cron.chronos.expected_audience` | this agent's fire-token audience |
| `cron.chronos.nas_jwks_url` | key set for verifying the inbound fire token |

If Chronos is misconfigured or the agent isn't logged into Nous,
`resolve_cron_scheduler()` falls back to the built-in ticker (logged warning) —
cron never loses its trigger. Recurring jobs re-arm after each fire; `repeat`-N
jobs stop cleanly when the count is exhausted (no orphaned one-shot). The full
agent↔Nous wire contract lives in `docs/chronos-managed-cron-contract.md`.

### Fresh Session Isolation

Each cron job runs in a completely fresh agent session:

- No conversation history from previous runs
- No memory of previous cron executions (unless persisted to memory/files)
- The prompt must be self-contained — cron jobs cannot ask clarifying questions
- The `cronjob` toolset is disabled (recursion guard)

## Skill-Backed Jobs

A cron job can attach one or more skills via the `skills` field. At execution time:

1. Skills are loaded in the specified order
2. Each skill's SKILL.md content is injected as context
3. The job's prompt is appended as the task instruction
4. The agent processes the combined skill context + prompt

This enables reusable, tested workflows without pasting full instructions into cron prompts. For example:

```
Create a daily funding report → attach "ai-funding-daily-report" skill
```

### Script-Backed Jobs

Jobs can also attach a Python script via the `script` field. The script runs *before* each agent turn, and its stdout is injected into the prompt as context. This enables data collection and change detection patterns:

```python
# ~/.hermes/scripts/check_competitors.py
import requests, json
# Fetch competitor release notes, diff against last run
# Print summary to stdout — agent analyzes and reports
```

The script timeout defaults to 3600 seconds (1 hour). `_get_script_timeout()` resolves the limit through a three-layer chain:

1. **Module-level override** — `_SCRIPT_TIMEOUT` (for tests/monkeypatching). Only used when it differs from the default.
2. **Environment variable** — `HERMES_CRON_SCRIPT_TIMEOUT`
3. **Config** — `cron.script_timeout_seconds` in `config.yaml` (read via `load_config()`)
4. **Default** — 3600 seconds (1 hour)

This timeout bounds the **pre-run script only**, not the agent. Skill-based / LLM-driven jobs run on a separate *inactivity*-based budget (`HERMES_CRON_TIMEOUT`, default 600s of idle time, `0` = unlimited) — they can run for hours as long as they keep calling tools or streaming tokens, and are only killed after the configured idle period with no activity. Scripts are dispatched to a persistent thread pool (not held under the tick lock), so a long-running script does not block other due jobs from firing.

### Provider Recovery

`run_job()` passes the user's configured fallback providers and credential pool into the `AIAgent` instance:

- **Fallback providers** — reads `fallback_providers` (list) or `fallback_model` (legacy dict) from `config.yaml`, matching the gateway's `_load_fallback_model()` pattern. Passed as `fallback_model=` to `AIAgent.__init__`, which normalizes both formats into a fallback chain.
- **Credential pool** — loads via `load_pool(provider)` from `agent.credential_pool` using the resolved runtime provider name. Only passed when the pool has credentials (`pool.has_credentials()`). Enables same-provider key rotation on 429/rate-limit errors.

This mirrors the gateway's behavior — without it, cron agents would fail on rate limits without attempting recovery.

## Delivery Model

Cron job results can be delivered to any supported platform.

A bare platform name (`slack`, `telegram`, …) delivers to that platform's configured **home channel**. To target a **specific** destination instead, append a target after a colon: `platform:<target>`. The target is resolved at fire time (not when the job is created), so a job can name a destination on a platform that isn't connected yet and start delivering once it comes online.

Most platforms also accept an optional thread/topic as a third segment: `platform:<chat_id>:<thread_id>`.

| Target | Syntax | Example |
|--------|--------|---------|
| Origin chat | `origin` | Deliver to the chat where the job was created |
| Local file | `local` | Save to `~/.hermes/cron/output/` |
| Telegram | `telegram`, `telegram:<chat_id>`, `telegram:<chat_id>:<thread_id>`, `telegram:@username` | `telegram:-1001234567890:17585` |
| Discord | `discord`, `discord:#channel`, `discord:<channel_id>`, `discord:<channel_id>:<thread_id>` | `discord:#engineering` |
| Slack | `slack`, `slack:#channel`, `slack:<channel_id>`, `slack:<channel_id>:<thread_ts>` | `slack:#engineering` |
| Matrix | `matrix`, `matrix:<!room_id:server>`, `matrix:<@user:server>` | `matrix:!abc123:example.org` |
| Feishu | `feishu`, `feishu:<chat_id>`, `feishu:<chat_id>:<thread_id>` | `feishu:oc_abc123def` |
| WhatsApp | `whatsapp`, `whatsapp:<jid>`, `whatsapp:+<E.164>` | `whatsapp:123456@g.us` |
| Signal | `signal`, `signal:group:<id>`, `signal:+<E.164>` | `signal:group:aBcD==` |
| SMS | `sms`, `sms:+<E.164>` | `sms:+<E.164 number>` |
| Email | `email`, `email:<address>` | `email:alerts@example.com` |
| Weixin | `weixin`, `weixin:<wxid>` | `weixin:wxid_abc123` |
| Mattermost | `mattermost` or `mattermost:<channel_id>` | Bare name delivers to Mattermost home |
| Home Assistant | `homeassistant` or `homeassistant:<conversation>` | Bare name delivers to HA conversation |
| DingTalk | `dingtalk` or `dingtalk:<chat_id>` | Bare name delivers to DingTalk |
| WeCom | `wecom` or `wecom:<chat_id>` | Bare name delivers to WeCom |
| BlueBubbles | `bluebubbles` or `bluebubbles:<chat_guid>` | Bare name delivers to iMessage via BlueBubbles |
| QQ Bot | `qqbot` or `qqbot:<chat_id>` | Bare name delivers to QQ (Tencent) via Official API v2 |

Platforms in the first group have explicit, validated target syntax — named channels (`#channel`), topics/threads, room/user IDs, group IDs, or phone numbers. The remaining platforms accept the generic `platform:<chat_id>` form (the value after the colon is used verbatim as the destination ID); a bare platform name always delivers to the home channel.

**Named channels** (`slack:#engineering`, `discord:#engineering`, or a friendly name like `slack:engineering`) are resolved against the channel directory the gateway builds from connected adapters, so the gateway must have discovered the channel for name resolution to succeed; raw IDs (`slack:C0123ABCD45`) always work.

For **Telegram topics**, use `telegram:<chat_id>:<thread_id>` (e.g., `telegram:-1001234567890:17585`). For **Slack threads**, the third segment is the parent message's `thread_ts` (e.g., `slack:C0123ABCD45:1700000000.000100`), so it only applies when replying under an existing message.

### Response Wrapping

By default (`cron.wrap_response: true`), cron deliveries are wrapped with:
- A header identifying the cron job name and task
- A footer noting the agent cannot see the delivered message in conversation

The `[SILENT]` prefix in a cron response suppresses delivery entirely — useful for jobs that only need to write to files or perform side effects.

### Session Isolation

Cron deliveries are NOT mirrored into gateway session conversation history. They exist only in the cron job's own session. This prevents message alternation violations in the target chat's conversation.

## Recursion Guard

Cron-run sessions have the `cronjob` toolset disabled. This prevents:
- A scheduled job from creating new cron jobs
- Recursive scheduling that could explode token usage
- Accidental mutation of the job schedule from within a job

## Locking

The short-lived tick lock prevents duplicate dispatch scans. Recurring jobs also
hold a per-job OS advisory lock from **before** `next_run_at` advancement through
execution, output save, delivery, and `mark_job_run`. Its path is derived from
the active `cron.jobs` store context, so equal IDs in two profile stores remain
independent while callers with different runtime homes still contend when they
address the same store. One-shots continue to use their durable store claim.

This full-run guarantee is deliberately limited to **same-host processes sharing
the active cron store on a filesystem with normal local advisory-lock semantics**.
It is not a distributed lease and does not promise correct full-run ownership on
NFS/SMB or between hosts. Lock contention skips without mutating the schedule;
lock open/backend/permission failures are logged with the path and fail that
recurring dispatch closed. On POSIX, inherited descriptors are closed in a bare
fork child without issuing an unlock; Python's non-inheritable descriptors also
prevent ownership leaking through subprocess `exec`.

## CLI Interface

The `hermes cron` CLI provides direct job management:

```bash
hermes cron list                    # Show all jobs
hermes cron create                  # Interactive job creation (alias: add)
hermes cron edit <job_id>           # Edit job configuration
hermes cron pause <job_id>          # Pause a running job
hermes cron resume <job_id>         # Resume a paused job
hermes cron run <job_id>            # Trigger immediate execution
hermes cron remove <job_id>         # Delete a job
```

## Related Docs

- [Cron Feature Guide](/user-guide/features/cron)
- [Gateway Internals](./gateway-internals.md)
- [Agent Loop Internals](./agent-loop.md)
