---
sidebar_position: 17
title: "Recurring Loops"
description: "Re-run a prompt on a recurring interval inside your session — Hermes' take on Claude Code's /loop."
---

# Recurring Loops (`/loop`)

`/loop` re-runs a prompt (or a slash command) on a recurring cadence **inside your current session**. Each wakeup is a real agent turn: Hermes reads the current state fresh — the latest CI result, the newest queue depth, the file as it is now — does the work, reports back, and goes quiet until the next tick.

It's Hermes' take on **Claude Code's `/loop`** (and its `/proactive` alias, which works here too). Where [`/goal`](./goals.md) is judge-driven — "keep working until this objective is achieved" — `/loop` is timer-driven: "do this again every N minutes (or whenever it makes sense) until something says stop."

## When to use it

- **Polling external state.** "Watch the deploy / the CI run / the queue and tell me when it changes." The canonical use case.
- **Iterate-until-green.** "Run the tests, fix what fails, repeat until they pass."
- **Monitoring during a work session.** Keep an eye on error rates or a long job's progress while you do something else in the same conversation.
- **Periodic housekeeping.** Re-run a lint pass or a status summary every N minutes during a long session.

When the work should run **unattended** — overnight, on a real schedule, surviving restarts of your terminal — use a [cron job](./cron.md) instead. `/loop` lives inside a session; cron lives outside all of them. And when the task is a single objective with a definition of done, [`/goal`](./goals.md) is usually the better fit.

## Quick start

```
/loop 5m check the deploy status and tell me if it's live yet
```

What you'll see:

1. **Loop accepted** — `↻ Loop set (every 5m): check the deploy status…`
2. **First wakeup in 5m** — while the session is idle, Hermes injects the wakeup and runs a normal turn against current state.
3. **Repeat** — every 5 minutes, until a stop condition fires or you stop it.

Loop a slash command just as easily:

```
/loop 10m /recap
```

### Run the first iteration immediately

By default the first wakeup waits one full interval. Pass `--start-now` to run
the first iteration right away, then continue on the cadence:

```
/loop 1h --start-now check the deploy status and tell me if it's live yet
```

The first wakeup fires on the next idle poll (gateway: the next 15s watcher
scan); subsequent wakeups follow the interval as usual. Works in both cadence
modes and with `/proactive`.

## The two cadence modes

**Fixed interval — you set the clock.** Give an interval (`30s`, `5m`, `2h`, `1h30m`) and the loop fires on that schedule. Use it when the thing you're watching changes on its own timeline:

```
/loop 2m poll the build at ci.example.com/job/42 and ping me the moment it finishes
```

**Self-paced — Hermes sets the clock.** Omit the interval and the loop paces itself: it starts at the floor (1 minute by default), and while the agent's replies stop changing it backs off exponentially — 2m, 4m, 8m, up to the ceiling (15 minutes by default). The moment a reply differs from the last one, cadence snaps back to the floor. Change detection is a local digest comparison (timestamps are ignored), so idle waits cost nothing extra:

```
/loop keep an eye on the migration and summarize progress
```

The rule of thumb: **fixed interval when an external clock drives the work; self-paced when the work drives the rhythm.**

## Stop conditions

A loop ends when any of these fires:

| Condition | How |
|---|---|
| The agent decides it's done | The wakeup prompt teaches the agent to end its reply with `LOOP_COMPLETE` on its own line when the task is finished or moot. |
| A run cap | `--times N` — stop after N wakeups. |
| An evidence-based condition | `--until <condition>` — after each wakeup, the same auxiliary judge that powers `/goal` checks the reply against your condition (fail-open: a broken judge never wedges the loop). |
| You | `/loop stop` (or `/loop pause` to keep it around). |
| The backstop budget | `loops.max_ticks` (default 100) pauses the loop so an unattended session can't burn tokens forever. `0` = unlimited. |

Examples:

```
/loop 2m poll CI --times 30
/loop 5m watch the queue --until queue depth reaches zero
```

## Commands

| Command | What it does |
|---|---|
| `/loop [interval] <prompt> [--times N] [--until <cond>] [--start-now]` | Start (or replace) the loop for this session. `--start-now` fires the first wakeup immediately. |
| `/loop` or `/loop status` | Show cadence, ticks fired, and time to the next wakeup. |
| `/loop pause` | Stop firing without losing the loop. |
| `/loop resume` | Pick it back up. |
| `/loop stop` | End the loop. |
| `/proactive …` | Alias for `/loop` (Claude Code parity). |

Works on the CLI, the TUI (`hermes --tui`), the web dashboard chat, the desktop app, and every gateway platform (Telegram, Discord, Slack, WhatsApp, …). On messaging platforms the gateway fires wakeups even between your messages — the loop belongs to the chat's session, and its results arrive as ordinary replies.

## Mixing with `/goal`

Both features inject synthetic turns at idle boundaries, so they follow one rule: **an active goal owns the session.** While a `/goal` is actively driving (judge saying "continue"), loop wakeups defer. The loop resumes using idle time as soon as the goal finishes, pauses, or parks itself on a wait barrier (`/goal wait`, or the judge's automatic WAIT verdict). A parked goal plus a `/loop` is a natural combo: the goal waits on the big async thing while the loop keeps a heartbeat on something else.

A real user message always wins over both — wakeups only fire while the session is idle and nothing of yours is queued.

## Behavior details

- **A wakeup is a normal user-role turn.** No system-prompt mutation, no toolset swap — prompt caching stays intact.
- **Survives `/resume` and compression.** Loop state persists per session and migrates across context-compression boundaries, same as `/goal`.
- **One loop per session.** Setting a new `/loop` replaces the old one. Run several loops by running several sessions (or use cron for a fleet of schedules).
- **Interrupting a wakeup turn (Ctrl+C) pauses the loop** — recoverable with `/loop resume`, so cancel actually means cancel.
- **Token cost scales with cadence.** Every tick is a full agent turn. Match the interval to how often the state actually changes; prefer self-pacing for idle waits.

## Configuration

```yaml
# ~/.hermes/config.yaml
loops:
  min_interval_seconds: 30       # floor for fixed intervals
  max_ticks: 100                 # backstop budget (0 = unlimited)
  self_paced_floor_seconds: 60   # self-paced starting cadence
  self_paced_ceiling_seconds: 900  # self-paced max backoff
```

The `--until` judge routes through the `goal_judge` auxiliary task, so `auxiliary.goal_judge.*` overrides (provider, model, max_tokens) apply to loop conditions too.

## `/loop` vs `/goal` vs cron

| | `/loop` | `/goal` | cron |
|---|---|---|---|
| **Trigger** | Timer (or self-paced) | Judge verdict after each turn | Schedule, outside any session |
| **Lives in** | Your current session | Your current session | Its own session per run |
| **Ends when** | Stop condition / caps / you | Goal achieved / budget / you | You remove the job |
| **Best for** | Polling, monitoring, periodic re-runs | One objective, iterate until done | Unattended, long-horizon schedules |
