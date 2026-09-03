---
sidebar_position: 17
title: "Session Heartbeats"
description: "A recurring prompt that re-enters your current session whenever it's idle — /heartbeat every 10m Check the deployment."
---

# Session Heartbeats (`/heartbeat`)

`/heartbeat` gives the **current session** one recurring instruction. Whenever the session is idle and the interval has elapsed, the prompt fires as a normal user turn — same conversation, same context, same prompt cache.

```
/heartbeat every 10m Check the deployment and report meaningful changes
```

Inspired by Prime-Agent's `/heartbeat`. The Hermes adaptation keeps the strict message-flow invariants: the heartbeat is injected only between turns (never mid-run), as a plain user-role message.

## Heartbeat vs cron: which one do I want?

They look similar but serve different jobs:

| | `/heartbeat` | [`hermes cron`](./cron) |
|---|---|---|
| Runs in | **This conversation** — full context, memory of the discussion | A fresh isolated session per tick |
| Survives process restart | State survives (SessionDB); firing resumes next time the session is driven | Yes — fully durable scheduler |
| How many | One per session | Unlimited jobs |
| Best for | "Keep an eye on X *in this thread* while we work" | Standing jobs, reports, watchdogs, deliveries |

Rule of thumb: if the recurring prompt needs the conversation's context, use `/heartbeat`. If it's a self-contained job, use cron.

## Commands

| Command | What it does |
|---|---|
| `/heartbeat every <interval> <prompt>` | Set (or replace) the session's heartbeat. Intervals: `90s`, `10m`, `2h`, `1d` (minimum 60s). |
| `/heartbeat` or `/heartbeat status` | Show the heartbeat, its interval, and time to next fire. |
| `/heartbeat pause` | Stop firing without clearing. |
| `/heartbeat resume` | Resume (re-anchors the timer — no instant stale fire). |
| `/heartbeat clear` | Remove the heartbeat. |

`/hb` is an alias. Works on the CLI and gateway platforms (on Slack, use `/hermes heartbeat …`).

## Behavior details

- **Idle-only.** A heartbeat never interrupts a running turn. If the agent is busy when the tick comes due, it fires at the next idle poll.
- **Missed ticks coalesce.** If the session was busy (or the process wasn't running) through several intervals, you get **one** heartbeat turn, not a backlog. The timer re-anchors on every fire.
- **User messages win.** A queued user message always takes priority; the heartbeat waits for the input queue to drain.
- **Cache-safe.** The injected prompt is an ordinary user message. No system-prompt mutation, no toolset change.
- **Persistence.** State lives in `SessionDB.state_meta` keyed by `heartbeat:<session_id>` — it survives `/resume` and rides across context-compression session rotations. Firing requires the owning process (CLI session or gateway) to be running; for schedules that must survive anything, use cron.
- **Don't-invent-work guard.** The injected prompt tells the agent to reply briefly and stop when nothing meaningful changed, so an idle heartbeat doesn't generate busywork.
- **Active hours (optional).** Restrict firing to a daily window via `config.yaml`:

  ```yaml
  heartbeat:
    active_hours:
      start: "08:00"   # inclusive
      end: "22:00"     # exclusive; use e.g. 22:00–06:00 for an overnight window
      timezone: "Asia/Kolkata"   # optional — falls back to your top-level `timezone`, else host-local
  ```

  Outside the window, due ticks are skipped (never queued) — the first poll inside the window fires immediately. Leave `start`/`end` empty (the default) for 24/7. Bounds must be distinct `HH:MM` values; anything else is ignored with a warning rather than guessed at.

## Example

```
You: /heartbeat every 15m Check whether the CI run for PR #1234 finished; summarize the result when it does

  ♥ Heartbeat set (every 15m): Check whether the CI run for PR #1234 finished; ...

[15 minutes of you working on other things in the same session]

Hermes: [Heartbeat — recurring instruction, fires every 15m]
  💻 gh pr checks 1234   (1.2s)
  CI is still running (14/37 checks complete). Nothing to report yet.
```

When the answer stops changing, `/heartbeat clear` it — or let it keep watch.
