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
| Survives process restart | State survives (SessionDB); gateway watches resume automatically after restart | Yes — fully durable scheduler |
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

`/hb` is an alias. Works on the CLI, the TUI / Desktop app, and gateway platforms (on Slack, use `/hermes heartbeat …`).

## Behavior details

- **Idle-only.** A heartbeat never interrupts a running turn. If the agent is busy when the tick comes due, it fires at the next idle poll. In the gateway, an idle watched session wakes proactively; no new inbound message is needed.
- **Missed ticks coalesce.** If the session was busy (or the process wasn't running) through several intervals, you get **one** heartbeat turn, not a backlog. The timer re-anchors on every fire.
- **User messages win.** A queued user message always takes priority; the heartbeat waits for the input queue to drain.
- **Cache-safe.** The injected prompt is an ordinary user message. No system-prompt mutation, no toolset change.
- **Gateway recovery.** Startup restores active heartbeats using the current persisted conversation and thread routing, in the owning profile. Each poll retries recovery after temporary storage failures or adapter downtime; paused and cleared heartbeats and suspended conversations do not restart. No new chat message is required.
- **Persistence and conversation boundaries.** State lives in `SessionDB.state_meta` keyed by `heartbeat:<session_id>` and follows context-compression session rotations. In the messaging gateway, leaving a conversation through reset, switch, or suspension clears its heartbeat; resuming that archived conversation does not resurrect it. Firing requires the owning process (CLI session or gateway) to be running. An already-admitted gateway tick is checked again after session resolution and before agent execution: it may follow a compression child, but cannot carry its old instruction into a reset or switched conversation.
- **Execution accounting.** The gateway reserves a due tick at adapter admission. If that exact attempt ends before entering the agent runner (including cancellation or a routing, authorization, emergency-stop, or preparation rejection), it refunds the tick unless the schedule has since changed. Once the agent runner is entered, the fire remains counted even if execution fails or is interrupted. This count is **not** proof of a successful model response or outbound delivery; abrupt process death can prevent the refund callback.
- **Don't-invent-work guard.** The injected prompt tells the agent to reply briefly and stop when nothing meaningful changed, so an idle heartbeat doesn't generate busywork.

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
