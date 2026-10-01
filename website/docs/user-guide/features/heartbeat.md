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
| `/heartbeat promote` | Lift this session's heartbeat to the profile, so new sessions start with it too. |
| `/heartbeat profile every <interval> <prompt>` | Set the **profile-wide** heartbeat that every new session in this profile inherits. |
| `/heartbeat profile status \| pause \| resume \| clear` | Manage the profile-wide heartbeat from any session. |

`/hb` is an alias. Works on the CLI, the TUI / Desktop app, and gateway platforms (on Slack, use `/hermes heartbeat …`).

## Profile heartbeats

A session heartbeat dies with its conversation: start a new session and you re-arm it by hand. A **profile heartbeat** is the standing version — it belongs to the profile, so any session of that profile picks it up already armed, and it carries that session's context when it fires (which `hermes cron` cannot, since each tick gets a fresh isolated session).

```
/heartbeat profile every 30m Check the staging deploy and report meaningful changes
```

What to expect:

- **One tick per interval, for the profile.** The profile heartbeat keeps its own clock, so exactly one session fires per interval — the first idle one to reach it. Two open sessions never both get woken for the same tick.
- **New sessions start armed.** A session with no heartbeat of its own adopts the profile one and follows its schedule, so a session that joins late waits for the next tick instead of firing a backlog on arrival. `/heartbeat status` marks it as inherited until it has fired once.
- **It keeps its own history once it fires.** The firing session records the heartbeat under its own key, so `/heartbeat status`, pause, resume and clear all work there — and an unstarted turn is still refunded, the profile tick included.
- **The profile stays in charge.** While a session's heartbeat is still the profile's, re-setting the profile heartbeat reaches it: it follows the new instruction and cadence rather than firing stale text. Setting, pausing, resuming or clearing a session heartbeat takes that session off the profile and onto its own clock.
- **`/heartbeat profile clear` is not a kill switch.** It stops the standing instruction for future sessions; a session already firing keeps its copy and carries on. Clear that session's own heartbeat to stop it.
- **A session heartbeat always wins.** Setting one in a session overrides the inherited default for that session only.
- **`promote` copies, it does not move.** `/heartbeat promote` gives the profile a copy and leaves the current session firing as before; the promoted copy starts its own clock.

State lives under the same per-profile store as session heartbeats, under a reserved `heartbeat:__profile__` key rather than a session id, so a profile switch picks the right one and `multiplex_profiles` routing is unaffected.

Not covered here yet: a `config.yaml` `heartbeat:` block, active/quiet hours, a dedicated model or provider for heartbeat turns, and notification targets. Those are tracked separately; this is the storage and inheritance primitive they build on.

## Behavior details

- **Idle-only.** A heartbeat never interrupts a running turn. If the agent is busy when the tick comes due, it fires at the next idle poll. In the gateway, an idle watched session wakes proactively; no new inbound message is needed.
- **Missed ticks coalesce.** If the session was busy (or the process wasn't running) through several intervals, you get **one** heartbeat turn, not a backlog. The timer re-anchors on every fire.
- **User messages win.** A queued user message always takes priority; the heartbeat waits for the input queue to drain.
- **Owned by the surface that set it.** A heartbeat set from a messaging chat fires from the gateway and replies into that chat even while the same session is open in the TUI / Desktop app; the viewer never claims the tick.
- **Cache-safe.** The injected prompt is an ordinary user message. No system-prompt mutation, no toolset change.
- **Gateway recovery.** Startup restores active heartbeats using the current persisted conversation and thread routing, in the owning profile. Each poll retries recovery after temporary storage failures or adapter downtime; paused and cleared heartbeats and suspended conversations do not restart. No new chat message is required.
- **Persistence and conversation boundaries.** State lives in `SessionDB.state_meta` keyed by `heartbeat:<session_id>` and follows context-compression session rotations. In the messaging gateway, leaving a conversation through reset, switch, or suspension clears its heartbeat; resuming that archived conversation does not resurrect it. Firing requires the owning process (CLI session or gateway) to be running. An already-admitted gateway tick is checked again after session resolution and before agent execution: it may follow a compression child, but cannot carry its old instruction into a reset or switched conversation.
- **Execution accounting.** The gateway reserves a due tick at adapter admission. If that exact attempt ends before entering the agent runner (including cancellation or a routing, authorization, emergency-stop, or preparation rejection), it refunds the tick unless the schedule has since changed. Once the agent runner is entered, the fire remains counted even if execution fails or is interrupted. This count is **not** proof of a successful model response or outbound delivery; abrupt process death can prevent the refund callback.
- **Quiet when there is nothing to say (gateway).** A scheduled heartbeat turn may end with a bare silence marker (`NO_REPLY` / `[SILENT]`): the gateway sends nothing, unlike a human message that returns only a marker (that still gets the visible "try again" notice). While the heartbeat works, no typing indicator, tool-progress, streamed draft or "still working" bubble is posted, and a result it does deliver is routed to the chat/topic without quoting the message that set the heartbeat. Approval prompts and failures stay visible.
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
