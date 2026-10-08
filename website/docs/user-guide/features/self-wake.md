---
sidebar_position: 17
title: "Self-Wake"
description: "Let the agent arm its own one-shot alarm — schedule_wake re-enters an idle session at a deadline with no human input."
---

# Self-Wake (`schedule_wake`)

`schedule_wake` is a tool the **agent** calls on itself: "wake me in 20 minutes with this instruction". When the deadline passes and the session is idle, the instruction fires as a normal user turn — same conversation, same context, same prompt cache — so a long orchestration (delegation batches, a gated release pipeline, waiting on a slow external job) re-enters its loop instead of sleeping forever because nobody typed anything.

```
Agent: I've dispatched the three fix lanes. Nothing to do until they report.
  ⏰ schedule_wake(delay_secs=1200, prompt="Check the three lanes, merge what's green, re-arm if any are still running")

[20 minutes later, no human input]

Hermes: [Wake — one-shot instruction you scheduled for yourself (1/100)]
  Check the three lanes, merge what's green, re-arm if any are still running
  💻 gh pr checks …
```

Inspired by ChatGPT Work's **dots**, which "decide when to pause and wake up to continue work; you don't need a fixed schedule for every follow-up". Hermes gives the same affordance to any session while keeping the strict message-flow invariants: the wake is injected only between turns, as a plain user-role message, and a real user message always wins.

## Self-wake vs heartbeat vs loop vs cron

| | `schedule_wake` | [`/heartbeat`](./heartbeat.md) | [`/loop`](./loops.md) | [`hermes cron`](./cron.md) |
|---|---|---|---|---|
| Who arms it | The **agent**, mid-task | You | You | You (or the agent via `cronjob_manage`) |
| Shape | One deadline, fires once | Recurring interval | Recurring interval / self-paced | Durable schedule |
| Runs in | This conversation | This conversation | This conversation | A fresh isolated session |
| Best for | "Nothing to do until X; come back then" | "Keep an eye on X while we work" | Re-running a prompt on a cadence | Standing jobs, reports, deliveries |

Rule of thumb: the agent reaches for `schedule_wake` when **it** knows the next useful moment; you reach for `/heartbeat` when **you** want a standing check.

## How it works

- **The agent arms it.** `schedule_wake(prompt, delay_secs | at_iso)` — at least 60 s out, exactly one armed wake per session (latest wins). Recurring wake-ups are refused (`/heartbeat` is the recurring surface); subagent sessions are refused (a child has no idle loop of its own; the orchestrator arms its own wake).
- **The owning surface fires it.** The classic CLI's idle hook, the TUI / Desktop / dashboard session-owner poller, and the messaging gateway's wake watcher each drive wakes for the sessions they own. A wake armed inside a Telegram/Discord/Slack chat captures that chat route and fires through the same bot into the same chat, surviving a gateway restart; CLI/TUI wakes are route-less and fire only in their own process.
- **One-shot, claim-first.** The fire is recorded before the prompt is queued, so a wake can be lost only on a crash, never double-fired. If the dispatch never actually started a turn (session became busy, prompt refused), the fire is refunded and the wake stays armed.
- **Never a command.** The injected text is the rendered `[Wake — one-shot instruction …]` message, so a model-authored prompt that happens to start with `/` or `!` is a chat turn, never a slash or shell command.
- **A real user message wins.** Wakes fire only into an idle session with an empty input queue; `hermes pause` holds them armed until `hermes resume`.
- **Fire budget.** Each session may fire at most `wake.max_fires` self-scheduled wakes (default 100, `0` = unlimited); the budget survives re-arms and compression rotations. Past it, `schedule_wake` refuses and tells the agent to ask you. This is the deterministic backstop against a model that keeps re-arming itself.
- **Not offered where it can't fire.** Cron runs (the session ends with the job), the API server and Kanban workers (the client owns the next turn) and ACP never see the tool, so a wake is never armed-but-dead.
- **Persistence.** State lives in `SessionDB.state_meta` under `wake:<session_id>` and follows context-compression session rotations, like heartbeats and loops.

## Configuration

```yaml
wake:
  max_fires: 100   # per-session budget of self-scheduled wakes; 0 = unlimited
```

Disable the tool entirely with `agent.disabled_toolsets: [wake]`.
