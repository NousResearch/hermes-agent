---
title: "Job Queue Protocol"
description: "Canonical enqueue and completion payloads for external queue backends like GBrain Minions"
---

# Job Queue Protocol

Hermes can hand work to an external queue/worker system instead of executing everything in-process. This page defines the **canonical payloads** for that handoff.

Use this when wiring Hermes to **GBrain Minions**, BullMQ workers, or any other durable execution layer.

## Why this exists

Without a contract, every queue bridge turns into bespoke JSON sludge.
That gets old fast.

The protocol solves three things:

1. **enqueue shape** — what Hermes sends to workers
2. **callback shape** — what workers send back after execution
3. **routing contract** — how a worker knows where the result should go

## Version

Current protocol version:

`1.0`

Workers should reject unknown major versions instead of guessing.

## Job kinds

Hermes currently emits three queueable job kinds:

- `background`
- `delegation`
- `cron`

All three share the same envelope.

## Enqueue envelope

Top-level fields:

- `version` — protocol version
- `kind` — `background | delegation | cron`
- `task_id` — Hermes-generated stable ID for this queued task
- `payload` — kind-specific body
- `callback` — how the worker should report completion

### Background job example

```json
{
  "version": "1.0",
  "kind": "background",
  "task_id": "bg_143022_a1b2c3",
  "payload": {
    "prompt": "Analyze the logs in /var/log and summarize any errors from today",
    "origin": "gateway",
    "platform": "telegram",
    "session_id": "bg_143022_a1b2c3",
    "user_id": "123",
    "user_name": "victor",
    "chat_id": "456",
    "thread_id": "789"
  },
  "callback": {
    "type": "platform",
    "target": {
      "platform": "telegram",
      "chat_id": "456",
      "thread_id": "789"
    }
  }
}
```

### Delegation job example

```json
{
  "version": "1.0",
  "kind": "delegation",
  "task_id": "delegate_1712345678_0",
  "payload": {
    "goal": "Research topic A",
    "context": "Focus on pricing and new launches",
    "toolsets": ["web", "file"],
    "model": "gpt-5.4",
    "max_iterations": 45,
    "task_index": 0,
    "task_count": 2
  },
  "callback": {
    "type": "session",
    "session_id": "sess_123",
    "platform": "telegram"
  }
}
```

### Cron job example

```json
{
  "version": "1.0",
  "kind": "cron",
  "task_id": "cron_job-123_20260421_154500",
  "payload": {
    "job_id": "job-123",
    "job_name": "daily digest",
    "prompt": "Check the feeds and summarize anything new.",
    "schedule_display": "0 9 * * *",
    "deliver": "telegram:-1001:55",
    "origin": {
      "platform": "telegram",
      "chat_id": "123",
      "thread_id": "44"
    },
    "model": null,
    "skills": ["blogwatcher"],
    "script": null
  },
  "callback": {
    "type": "platform",
    "target": {
      "platform": "telegram",
      "chat_id": "-1001",
      "thread_id": "55"
    }
  }
}
```

## Callback contract

Workers should return completion data using the same routing callback Hermes originally provided.

Top-level fields:

- `version`
- `kind`
- `task_id`
- `status`
- `summary`
- `final_output`
- `error` (only when present)
- `artifacts`
- `metadata`
- `callback`

### Completion example

```json
{
  "version": "1.0",
  "kind": "background",
  "task_id": "bg_143022_a1b2c3",
  "status": "succeeded",
  "summary": "Found 3 production errors",
  "final_output": "Found 3 production errors in the last 24h...",
  "artifacts": [
    {"kind": "file", "path": "/tmp/error-report.md"}
  ],
  "metadata": {
    "attempt": 1,
    "worker": "minion-7"
  },
  "callback": {
    "type": "platform",
    "target": {
      "platform": "telegram",
      "chat_id": "456",
      "thread_id": "789"
    }
  }
}
```

## Callback types

### `platform`

Worker should send the final result to a platform destination.

Supported target shape:

```json
{
  "type": "platform",
  "target": {
    "platform": "telegram",
    "chat_id": "456",
    "thread_id": "789"
  }
}
```

Hermes includes a helper in `agent.job_callbacks.deliver_completion()` that converts this into the same delivery path cron uses.

### `session`

Worker completed a delegated subtask that should be reconciled back into a Hermes session.

Current behavior:
- Hermes appends the result into the session transcript (`~/.hermes/sessions/<session_id>.jsonl` and SQLite when available)
- if the session can be resolved back to a messaging origin, Hermes also delivers the final text back to that chat/thread

That gives queued delegation a real closed loop instead of a dead drop.

### `none`

Worker should not deliver anything back automatically.

## Worker expectations

A clean Minions worker should:

1. parse the envelope
2. branch on `kind`
3. execute using the `payload`
4. build a completion envelope
5. honor `callback`

## Recommended Minions adapter shape

For a BullMQ-style worker, keep the bridge thin:

- Hermes enqueue command: stdin envelope -> queue add
- Minions worker: queue job -> execute -> completion envelope
- Delivery step: call Hermes helper or your own platform sender

Do **not** mutate the protocol per queue backend. BullMQ, Redis streams, Postgres jobs, and SQS should all carry the same logical envelope.

## Code references

Canonical builders live in:

- `agent.job_protocol`
- `agent.background_jobs`
- `agent.job_callbacks`

For a tiny working example, see:

- `agent.minions_reference`
- `scripts/minions_reference_enqueue.py`
- `scripts/minions_reference_worker.py`
- [Reference Minions Worker](/docs/developer-guide/reference-minions-worker)

For the real HTTP transport shape, see:

- `agent.gbrain_minions_transport`
- `scripts/gbrain_minions_enqueue.py`
- `scripts/gbrain_minions_complete.py`
- [GBrain / Minions HTTP Transport](/docs/developer-guide/gbrain-minions-http-transport)

For deterministic zero-token cron work, see:

- `agent.gbrain_cli_jobs`
- `cron/scheduler.py` (`job.shell` support + auto-routing for script-only crons)
- [Morph GBrain Shell Jobs into Hermes Cron](/docs/developer-guide/gbrain-shell-jobs-for-hermes-cron)

Those are the source of truth. If docs and code ever disagree, the code wins.
