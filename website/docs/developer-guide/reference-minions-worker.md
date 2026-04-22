---
title: "Reference Minions Worker"
description: "Tiny file-spool worker and enqueue shim for wiring Hermes to external queues during development"
---

# Reference Minions Worker

This is the smallest working bridge between Hermes and an external worker lane.

It is not fancy. That’s the point.

Use it to prove the queue contract end to end before wiring Hermes into real GBrain Minions, BullMQ, Railway workers, or whatever else you want to run.

## What it gives you

- one enqueue shim that accepts Hermes job envelopes on stdin
- one worker that processes queued jobs from a local spool directory
- one default runner that turns queue payloads into completion envelopes
- callback delivery through Hermes' own callback helper

Files:

- `scripts/minions_reference_enqueue.py`
- `scripts/minions_reference_worker.py`
- `agent/minions_reference.py`

## Spool layout

By default the reference worker uses:

`~/.hermes/minions-reference/`

Structure:

- `queue/<kind>/<task_id>.json`
- `completed/<kind>/<task_id>.json`

Set `HERMES_MINIONS_SPOOL_DIR` to override it.

## Wire Hermes into the reference enqueue shim

Point Hermes at the generic enqueue script:

```bash
export HERMES_BACKGROUND_BACKEND=command
export HERMES_BACKGROUND_ENQUEUE_CMD='python scripts/minions_reference_enqueue.py'

export HERMES_DELEGATION_BACKEND=command
export HERMES_DELEGATION_ENQUEUE_CMD='python scripts/minions_reference_enqueue.py'

export HERMES_CRON_BACKEND=command
export HERMES_CRON_ENQUEUE_CMD='python scripts/minions_reference_enqueue.py'
```

That’s enough for Hermes to stop executing those lanes locally and start dropping envelopes into the spool queue.

## Run the worker

Process one queued job:

```bash
python scripts/minions_reference_worker.py --once
```

Typical output:

```json
{"version":"1.0","kind":"background","task_id":"bg_...","status":"succeeded",...}
```

If no work is waiting:

```json
{"status":"idle"}
```

## End-to-end smoke test

1. export the backend env vars above
2. start Hermes
3. run `/background summarize this repo`
4. in another shell, run:

```bash
python scripts/minions_reference_worker.py --once
```

Expected result:
- Hermes enqueues the job instead of running it inline
- worker writes a completion file under `completed/background/`
- callback delivery sends the result back through Hermes

## What the default worker actually does

The reference runner is intentionally dumb:

- background -> echoes the prompt as completed work
- delegation -> echoes the delegated goal
- cron -> echoes the cron job name

That makes it runnable with zero external dependencies.

If you want real execution, replace the default runner with your own worker logic.

## Recommended upgrade path

Use this reference lane only to validate the protocol and callback flow.
Then replace the storage/runner pieces in this order:

1. file spool -> BullMQ / GBrain / Postgres queue
2. default runner -> real task executor
3. ad hoc worker loop -> supervised process / container / serverless job

Do not change the envelope format while upgrading. The protocol is the stable seam.

## Why this exists

Because architecture diagrams don’t execute shit.
You need a tiny working lane before you build the serious one.
