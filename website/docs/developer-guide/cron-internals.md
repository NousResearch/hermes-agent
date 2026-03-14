---
sidebar_position: 11
title: "Cron Internals"
description: "How Hermes stores, schedules, locks, and delivers cron jobs"
---

# Cron Internals

Hermes cron support is implemented primarily in:

- `cron/jobs.py`
- `cron/scheduler.py`
- `gateway/run.py`

## Scheduling model

Hermes supports:

- one-shot delays
- intervals
- cron expressions
- explicit timestamps

## Job storage

Cron jobs are stored in Hermes-managed local state with atomic save/update semantics.

## Runtime behavior

The scheduler:

- loads jobs
- computes due work
- executes jobs in fresh agent sessions
- handles repeat counters
- updates next-run metadata

In gateway mode, cron ticking is integrated into the long-running gateway loop.

## Delivery model

Cron jobs can deliver to:

- origin chat
- local files
- platform home channels
- explicit platform/chat IDs

## Locking

Hermes uses lock-based protections so concurrent cron ticks or overlapping scheduler processes do not corrupt job state.

## Related docs

- [Cron feature guide](../user-guide/features/cron.md)
- [Gateway Internals](./gateway-internals.md)
