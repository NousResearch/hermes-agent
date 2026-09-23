---
title: "Kanban Multi-Gateway Deployment"
description: "Running one kanban board across several per-profile gateways: single dispatcher, profile-owned delivery"
---

# Multi-gateway deployment

Hermes supports multiple gateway processes running concurrently — one per profile
(default, writer, admin, coder, researcher). Each gateway opens its own connection
to platform APIs and delivers messages for its profile's subscribers.

Task subscriptions also cover review feedback. A `changes_requested` review
event is delivered as an actionable review-BLOCK notification. Subscriptions
using `notify+wake` additionally wake the exact originating chat/thread/session
so the controller inspects the existing card and current run; `notify` remains
passive-only and `wake` remains wake-only. Review feedback never creates,
unblocks, requeues, or otherwise mutates a task.

## Single-dispatcher posture

Only one gateway owns the kanban dispatcher. The owning gateway keeps
`kanban.dispatch_in_gateway: true` (the default); every other gateway sets it
to `false`.

**Why this matters:** dispatching is single-owner so multiple gateways do not
race to spawn the same work. Notification delivery is profile-owned instead:
each gateway polls only subscriptions for profiles whose platform adapters it
hosts. The atomic event claim prevents duplicate delivery across watcher
processes.

### Ownership is re-evaluated every tick

The flag and the machine-global singleton lock (`<kanban home>/kanban/
.dispatcher.lock`) are read together on **every dispatcher tick**, not once at
boot. Three consequences worth relying on:

- Flipping `dispatch_in_gateway` takes effect on the next tick. The gateway
  losing the role releases the lock; the gateway gaining it picks the lock up.
  Neither side needs a restart.
- A gateway that loses the startup race keeps retrying instead of giving up for
  its process lifetime, so a freed lock is always picked up by a gateway that
  wants it.
- A gateway whose config enables dispatch but cannot get the lock logs a
  **WARNING** naming the lock path, repeated hourly while the condition holds.
  A silent fleet with a full ready queue is the failure mode this replaces.

`HERMES_KANBAN_DISPATCH_IN_GATEWAY=false` remains a permanent per-process
opt-out: it is read once at startup and that gateway never touches the lock.

## Configuration

On the dispatch-owning gateway (typically the `default` profile), no change is
needed. On every other profile gateway, add to `~/.hermes/config.yaml`:

```yaml
kanban:
  dispatch_in_gateway: false
```

Or set the env var: `HERMES_KANBAN_DISPATCH_IN_GATEWAY=false`

## What each gateway does

| Gateway role | dispatch_in_gateway | Opens subscribed board DBs? | Dispatcher | Notifier |
|---|---|---|---|---|
| default (confirmed dispatch-lock owner) | true (default) | yes | yes | owned profiles + legacy unstamped subscriptions |
| writer, admin, coder, etc. | false | yes, when the profile has subscriptions | no | that gateway's owned profiles |

Non-dispatch gateways still deliver messages for their own platform adapters
(Telegram, Discord, etc.). They do not dispatch tasks, and they skip boards
that have no subscriptions owned by their profiles.
