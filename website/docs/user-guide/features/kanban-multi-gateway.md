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

## Shared boards between installs (assignee namespaces)

Two Hermes installs on one host (different OS users, different `~/.hermes`) can
each see the same board by symlinking the board directory into both
`<root>/kanban/boards/` roots. A profile's identity there is the PAIR
`<namespace>:<profile>`; the namespace names the install whose dispatcher should
run the card.

```yaml
# ~/.hermes/config.yaml — this install's namespace
kanban:
  namespace: em          # ours; the other install uses its own token, e.g. yummi
```

```json
// the shared board's board.json — the namespace a BARE assignee means here
{ "slug": "shared-project", "namespace": "yummi" }
```

Resolution (`kanban_db_dispatch.resolve_assignee`), applied in the dispatcher
immediately before the profile-exists check:

- `ns:profile` — canonical, and an explicit override: claimed only when `ns` is
  one of this install's namespace tokens; the spawned profile is `profile`.
- bare name on a board that declares a namespace — resolves to
  `<board namespace>:profile`, so it is claimed by that install only.
- bare name on a board with no declared namespace — unchanged behaviour
  (partition by profile name), EXCEPT an install-relative name (`default`), which
  is ambiguous by construction and therefore refused.
- the `default` board never needs a declaration: its DB is
  `<this install root>/kanban.db` by construction, so no other install can reach
  it.

Every refusal is visible: a `kanban dispatch: task … NOT claimed` warning each
tick plus one durable `dispatch_skipped` event per (task, assignee, reason) with
the remedy, and a `skipped_namespace` line in `hermes kanban dispatch` output —
never a silent bucket. Tokens are parsed case-insensitively (`Yummi:default` ==
`yummi:default`).
