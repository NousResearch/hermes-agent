---
title: Scoped task surfaces
---

# Scoped task surfaces

These optional host interfaces let an external presentation plugin render live todo progress,
authored proposal briefs, durable canonical task cards, bounded blocker decisions and read-only
detail. They do not add a model tool, polling loop or task database. The Telegram Experience
plugin is the concrete consumer; its tools, renderers and browser assets remain outside the host
repository.

## Registration and ownership

`PluginContext` exposes four local capability versions, each currently `2`:
`live_todo_capability`, `task_card_capability`, `task_decision_capability` and
`task_read_capability`. It also exposes `work_presentation_capability = 1`. A consumer checks the
capability it needs before registering; it must not fall back to monkey-patching an unsupported
host.

| Registration | Ownership |
| --- | --- |
| `register_work_presentation(scope=...)` | Host validates bounded authored proposals and task publications, binds them to trusted ingress and exact runs, and owns proposal lifecycle/delivery state. The plugin supplies state and rendering. |
| `register_live_todo(factory, scope=...)` | Host binds one live run and transport; plugin renders todo snapshots. |
| `register_task_cards(factory, scope=...)` | Host owns canonical snapshots, subscription generation, receipt leases and reconciliation. |
| `register_task_decisions(callback_prefix="task:")` | Requires the same plugin's active card registration; host authenticates native callbacks and performs the bounded transition. |
| `register_task_detail(scope=...)` | Host authenticates Telegram identity and projects bounded canonical fields through the existing API application. |

Factories receive host-owned delivery handles, not bot credentials or an unrestricted adapter.
Unload callbacks revoke new admissions, including cleanup after a failed plugin registration.
A configuration-only disable is not a substitute for unloading/restarting the running consumer.

A plugin that removes its own native handlers during unload can register its platform factory
with `reload_safe=True` (also supported by `register_telegram_handler`). The host then keys wiring
to that registration generation, allowing the replacement to wire once on the same native client.
This is an explicit cleanup contract: the plugin must remove its old handlers. The default remains
qualname-based deduplication across rediscovery, preserving existing plugins that do not own cleanup.
Rebuilding the native client still wires the current factories afresh.

## Exact admission scope

Routes are always required. Exact task resources are optional for authored work briefs and
todo-only consumers; legacy card/detail registration still requires them:

```yaml
scope:
  routes:
    - profile: default
      platform: telegram
      chat_id: "-1000000000001"
      thread_id: "7"
  task_resources: []
```

These are synthetic examples. Profile, platform, chat and topic match exactly. A null topic is an
exact no-topic route, never a wildcard. Missing, invalid or duplicate entries deny registration.
The host checks scope before todo binding, card receipt/cursor ownership, proposal transport and
read admission. A work-presentation registration can create cards for its own exact published
tasks without adding each task to YAML. Out-of-scope subscriptions retain their existing ordinary
notification path.

Scope does not grant reads or actions. With `work_briefs: true`, a signed read additionally
requires the configured bot to be an administrator and the requesting actor to be a current
member of the exact group. Membership and route/resource identity are checked before and after
projection, without a positive cross-request cache. Read authority never grants actions.

When decisions are enabled, a trusted publication can derive one exact blocker action grant for
its initiating actor, task incarnation, route, message and current control generation. Missing or
ambiguous initiators deny controls. Existing `kanban.decision_grants` and `kanban.read_grants`
remain the compatibility path when authored work presentation is not registered. Telegram-signed
initData establishes identity, not resource authority.

Proposal approval binds the stored source digest, proposal incarnation and revision and requires
that exact revision to have confirmed delivery to its durable card. A pending, failed or unknown
edit cannot be approved as though it were displayed. Approval does not rewrite the authored brief.

## Persistence and recovery

Task publications are explicit bounded payloads on canonical task events; arbitrary bodies,
comments, transcripts and raw results are never projected. Proposal records use the registering
plugin's existing state facade. Durable task receipts and action audit records use the existing
canonical board database. No second task store or polling loop is introduced.

A sent card is reconciled by its verified message ID and normalized full rendered payload. Action
controls use a separate hash populated only by confirmed transport settlement. Ambiguous attempts
clear confirmed equivalence and remain quarantined rather than being blindly resent. A newer
revision can recover only the bounded, positively classified known-message/not-modified case.
Legacy receipt migration does not invent a successful send or action authority. Live todo bubbles
are run-scoped and are not reconstructed after a process crash.

Read-only detail excludes task bodies, transcripts and arbitrary metadata. It never mutates
canonical task/event/action data; normal SQLite read-only WAL access may create coordination files.
Do not use SQLite `immutable=1` on a live WAL database.

A host rollback must preserve current databases and legitimate activity. Do not restore an old
whole-profile snapshot as a routine code rollback. Prove old-host reopening for the exact tested
schema and retain the host/patch identity alongside the separately installed plugin artifact.

## Validation

Host contract tests cover admission, policy revocation, transport fences, durable receipts,
recovery and bounded decisions. Tests requiring the separate plugin skip when it is absent;
the plugin repository's installed-host CI runs those against its wheel. Browser assets and
real Telegram/Mini App/mobile acceptance belong to that consumer's delivery gate. Unit tests
or a patched host alone do not establish live client behavior or catalog availability.
