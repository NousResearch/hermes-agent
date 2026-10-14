---
sidebar_position: 18
title: Same-worker model recovery
---

# Same-worker model recovery

Gateway goal workers can try a different approved model at a completed-turn boundary without replacing the conversation, implementation owner, checkpoint or original goal budget. This is opt-in and **gateway-only**. CLI/TUI/Desktop retain normal goal recovery; they do not claim automatic worker model handoff support.

## Configure deliberately

Append this under your existing `goal_recovery` settings using Hermes' config writer. Do not replace other configuration or copy credentials between profiles.

```yaml
goal_recovery:
  auto_effects: [read_only, local_edit]
  judge_backoff: true
  model_handoff:
    enabled: false
    session_ids: []   # exact current authoritative goal session IDs; empty means no workers
    max_attempts: 3   # 1–3; never more than three
    routes: []       # ordered {provider: <configured-provider>, model: <configured-model-id>}
```

Populate only already configured routes and owning-session IDs. Then enable after a qualified pilot. API keys, endpoint URLs and extra route fields are rejected. Defaults do not automatically change any existing worker's model.

An existing recovery policy must supply an automatic, evidenced, authorized low-risk read-only/local-edit option for a service, dependency or interpretation blocker. Model switching does not replace that policy or lower its confidence/authorization thresholds. Permission/ownership and hard resource-floor blocks do not trigger this feature. If no option qualifies, the goal retains its ordinary pause/choices.

## What happens

1. Save a revision-bound reservation and prior/target non-secret route. The original worker does not continue until commit.
2. Recheck current owner, profile, session routing, user goal controls and allowed routes.
3. Resolve the target through the existing model-switch path. Run one bounded harmless tool-generation probe; no tool is executed. Reject unavailable routes, unqualified adoption and provider/model fallback.
4. Persist the non-secret session override, apply the cached-agent model switch and checkpoint the committed transition.
5. Queue a canonical standing-goal continuation for the **same worker**. It names the recommended bounded step and required verification; approvals/gates remain intact.
6. Claim that continuation once before execution. User input wins and is never overwritten by an automated prompt.
7. A repeated qualified block may try the next untried route, up to three reserved attempts. The count survives reload/restart. Failures are audited and bounded rather than silently retried.

Current goal requirements, completion checks, owners, turn caps and tool approvals are not reset. A model change can lose that provider's prompt cache, but canonical history is not rewritten and global defaults are not changed.

## Restart and user control

A durable pending reservation is restored through the gateway's existing authorized restart-resume queue. A previously persisted target is adopted only when its successful tool-generation qualification was recorded. No new reservation or duplicate worker is created.

New pause/clear/model changes win over pending automation. A normal user event is kept intact; a stale tagged automated event is rejected. Explicit user resume can start a new recovery episode while retaining the audit. Automated recovery does not call resume or extend the original budget.

## Inspect and rollback

Goal status/notifications show attempt N/3, target model and recommended action. Goal state retains the attempt audit and exact transition. After the cap or a failed/unavailable route, the checkpoint is preserved and a specific input/repair is requested.

Disable `goal_recovery.model_handoff.enabled` to prevent further automatic handoffs. Do not delete the goal, its history, keys or worktree. A manually chosen valid model is not replaced with a global default.

## Deployment boundary

Code must be loaded by the gateway before enabling this configuration. Do not forcibly restart healthy project workers just to load it. Validate provider generation and use a checkpointed activation/pilot. Doctor storage/auth/dependency issues are separate; changing a model never substitutes for repairing them.
