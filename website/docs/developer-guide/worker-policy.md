---
sidebar_position: 42
title: Delegation worker policy
---

# Delegation worker policy

Out of the box, `delegate_task` children run on the configured delegation route. A plugin can add
named worker lanes (different models or efforts per task) and decide who may use them, by
registering one worker policy:

```python
def register(ctx):
    ctx.register_worker_policy("my-lanes", MyWorkerPolicy())
```

## Lifecycle

1. **Admission** - `admit(tasks, routing_cfg, delegation_cfg, *, parent, home)` runs once per batch,
   before any per-task credential resolution, provider call or child construction. `parent` is a
   frozen `ParentOrigin` snapshot (depth, session id, session DB path, model route); the policy never
   receives the live agent. Raise `ValueError` to refuse; return `(routing_cfg, tasks, None)`.
2. **Per-task route** - `task_route(task, base_cfg)` returns the credentials config for a task that
   carries selectors; `validate_credentials(task, credentials)` may still refuse after resolution.
3. **Construction** - `child_routing(task, routing_cfg)` may narrow routing (e.g. drop fallback hops);
   `configure_child(child, task, *, parent, home)` shapes the child. The host pins the admitting
   policy on the child.
4. **Lease** - every child conversation (including the schema-retry turn) runs through
   `run_child(child, **kwargs)`, so acquisition, cancellation, the conversation and release share one
   owner. A child marked with a named lane whose pinned policy is gone is refused, never run unpermitted.
5. `refresh_effort(agent)`, `parent_authorized(parent, home)` and `task_schema()` (extra
   `delegate_task` task fields) complete the contract.

## Failure behaviour

No policy, more than one policy, a disabled one, or a policy raising anything other than
`ValueError` means **default-worker-only**: any `worker`, `effort` or `work_class` selector is refused
and default tasks run on the configured route. A broken plugin can never widen a route.
