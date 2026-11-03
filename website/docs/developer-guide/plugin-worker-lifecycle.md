---
title: Plugin worker lifecycle
description: Cooperative background threads with reload fencing and confirmed shutdown.
---

# Plugin worker lifecycle

Use `ctx.worker_scope()` when a Python plugin owns background threads whose old
and new loads must not overlap (for example a durable notification sender).
`on_unload()` is a cleanup callback, not a confirmation that arbitrary threads
stopped; `spawn_task()` retains its existing asyncio cancellation-only contract.
This API does not add a scheduler, retry policy, notification service or tool.

```python
def register(ctx):
    # Acquiring stops and joins the preceding load before returning. A shutdown
    # timeout raises RuntimeError and no replacement scope is granted.
    workers = ctx.worker_scope(timeout=5.0)

    def consume(stop):
        while not stop.wait(0.2):
            if not workers.is_active():
                continue
            # Bounded work. Recheck is_active immediately before external I/O.
            # Use the plugin's own durable claim/commit and recovery protocol.

    workers.start_thread(consume, name="my-plugin-worker")
    # Finish durable initialization/recovery before enabling external effects.
    if not workers.activate():
        raise RuntimeError("plugin worker scope was revoked")
```

## Contract

- One cached scope per context, one current owner per profile home and plugin
  identity in the execution process. The coordinator belongs to Hermes, not the
  reloadable plugin module. `instance_id` is a fresh opaque identifier for this
  load; it is not a conversation/turn generation or an authorization credential.
- `start_thread(target, name=...)` passes a shared `threading.Event` to `target`,
  binds the owning profile and preserves the spawning context. Threads are
  daemon threads. All participating worker threads must be created this way;
  arbitrary plugin-created threads, subprocesses and asyncio tasks are not tracked.
- The scope starts inactive. `activate()` returns whether it still owns the slot;
  `is_active()` reads current host-owned readiness, including from old closures.
  An inactive scope must not send. Already admitted network I/O may complete
  during shutdown; this is not transactional cancellation of remote side effects.
- Unload invokes `close()`, which revokes readiness before signalling stop and
  joining every managed thread. Its total join budget is `timeout` seconds
  (finite, greater than zero and at most 30). This is the scope's own shutdown
  budget, not a deadline on acquisition: acquiring a replacement honors the
  preceding scope's configured shutdown budget, plus serialization time. It
  returns `True` only if all managed threads have exited. A closed scope cannot
  be reactivated.
- Replacement acquisition also closes and joins the preceding owner, even if its
  unload callback never ran. If any old thread remains alive, acquisition raises
  and the old scope stays revoked. Once those threads really exit a fresh context
  may acquire a replacement. Python threads cannot be forcibly terminated.
- Scope operations are serialized against worker creation and replacement. Closing
  an old scope after replacement cannot revoke the new owner. A worker calling
  its own scope's `close()` receives `False`, not a successful self-join.
- Under `plugins.isolation: host`, the scope and thread handles stay inside the
  plugin host process, not across RPC. The supervisor confirms an old host process
  has exited before starting its replacement after transport loss.

Check `callable(getattr(ctx, "worker_scope", None))` on older hosts. If unavailable,
disable the affected background channel and report the unsupported capability;
do not invent a module-global coordinator or fall back to untracked threads.
This is lifecycle management for trusted plugin code, not a security sandbox.
