"""Whether a background process will still report back to the agent that spawned it.

The session reapers keep an owner alive while this is true: the notice reaches the model only through that owner's
notification poller, and closing the owner kills the process (``agent_close``).

Pending is read from live state, not estimated with a timer: the process is still running and notifies, its exit is
recorded but its completion not yet published, or an event it produced is still queued. Each state ends on its own
(the process exits, the publish lands, a poller dequeues the event), so an owner is never pinned after its notice is
handed over, and pruning retained process history cannot release an owner whose notice is still in flight.
"""

from __future__ import annotations


def _notifies(s) -> bool:
    return bool(s.notify_on_complete or (s.watch_patterns and not s._watch_disabled))


def awaits_notification(registry, owner_task_ids, *, include_queued: bool = True) -> bool:
    """Whether a process spawned by one of ``owner_task_ids`` still has a notice on its way. Keyed on the RAW
    spawning owner, the same field agent close kills by, so it survives compression rotating the session key.
    ``include_queued=False`` ignores already-published events — for an owner whose poller is holding them back
    (the user pressed Stop) until a prompt that only the user can send."""
    owners = {o for o in owner_task_ids if o}
    if not owners:
        return False
    # Running. _any_running refreshes checkpoint-recovered entries (outside the registry lock) first, so a recovered
    # PID that died unobserved reads as gone instead of pinning its owner.
    if registry._any_running(lambda s: s.owner_task_id in owners and _notifies(s)):
        return True
    with registry._lock:
        # Exit recorded, completion not yet queued: _move_to_finished sets _completion_event only after the publish.
        # Entries closed without a collected exit (recovered PID gone or reused) set it too, so they never count.
        if any(s.owner_task_id in owners and s.notify_on_complete and s.exited and not s._completion_event.is_set()
               for store in (registry._running, registry._finished) for s in store.values()):
            return True
    if not include_queued:
        return False
    queue = registry.completion_queue
    with queue.mutex:  # published, not yet dequeued (or requeued while the owner was busy)
        return any(isinstance(evt, dict) and evt.get("owner_task_id") in owners for evt in queue.queue)
