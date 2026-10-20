"""Whether a background process will still report back to the agent that spawned it.

The session reapers keep an owner alive while this is true: the notice reaches the model only through that owner's
notification poller, and closing the owner kills the process (``agent_close``).

Pending is read from live state, never estimated with a timer. A notice is pending from the moment its process can
produce it until a consumer has finished routing it:

1. a notifying process is still running;
2. its exit is recorded but the completion is not yet published (``_completion_event`` unset; the registry keeps
   such entries out of pruning);
3. the event sits in ``completion_queue``;
4. a consumer has taken it off the queue and is still routing it (an in-flight hold, taken atomically with the
   dequeue under the queue's own lock and released only after routing returns).

Routing ends by admitting a turn (``running`` then protects the session), requeueing (state 3 again) or dropping the
notice, so every state hands over to the next without a gap and none outlives delivery.
"""

from __future__ import annotations

import collections
import contextlib
import threading

# Owner task id → number of dequeued-but-unrouted events. take_ready increments it under the queue's mutex in the same
# critical section as the dequeue, and awaits_notification reads queue and counter under that mutex too, so a reader
# sees each event either queued or held, never neither. Decrements (release) happen only after any requeue.
_in_flight: collections.Counter = collections.Counter()
_in_flight_lock = threading.Lock()  # always taken after the queue mutex, never before


def _owner(evt) -> str:
    return str(evt.get("owner_task_id") or "") if isinstance(evt, dict) else ""


def _adjust(owners, delta: int) -> None:
    with _in_flight_lock:
        for owner in owners:
            if not owner:
                continue
            _in_flight[owner] += delta
            if _in_flight[owner] <= 0:
                del _in_flight[owner]


def take_ready(queue, timeout: float) -> list:
    """Dequeue everything queued (waiting up to ``timeout`` for the first event) and hold each event's owner in the
    same critical section, so no reaper check can observe an event as neither queued nor held. The caller MUST pass
    the result to ``release`` once routing returns."""
    with queue.not_empty:
        if not queue._qsize() and timeout > 0:
            queue.not_empty.wait(timeout)
        events = list(queue.queue)
        queue.queue.clear()
        if events:
            queue.not_full.notify_all()
            _adjust((_owner(e) for e in events), +1)
    return events


def release(events) -> None:
    """End the in-flight hold ``take_ready`` took for ``events`` (after any requeue, so coverage is continuous)."""
    if events:
        _adjust((_owner(e) for e in events), -1)


@contextlib.contextmanager
def holding(owner_task_ids):
    """Hold ``owner_task_ids`` for a drain that dequeues through another API (``drain_notifications``)."""
    owners = [o for o in owner_task_ids if o]
    _adjust(owners, +1)
    try:
        yield
    finally:
        _adjust(owners, -1)


def _notifies(s) -> bool:
    return bool(s.notify_on_complete or (s.watch_patterns and not s._watch_disabled))


def awaits_notification(registry, owner_task_ids) -> bool:
    """Whether a process spawned by one of ``owner_task_ids`` still has a notice on its way. Keyed on the RAW
    spawning owner, the same field agent close kills by, so it survives compression rotating the session key."""
    owners = {o for o in owner_task_ids if o}
    if not owners:
        return False
    # (1) Running. _any_running refreshes checkpoint-recovered entries (outside the registry lock) first, so a
    # recovered PID that died unobserved reads as gone instead of pinning its owner.
    if registry._any_running(lambda s: s.owner_task_id in owners and _notifies(s)):
        return True
    with registry._lock:
        # (2) Exit recorded, completion not yet published. Entries closed without a collected exit (recovered PID
        # gone or reused) set _completion_event too, so they never count.
        if any(s.owner_task_id in owners and s.notify_on_complete and s.exited and not s._completion_event.is_set()
               for store in (registry._running, registry._finished) for s in store.values()):
            return True
    queue = registry.completion_queue
    with queue.mutex:  # (3) queued and (4) in flight, read together: take_ready moves events between them here
        with _in_flight_lock:
            if any(_in_flight.get(o) for o in owners):
                return True
        return any(_owner(evt) in owners for evt in queue.queue)
