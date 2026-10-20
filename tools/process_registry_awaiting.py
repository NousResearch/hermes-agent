"""Whether a background process will still report back to the agent that spawned it.

The session reapers keep an owner alive while this is true: the completion reaches the model only through that
owner's notification poller, and closing the owner kills the process (``agent_close``).
"""

from __future__ import annotations

import time

# How long an owner stays "awaited" after its notifying process exits: covers publish → poller dequeue →
# turn claim (sub-second when idle). Bounded so an undeliverable notice cannot pin its owner.
NOTIFY_DELIVERY_GRACE_SECONDS = 60.0


def awaits_notification(registry, owner_task_ids, *, delivery_grace: float = NOTIFY_DELIVERY_GRACE_SECONDS) -> bool:
    """Whether a process spawned by one of ``owner_task_ids`` will still report back: it notifies
    (``notify_on_complete`` or live ``watch_patterns``) and is running, or exited less than ``delivery_grace``
    seconds ago — its completion is published after the exit is recorded and becomes an agent turn only once a
    poller dequeues it, so the owner must outlive that hand-off too. Keyed on the RAW spawning owner, the same
    field agent close kills by, so it survives compression rotating the session key."""
    owners = {o for o in owner_task_ids if o}
    if not owners:
        return False
    now = time.time()
    with registry._lock:
        sessions = [*registry._running.values(), *registry._finished.values()]
    return any(
        s.owner_task_id in owners
        and (s.notify_on_complete or (s.watch_patterns and not s._watch_disabled))
        and (not s.exited or now - (s.exited_at or now) < delivery_grace)
        for s in sessions)
