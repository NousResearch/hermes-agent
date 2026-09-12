"""Owner-lifetime observations for the UI backend's automatic session cleanup."""


def has_owned_work(registry, owner_task_ids) -> bool:
    """In-memory only: callers may hold lifecycle locks; never probe a PID here.

    A reader can mark exited before moving/publishing its result. Keep the owner
    through that transition and, for notify jobs, until the continuation accepts
    the result. Polling is observation, not notification delivery.
    """
    owners = frozenset(owner_task_ids)
    with registry._lock:
        if any(s.owner_task_id in owners for s in registry._running.values()):
            return True
        return any(s.owner_task_id in owners and (
            not s._completion_event.is_set() or (
                s.notify_on_complete and not s.notification_delivered
                and s.id not in registry._completion_consumed))
            for s in registry._finished.values())


def refresh_owned_work(registry, owner_task_ids) -> None:
    """Refresh recovered processes outside the UI lifecycle locks."""
    owners = frozenset(owner_task_ids)
    with registry._lock:
        detached = [s for s in registry._running.values() if s.owner_task_id in owners and s.detached]
    for session in detached:
        registry._refresh_detached_session(session)


def acknowledge_notifications(registry, events) -> None:
    """Admission receipt, separate from wait/log consumption of process output."""
    with registry._lock:
        for event in events:
            session = registry._finished.get(event.get("session_id"))
            if session is not None:
                session.notification_delivered = True
