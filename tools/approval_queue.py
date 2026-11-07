"""Approval queue predicates; caller holds tools.approval._lock."""


def resolved_surface_is_idle(queues, session_key, targets) -> bool:
    """Only resume a surface when every currently queued wait for it is selected."""
    groups = {entry.wait_group if entry.wait_group is not None else session_key for entry in targets}
    selected = set(targets)
    return not any(
        entry not in selected and (entry.wait_group if entry.wait_group is not None else key) in groups
        for key, queue in queues.items() for entry in queue
    )


def clear_idle_typing_pause(adapter, chat_id) -> None:
    """Turn cleanup cannot release another session's shared approval pause."""
    from tools import approval

    # Serialize the check and discard with enqueue; worker-thread pauses may
    # otherwise race cleanup just as they can race approval resolution.
    with approval._lock:
        group = (adapter, chat_id)
        if not any(entry.wait_group == group
                   for queue in approval._gateway_queues.values() for entry in queue):
            adapter._typing_paused.discard(chat_id)
