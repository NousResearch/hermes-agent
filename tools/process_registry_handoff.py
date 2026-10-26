"""Transfer a delegated child's live process authority durably to its parent."""

from typing import Optional


_MAX_HANDOFFS_PER_CHILD = 3
OWNERSHIP_FIELDS = ('owner_task_id', 'task_id', 'session_key', 'parent_session_id', 'handoff_note', 'ownership_generation')


def merge_ownership(entries, previous):
    """A native process's committed handoff survives older checkpoint/result writers."""
    prior = {entry.get('session_id', entry.get('id')): entry for entry in previous}
    updates = {}
    for entry in entries:
        sid = entry.get('session_id', entry.get('id'))
        old = prior.get(sid)
        if not old or old.get('ownership_generation', 0) <= 0:
            continue
        if old['ownership_generation'] < entry.get('ownership_generation', 0):
            continue
        identity = (entry.get('pid'), entry.get('host_start_time'))
        if (any(type(value) is not int or value <= 0 for value in identity)
                or identity != (old.get('pid'), old.get('host_start_time'))):
            raise ValueError('handed process identity mismatch')
        update = {field: old.get(field, '' if field != 'ownership_generation' else 0) for field in OWNERSHIP_FIELDS}
        entry.update(update)
        updates[sid] = update
    return updates


def transfer_ownership(registry, session_id, *, from_owner, to_owner, to_task_id,
                       to_session_key, to_parent_session_id, note, child_stop_event):
    session = registry.get(session_id)
    with registry._lock:
        if session is None or session.exited or session.owner_task_id != from_owner or not to_parent_session_id:
            return None
        if child_stop_event is not None and child_stop_event.is_set():
            return None
        if any(type(value) is not int or value <= 0 for value in (session.pid, session.host_start_time)):
            return None
        previous = tuple(getattr(session, field) for field in OWNERSHIP_FIELDS)
        target = (to_owner, to_task_id, to_session_key, to_parent_session_id, note, session.ownership_generation + 1)
        for field, value in zip(OWNERSHIP_FIELDS, target):
            setattr(session, field, value)
        # A reader may mark exited while settlement waits for the registry lock.
        # Its native result must inherit this committed parent before retirement.
        committed = registry._write_checkpoint(retain_ids=(session.id,), _lock_held=True)
        if not committed:
            for field, value in zip(OWNERSHIP_FIELDS, previous):
                setattr(session, field, value)
        if not committed or tuple(getattr(session, field) for field in OWNERSHIP_FIELDS) != target:
            return None
        return session


def handoff_process(session_id: str, args: dict, task_id: Optional[str]) -> dict:
    """Require a live child owner and parent before moving completion routing."""
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
    from tools.process_registry import process_registry
    from tools.terminal_tool import _resolve_container_task_id

    with _active_subagents_lock:
        record = _active_subagents.get(str(task_id or ""))
    child = record.get("agent") if record else None
    parent_ref = getattr(child, "_delegate_parent_ref", None)
    parent = parent_ref() if callable(parent_ref) else None
    if parent is None:
        return {"error": "handoff is only available to a running subagent with a live parent; you are not one."}
    parent_owner = str(getattr(parent, "_current_task_id", "") or getattr(parent, "session_id", "") or "")
    if not parent_owner:
        return {"error": "parent has no process owner id yet; retry after the parent's turn has started."}
    parent_session = str(record.get('owner_session_id') or getattr(parent, 'session_id', '') or '')
    if not parent_session:
        return {'error': 'parent has no native session identity for this process handoff.'}
    handed = getattr(child, "_handed_off_processes", None)
    if handed is None:
        handed = child._handed_off_processes = []
    if len(handed) >= _MAX_HANDOFFS_PER_CHILD:
        return {"error": f"handoff cap reached ({_MAX_HANDOFFS_PER_CHILD} per subagent); wait on or kill the rest yourself."}
    note = str(args.get("data") or "").strip()
    if not note:
        return {"error": "handoff requires `data`: one sentence saying what the process is for and what the parent should do with its result."}
    session = process_registry.transfer_ownership(
        session_id, from_owner=str(task_id or ""), to_owner=parent_owner,
        to_task_id=_resolve_container_task_id(parent_owner),
        to_session_key=str(getattr(parent, "_gateway_session_key", "") or getattr(parent, "session_id", "") or ""),
        to_parent_session_id=parent_session,
        note=note,
        child_stop_event=getattr(child, "_hard_interrupt_requested", None))
    if session is None:
        return {"error": f"cannot hand off {session_id}: process exited, ownership changed, or its child was stopped; read its result with poll/log and report it instead."}
    handed.append({"session_id": session.id, "command": session.command, "note": note})
    return {"status": "handed_off", "session_id": session.id, "command": session.command,
            "note": "Your parent now owns this process and will receive its completion; you will not. Mention the handoff "
                    "in your final answer."}


def ownership_identity(session):
    """Immutable native scope captured by an owner-selected control."""
    return tuple(getattr(session, field) for field in (*OWNERSHIP_FIELDS, 'pid', 'host_start_time'))


def refresh_ownership(registry):
    """Read committed profile-local identity before a cached owner selects work.

    The file-locked observation is the control's ownership boundary. Release it
    before registry callbacks or process waits; a concurrent transfer afterward
    may legitimately follow this control in the native ordering.
    """
    import json
    from hermes_cli.active_sessions import _FileLock
    from tools.process_registry import _checkpoint_path
    from tools.process_registry_results import completed_ownership
    with registry._lock:
        entries = [dict(session_id=p.id, **dict(zip((*OWNERSHIP_FIELDS, 'pid', 'host_start_time'), ownership_identity(p))))
                   for p in (*registry._running.values(), *registry._finished.values())]
    if not entries:
        return True
    try:
        path = _checkpoint_path()
        with _FileLock(path.with_name(path.name + '.lock')):
            try:
                previous = json.loads(path.read_text(encoding='utf-8-sig'))
            except FileNotFoundError:
                previous = []
            if not isinstance(previous, list) or any(not isinstance(entry, dict) for entry in previous):
                raise ValueError('invalid native process checkpoint')
            updates = merge_ownership(entries, previous)
            updates.update(merge_ownership(entries, completed_ownership(entries)))
        identities = {entry['session_id']: (entry['pid'], entry['host_start_time']) for entry in entries}
        with registry._lock:
            for sid, update in updates.items():
                current = registry._running.get(sid) or registry._finished.get(sid)
                if current is not None and current.ownership_generation <= update['ownership_generation']:
                    if (current.pid, current.host_start_time) != identities[sid]:
                        return False
                    for field, value in update.items():
                        setattr(current, field, value)
        return True
    except (OSError, ValueError, TypeError):
        return False


def check_control_ownership(registry, session, expected, consume_output):
    """Revalidate a selected native target immediately before signalling it."""
    if expected is None:
        return None  # Explicit profile-wide process_manage actions retain their native policy.
    if not refresh_ownership(registry):
        return {'status': 'error', 'error': 'process_stop_unknown', 'session_id': session.id}
    with registry._lock:
        current = ownership_identity(session)
        if current != expected:
            generation = OWNERSHIP_FIELDS.index('ownership_generation')
            # Only a committed ownership transfer makes this job foreign. A
            # changed/unproved process birth still belongs behind the Stop fence.
            if current[generation] > expected[generation] and current[:4] != expected[:4]:
                return {'status': 'ownership_changed', 'session_id': session.id}
            return {'status': 'error', 'error': 'process_stop_unknown', 'session_id': session.id}
        if consume_output:
            registry._completion_consumed.add(session.id)
    return None
