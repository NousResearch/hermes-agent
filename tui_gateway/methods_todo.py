"""Live, user-initiated todo state changes for shared clients."""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("todo.list_open_sessions")
def _list_open_sessions(rid, params):
    """Stored sessions with an unfinished todo list, for the desktop Task Overview sidebar's
    startup backfill (`session-todos-overview.ts` only captures sessions its own window has
    actually observed live; this covers everything else already sitting in the DB)."""
    limit = params.get("limit", 100) if isinstance(params, dict) else 100
    try:
        limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        limit = 100
    with _profile_db(params) as db:
        if db is None:
            return _err(rid, 5006, "session store unavailable")
        try:
            sessions = db.sessions_with_open_todos(limit=limit)
        except Exception as e:
            return _err(rid, 5006, str(e))
    return _ok(rid, {"sessions": sessions})


def _find_owned_session(stored_session_id, transport):
    """Live session (dict) owned by ``stored_session_id`` on the calling transport, or None."""
    with _sessions_lock:
        return next((candidate for candidate in _sessions.values()
                     if candidate.get("session_key") == stored_session_id
                     and _session_transport_contains(candidate, transport)), None)


@method("todo.cancel_item")
def _cancel_item(rid, params):
    """Cancel one todo on an owned live session and steer an active agent away from it."""
    stored_session_id = _str_param(params, "stored_session_id")
    item_id = _str_param(params, "item_id")
    if not stored_session_id:
        return _err(rid, 4000, "stored_session_id required")
    if not item_id:
        return _err(rid, 4000, "item_id required")

    transport = current_transport()
    if transport is None:
        return _err(rid, 4001, "live session not found or not owned by this transport")
    session = _find_owned_session(stored_session_id, transport)
    agent = session.get("agent") if session else None
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return _err(rid, 4001, "live session not found or not owned by this transport")

    item = next((todo for todo in store.read() if todo.get("id") == item_id), None)
    if item is None:
        return _err(rid, 4004, "todo item not found")

    todos = store.write([{"id": item_id, "status": "cancelled"}], merge=True)
    revision = store.snapshot()["revision"]
    notice = (
        f"[The user cancelled todo item {item_id}: {item['content']}. "
        "Do not continue this task; re-read todo_list before choosing further work.]"
    )
    notified = False
    if getattr(session, "get", None) and session.get("running"):
        try:
            notified = bool(agent.steer(notice))
        except Exception:
            logger.debug("todo cancellation steer failed", exc_info=True)
    _cache_todo_state(session, {"todos": todos, "revision": revision})
    _emit("todo.updated", next(sid for sid, candidate in _sessions.items() if candidate is session), {
        "todos": todos,
        "revision": revision,
    })
    return _ok(rid, {"item_id": item_id, "status": "cancelled", "revision": revision, "notified": notified})


@method("todo.move_item")
def _move_item(rid, params):
    """Move one todo item from one owned live session's list to another's (desktop Task
    Overview drag-and-drop). Both sessions must be live and owned by the calling transport —
    the TodoStore is in-memory per AIAgent, so a session with no live agent has nothing to
    move to/from. The source item is actually REMOVED (unlike cancel, which only flips
    status) since a moved item belongs to its new session now, not a cancelled leftover in
    the old one. The destination gets a fresh id (todo ids are only unique within one
    session's list) with the source's content/parent dropped (a subtask's parent id is
    meaningless without its former siblings)."""
    from_session_id = _str_param(params, "from_session_id")
    to_session_id = _str_param(params, "to_session_id")
    item_id = _str_param(params, "item_id")
    if not from_session_id:
        return _err(rid, 4000, "from_session_id required")
    if not to_session_id:
        return _err(rid, 4000, "to_session_id required")
    if not item_id:
        return _err(rid, 4000, "item_id required")
    if from_session_id == to_session_id:
        return _err(rid, 4000, "from_session_id and to_session_id must differ")

    transport = current_transport()
    if transport is None:
        return _err(rid, 4001, "live session not found or not owned by this transport")

    from_session = _find_owned_session(from_session_id, transport)
    from_agent = from_session.get("agent") if from_session else None
    from_store = getattr(from_agent, "_todo_store", None)
    if from_store is None:
        return _err(rid, 4001, "source session not live or not owned by this transport")

    to_session = _find_owned_session(to_session_id, transport)
    to_agent = to_session.get("agent") if to_session else None
    to_store = getattr(to_agent, "_todo_store", None)
    if to_store is None:
        return _err(rid, 4001, "destination session not live or not owned by this transport")

    from_items = from_store.read()
    item = next((todo for todo in from_items if todo.get("id") == item_id), None)
    if item is None:
        return _err(rid, 4004, "todo item not found in source session")

    # Drop the moved item AND anything whose parent was it (an orphaned subtask
    # referencing a parent id that no longer exists in this list is nonsensical).
    remaining = [todo for todo in from_items if todo.get("id") != item_id and todo.get("parent") != item_id]
    from_todos = from_store.write(remaining)
    from_revision = from_store.snapshot()["revision"]

    to_todos_before = to_store.read()
    existing_ids = {todo.get("id") for todo in to_todos_before}
    new_id = item_id if item_id not in existing_ids else f"{item_id}-moved"
    suffix = 2
    while new_id in existing_ids:
        new_id = f"{item_id}-moved-{suffix}"
        suffix += 1
    to_todos = to_store.write(
        to_todos_before + [{"id": new_id, "content": item["content"], "status": item["status"]}], merge=False)
    to_revision = to_store.snapshot()["revision"]

    _cache_todo_state(from_session, {"todos": from_todos, "revision": from_revision})
    _cache_todo_state(to_session, {"todos": to_todos, "revision": to_revision})
    from_runtime_id = next(sid for sid, candidate in _sessions.items() if candidate is from_session)
    to_runtime_id = next(sid for sid, candidate in _sessions.items() if candidate is to_session)
    _emit("todo.updated", from_runtime_id, {"todos": from_todos, "revision": from_revision})
    _emit("todo.updated", to_runtime_id, {"todos": to_todos, "revision": to_revision})
    return _ok(rid, {
        "from_session_id": from_session_id, "to_session_id": to_session_id,
        "item_id": new_id, "from_revision": from_revision, "to_revision": to_revision,
    })


def register(server):
    bind_module(globals(), server)
