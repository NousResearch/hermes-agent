"""Session-scoped roster and bounded live transcript snapshots for shared clients.

Async projection adapted from JoaoMarcos44's PR #70899; controls reuse the
existing subagent.steer RPC rather than introducing a second steering runtime.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

_SUBAGENT_SNAPSHOT_FIELDS = (
    "subagent_id", "parent_id", "depth", "goal", "delegation_id", "model",
    "started_at", "status", "tool_count", "last_tool", "accepting_steer",
)
_SUBAGENT_TAIL_BYTES = 16384


def _owned_subagent_records(session_id, transport, owner):
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _subagent_transport_matches

    with _active_subagents_lock:
        return [dict(r) for r in _active_subagents.values()
                if r.get("owner_session_id") == session_id
                and _subagent_transport_matches(r, transport)
                and r.get("owner_session_record") is owner]


def _visible_subagent_records(session_id, transport, owner):
    """Read-only roster for ``subagent.list``: the exact-owner records PLUS children whose durable
    conversation lineage (``owner_agent_session_id`` resolved to its compression tip) is this
    session's agent — the same spine ``delegate_task(action="list")`` walks in-process.

    Spawn freezes ``owner_session_id`` to the UI session id of that moment; a Desktop reconnect /
    resume remints the id and rebuilds the session record, and compression rotates the durable key,
    so the exact match alone hid every still-running child from the panel for good (#114909).
    Control RPCs (steer / interrupt / tail) keep the exact generation authority."""
    from tools.delegate_tool_registry import (
        _active_subagents, _active_subagents_lock, _owns_subagent_record, _subagent_transport_matches,
    )

    with _active_subagents_lock:
        records = [dict(r) for r in _active_subagents.values()]
    agent = owner.get("agent")
    # Lineage resolution may read the session DB — evaluated outside the registry lock.
    return [r for r in records
            if (r.get("owner_session_id") == session_id
                and _subagent_transport_matches(r, transport)
                and r.get("owner_session_record") is owner)
            or _owns_subagent_record(r, agent)]


@method("subagent.list")
def _(rid, params):
    session_id = _str_param(params, "session_id")
    transport, owner = _current_session_steer_authority(session_id)
    if transport is None or owner is None:
        return _err(rid, 4001, "session not found or not owned by this transport")
    live = _visible_subagent_records(session_id, transport, owner)
    return _ok(rid, {
        "subagents": [{key: r.get(key) for key in _SUBAGENT_SNAPSHOT_FIELDS} for r in live],
        "delegations": _failed_delegations(session_id, owner),
    })


def _live_work_owners():
    """``(kind, owner candidates)`` for every live top-level child and running background process in this
    process. A child names its durable conversation (``owner_agent_session_id``, the spine completion delivery
    routes by). A process names the conversation that spawned it (``parent_session_id``, from
    ``HERMES_SESSION_ID``) and its routing ``session_key``, which on the gateway path is a route
    (``agent:main:local:dm:...``), never a stored id."""
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
    from tools.process_registry import process_registry

    with _active_subagents_lock:
        owners = [("subagents", (str(r.get("owner_agent_session_id") or ""), ""))
                  for r in _active_subagents.values() if not r.get("depth")]
    for entry in process_registry.list_sessions():
        proc = process_registry.get(entry.get("session_id"))
        if proc is not None and not getattr(proc, "exited", True):
            owners.append(("background", (str(getattr(proc, "parent_session_id", "") or ""),
                                          str(getattr(proc, "session_key", "") or ""))))
    return owners


def _route_namespaces(profile_home):
    """Session-key namespaces (the ``agent:<ns>`` segment) a route of the requested profile can carry.
    Multiplexed routes are namespaced per profile; ``agent:main`` is also every unmultiplexed and legacy
    route, so it is trusted for a named profile only when that profile is the sole one this process hosts."""
    from gateway.session import _session_key_namespace
    from gateway.session_authorities import served_profile_name

    own = _session_key_namespace(served_profile_name(profile_home or _hermes_home)).split(":", 1)[1]
    sole = profile_home is None and not _served_profile_homes
    return {own, "main"} if sole else {own}


def _stored_owner(db, owner_ids, namespaces):
    """The conversation in ``db`` (the requested profile's state.db) that owns a piece of work; "" when none
    is ours. A stored id (child lineage, process ``parent_session_id``) is authoritative: present here or
    not ours. Only a process without one falls back to its route key, fenced to this profile's namespaces,
    since an identical route (a legacy ``agent:main`` key, the same DM in two profiles) can sit in another
    profile's store."""
    stored_id, route = owner_ids
    if stored_id:
        return stored_id if db.get_session(stored_id) else ""
    parts = route.split(":")
    if len(parts) < 3 or parts[0] != "agent" or parts[1] not in namespaces:
        return ""
    row = db.find_latest_gateway_session_for_peer(source=parts[2], session_key=route)
    return str(row["id"]) if row and row.get("id") else ""


@method("subagent.roster")
def _(rid, params):
    """App-wide sidebar feed: which conversations still have children or background processes running.
    ``subagent.list`` needs the conversation's live, transport-attached session record, so it cannot answer
    for a row that is not open or for any row right after a renderer reload; this read needs neither. One
    process hosts several profiles, so only conversations stored in the requested profile's state.db are
    returned — a caller never learns another profile's session ids."""
    owners = _live_work_owners()
    if not owners:
        return _ok(rid, {"sessions": []})
    scope = params if isinstance(params, dict) else {}
    if not str(scope.get("profile") or "").strip():
        # A legacy-fallback socket ticketed to a non-launch profile answers for that profile only.
        from tui_gateway.ws_legacy_fallback import connection_profile_home
        if home := connection_profile_home():
            scope = {"profile": Path(home).name}
    counts = {}
    namespaces = _route_namespaces(_profile_home(scope.get("profile")))
    with _profile_db(scope) as db:
        if db is None:
            return _ok(rid, {"sessions": []})
        for kind, owner_ids in owners:
            if sid := _stored_owner(db, owner_ids, namespaces):
                counts.setdefault(sid, {"subagents": 0, "background": 0})[kind] += 1
    return _ok(rid, {"sessions": [{"session_id": sid, **c} for sid, c in counts.items()]})


def _failed_delegations(session_id, owner):
    """Recently failed async delegation tasks for this session from the durable store (the live
    roster forgets ended children and dies with a renderer reload, #97202). Read under the session's
    profile home, where its delegations were persisted; a store error degrades to no rows."""
    from tools.async_delegation import failed_delegations_for_session

    agent_session_id = str(getattr(owner.get("agent"), "session_id", "") or "")
    try:
        with _session_home_scope(owner):
            return failed_delegations_for_session(session_id, agent_session_id)
    except Exception:
        logger.debug("subagent.list: failed-delegation read failed for %s", session_id, exc_info=True)
        return []


@method("subagent.interrupt")
def _(rid, params):
    from agent.interrupt_compat import request_hard_interrupt

    subagent_id = _str_param(params, "subagent_id")
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    session_id = _str_param(params, "session_id")
    transport, owner = _current_session_steer_authority(session_id)
    if transport is None or owner is None:
        return _err(rid, 4001, "session not found or not owned by this transport")
    record = next((r for r in _owned_subagent_records(session_id, transport, owner)
                   if r.get("subagent_id") == subagent_id), None)
    agent = record.get("agent") if record else None
    # Interrupt the authorized object, never re-resolve a globally recyclable id.
    found = False
    if agent is not None:
        try:
            found = bool(request_hard_interrupt(agent, f"Interrupted via TUI ({subagent_id})"))
        except Exception:
            logger.debug("subagent interrupt failed", exc_info=True)
    return _ok(rid, {"found": found, "subagent_id": subagent_id})


@method("subagent.tail")
def _(rid, params):
    session_id = _str_param(params, "session_id")
    subagent_id = _str_param(params, "subagent_id")
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    transport, owner = _current_session_steer_authority(session_id)
    if transport is None or owner is None:
        return _err(rid, 4001, "session not found or not owned by this transport")
    result = {"subagent_id": subagent_id, "available": False, "text": "", "truncated": False}
    record = next((r for r in _owned_subagent_records(session_id, transport, owner)
                   if r.get("subagent_id") == subagent_id), None)
    path = getattr(record.get("agent"), "_live_transcript_path", None) if record else None
    if not path:
        return _ok(rid, result)
    try:
        with open(path, "rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - _SUBAGENT_TAIL_BYTES))
            text = stream.read(_SUBAGENT_TAIL_BYTES).decode("utf-8", errors="ignore")
    except OSError:
        # Creation/cleanup races are normal while a child starts or ends.
        return _ok(rid, result)
    return _ok(rid, {**result, "available": True, "text": text, "truncated": size > _SUBAGENT_TAIL_BYTES})


def register(server):
    bind_module(globals(), server)
