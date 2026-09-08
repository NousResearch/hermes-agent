"""Session-scoped roster and bounded live transcript snapshots for shared clients.

Async projection adapted from JoaoMarcos44's PR #70899; controls reuse the
existing subagent.steer RPC rather than introducing a second steering runtime.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

_SUBAGENT_SNAPSHOT_FIELDS = (
    "subagent_id", "parent_id", "depth", "goal", "delegation_id", "model",
    "started_at", "status", "tool_count", "current_tool", "accepting_steer",
)
_ASYNC_SNAPSHOT_FIELDS = (
    "delegation_id", "goal", "role", "model", "status", "dispatched_at", "completed_at", "is_batch",
)
_SUBAGENT_TAIL_BYTES = 16384


def _owned_subagent_records(session_id, transport, owner):
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

    with _active_subagents_lock:
        return [dict(r) for r in _active_subagents.values()
                if r.get("owner_session_id") == session_id
                and r.get("owner_transport") is transport
                and r.get("owner_session_record") is owner]


@method("subagent.list")
def _(rid, params):
    session_id = _str_param(params, "session_id")
    transport, owner = _current_session_steer_authority(session_id)
    if transport is None or owner is None:
        return _err(rid, 4001, "session not found or not owned by this transport")
    from tools.async_delegation import _records, _records_lock

    live = _owned_subagent_records(session_id, transport, owner)
    # Read only projected fields, without invoking unrelated sessions' progress callbacks.
    with _records_lock:
        delegations = []
        for record in _records.values():
            if record.get("origin_ui_session_id") != session_id:
                continue
            item = {key: record.get(key) for key in _ASYNC_SNAPSHOT_FIELDS}
            item["subagent_ids"] = [r["subagent_id"] for r in live
                                    if r.get("delegation_id") == record.get("delegation_id")]
            delegations.append(item)
    return _ok(rid, {
        "subagents": [{key: r.get(key) for key in _SUBAGENT_SNAPSHOT_FIELDS} for r in live],
        "delegations": delegations,
    })


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
