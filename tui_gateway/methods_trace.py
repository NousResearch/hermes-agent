"""Execution traces: the NeMo Relay events recorded for a session, fetched and followed live.

``trace.events`` returns what ``hermes_cli/observability/relay_traces.py`` wrote for the session
(and its delegated descendants) and, for a live gateway session, subscribes the session so every
further recorded event reaches its client as ``trace.event``. The desktop folds both into one
waterfall; there is no second, gateway-shaped trace.

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _traced_session_id(session: dict) -> str:
    agent = session.get("agent")
    return str(getattr(agent, "session_id", "") or session.get("session_key") or "")


@method("trace.events")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from agent import relay_runtime
    from hermes_cli.observability import relay_traces
    from hermes_constants import get_hermes_home

    raw = str(params.get("session_id") or "").strip()
    if not raw:
        return _err(rid, 4000, "session_id required")
    session = _sessions.get(raw)
    stored_id = _traced_session_id(session) if isinstance(session, dict) else raw
    if isinstance(session, dict):
        session["trace_profile_key"] = relay_runtime.current_profile_key()
    if not stored_id:
        return _ok(rid, {"session_id": "", "events": [], "truncated": False})
    try:
        recording = relay_traces.policy().get("enabled", True) is not False
    except Exception:
        recording = True
    events, truncated = relay_traces.read_session_events(get_hermes_home(), stored_id)
    return _ok(rid, {"session_id": stored_id, "events": events, "truncated": truncated, "recording": recording})


def _forward_trace_event(profile_key: str, root_session_id: str, session_id: str, event: dict) -> None:
    """Relay publication thread → every client tracing the event's root session."""
    del session_id
    with _sessions_lock:
        sessions = list(_sessions.items())
    for sid, session in sessions:
        if (
            isinstance(session, dict)
            and session.get("trace_profile_key") == profile_key
            and _traced_session_id(session) == root_session_id
        ):
            _emit("trace.event", sid, {"trace_session_id": root_session_id, "event": event})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
    from hermes_cli.observability import relay_traces

    previous = getattr(server, "_trace_listener_unsubscribe", None)
    if callable(previous):
        previous()
    server._trace_listener_unsubscribe = relay_traces.add_listener(vars(server)["_forward_trace_event"])
