"""``session.render_snapshot`` (``methods_session`` split): what the agent saw for one live session.

The payload is ``agent.render_snapshot.build_render_snapshot`` over the session's own state.db row:
the stored prompt hash, never a prompt rebuild. Bodies close over server.py globals through
``method_ctx.bind_module``; publication runs from the parent's ``register()``.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("session.render_snapshot")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    from agent.render_snapshot import build_render_snapshot
    with _session_profile_runtime_scope(session), _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        return _ok(rid, build_render_snapshot(db, session["session_key"], agent=session.get("agent")))


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
