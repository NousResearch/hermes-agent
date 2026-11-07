"""Native memory review; explicit slash/CLI commands keep their existing semantics."""
from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
def _memory_review_scoped(handler):
    # Validate before entering scope: a retired runtime must never read launch state.
    def wrapper(rid, params):
        if params.get("session_id") is None:
            return _profile_scoped(handler)(rid, params)
        session = _sessions.get(params.get("session_id"))
        # None is the canonical launch-profile owner; an absent key is unowned.
        if not isinstance(session, dict) or "profile_home" not in session:
            return _err(rid, 4004, "Session not found or profile unavailable")
        with _session_profile_runtime_scope(session):
            return handler(rid, params)
    return wrapper

@method("memory.pending")
@_memory_review_scoped
def _(rid, params):
    from tools.memory_review import list_memory_reviews
    return _ok(rid, list_memory_reviews())

@method("memory.decide")
@_memory_review_scoped
def _(rid, params):
    from tools.memory_review import decide_memory_review
    return _ok(rid, decide_memory_review(params["id"], params["decision"], params["revision"]))

def register(server):
    bind_module(globals(), server, skip=("_",))
