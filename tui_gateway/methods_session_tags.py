"""Persistent session-tag RPCs (catalogue and assignments share the profile DB)."""
from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


@method("session.tags.list")
@_profile_scoped
def _tags_list(rid, params):
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        return _ok(rid, {"tags": db.list_session_tags()})


@method("session.tags.set")
@_profile_scoped
def _tags_set(rid, params):
    session = _sessions.get(params["session_id"])
    if session is not None and params.get("profile"):
        requested = _profile_home(params["profile"]) or Path(_hermes_home)
        owner = Path(session.get("profile_home") or _hermes_home)
        if requested.resolve() != owner.resolve():
            return _err(rid, 4001, "session not found in requested profile")
    with (_session_db(session) if session is not None else _profile_db(params, writer=True)) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        # Stored IDs are exact: a typo must not tag a different conversation by prefix/title.
        key = session["session_key"] if session is not None else params["session_id"]
        if db.get_session(key) is None:
            return _err(rid, 4001, "session not found")
        try:
            tags = db.set_session_tag(key, params["tag"], params["assigned"])
        except ValueError as exc:
            return _err(rid, 4000, str(exc))
        return _ok(rid, {"tags": tags})


def register(server):
    bind_module(globals(), server)
