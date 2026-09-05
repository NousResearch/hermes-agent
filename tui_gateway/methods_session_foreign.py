"""Desktop foreign-history browsing, scoped to the serving backend and profile."""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("session.foreign.list")
def _foreign_list(rid, params):
    from hermes_cli.foreign_session_browser import list_foreign_sessions
    try:
        return _ok(rid, list_foreign_sessions(params.get("source"), params.get("offset", 0), params.get("limit", 25)))
    except ValueError as exc:
        return _err(rid, -32602, str(exc))
    except OSError:
        return _err(rid, -32000, "Could not read session folders on this backend")


def _foreign_history_request(rid, params, importing):
    from hermes_cli.foreign_session_browser import import_browser_session, preview_foreign_session
    try:
        with _profile_db(params) as db:
            if db is None:
                return _db_unavailable_error(rid, code=-32000)
            result = (import_browser_session(params.get("id"), db, _response_profile_name(params.get("profile")))
                      if importing else preview_foreign_session(params.get("id"), db))
            return _ok(rid, result)
    except ValueError as exc:
        return _err(rid, -32602, str(exc))
    except OSError:
        return _err(rid, -32000, "Could not read this session on the backend")


@method("session.foreign.preview")
def _foreign_preview(rid, params):
    return _foreign_history_request(rid, params, False)


@method("session.foreign.import")
def _foreign_import(rid, params):
    return _foreign_history_request(rid, params, True)


def register(server):
    bind_module(globals(), server)
