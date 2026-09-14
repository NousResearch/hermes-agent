"""Desktop foreign-history browsing, scoped to the serving backend and profile."""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("session.foreign.list")
def _foreign_list(rid, params: SessionForeignListParams) -> SessionForeignListResult | dict:
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import SessionForeignListResult
    from hermes_cli.foreign_sessions_browser import list_foreign_sessions
    try:
        return SessionForeignListResult.model_validate(list_foreign_sessions(params.source.value if params.source else None, params.offset, params.limit))
    except ValueError as exc:
        return _err(rid, -32602, str(exc))
    except OSError:
        return _err(rid, -32000, "Could not read session folders on this backend")


def _foreign_history_request(rid, params, importing):
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import (
        SessionForeignImportResult, SessionForeignPreviewResult)
    from hermes_cli.foreign_sessions_browser import import_browser_session, preview_foreign_session
    try:
        with _profile_db(params, writer=importing) as db:
            if db is None:
                return _db_unavailable_error(rid, code=-32000)
            result = (import_browser_session(params.id, db, _response_profile_name(params.profile))
                      if importing else preview_foreign_session(params.id, db))
            return (SessionForeignImportResult if importing else SessionForeignPreviewResult).model_validate(result)
    except ValueError as exc:
        return _err(rid, -32602, str(exc))
    except OSError:
        return _err(rid, -32000, "Could not read this session on the backend")


@method("session.foreign.preview")
def _foreign_preview(rid, params: SessionForeignIdParams) -> SessionForeignPreviewResult | dict:
    return _foreign_history_request(rid, params, False)


@method("session.foreign.import")
def _foreign_import(rid, params: SessionForeignIdParams) -> SessionForeignImportResult | dict:
    return _foreign_history_request(rid, params, True)


def register(server):
    bind_module(globals(), server)
