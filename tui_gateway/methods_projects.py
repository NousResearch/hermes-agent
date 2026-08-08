"""Projects JSON-RPC surface helpers (moved verbatim from server.py).

Handler bodies are byte-identical to their pre-split server.py form;
they are rebound onto server.py's globals at install time — see method_ctx.py.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


_E_PROJECTS = 5061  # generic failure
_E_NO_PROJECT = 5062  # id resolved to nothing
_E_PROJECT_ARG = 5063  # invalid argument (e.g. bad name/slug)


class _NoProject(Exception):
    """Raised inside a projects handler when ``params['id']`` resolves to None."""


def _projects_payload(conn) -> dict:
    from hermes_cli import projects_db as pdb

    return {
        "projects": [p.to_dict() for p in pdb.list_projects(conn, include_archived=True)],
        "active_id": pdb.get_active_id(conn),
    }


def _projects_method(name: str):
    """Register a projects RPC, injecting (pdb, conn) and unifying error mapping.

    Every project CRUD handler opened the per-profile DB, mapped a missing id to
    5062, bad args to 5063, and everything else to 5061. This collapses that
    boilerplate so each handler is just its one meaningful operation.
    """

    def decorator(fn):
        @method(name)
        def handler(rid, params: dict) -> dict:
            try:
                from hermes_cli import projects_db as pdb

                with pdb.connect_closing() as conn:
                    return fn(rid, params, pdb, conn)
            except _NoProject:
                return _err(rid, _E_NO_PROJECT, "no such project")
            except ValueError as e:
                return _err(rid, _E_PROJECT_ARG, str(e))
            except Exception as e:
                return _err(rid, _E_PROJECTS, str(e))

        return handler

    return decorator


def _require_project(pdb, conn, params: dict):
    """The project named by ``params['id']`` (or raise ``_NoProject``)."""
    proj = pdb.get_project(conn, str(params.get("id") or ""))
    if proj is None:
        raise _NoProject
    return proj


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
