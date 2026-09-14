"""Workstation resource projection for non-REST clients.

This is a read-only adapter over the same Electron controller used by the
Dashboard. It intentionally returns the controller's resource envelope and
does not create a gateway-owned task/session store.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


@method("workstation.resources")
def _(rid, params: dict) -> dict:
    try:
        from workstation.client import get_workstation_resources

        return _ok(rid, get_workstation_resources())
    except Exception as exc:
        return _err(rid, 5010, str(exc))


@method("workstation.events")
def _(rid, params: dict) -> dict:
    task_id = params.get("task_id")
    limit = params.get("limit", 200)
    if task_id is not None and not isinstance(task_id, str):
        return _err(rid, 5011, "task_id must be a string")
    if isinstance(limit, bool) or not isinstance(limit, int):
        return _err(rid, 5011, "limit must be an integer")
    try:
        from workstation.client import get_workstation_events

        return _ok(rid, get_workstation_events(task_id=task_id, limit=limit))
    except Exception as exc:
        return _err(rid, 5010, str(exc))


def register(server) -> None:
    _registry.install(server)
