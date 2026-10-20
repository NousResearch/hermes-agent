"""Coalesce concurrent update RPCs without restarting their publication transaction."""
from concurrent.futures import Future
from threading import Lock

from hermes_constants import hermes_home_key

_lock = Lock()
_pending: dict[tuple[str, str], Future] = {}


def join_update(name: str, operation):
    """Share the running result within one profile; completed calls re-check the catalog pin.

    The first request owns consent. A joining request never changes its decision,
    so a refused update still returns consent_required and must be submitted again.
    """
    key = (hermes_home_key(), name)
    with _lock:
        future = _pending.get(key)
        owner = future is None
        if owner:
            future = _pending[key] = Future()
    if not owner:
        return future.result()
    try:
        result = operation()
        future.set_result(result)
        return result
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        with _lock:
            del _pending[key]


def plugins_update(rid, params):
    """Join the profile's update and address its response to this request."""
    name = (params.get("name") or "").strip()
    result = join_update(name, lambda: _update_once(rid, params))
    return {**result, "id": rid}


def _update_once(rid, params):
    """Catalog installs only: re-pin to the current catalog SHA (non-catalog installs update via the CLI).
    A pin that widens the plugin (new tools/hooks/deps/capabilities/Desktop half) answers
    ``{ok: false, consent_required: true, delta, delta_lines}`` with nothing changed; the client shows the
    delta and retries with ``accept_capabilities: true``."""
    from tui_gateway.server import (
        _ensure_plugin_activation_listener, _err, _ok, _tools_mod, _with_activation,
    )

    name = (params.get("name") or "").strip()
    if not name:
        return _err(rid, 4019, "plugins.update requires a 'name'")
    pc, cat = _tools_mod("hermes_cli.plugins_cmd"), _tools_mod("hermes_cli.plugins_cmd_catalog")
    target = pc._plugins_dir() / name
    sidecar = cat.catalog_install_record(target) if target.is_dir() else None
    if not sidecar:
        return _err(rid, 4020, f"'{name}' is not a catalog install — update it via the CLI")
    try:
        result = cat.repin_catalog_plugin(
            target, sidecar, consent_cb=(lambda _delta: True) if params.get("accept_capabilities") else None)
    except cat.RepinConsentRequired as e:
        return _ok(rid, {"ok": False, "consent_required": True, "name": e.name, "sha": e.sha, "delta": e.delta,
                         "delta_lines": cat.surface_delta_lines(e.delta), "error": str(e)})
    except pc.PluginOperationError as e:
        return _err(rid, 4021, str(e))
    payload = {"ok": True, "unchanged": not result.changed, "sha": result.sha, "name": result.installed_name,
               "warnings": list(result.warnings)}
    if result.changed:
        _ensure_plugin_activation_listener()
        activate = _tools_mod("hermes_cli.plugins_activation").activate_plugin_now
        payload = _with_activation({**payload, **activate(result.installed_name)}, result.installed_name)
    return _ok(rid, payload)
