"""Bot Desktop JSON-RPC handlers: the Desktop app's door to a profile's headless screen.

``display.status`` reports runtime + lease; ``display.start`` / ``display.stop`` manage the Xvnc/Xfce
process; ``display.observe`` mints a single-use ticket the renderer redeems on ``/api/display/ws``
(``hermes_cli.web_routers.display``) to stream raw RFB; ``display.lease.acquire`` / ``release`` are
Take over / Hand back. Every handler is profile-scoped so a multiplexed gateway answers for the bot
the pane is looking at. Lease transitions fan out as the global ``display.lease`` event so every
connected client repaints (badge on the bot row, red border on the viewer, agent handoff prompt).

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

import logging
import threading

from .method_ctx import HandlerRegistry, bind_module

logger = logging.getLogger(__name__)
_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

_DISPLAY_ERR = 5300
_lease_listener_installed = threading.Event()


def _display_snapshot() -> dict:
    from hermes_constants import hermes_home_key
    from tools.bot_desktop import lease as _bd_lease, runtime as _bd_runtime
    st = _bd_runtime.status()
    return {**st.as_dict(), "lease": _bd_lease.get().as_dict(), "profile_key": hermes_home_key()}


def _install_lease_listener() -> None:
    """Once per process: broadcast every lease change to all connected clients."""
    if _lease_listener_installed.is_set():
        return
    _lease_listener_installed.set()
    from tools.bot_desktop import lease as _bd_lease

    def _on_change(profile_key: str, lease) -> None:
        _broadcast_global_event("display.lease", {"profile_key": profile_key, "lease": lease.as_dict()})
    _bd_lease.on_change(_on_change)


@method("display.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    _install_lease_listener()
    try:
        return _ok(rid, _display_snapshot())
    except Exception as e:
        return _err(rid, _DISPLAY_ERR, str(e))


@method("display.start")
@_profile_scoped
def _(rid, params: dict) -> dict:
    _install_lease_listener()
    from tools.bot_desktop import runtime as _bd_runtime
    try:
        _bd_runtime.start()
        return _ok(rid, _display_snapshot())
    except Exception as e:
        return _err(rid, _DISPLAY_ERR, str(e))


@method("display.stop")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from tools.bot_desktop import lease as _bd_lease, runtime as _bd_runtime
    try:
        _bd_lease.release()
        stopped = _bd_runtime.stop()
        return _ok(rid, {**_display_snapshot(), "stopped": stopped})
    except Exception as e:
        return _err(rid, _DISPLAY_ERR, str(e))


@method("display.observe")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Mint a single-use, 30 s ticket for ``/api/display/ws``. The ticket carries the profile home so
    the bridge dials THIS profile's RFB socket, and the viewer id so the lease can name the holder."""
    from hermes_constants import get_hermes_home
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tools.bot_desktop import runtime as _bd_runtime
    try:
        if _bd_runtime.rfb_socket_path() is None:
            return _err(rid, _DISPLAY_ERR, "this profile's Bot Desktop is not running; call display.start first")
        viewer_id = str(params.get("viewer_id") or "").strip() or f"viewer-{rid}"
        ticket = mint_ticket(user_id=f"display:{viewer_id}", provider="bot-desktop",
                             extra={"hermes_home": str(get_hermes_home()), "viewer_id": viewer_id})
        return _ok(rid, {"ticket": ticket, "path": "/api/display/ws", "viewer_id": viewer_id,
                         **_display_snapshot()})
    except Exception as e:
        return _err(rid, _DISPLAY_ERR, str(e))


@method("display.lease.acquire")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from tools.bot_desktop import lease as _bd_lease
    viewer_id = str(params.get("viewer_id") or "").strip()
    if not viewer_id:
        return _err(rid, _DISPLAY_ERR, "viewer_id required")
    lease = _bd_lease.acquire(viewer_id, reason=str(params.get("reason") or ""))
    return _ok(rid, {"lease": lease.as_dict()})


@method("display.lease.release")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from tools.bot_desktop import lease as _bd_lease
    viewer_id = str(params.get("viewer_id") or "").strip() or None
    lease = _bd_lease.release(viewer_id)
    return _ok(rid, {"lease": lease.as_dict()})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
