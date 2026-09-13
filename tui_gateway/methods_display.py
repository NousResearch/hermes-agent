"""Bot Desktop JSON-RPC handlers: the Desktop app's door to a profile's headless screen.

``display.status`` reports runtime + lease; ``display.start`` / ``display.stop`` manage the Xvnc/Xfce
process; ``display.observe`` mints a single-use ticket the renderer redeems on ``/api/display/ws``
(``hermes_cli.web_routers.display``) to stream raw RFB; ``display.lease.acquire`` / ``release`` are
Take over / Hand back. ``display.install`` runs the distro package install on the gateway host: sudo
privilege is asked for through the masked ``display.install.sudo.request`` card (same ``_block`` bridge as
the terminal tool's sudo prompt), stdout streams as ``display.install.log`` and the run ends with
``display.install.done`` carrying a fresh status snapshot. Every handler is profile-scoped so a multiplexed gateway answers for the bot
the pane is looking at. Lease transitions fan out as the global ``display.lease`` event so every
connected client repaints (badge on the bot row, red border on the viewer, agent handoff prompt).

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

import logging
import threading
import weakref

from .method_ctx import HandlerRegistry, bind_module

logger = logging.getLogger(__name__)
_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

_DISPLAY_ERR = 5300
_DISPLAY_FORBIDDEN = 4403
_lease_listener_installed = threading.Event()


def _lease_view(lease) -> dict:
    """The lease as clients may see it: the holder's viewer id is a capability (whoever presents it
    co-drives or releases the lease), so it is replaced by a short hash the holder can match against
    its own id to know it is the one in control."""
    import hashlib

    d = lease.as_dict()
    vid = d.pop("viewer_id")
    d.pop("resume_hash", None)
    d["viewer_id"] = None
    d["viewer_hash"] = hashlib.sha256(vid.encode()).hexdigest()[:12] if vid else None
    return d


def _display_snapshot() -> dict:
    from hermes_constants import hermes_home_key
    from tools.bot_desktop import lease as _bd_lease, runtime as _bd_runtime

    st = _bd_runtime.status()
    return {
        **st.as_dict(),
        "lease": _lease_view(_bd_lease.get()),
        "profile_key": hermes_home_key(),
    }


def _install_lease_listener() -> None:
    """Once per process: broadcast every lease change to all connected clients."""
    if _lease_listener_installed.is_set():
        return
    from tools.bot_desktop import lease as _bd_lease

    def _on_change(profile_key: str, lease) -> None:
        _broadcast_global_event("display.lease", {"profile_key": profile_key, "lease": _lease_view(lease)})

    _bd_lease.on_change(_on_change)
    _lease_listener_installed.set()  # only once the subscription exists, or a failed import would silence every client


@method("display.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    _install_lease_listener()
    try:
        return _ok(rid, _display_snapshot())
    except Exception as e:
        return _err(rid, _DISPLAY_ERR, str(e))


@method("display.thumbnail")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """One JPEG grab of the bot's screen (``data_url``: null while stopped). Read-only: no lease change.
    Suppressed while a human holds the lease — the frame may show what they are typing."""
    try:
        from tools.bot_desktop import lease as _bd_lease

        if _bd_lease.human_holds():
            return _ok(rid, {"data_url": None, "suppressed": "human_has_control"})
        from tools.bot_desktop.thumbnail import thumbnail_data_url

        return _ok(rid, {"data_url": thumbnail_data_url()})
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
        if _bd_lease.human_holds():
            return _err(
                rid,
                _DISPLAY_FORBIDDEN,
                "private takeover is active; hand back from the holding viewer first",
            )
        _bd_lease.release()
        stopped = _bd_runtime.stop()
        return _ok(rid, {**_display_snapshot(), "stopped": stopped})
    except Exception as e:
        return _err(rid, _DISPLAY_ERR, str(e))


# Viewer ids and their recovery hashes are scoped to both the authenticated transport and profile
# that minted them. The stdio transport cannot be weak-referenced; it is one process-local peer.
_minted_viewer_ids: "weakref.WeakKeyDictionary[object, dict[str, dict[str, str]]]" = weakref.WeakKeyDictionary()
_stdio_minted_viewer_ids: dict[str, dict[str, str]] = {}


def _current_transport_viewer_ids() -> dict[str, str] | None:
    from hermes_constants import hermes_home_key

    transport = current_transport()
    if transport is None:
        return None
    try:
        by_profile = _minted_viewer_ids.setdefault(transport, {})
    except TypeError:
        if transport is not _stdio_transport:
            return None
        by_profile = _stdio_minted_viewer_ids
    return by_profile.setdefault(hermes_home_key(), {})


def _viewer_id_owned_by_current_transport(viewer_id: str) -> bool:
    mine = _current_transport_viewer_ids()
    return mine is not None and viewer_id in mine


def _viewer_resume_hash(viewer_id: str) -> str:
    mine = _current_transport_viewer_ids()
    return mine.get(viewer_id, "") if mine is not None else ""


def _capability_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _mint_viewer_identity(requested: str, resume_token: str) -> tuple[str, str]:
    """Return a server-owned viewer id plus a client-only recovery capability.

    A new transport may recover the current holder only by proving the random token whose hash is
    stored in the lease. The raw viewer id remains identity, never authorization.
    """
    import secrets

    mine = _current_transport_viewer_ids()
    if mine is not None and resume_token:
        from tools.bot_desktop import lease as _bd_lease

        held = _bd_lease.get()
        candidate_hash = _capability_hash(resume_token)
        if (
            held.holder == _bd_lease.HUMAN
            and held.viewer_id
            and held.resume_hash
            and secrets.compare_digest(candidate_hash, held.resume_hash)
        ):
            mine[held.viewer_id] = held.resume_hash
            return held.viewer_id, resume_token
    viewer_id = requested if mine is not None and requested in mine else secrets.token_urlsafe(16)
    resume_token = secrets.token_urlsafe(32)
    if mine is not None:
        mine[viewer_id] = _capability_hash(resume_token)
    return viewer_id, resume_token


@method("display.observe")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Mint a single-use, 30 s ticket for ``/api/display/ws``. The ticket carries the profile home so
    the bridge dials THIS profile's RFB socket, and a server-minted viewer id (returned to the caller,
    who passes it to ``display.lease.acquire`` / ``release``) so the lease can name the holder."""
    from hermes_constants import get_hermes_home
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tools.bot_desktop import runtime as _bd_runtime

    try:
        if _bd_runtime.rfb_socket_path() is None:
            return _err(
                rid,
                _DISPLAY_ERR,
                "this profile's Bot Desktop is not running; call display.start first",
            )
        viewer_id, resume_token = _mint_viewer_identity(
            str(params.get("viewer_id") or "").strip(),
            str(params.get("resume_token") or "").strip(),
        )
        ticket = mint_ticket(
            user_id=f"display:{viewer_id}",
            provider="bot-desktop",
            extra={"hermes_home": str(get_hermes_home()), "viewer_id": viewer_id},
        )
        return _ok(
            rid,
            {
                "ticket": ticket,
                "path": "/api/display/ws",
                "viewer_id": viewer_id,
                "resume_token": resume_token,
                **_display_snapshot(),
            },
        )
    except Exception as e:
        return _err(rid, _DISPLAY_ERR, str(e))


@method("display.install")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Start the package install in the background; the renderer follows ``display.install.log`` /
    ``display.install.done``. Refused while one is already running for this profile."""
    from hermes_constants import hermes_home_key
    from tools.bot_desktop import install as _bd_install, runtime as _bd_runtime

    if not _bd_runtime.is_supported_host():
        return _err(rid, _DISPLAY_ERR, "Bot Desktop runs on Linux gateway hosts only")
    if _bd_runtime.install_command() is None:
        return _err(
            rid,
            _DISPLAY_ERR,
            "no supported package manager (apt-get, dnf, pacman) on this host",
        )
    profile_key = hermes_home_key()
    sid = str(params.get("session_id") or "")

    def _ask_password() -> str:
        return _block(
            "display.install.sudo.request",
            sid,
            {"profile_key": profile_key},
            timeout=300,
        )

    def _line(text: str) -> None:
        _broadcast_global_event("display.install.log", {"profile_key": profile_key, "line": text})

    # The worker thread inherits NO context: the caller's transport (so the sudo card reaches the
    # CLIENT that clicked Install) and the profile scope `_profile_scoped` installed (so status, lock
    # and events all speak for the requested profile) are carried across with copy_context().
    import contextvars

    ctx = contextvars.copy_context()

    def _run() -> None:
        try:
            code = _bd_install.install_packages(ask_password=_ask_password, on_line=_line, claimed=True)
        except Exception as e:
            _line(f"install failed: {e}")
            code = 1
        _broadcast_global_event(
            "display.install.done",
            {"profile_key": profile_key, "code": code, "status": _display_snapshot()},
        )

    try:
        _bd_install.claim()  # atomic: two fast clicks cannot both start a package manager
    except _bd_install.InstallBusy as e:
        return _err(rid, _DISPLAY_ERR, str(e))
    threading.Thread(
        target=ctx.run,
        args=(_run,),
        name=f"bot-desktop-install:{profile_key}",
        daemon=True,
    ).start()
    return _ok(
        rid,
        {
            "started": True,
            "command": _bd_runtime.install_command(),
            "profile_key": profile_key,
        },
    )


@method("display.lease.acquire")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from tools.bot_desktop import lease as _bd_lease

    viewer_id = str(params.get("viewer_id") or "").strip()
    if not viewer_id:
        return _err(rid, _DISPLAY_ERR, "viewer_id required")
    if not _viewer_id_owned_by_current_transport(viewer_id):
        return _err(
            rid,
            _DISPLAY_FORBIDDEN,
            "viewer_id is not owned by this connection",
            data={"code": "viewer_not_owned"},
        )
    try:
        lease = _bd_lease.acquire(
            viewer_id,
            reason=str(params.get("reason") or ""),
            resume_hash=_viewer_resume_hash(viewer_id),
        )
    except _bd_lease.ViewerLeaseHeld as exc:
        return _err(rid, _DISPLAY_FORBIDDEN, str(exc), data={"code": "lease_held"})
    return _ok(rid, {"lease": _lease_view(lease)})


@method("display.lease.release")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from tools.bot_desktop import lease as _bd_lease

    viewer_id = str(params.get("viewer_id") or "").strip()
    if not viewer_id or not _viewer_id_owned_by_current_transport(viewer_id):
        return _err(
            rid,
            _DISPLAY_FORBIDDEN,
            "viewer_id is not owned by this connection",
            data={"code": "viewer_not_owned"},
        )
    lease = _bd_lease.release(viewer_id)
    if lease.holder == _bd_lease.HUMAN:
        return _err(
            rid,
            _DISPLAY_FORBIDDEN,
            "only the holding viewer may hand back control",
            data={"code": "viewer_mismatch"},
        )
    return _ok(rid, {"lease": _lease_view(lease)})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
