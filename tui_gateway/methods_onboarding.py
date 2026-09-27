"""Onboarding JSON-RPC handlers: the backend owns the setup profile (``hermes_cli.setup_profile``).
Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("onboarding.ensure_setup_profile")
def _(rid, params: dict) -> dict:
    """Create-or-read the setup profile. Takes no name: the backend picks it and finds it by role."""
    from hermes_cli.setup_profile import ensure_setup_profile
    try:
        setup = ensure_setup_profile()
        if setup.created:
            # Same credential mirroring profiles.create gives the desktop's clones; auth.json stays
            # shared with the root so a token refresh never forks.
            _mirror_launch_credentials(setup.path, {"share_auth": True})
    except Exception as e:
        return _err(rid, 5073, str(e))
    return _ok(rid, {"name": setup.name, "path": str(setup.path), "created": setup.created, "role": "setup"})


@method("onboarding.reset_setup_profile")
def _(rid, params: dict) -> dict:
    """Restore the setup profile to its created state in place; clears its session history."""
    from hermes_cli.setup_profile import find_setup_profile, reset_setup_profile
    found = find_setup_profile()
    if found is None:
        return _err(rid, 4072, "no setup profile to reset")
    _clear_setup_sessions(found[1])
    try:
        setup = reset_setup_profile()
    except Exception as e:
        return _err(rid, 5074, str(e))
    return _ok(rid, {"name": setup.name, "path": str(setup.path), "reset": True})


def _clear_setup_sessions(profile_dir) -> None:
    """Close this process's live sessions in the setup profile, then delete its stored sessions."""
    target = Path(profile_dir).resolve()
    with _sessions_lock:
        live = [sid for sid, sess in _sessions.items()
                if Path(sess.get("profile_home") or _hermes_home).resolve() == target]
    for sid in live:
        _close_session_by_id(sid, end_reason="setup_reset")
    from hermes_state_registry import acquire, release_or_close
    db = acquire(target / "state.db")
    try:
        ids = [row[0] for row in db._read_all("SELECT id FROM sessions")]
        db.delete_sessions(ids, sessions_dir=target / "sessions")
    finally:
        release_or_close(db)


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
