"""Host-owned execution of native plugin session commands under exact control leases."""

from __future__ import annotations

import contextlib

from .method_ctx import bind_module


class _PluginSessionCommandBusy(RuntimeError):
    """A host control lease could not be acquired; adapters return RPC 4009."""


@contextlib.contextmanager
def _plugin_session_command_db(target, request, session_view):
    live_db = getattr(target.agent, "_session_db", None)
    if live_db is not None:
        yield live_db
        return
    if request.write:
        with _session_db(session_view) as db:
            yield db
        return
    from hermes_state import SessionDB

    path = Path(target.profile_root) / "state.db"
    if not path.exists():
        yield None
        return
    with contextlib.closing(SessionDB(db_path=path, read_only=True)) as db:
        yield db


def _plugin_session_ensure(target, session_view):
    """Use the canonical writer only while the frozen target still owns its identity."""
    def ensure():
        with _sessions_lock:
            with target.session["history_lock"]:
                if (
                    _sessions.get(target.sid) is not target.session
                    or target.session.get("_session_control_inflight") is not target.token
                    or target.session.get("agent") is not target.agent
                    or target.session.get("session_key") != target.session_key
                    or str(getattr(target.agent, "session_id", "") or "") != target.agent_session_id
                    or int(target.session.get("session_generation", 0) or 0) != target.generation
                    or (
                        str(target.session.get("profile_home"))
                        if target.session.get("profile_home") is not None else None
                    ) != target.profile_home
                ):
                    raise ValueError("session identity changed; refusing to mutate")
        if target.agent is not None:
            callback = getattr(target.agent, "_ensure_db_session", None)
            if not callable(callback):
                raise ValueError("canonical session persistence is not available")
            callback()
        elif _ensure_session_db_row(session_view) is False:
            raise ValueError("canonical session persistence is not available")
    return ensure


def _execute_plugin_session_command(handler, request, target):
    from hermes_cli.session_commands import make_session_context, run_session_command

    # Row creation keeps the backend's canonical metadata while routing and
    # identity are frozen for the duration of this host control lease.
    session_view = dict(target.session)
    session_view.update(session_key=target.session_key, profile_home=target.profile_root)
    state = target.session.get(_PLUGIN_SESSION_CONTROL_STATE_KEY)
    with _session_profile_runtime_scope(session_view):
        config = _session_plugin_config_loader(session_view)()
        if not isinstance(config, dict):
            raise ValueError("could not read global configuration")
        with _plugin_session_command_db(target, request, session_view) as db:
            context = make_session_context(
                agent=target.agent,
                session_db=db,
                session_id=target.session_key,
                ensure_session=_plugin_session_ensure(target, session_view),
                config_loader=lambda: config,
                profile_home=target.profile_root,
            )
            tokens = _set_session_context(target.session_key, ui_session_id=target.sid,
                                          cwd=str(session_view.get("cwd") or ""))
            try:
                return str(run_session_command(handler, request, context) or "")
            finally:
                # set_override records only committed writes, even when a
                # later live-policy callback or the command itself raises.
                if isinstance(state, dict):
                    state["writes"] = dict(context.writes)
                _clear_session_context(tokens)


def _run_plugin_session_command(handler, request, session, sid):
    target, claim_error = _claim_live_plugin_session_control(sid, session)
    if claim_error is _PLUGIN_SESSION_TURN_BUSY:
        raise _PluginSessionCommandBusy(_PLUGIN_SESSION_TURN_BUSY_MESSAGE)
    if claim_error is _SESSION_CONTROL_BUSY:
        raise _PluginSessionCommandBusy(_PLUGIN_SESSION_CONTROL_BUSY_MESSAGE)
    if isinstance(claim_error, str):
        return claim_error
    if target is None:
        raise ValueError("could not safely reserve the authorized session")

    response = ""
    try:
        response = _execute_plugin_session_command(handler, request, target)
    except Exception as exc:
        logger.exception("Plugin session command execution failed: command=%s session_id=%s",
                         handler.name, sid)
        response = f"Plugin command error: {exc}"
    finally:
        release = _release_live_plugin_session_control(target)
    warning = str(release.warning or "")
    if warning:
        return f"{response}\n\n{warning}".strip()
    if not release:
        return "Session command error: live session changed while the command was running"
    return response


def _dispatch_plugin_session_slash(rid, params, session, base, arg):
    """Return a session-plugin slash response, or None for the existing slash routes."""
    from hermes_cli.session_commands import is_session_command

    with _session_home_scope(session):
        handler = _plugin_command_handler(base) if base else None
        if not is_session_command(handler):
            return None
        try:
            output = _run_plugin_command(handler, arg, session, sid=params.get("session_id", ""))
            return _ok(rid, {"output": output or "(no output)"})
        except _PluginSessionCommandBusy as exc:
            return _err(rid, 4009, str(exc))
        except Exception as exc:
            logger.exception("Plugin session slash dispatch failed: command=%s session_id=%s",
                             base, params.get("session_id", ""))
            return _ok(rid, {"output": f"Plugin command error: {exc}"})


def register(server) -> None:
    bind_module(globals(), server)
