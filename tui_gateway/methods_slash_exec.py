"""Slash execution on the live backend or its persistent worker."""

from __future__ import annotations

import contextlib
import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


def _live_slash_response(rid, sid, session, base, arg):
    live_output = _live_slash_command_output(sid, session, base, arg)
    if live_output is not None:
        if (
            base in {"compress", "compact"}
            and isinstance(live_output, str)
            and live_output.lower().startswith("session busy")
        ):
            return _err(rid, 4009, live_output)
        return _ok(rid, {"output": live_output or "(no output)"})
    return None


@method("slash.exec")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")
    # Skill/bundle and _PENDING_INPUT_COMMANDS must NOT reach the slash worker. Plugin
    # commands also bypass it but return normal slash.exec output (TUI keeps the pager path).
    parts = cmd.lstrip("/").split(maxsplit=1)
    base = (parts[0] if parts else "").lower()
    arg = parts[1] if len(parts) > 1 else ""
    sid = params.get("session_id", "")
    if response := _dispatch_plugin_session_slash(rid, params, session, base, arg):
        return response
    if response := _live_slash_response(rid, sid, session, base, arg):
        return response
    if base in _WORKER_BLOCKED_COMMANDS and _is_snapshot_restore(arg):
        return _err(rid, 4018, "snapshot restore mutates live config/state; use command.dispatch for /snapshot restore")
    # Pending-input built-ins route straight to command.dispatch (some clients fail the
    # error-then-retry fallback); bundles go the same way under their resolved key.
    with _session_home_scope(session):  # a secondary-only bundle must route too (#110695)
        target = base if base in _PENDING_INPUT_COMMANDS else _bundle_key_for(base)
    if target is not None:
        return _methods["command.dispatch"](rid, {"name": target.lstrip("/"), "arg": arg, "session_id": sid})
    # Recognized skills keep the 4018 gate so clients command.dispatch. A scan
    # exception must not fail open into the worker: return the dispatch payload
    # (or a hard error) here, or the loading banner swallows the prompt.
    skill_hit = _profile_skill_command(session, base)
    if skill_hit is True:
        return _err(rid, 4018, f"skill command: use command.dispatch for /{base}")
    if skill_hit is None:
        dispatched = _skill_dispatch_or_refuse(rid, sid, base, arg)
        # A built-in cannot collide with a skill slug, so the worker may still
        # run it. Anything else might be the skill the scan failed to see.
        if (dispatched.get("result") or {}).get("type") or not _is_registry_command(base):
            return dispatched
    if plugin_handler := _plugin_command_handler(base) if base else None:
        try:
            return _ok(rid, {"output": _run_plugin_command(plugin_handler, arg, session) or "(no output)"})
        except Exception as e:
            logger.debug("Slash command execution failed", exc_info=True)
            return _ok(rid, {"output": f"Plugin command error: {e}"})
    worker = session.get("slash_worker")
    if not worker:
        # slash.exec runs on the RPC pool: two concurrent commands could both see slash_worker=None
        # and each fork a full MCP-fleet worker (the loser leaks). Serialize first-use spawn.
        with _sessions_lock:
            spawn_lock = session.setdefault("_slash_spawn_lock", threading.Lock())
        with spawn_lock:
            worker = session.get("slash_worker")
            if not worker:
                try:
                    worker = _SlashWorker(
                        session["session_key"], getattr(session.get("agent"), "model", _resolve_model()),
                        profile_home=session.get("profile_home"),
                        provider=getattr(session.get("agent"), "provider", None) or None)
                    _attach_worker(sid, session, worker)
                except Exception as e:
                    logger.debug("Slash command execution failed", exc_info=True)
                    return _err(rid, 5030, f"slash worker start failed: {e}")
    try:
        output = worker.run(cmd)
        if seed := (worker.pop_seed() if hasattr(worker, "pop_seed") else ""):
            # /prompt//blueprint composed a next-turn prompt in the worker; route it as a
            # send dispatch (both Desktop and TUI clients already handle {type:"send"}).
            return _ok(rid, {"type": "send", "message": seed})
        payload = {"output": output or "(no output)"}
        if warning := _mirror_slash_side_effects(sid, session, cmd):
            payload["warning"] = warning
        if base in _SESSION_CONTROL_SLASHES:
            _publish_session_control_snapshot(sid, session)
        return _ok(rid, payload)
    except Exception as e:
        logger.debug("Slash command execution failed", exc_info=True)
        if _worker_refused_skill(e):
            # Refused before process_command; the worker is still healthy.
            return _skill_dispatch_or_refuse(rid, sid, base, arg)
        with contextlib.suppress(Exception):
            worker.close()
        session["slash_worker"] = None
        return _err(rid, 5030, str(e))



def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
