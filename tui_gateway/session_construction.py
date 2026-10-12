"""Deferred construction and exact-owner attachment of live session agents."""

from __future__ import annotations

import contextlib
import threading

from .method_ctx import bind_module


def _attach_built_agent(sid: str, current: dict, agent) -> bool:
    """Attach a freshly built agent to its live record (session DB row deferred to first run_conversation()).
    False when ``session.close`` popped this record mid-build: teardown saw ``agent=None`` and closed
    nothing, so the caller owns closing the orphan (#49852)."""
    # Bot Mode gate hint: the DB title lands post-first-turn but the system prompt builds at turn START.
    if _title_hint := str(current.get("pending_title") or "").strip():
        agent._session_title_hint = _title_hint
    # Construction and session commands share admission, including the final attach.
    with _sessions_lock:
        with current.setdefault("history_lock", threading.Lock()):
            if (_sessions.get(sid) is not current
                    or current.get("_session_control_inflight") is not None):
                return False
            current["agent"] = agent
    # A workspace move can land while construction is still in flight.
    _register_session_cwd(current)
    _session_todo_state(current)
    # Baseline for the per-turn config sync (profile home override still active).
    current["config_model_seen"] = _config_model_target()
    return True

def _start_agent_build(sid: str, session: dict) -> None:
    """Start building the real AIAgent for a TUI session, once. Deferred until the first prompt (or any
    command needing the agent) so the composer isn't blocked on tool discovery / model metadata;
    the ready/error event contract is unchanged."""
    ready = session.get("agent_ready")
    if ready is None:
        return
    # A lazy watch session spectating an in-flight child must stay lazy so the subagent live-mirror keeps
    # flowing (it bails once agent is set); incidental RPCs via _sess() would upgrade it mid-stream.
    if session.get("lazy") and _child_run_active(
            str(session.get("session_key") or ""), session.get("profile_home") or None):
        return
    with session.setdefault("history_lock", threading.Lock()):
        if session.get("_session_control_inflight") is not None:
            return
        with session.setdefault("agent_build_lock", threading.Lock()):
            if ready.is_set() or session.get("agent_build_started"):
                return
            session["agent_build_started"] = True
            session.pop("lazy", None)  # now genuinely mid-construction: restore the "still starting" eviction exemption
    key = session["session_key"]

    def _build() -> None:
        with _sessions_lock:
            current = _sessions.get(sid)
        if current is None:
            # Closed/reaped before the build started: nothing will ever attach an agent to this
            # record, yet ``agent_ready`` must be set so a prompt waiting on it fails instead of hanging.
            session["agent_error"] = AGENT_BUILD_ABANDONED
            ready.set()
            return
        notify_registered, scopes, session_db = False, None, None
        profile_home = current.get("profile_home")
        try:
            if not _await_resume_history(sid, current):
                # Replaced mid-build: the finally still sets ``agent_ready`` with ``agent`` None, so record
                # why — a turn admitted against this record refuses with the real reason (#111531).
                current["agent_error"] = AGENT_BUILD_ABANDONED
                return
            tokens = _set_session_context(key, cwd=_session_cwd(current))
            # Global-remote: bind the session profile's HERMES_HOME and hand the agent that profile's db —
            # DEDICATED and ours until _transfer_db_to_agent in the finally; FAIL CLOSED rather than
            # binding the launch DB and bleeding rows into the wrong state.db.
            scopes = _bind_build_profile_scopes(profile_home)
            if profile_home:
                session_db = _open_profile_session_db(profile_home)
            try:
                from tui_gateway.entry import ensure_mcp_discovery_started
                ensure_mcp_discovery_started()
            except Exception:
                logger.warning("MCP discovery startup failed", exc_info=True)
            try:
                agent = _make_agent(sid, key, **_deferred_build_agent_kwargs(current, session_db))
            finally:
                _clear_session_context(tokens)
            # Attach atomically against session teardown: ``session.close`` may have popped this
            # session while the expensive build was in flight, in which case teardown could not close
            # an agent that did not exist yet. Release the orphan immediately and do not keep wiring
            # workers/callbacks for a dead session (#49852).
            if not _attach_built_agent(sid, current, agent):
                # Same contract as the replaced-before-attach exit above: a turn admitted against
                # this record must refuse with the real reason rather than a generic missing agent.
                current["agent_error"] = AGENT_BUILD_ABANDONED
                with contextlib.suppress(Exception):
                    if hasattr(agent, "close"):
                        agent.close()
                return
            # No eager slash-worker pre-warm (slash.exec spawns on demand): each worker forks the full stdio
            # MCP fleet, and live-transport sessions are never reaped, so fleets would accumulate.
            notify_registered = _wire_session_agent(sid, key, agent)
            _announce_built_agent(sid, key, current, agent)
        except Exception as e:
            from agent.auxiliary_unavailable import ProviderNotConfiguredError
            current["agent_error"] = str(e)
            # A client can route "no provider is set up" to its setup flow instead of a dead-end
            # error toast — but only if it can tell. The sentence is for the reader, the code is
            # for the client; older clients keep matching the text.
            _emit("error", sid, {
                "message": agent_init_failed_message(e),
                **({"code": "provider_not_configured"} if isinstance(e, ProviderNotConfiguredError) else {})})
        finally:
            _finish_agent_build(
                sid, key, current, notify_registered=notify_registered, scopes=scopes, session_db=session_db)
            ready.set()

    build_thread = threading.Thread(target=_build, daemon=True)
    # _wait_agent_for_prompt handle: dead thread + unset agent_ready = died hard; waiters must not sit out the cap.
    session["_agent_build_thread"] = build_thread
    build_thread.start()


def register(server) -> None:
    bind_module(globals(), server)
