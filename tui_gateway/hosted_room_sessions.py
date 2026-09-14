"""Local hosted-driver ownership: retire writers, then cold-resume under a lease.

These are in-process operations, not additional RPCs. Ordinary viewer resumes keep
their lazy admission semantics; a driver is already preparing an actual task.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)
_reservation_lock = threading.Lock()
_reservation_ids: set[str] = set()


def reserved_lease_ids() -> set[str]:
    """Hydration owns real leases even before its live record is published."""
    with _reservation_lock:
        return set(_reservation_ids)


def retire_hosted_session(server, sid: str, session: dict, *, turn_finished: bool = False) -> bool:
    """Soft-retire an unwatched, quiescent hidden writer without ending its history.

    A receipt is not quiescence. Only the turn's final bookend may retire its own
    still-alive thread; other callers require that thread to have exited. Unowned
    viewer caches may be discarded, but must NEVER flush their stale transcript.
    """
    # Reattachment must see either the intact viewer or a cold-resumable durable
    # row, never a record halfway through releasing its agent. No thread joins or
    # hard task-resource cleanup run while this ownership boundary is held.
    with server._session_resume_lock:
        return _retire_locked(server, sid, session, turn_finished=turn_finished)


def _retire_locked(server, sid: str, session: dict, *, turn_finished: bool) -> bool:
    # Activation takes history before the registry; never hold the global
    # registry while waiting for this session's locks in the opposite order.
    with session.setdefault("agent_build_lock", threading.Lock()), session["history_lock"], server._sessions_lock:
        if server._sessions.get(sid) is not session or session.get("_closing"):
            return False
        if session.get("source") != "bot_room" or session.get("running"):
            return False
        worker = session.get("_run_thread")
        if worker is not None and not (turn_finished or session.get("_hosted_retirement_pending")):
            return False
        if worker is not None and worker.is_alive() and not (
                turn_finished and worker is threading.current_thread()):
            return False
        if session.get("_hosted_room_task") or session.get("_active_turn_marker_key"):
            return False
        if session.get("queued_prompt") or session.get("queued_prompts"):
            return False
        build = session.get("_agent_build_thread")
        if build is not None and build.is_alive():
            return False
        ready = session.get("agent_ready")
        if session.get("agent_build_started") and ready is not None and not ready.is_set():
            return False
        if session.get("resume_hydrating") or server._session_pending_kind(sid):
            return False
        transport = session.get("transport")
        if transport is not server._stdio_transport and not server._transport_is_dead(transport):
            return False  # An attached viewer keeps its live runtime and ownership.
        if any(viewer is not server._stdio_transport and not server._transport_is_dead(viewer)
               for viewer in (session.get("viewers") or {})):
            return False
        from tools.approval import get_pending_gateway_approval
        if get_pending_gateway_approval(str(session.get("session_key") or "")):
            return False
        if server._session_has_active_delegations(sid, session):
            return False
        with server._session_db(session) as db:
            row = db.get_session(session["session_key"]) if db is not None else None
        title = str((row or {}).get("title") or session.get("pending_title") or "")
        if not row or not row.get("hidden") or not title.startswith("Group: "):
            return False
        # Mark before releasing locks: stale RPC/poller references must fail admission.
        # Keep the record registered until cleanup succeeds so orphan sweeps cannot
        # release a lease while its last persistence operation is still in progress.
        session["_closing"] = True
        session["agent_build_started"] = True  # fence a not-yet-started pre-warm timer
        if stop := session.get("_notif_stop"):
            stop.set()
    scopes = None
    try:
        if session.get("profile_home"):
            scopes = server._bind_build_profile_scopes(session["profile_home"])
        agent = session.get("agent")
        if agent is not None:
            if session.get("active_session_lease") is not None and worker is not None:
                agent._persist_session(agent._session_messages)
                # _persist_session historically swallows a failed SQLite flush.
                # Require the flush's success receipt, not a potentially stale index.
                if agent._flush_messages_to_session_db(agent._session_messages) is not True:
                    raise RuntimeError("hosted transcript has not finished persisting")
                if session.get("pending_title"):
                    with server._session_db(session) as db:
                        if db is None or not db.set_session_title(session["session_key"], title):
                            raise RuntimeError("hosted session identity could not be persisted")
            # Unlike close(), this preserves terminal/browser/background task resources.
            agent.release_clients()
            if getattr(agent, "_owns_session_db", False):
                from hermes_state_registry import release_or_close
                release_or_close(agent._session_db)
                agent._owns_session_db = False
        if worker := session.get("slash_worker"):
            worker.close()
        from tools.approval import unregister_gateway_notify
        unregister_gateway_notify(session["session_key"])
        with session["history_lock"], server._sessions_lock:
            if not server._release_active_session_slot(session):
                raise RuntimeError("hosted session lease could not be released")
            session["_finalized"] = True
            server._sessions.pop(sid, None)
        return True
    except Exception:
        # Retain the closed-to-admission record AND lease; unknown persistence must
        # never be turned into permission for a new writer by an orphan sweep.
        logger.exception("Hosted session retirement incomplete: sid=%s", sid)
        return False
    finally:
        if scopes is not None:
            server._release_build_profile_scopes(scopes)


def resume_hosted_session(server, rid, params: dict) -> dict:
    """Use the canonical cold-resume primitives, reserving before any history read.

    A lease-less cached viewer has no freshness guarantee. Discard it, then hydrate
    with the exclusive lease held; never promote its old AIAgent to the next writer.
    No automatic continuation: the hosted task driver owns the next prompt.
    """
    ctx = server._Resume(rid, params, params["session_id"])
    ctx.db, ctx.owns_db = server._profile_session_db(ctx.profile_home)
    lease = None
    reservation_id = None
    try:
        if ctx.db is None:
            return server._db_unavailable_error(rid, code=5000)
        if (response := server._resume_locate(ctx)) is not None:
            return response
        server._resume_follow_tip(ctx)
        if (response := server._resume_guard(ctx)) is not None:
            return response
        with server._session_resume_lock:
            live = server._find_live_session_by_key(ctx.target, ctx.profile_home)
        if live is not None:
            sid, session = live
            if session.get("_closing"):
                return server._err(rid, 4090, "Hosted session retirement is still pending")
            if session.get("active_session_lease") is not None:
                # Internal task preparation must not rebind an attached frontend
                # to stdio. The driver only needs the already-owned live identity.
                with server._session_resume_lock, session["history_lock"]:
                    if server._sessions.get(sid) is not session or session.get("_closing"):
                        return server._err(rid, 4090, "Hosted session retirement is still pending")
                    if session.get("_client_gone_interrupt_requested"):
                        return server._err(rid, 4009, "Hosted session disconnect is still settling")
                    server._cancel_ws_orphan_reap(sid)
                    return server._ok(rid, {"session_id": sid, "stored_session_id": session["session_key"]})
            if not retire_hosted_session(server, sid, session):
                return server._err(rid, 4090, "Hosted session is not ready for ownership handoff")
        ctx.profile_resume_cwd = str(ctx.found.get("cwd") or "") or server._profile_configured_cwd(ctx.profile_home)
        sid, source, cwd = ctx.mint()
        lease, refusal = server._claim_active_session_slot(
            ctx.target, live_session_id=sid, surface=source, profile_home=ctx.profile_home)
        if refusal is not None or lease is None:
            return server._err(rid, 4090, str(refusal or "Hosted session ownership unavailable"))
        reservation_id = str(lease.lease_id)
        with _reservation_lock:
            _reservation_ids.add(reservation_id)
        # Compression may have transferred the former owner's lease between metadata
        # lookup and acquisition. Do not read/write an already superseded parent.
        if ctx.db.resolve_resume_session_id(ctx.target) != ctx.target:
            return server._err(rid, 4090, "Hosted session continuation changed; resolve it again")
        history, display, raw = ctx.restore()
        overrides = server._stored_session_runtime_overrides(ctx.found)
        record = ctx.record(source, cwd, history, overrides, display_history_prefix=ctx.display_prefix(),
                            todo_state=server._todo_state_from_history(history))
        record["active_session_lease"] = lease
        # A viewer may have won the registration while history was loading. Refuse
        # instead of reusing a snapshot whose read began before our reservation.
        if server._claim_or_reuse_live(sid, ctx.target, record, lease) is not None:
            return server._err(rid, 4090, "Hosted session was reattached during handoff")
        lease = None  # registered record now owns it; prompt/attachment builds on demand
        return server._resume_response(
            ctx, sid, record, info=ctx.info(cwd, overrides), display=display,
            count_source=raw, status="idle")
    finally:
        if reservation_id is not None:
            with _reservation_lock:
                _reservation_ids.discard(reservation_id)
        if lease is not None:
            lease.release()
        if ctx.owns_db and ctx.db is not None:
            server._release_db(ctx.db)
