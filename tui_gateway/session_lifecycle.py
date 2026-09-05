"""Session lifecycle: active-session slot leases, finalize/teardown/close, turn interrupt,
WS-orphan reap scheduling, transport-scoped close. Bodies are rebound onto server.py's
globals at install time (method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import contextlib

from .method_ctx import bind_module


def _notify_session_boundary(event_type: str, session_id: str | None, platform: str | None = None) -> None:
    """Fire session lifecycle hooks with CLI parity."""
    with contextlib.suppress(Exception):
        from hermes_cli.lifecycle import finalize_session, invoke_hook
        if event_type == "on_session_finalize":
            finalize_session(session_id=session_id, platform=_resolve_agent_platform(platform))
        else:
            invoke_hook(event_type, session_id=session_id, platform=_resolve_agent_platform(platform))


_SESSION_OWNERSHIP_UNAVAILABLE = "Hermes could not safely reserve this session. Try again."
_AUTOMATIC_SESSION_END_REASONS = frozenset({"ws_orphan_reap", "ws_disconnect", "idle_timeout", "lru_evict", "tui_shutdown"})


def _claim_active_session_slot(
    session_key: str, *, live_session_id: str, surface: str = "tui", profile_home: str | Path | None = None
) -> tuple[Any, str | None]:
    try:
        from hermes_cli.active_sessions import try_acquire_active_session
        return try_acquire_active_session(
            session_id=session_key, surface=surface, config=_load_cfg(), registry_home=profile_home,
            metadata={"live_session_id": live_session_id},
            track_liveness=str(surface or "").strip().lower() == "desktop")
    except Exception as exc:
        logger.warning("Failed to claim active session slot: %s", exc)
        # Fail CLOSED: an errored claim has NOT proven the session unowned; lease-less = silent double-writer hole.
        # Fail CLOSED regardless of surface: per-session exclusivity is a correctness guarantee (see
        # PER_SESSION_EXCLUSIVE_SUBMIT), and a claim that errors out has NOT proven the session is unowned.
        # Proceeding without a lease here is the silent double-writer hole flagged in the #94595 review
        # (blocker 2).
        return (None, _SESSION_OWNERSHIP_UNAVAILABLE)


def _ensure_active_session_slot(sid: str, session: dict) -> str | None:
    """Claim this session's cap slot on its first real turn; None when ok. session.create/resume deliberately
    do NOT claim: tile paints, reconnect-resumes and abandoned drafts would hold invisible slots (no DB row)
    that starve the messaging gateway sharing the cap. Anything holding a slot must be user-visible."""
    if session.get("active_session_lease") is not None:
        return None
    lease, limit_message = _claim_active_session_slot(
        str(session.get("session_key") or ""), live_session_id=sid,
        surface=_session_source(session), profile_home=session.get("profile_home"))
    if limit_message is None:
        session["active_session_lease"] = lease
    return limit_message


def _install_delegated_active_session_lease(session: dict, frame: dict) -> None:
    """Install a disabled sentinel lease for a turn admitted by the serving process.

    Turn-isolation Design 1: a named-profile turn is admitted ONCE, in the
    serving (dashboard/gateway) process, which claims the real active-session
    lease. The turn is then dispatched to a compute-host child, whose rebuilt
    session dict carries no lease. The child re-crosses the ownership chokepoint
    in ``_run_prompt_submit`` (the #94778 double-writer guard) and would
    otherwise re-claim -- and be refused, because ``_is_same_writer`` cannot
    match across a process boundary (it requires an equal pid).

    When the frame says the turn was already admitted, install a DISABLED
    ``ActiveSessionLease`` so ``_ensure_active_session_slot`` short-circuits on
    the ``active_session_lease is not None`` fast path -- no second registry
    claim. The lease is disabled (``enabled=False``), so ``release()`` is a
    no-op and ``transfer_active_session`` (fired when auto-compression rotates
    the session id mid-turn) takes its disabled-lease no-op branch instead of
    the fallback that would claim a real second slot in the child. It must NOT
    be marked ``released``: a released lease trips ``transfer_active_session``'s
    ``if lease.released`` early-return, which bypasses the disabled branch and
    falls through to that real re-claim. A real lease already present (native,
    non-isolated path) is never overwritten.

    Blocker 2 (#99719): admission is now carried as the qualified
    ``active_session_admission`` record ``{lease_id, session_id, generation}``.
    The old bare ``active_session_admitted`` boolean is derived from
    ``admission is not None`` for back-compat during rollout. The record is
    stashed on the session (``_delegated_admission``) so the child can quote it
    back verbatim in its mid-turn re-anchor proposal.
    """
    admission = frame.get("active_session_admission")
    # Back-compat: derive from either the qualified record or the legacy bool.
    admitted = admission is not None or bool(frame.get("active_session_admitted"))
    if not admitted:
        return
    if session.get("active_session_lease") is not None:
        return
    from hermes_cli.active_sessions import ActiveSessionLease

    if isinstance(admission, dict):
        session["_delegated_admission"] = dict(admission)
    session["active_session_lease"] = ActiveSessionLease(
        lease_id=f"delegated:{frame.get('session_key') or ''}",
        session_id=str(frame.get("session_key") or ""),
        surface="compute-host",
        enabled=False,
        released=False,
    )


def _lease_retry(attempts: int, fn) -> Exception | None:
    """Call ``fn`` up to ``attempts`` times (50ms*n backoff: registry writes contend across processes); last
    exception when every try failed, else None."""
    for attempt in range(attempts):
        try:
            fn()
            return None
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.05 * (attempt + 1))
    return last_error


def _release_active_session_slot(session: dict | None) -> bool:
    """Release the session's lease (liveness-tracked leases get 3 tries); True when released."""
    lease = session.get("active_session_lease") if session else None
    if lease is None:
        return True
    if (err := _lease_retry(3 if getattr(lease, "track_liveness", False) else 1, lambda: lease.release())) is not None:
        logger.warning("Failed to release active session slot", exc_info=err)
        return False
    if not (getattr(lease, "released", True) or not getattr(lease, "enabled", True)):
        return False
    if session.get("active_session_lease") is lease:
        session.pop("active_session_lease", None)
    return True


def _own_live_lease_ids(*, exclude=None) -> set[str]:
    """Snapshot leases still backed by this process's live session records."""
    with _sessions_lock:
        return {str(lease.lease_id) for session in _sessions.values()
                if (lease := session.get("active_session_lease")) is not None and lease is not exclude}


def _release_active_session_lease(lease) -> bool:
    """Release a bare lease object (no session dict), liveness-tracked leases get 3 tries; True when released."""
    if lease is None:
        return True
    if (err := _lease_retry(3 if getattr(lease, "track_liveness", False) else 1, lambda: lease.release())) is not None:
        logger.warning("Failed to release active session slot", exc_info=err)
        return False
    return bool(getattr(lease, "released", True) or not getattr(lease, "enabled", True))


def _ownership_owed_to_live_child(session: dict | None) -> bool:
    """True when the real lease must be HELD past close because an isolated
    compute-host child may still be writing this session.

    Blocker 1 (#99719): the child re-crosses the ownership chokepoint with only
    a DISABLED sentinel; its write-exclusivity rests ENTIRELY on this serving
    process holding the one real enabled lease for the whole turn. Releasing it
    while ``running`` is still true (the child has not reported turn.end) drops
    the registry entry out from under a live writer and reopens the
    double-writer window. In that case ownership is DETACHED-and-deferred, not
    released.
    """
    if not session:
        return False
    lease = session.get("active_session_lease")
    if lease is None or not getattr(lease, "enabled", False):
        return False
    if not (session.get("running") or session.get("_compute_host_active")):
        return False
    try:
        return bool(_session_uses_compute_host(session))
    except Exception:
        return False


def _detached_lease_key(sid: str, lease) -> str:
    return f"{sid}:{getattr(lease, 'lease_id', '')}"


def _detach_lease_for_deferred_release(
    sid: str, lease, *, session: dict | None = None
) -> str:
    """Move the still-enabled real lease out of the session dict into the
    process-global detached table, as ONE critical section under
    ``_sessions_lock`` that inserts into ``_detached_leases`` BEFORE removing it
    from the session dict (insert-before-remove).

    Returns the detached-table key. Idempotent for a given (sid, lease).
    """
    key = _detached_lease_key(sid, lease)
    with _sessions_lock:
        # Insert first: a reaper tick that acquires _sessions_lock between the
        # two mutations must never find the lease in NEITHER map.
        _detached_leases[key] = lease
        if session is not None and session.get("active_session_lease") is lease:
            session.pop("active_session_lease", None)
    return key


def _release_detached_leases_for_dead_child(sid: str | None = None) -> int:
    """Release detached leases whose isolated child has exited.

    Called from the per-turn done receipt (``_on_compute_host_turn_done`` for a
    single sid), the child-death paths (``host_supervisor._wait_for_exit`` ahead
    of the ``_closing`` guard, and ``shutdown()``), and the crash callback. When
    ``sid`` is None every detached lease is released (whole-child exit). Each
    release is an idempotent keyed pop, so overlapping callers are safe.
    """
    with _sessions_lock:
        if sid is None:
            popped = list(_detached_leases.items())
            _detached_leases.clear()
        else:
            prefix = f"{sid}:"
            keys = [k for k in _detached_leases if k.startswith(prefix)]
            popped = [(k, _detached_leases.pop(k)) for k in keys]
    for _key, lease in popped:
        _release_active_session_lease(lease)
    return len(popped)


@contextlib.contextmanager
def _other_runtime_lease_guard(session_id: str, session: dict):
    """Release this runtime and lock sibling ownership through the DB write. Yields True (another runtime owns
    the lifecycle -> preserve) when the guard can't be loaded/entered in 3 tries: unknown ownership never ends a row."""
    lease = session.get("active_session_lease")
    try:
        from hermes_cli.active_sessions import active_session_liveness_guard, release_active_session_liveness_guard
    except Exception as exc:
        logger.warning("Failed to load active session ownership guard; preserving session %s: %s", session_id, exc)
        yield True
        return
    stack = contextlib.ExitStack()
    active: list = []
    own_live_lease_ids = _own_live_lease_ids(exclude=lease)

    def _enter() -> None:
        stack.close()  # drop anything a half-failed previous attempt left behind
        if lease is not None and getattr(lease, "enabled", False):
            guard = release_active_session_liveness_guard(lease, session_id, own_live_lease_ids=own_live_lease_ids)
        else:
            guard = active_session_liveness_guard(
                session_id, registry_home=session.get("profile_home"), own_live_lease_ids=own_live_lease_ids)
        active[:] = [stack.enter_context(guard)]

    if (last_error := _lease_retry(3, _enter)) is not None:
        stack.close()
        logger.warning("Failed to inspect active session leases; preserving session %s: %s", session_id, last_error)
        yield True
        return
    try:
        yield active[0]
    finally:
        stack.close()
        if lease is not None and getattr(lease, "released", False) and session.get("active_session_lease") is lease:
            session.pop("active_session_lease", None)


def _transfer_active_session_slot(sid: str, session: dict, *, new_session_id: str) -> bool:
    if not new_session_id:
        return False
    lease = session.get("active_session_lease")
    if lease is None:
        return True
    try:
        from hermes_cli.active_sessions import transfer_active_session
        if transfer_active_session(lease, session_id=new_session_id, metadata={"live_session_id": sid}):
            return True
    except Exception:
        logger.debug("Failed to transfer active session slot", exc_info=True)
    if getattr(lease, "track_liveness", False):
        return False
    # Fallback (entry pruned / pid-check transiently failed): reserve the new slot BEFORE releasing the old one so
    # a gateway at the cap can't grab the freed slot and leave this session lease-less; on failure KEEP the old lease.
    # See #49041.
    new_lease, limit_message = _claim_active_session_slot(
        new_session_id, live_session_id=sid, surface=_session_source(session), profile_home=session.get("profile_home"))
    if new_lease is None:
        if limit_message:
            logger.warning("Compression session lease re-anchor failed (kept old lease): sid=%s new_session_id=%s reason=%s",
                           sid, new_session_id, limit_message)
        return False
    if (old := session.pop("active_session_lease", None)) is not None and (err := _lease_retry(1, old.release)):
        logger.debug("Failed to release stale active session slot", exc_info=err)
    session["active_session_lease"] = new_lease
    return True


# Sources this backend must never end in state.db: the messaging gateway owns those sessions and the TUI is only
# a viewer (ending one causes the Groundhog Day loop, see _finalize_session). Self-created/CLI sources are NOT gateway-owned.
# Sources the TUI backend itself creates ("tui", plus whatever a client passes as its own ``source``) and
# the CLI's own sessions are NOT gateway-owned. See #60609.
_NON_GATEWAY_SOURCES = frozenset({
    "", "tui", "cli", "webui", "desktop", "cron", "kanban", "subagent", "test",
    "local", "acp", "webhook", "api_server", "msgraph_webhook"})


def _is_gateway_owned_source(source: str) -> bool:
    """True when ``source`` resolves to a gateway ``Platform`` (enum member or plugin via ``Platform._missing_``, so
    new platforms are covered automatically); self-owned Platform members (local/webhook/api_server) are excluded."""
    src = (source or "").strip().lower()
    if src in _NON_GATEWAY_SOURCES:
        return False
    try:
        from gateway.config import Platform
        Platform(src)  # raises ValueError for arbitrary non-platform strings
        return True
    except Exception:
        return False


def _lifecycle_own_sid(session: dict, sid_hint: str = "") -> str:
    """Live UI sid for ``session``: hint, stamped ``_sid``, else registry scan."""
    own_sid = str(sid_hint or session.get("_sid") or "")
    if not own_sid:
        with contextlib.suppress(Exception), _sessions_lock:
            own_sid = next((cand_sid for cand_sid, cand in _sessions.items() if cand is session), "")
    return own_sid


def _finalize_session(session: dict | None, end_reason: str = "tui_close") -> None:
    """Best-effort finalize hook + memory commit; mirrors the CLI exit path so a force-quit mid-turn (double
    Ctrl-C, terminal close, SIGHUP) loses nothing."""
    if not session or session.get("_finalized"):
        return
    session["_finalized"] = True
    if (history_ready := session.get("resume_history_ready")) is not None and not history_ready.is_set():
        session["resume_history_error"] = "session resume cancelled"
        history_ready.set()
    _desktop_automatic_cleanup = (
        end_reason in _AUTOMATIC_SESSION_END_REASONS and _session_source(session).strip().lower() == "desktop")
    # Automatic Desktop cleanup releases its lease inside the lifecycle guard below; other paths keep force/end semantics.
    if not _desktop_automatic_cleanup:
        # Blocker 1 (#99719): do NOT release the canonical lease while an
        # isolated compute-host child may still be writing this session. When
        # ownership is still owed to a live child, DETACH-and-defer the enabled
        # lease instead -- the registry entry keeps protecting the id until the
        # child actually exits (turn.end / crash / child-death), reconciled with
        # the idle reaper (which skips detached leases) and the child-death
        # release paths. Otherwise a bounded close would drop the lease out from
        # under a live writer and reopen the double-writer window.
        if _ownership_owed_to_live_child(session):
            lease = session.get("active_session_lease")
            _detach_lease_for_deferred_release(
                str(session.get("_sid") or ""), lease, session=session
            )
        else:
            _release_active_session_slot(session)
    if (stop_event := session.get("_notif_stop")) is not None:
        stop_event.set()
    agent = session.get("agent")
    with (session.get("history_lock") or contextlib.nullcontext()):
        history = list(session.get("history", []))
    # Persist via ``_persist_session``'s marker-based dedup (gateway-shutdown flush contract). Do NOT pass
    # ``conversation_history``: ``session["history"]`` and ``_session_messages`` alias the SAME list after a turn, so
    # the flush would treat every message as durable and skip it — data loss when finalize is the sole persist path.
    if hasattr(agent, "_persist_session") and (snapshot := getattr(agent, "_session_messages", None)):
        with contextlib.suppress(Exception):
            agent._persist_session(snapshot)
    # interrupted=True so crash-recovery plugins can flush state (mirrors cli.py atexit).
    if agent is not None:
        with contextlib.suppress(Exception):
            from hermes_cli.lifecycle import invoke_hook
            invoke_hook(
                "on_session_end", completed=False, interrupted=True,
                session_id=getattr(agent, "session_id", None) or session.get("session_key", ""),
                model=getattr(agent, "model", "unknown"), platform=getattr(agent, "platform", None) or "tui")
    if agent is not None and history and hasattr(agent, "commit_memory_session"):
        with contextlib.suppress(Exception):
            agent.commit_memory_session(history)

    session_key = session.get("session_key")
    session_id = getattr(agent, "session_id", None) or session_key
    _notify_session_boundary("on_session_finalize", session_id, _session_source(session))
    # End the state.db row so it doesn't linger as a ghost in /resume. Use session_id (agent.session_id), not
    # session_key: after compression the key may be the stale ended parent while session_id is the live continuation.
    # Fix for #20001.
    if _desktop_automatic_cleanup and not session_id:
        _release_active_session_slot(session)
    _lifecycle_guard = (_other_runtime_lease_guard(session_id, session)
                        if _desktop_automatic_cleanup and session_id else contextlib.nullcontext(False))
    with _lifecycle_guard as _other_runtime_owns_lifecycle:
        _tui_owns_lifecycle = not _other_runtime_owns_lifecycle
        if _other_runtime_owns_lifecycle:
            logger.info("Preserving session %s during %s: another backend owns an active lease", session_id, end_reason)
        if session_id:
            # The *session's* profile state.db (app-global remote mode), not the launch profile's.
            with contextlib.suppress(Exception), _session_db(session) as db:
                if db is not None:
                    # Never end gateway-originated sessions: Groundhog Day loop (gateway self-heals to the parent,
                    # compression splits back to the reaped child, forever).
                    if _is_gateway_owned_source((db.get_session(session_id) or {}).get("source", "")):
                        _tui_owns_lifecycle = False
                    elif _tui_owns_lifecycle:
                        db.end_session(session_id, end_reason)
    # In-flight async delegations end WITH the session (no return address left). Always interrupt by THIS live UI
    # sid; by durable session_key only when the TUI owns the lifecycle — a viewer tab must not kill gateway work.
    with contextlib.suppress(Exception):
        from tools.async_delegation import interrupt_for_session
        interrupt_for_session(
            session_key=str(session_key or "") if _tui_owns_lifecycle else "",
            origin_ui_session_id=_lifecycle_own_sid(session), reason=end_reason)
    # Close the slash-worker in this single ``_finalized``-guarded chokepoint (a direct caller can't leak it); idempotent.
    with contextlib.suppress(Exception):
        if worker := session.get("slash_worker"):
            worker.close()


# End reasons where the BACKEND reclaimed a session the client never asked to close (else its next prompt fails
# against a forgotten id). Client-initiated reasons (``tui_close`` etc.) are deliberately absent.
_RECLAIM_END_REASONS = frozenset({"idle_timeout", "lru_evict", "ws_orphan_reap"})


def _announce_session_reclaimed(session: dict, end_reason: str) -> None:
    """Tell connected clients a session was reclaimed out from under them. Broadcast, not session-targeted: reap
    paths run on timer threads with no contextvar binding and no live transport, so ``_emit`` would hit stdio."""
    if end_reason not in _RECLAIM_END_REASONS:
        return
    try:
        _broadcast_global_event("session.reclaimed", {
            "session_id": str(session.get("_sid") or ""),
            "stored_session_id": str(session.get("session_key") or ""),
            "reason": end_reason})
    except Exception:
        logger.debug("session.reclaimed broadcast failed", exc_info=True)


def _teardown_session(session: dict | None, *, end_reason: str = "tui_close") -> None:
    """Fully tear down a session: finalize, unregister notifier, close agent (``session.close`` + WS reaper). The
    slash-worker is closed in ``_finalize_session`` (the single chokepoint), NOT here. Idempotent via ``_finalized``."""
    if not session:
        return
    _finalize_session(session, end_reason=end_reason)
    _announce_session_reclaimed(session, end_reason)
    with contextlib.suppress(Exception):
        from tools.approval import unregister_gateway_notify
        if key := session.get("session_key"):
            unregister_gateway_notify(key)
    with contextlib.suppress(Exception):
        if hasattr(agent := session.get("agent"), "close"):
            agent.close()


def _attach_worker(sid: str, session: dict, worker) -> None:
    """Store worker on session iff sid still maps to it, else close it (a concurrent teardown popped the session)."""
    with _sessions_lock:
        if _sessions.get(sid) is session:
            session["slash_worker"] = worker
            return
    worker.close()


def _pop_session_by_id(sid: str) -> dict | None:
    """Atomically detach one live session from the registry — the ownership claim for teardown (a concurrent
    close/reaper no-ops). Separate from ``_teardown_session``: slow finalization must not run under the resume lock."""
    with _sessions_lock:
        session = _sessions.pop(sid, None)
        if session is not None:
            session["_closing"] = True
            session["_sid"] = sid  # out of _sessions now, so teardown can't recover the live id by scanning
    return session


def _teardown_popped_session(session: dict | None, *, end_reason: str = "tui_close") -> bool:
    """Finish a close after the caller has atomically detached the session."""
    if session is None:
        return False
    # An isolated (compute-host) turn runs in a CHILD process, not in this
    # process's _run_thread, so the settle-grace join below cannot see it. Its
    # safety while running rests entirely on this serving process holding the
    # real active-session lease (the child's own ownership chokepoint is a
    # no-op via the delegated sentinel). _finalize_session releases that lease
    # unconditionally, so interrupt the child FIRST -- otherwise the lease
    # drops while the child keeps writing, reopening a narrow #94778 window for
    # a sibling backend sharing HERMES_HOME. The automatic reapers already gate
    # on `running`, so this only matters for an explicit close mid-turn.
    if (
        end_reason != "tui_shutdown"
        and (session.get("running") or session.get("_compute_host_active"))
        and _session_uses_compute_host(session)
    ):
        try:
            _interrupt_session_turn(str(session.get("_sid") or ""), session)
            # interrupt() is fire-and-forget (it just sends a frame to the
            # child), so wait -- bounded by the same settle grace the in-process
            # path uses for _run_thread -- for the child to report the turn done
            # (_on_compute_host_turn_done clears `running`). This keeps the real
            # lease held until the child actually stops writing, not merely until
            # the interrupt was dispatched.
            deadline = time.monotonic() + _TURN_SETTLE_BEFORE_CLOSE_SECONDS
            while session.get("running") and time.monotonic() < deadline:
                time.sleep(0.02)
            if session.get("running"):
                logger.warning(
                    "compute-host turn still running after %.1fs close grace",
                    _TURN_SETTLE_BEFORE_CLOSE_SECONDS,
                )
        except Exception:
            logger.debug("failed to interrupt compute-host turn on close", exc_info=True)
    run_thread = session.get("_run_thread")
    if end_reason != "tui_shutdown" and run_thread is not None and run_thread is not threading.current_thread():
        try:
            if run_thread.is_alive():
                run_thread.join(timeout=_TURN_SETTLE_BEFORE_CLOSE_SECONDS)
            if run_thread.is_alive():
                logger.warning(
                    "session turn thread still alive after %.1fs teardown grace", _TURN_SETTLE_BEFORE_CLOSE_SECONDS)
        except Exception:
            logger.debug("failed waiting for session turn thread", exc_info=True)
    _teardown_session(session, end_reason=end_reason)
    return True


def _close_session_by_id(
    sid: str, *, end_reason: str = "tui_close", predicate: Callable[[dict], bool] | None = None) -> bool:
    """Idempotent teardown funnel for callers with no resume race (resume-sensitive callers pop under
    ``_session_resume_lock`` and call ``_teardown_popped_session`` after releasing it). Automatic reapers pass
    ``predicate`` to revalidate under ``_sessions_lock`` right before the claim, so a stale scan can't close a
    session that reattached."""
    with _sessions_lock:  # RLock: predicate + claim in one critical section
        current = _sessions.get(sid)
        if predicate is not None and (current is None or not predicate(current)):
            return False
        session = _pop_session_by_id(sid)
    return _teardown_popped_session(session, end_reason=end_reason)


def _ws_session_is_detached(session: dict | None) -> bool:
    """True if a live session is still bound to the disconnected-WS sentinel."""
    return bool(session and not session.get("_finalized") and session.get("transport") is _detached_ws_transport)


def _ws_session_is_orphaned(session: dict | None) -> bool:
    """True if a WS session sits on ``_detached_ws_transport`` (where ``handle_ws`` parks disconnected clients), idle."""
    return bool(_ws_session_is_detached(session) and not session.get("running"))


def _interrupt_session_turn(sid: str, session: dict, *, request_id: str | None = None) -> bool:
    """Apply the shared ``session.interrupt`` contract to one claimed session; returns whether the compute-host control
    channel was used. The WS orphan reaper reuses this so a dead client gets the same partial-history/queue semantics."""
    use_compute_host = _session_uses_compute_host(session)
    should_interrupt = bool(session.get("running"))
    run_thread_alive = False
    if use_compute_host:
        # The host owns the live turn (parent `running` can lag a blocked tool), so let it decide. Gate on
        # `_compute_host_active`: HostSupervisor.interrupt() calls start(), so a lazy session would spawn a child to interrupt.
        if should_interrupt or session.get("_compute_host_active"):
            _get_compute_host_supervisor().interrupt(sid, request_id=request_id)
    else:
        run_thread_alive = (rt := session.get("_run_thread")) is not None and rt.is_alive()
    with session["history_lock"]:
        session["_turn_cancel_requested"] = True
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)
        session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
    if not use_compute_host:
        if should_interrupt:
            from agent.interrupt_compat import request_hard_interrupt
            request_hard_interrupt(session.get("agent"))
        if not run_thread_alive:
            with session["history_lock"]:
                if session.get("running"):
                    session["running"] = False
                    _clear_inflight_turn(session)
    _clear_pending(sid)
    with contextlib.suppress(Exception):
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    return use_compute_host


def _session_has_active_delegations(sid: str, session: dict | None = None) -> bool:
    """True when UI session ``sid`` still owns live background work — by live UI sid AND, when the TUI owns the durable
    lifecycle (never for gateway-viewer tabs), by session_key so a delegation from an earlier tab keeps it alive.

    See #60609.
    """
    if session is None:
        with _sessions_lock:
            session = _sessions.get(sid)
    if not session:
        return False
    own_sid = _lifecycle_own_sid(session, sid)
    owned_session_key = session_key = str(session.get("session_key") or "")
    session_id = getattr(session.get("agent"), "session_id", None) or session_key
    if session_id:
        # Only when this session may end its durable row by key — never for gateway-originated sessions (TUI is a
        # viewer there). Unknown DB state -> assume ownership.
        with contextlib.suppress(Exception):
            db = _get_db()
            if db is not None and _is_gateway_owned_source((db.get_session(session_id) or {}).get("source", "")):
                owned_session_key = ""
    if not own_sid and not owned_session_key:
        return False
    try:
        from tools.async_delegation import has_live_for_session
        return has_live_for_session(session_key=owned_session_key, origin_ui_session_id=own_sid)
    except Exception:
        logger.debug("Failed to query active delegations for UI session %s", sid, exc_info=True)
        return True  # a transient registry/import failure must not become destructive cleanup


# One pending WS-orphan reap Timer per live sid; guarded by _sessions_lock. Cancelled by _cancel_ws_orphan_reap from
# every resume/reuse/transport-rebind path — else a reap on a reattached session triggers a reap->broadcast->resume storm.
_pending_ws_reaps: dict[str, threading.Timer] = {}


def _cancel_ws_orphan_reap(sid: str) -> None:
    """Cancel a pending WS-orphan reap for ``sid`` (client came back). Called from every path that re-binds a live
    transport; closes the fired-but-not-run Timer race and stops dead Timers accumulating on flappy clients."""
    with _sessions_lock:
        timer = _pending_ws_reaps.pop(sid, None)
    if timer is not None:
        with contextlib.suppress(Exception):
            timer.cancel()


def _ws_orphan_turn_activity_is_fresh(session: dict) -> bool:
    """Whether a detached RUNNING turn's activity clock (``_touch_activity``) is still fresh — the reaper must NOT
    interrupt healthy detached work (closed laptop). Conservative: disabled threshold, missing/opaque agent, unreadable
    summary or never-stamped clock all report NOT fresh (eligible for interrupt-at-grace) to keep the wedged-turn net.

    Reuses the agent's existing activity summary (``_touch_activity`` is stamped by API waits, stream
    tokens, and tool heartbeats — the same clock the turn-liveness watchdog samples; see
    agent/turn_liveness.py). See #100325, #98028.
    """
    if _WS_ORPHAN_ACTIVITY_STALE_S <= 0:
        return False
    if not callable(summary_fn := getattr(session.get("agent"), "get_activity_summary", None)):
        return False
    try:
        elapsed = summary_fn().get("seconds_since_activity")
        return elapsed is not None and float(elapsed) < _WS_ORPHAN_ACTIVITY_STALE_S
    except Exception:
        return False


def _schedule_ws_orphan_reap(sid: str, *, delay_s: float | None = None) -> None:
    """After a grace window, reap session ``sid`` iff it's still orphaned. Called from the WS-disconnect path; a
    reconnect or ``session.resume`` cancels the reap by re-binding a live transport. Disabled when grace is 0."""
    if _WS_ORPHAN_REAP_GRACE_S <= 0:
        return

    def _reap() -> None:
        # Serialize the re-check against session.resume (rebinds under _session_resume_lock). Claim teardown by popping
        # under both locks, then release the resume lock before slow finalization. Order: resume_lock -> sessions_lock.
        reschedule_delay = interrupt_session = session = None
        with _session_resume_lock:
            # Drop this Timer's registration so a concurrent _cancel_ws_orphan_reap can't cancel a dead Timer while a
            # rescheduled one (registered below) is the owner.
            with _sessions_lock:
                _pending_ws_reaps.pop(sid, None)
            current = _sessions.get(sid)
            if current is None or not _ws_session_is_detached(current):
                return
            if _session_has_active_delegations(sid, current):
                reschedule_delay = _WS_ORPHAN_REAP_GRACE_S
            elif not current.get("running"):
                session = _pop_session_by_id(sid)
            elif not current.get("_client_gone_interrupt_requested") and _ws_orphan_turn_activity_is_fresh(current):
                # Client-absent but producing: keep running detached (the sentinel buffers emits), re-check each grace.
                logger.debug("client_gone sid=%s action=defer (turn activity fresh; stale threshold %.0fs)",
                             sid, _WS_ORPHAN_ACTIVITY_STALE_S)
                reschedule_delay = _WS_ORPHAN_REAP_GRACE_S
            else:
                # Mid-turn detached sessions must never drop the single Timer: interrupt once after grace, then poll
                # until turn-finalization settles.
                polls = current["_client_gone_interrupt_polls"] = int(current.get("_client_gone_interrupt_polls") or 0) + 1
                # See #85578.
                if polls > _WS_ORPHAN_INTERRUPT_REAP_MAX_POLLS:
                    # Never settled inside the budget — force-reap rather than park forever.
                    logger.error(
                        "client_gone sid=%s: turn did not settle after %d interrupt polls (%.0fs) — force-reaping detached session",
                        sid, polls - 1, (polls - 1) * _WS_ORPHAN_INTERRUPT_REAP_POLL_S)
                    session = _pop_session_by_id(sid)
                else:
                    if not current.get("_client_gone_interrupt_requested"):
                        current["_client_gone_interrupt_requested"] = True
                        interrupt_session = current
                    reschedule_delay = _WS_ORPHAN_INTERRUPT_REAP_POLL_S
        if interrupt_session is not None:
            try:
                isolated = _interrupt_session_turn(sid, interrupt_session, request_id=f"client-gone-{sid}")
                logger.info("client_gone sid=%s action=interrupt turn_isolation=%s", sid, isolated)
            except Exception:
                logger.exception("client_gone interrupt failed sid=%s", sid)
                with _sessions_lock:
                    if _sessions.get(sid) is interrupt_session:
                        interrupt_session.pop("_client_gone_interrupt_requested", None)
        if reschedule_delay is not None:
            _schedule_ws_orphan_reap(sid, delay_s=reschedule_delay)
            return
        if session is not None and session.get("_client_gone_interrupt_requested"):
            logger.info("client_gone sid=%s action=reap", sid)
        _teardown_popped_session(session, end_reason="ws_orphan_reap")

    timer = threading.Timer(_WS_ORPHAN_REAP_GRACE_S if delay_s is None else max(0.0, delay_s), _reap)
    timer.daemon = True
    with _sessions_lock:
        prior = _pending_ws_reaps.pop(sid, None)
        _pending_ws_reaps[sid] = timer
    if prior is not None:
        with contextlib.suppress(Exception):
            prior.cancel()
    timer.start()


def _close_sessions_for_transport(transport, *, end_reason: str = "ws_disconnect") -> tuple[int, int]:
    """Single WS-disconnect teardown entry point: reap close_on_disconnect sessions (sidecar/dashboard) immediately;
    re-point the rest at the detached transport (later emits miss the dead socket) for the grace-windowed WS-orphan
    reaper. Returns ``(reaped, detached)`` counts."""
    with _sessions_lock:
        owned = [(sid, s) for sid, s in _sessions.items() if s.get("transport") is transport]
    reaped = detached = 0
    for sid, session in owned:
        claimed_for_teardown = None
        should_schedule_reap = False
        # session.resume fast-path rebinds under _session_resume_lock: take it so a reconnect can't move the transport
        # between check and claim.
        with _session_resume_lock, _sessions_lock:
            current = _sessions.get(sid)
            if current is not session:
                continue
            if current.get("transport") is not transport:
                # The reconnect owns this session now; drop only the old viewer registration.
                (current.get("viewers") or {}).pop(transport, None)
                continue
            if current.get("close_on_disconnect"):
                claimed_for_teardown = _pop_session_by_id(sid)
            else:
                # Point at the drop sentinel (NOT real stdio) so _ws_session_is_orphaned recognizes it; standalone
                # `hermes --tui` keeps real _stdio. UNLESS another window (pop-out viewer) still shows the session:
                # re-bind to the most recent surviving viewer instead.
                viewers = current.get("viewers") or {}
                # See #83716.
                viewers.pop(transport, None)
                live = [vt for vt, ts in sorted(viewers.items(), key=lambda kv: kv[1]) if not _transport_is_dead(vt)]
                if live:
                    current["transport"] = live[-1]
                else:
                    current["transport"] = _detached_ws_transport
                    current.pop("_client_gone_interrupt_requested", None)
                    should_schedule_reap = True
        if claimed_for_teardown is not None:
            reaped += _teardown_popped_session(claimed_for_teardown, end_reason=end_reason)
        elif should_schedule_reap:
            detached += 1
            with contextlib.suppress(Exception):
                _schedule_ws_orphan_reap(sid)
    return reaped, detached


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
