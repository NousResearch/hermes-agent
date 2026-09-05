"""Session lifecycle: active-session slot leases, finalize/teardown/close, turn interrupt,
WS-orphan reap scheduling, transport-scoped close. Bodies are rebound onto server.py's
globals at install time (method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import logging

import contextlib
import time

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
_AUTOMATIC_SESSION_END_REASONS = frozenset({
    "ws_orphan_reap", "ws_disconnect", "idle_timeout", "lru_evict", "tui_shutdown",
    # Preemptible leases: the backend's own maintenance watcher closes a session whose
    # lease was stolen — automatic (the user never asked), and the client must learn the
    # session moved (session.reclaimed broadcast below).
    "lease_preempted"})


def _claim_active_session_slot(
    session_key: str, *, live_session_id: str, surface: str = "tui", profile_home: str | Path | None = None,
    mark_busy: bool = True,
) -> tuple[Any, str | None]:
    try:
        from hermes_cli.active_sessions import try_acquire_active_session
        return try_acquire_active_session(
            session_id=session_key, surface=surface, config=_load_cfg(), registry_home=profile_home,
            metadata={"live_session_id": live_session_id, "bot_live_delivery_consumer": True},
            track_liveness=str(surface or "").strip().lower() == "desktop",
            # Every gateway claim is a turn admission (or a mid-turn compression re-anchor):
            # publishing busy from the first instant prevents two requesters ping-ponging a
            # stolen lease inside the first turn (preemptible leases).
            mark_busy=mark_busy)
    except Exception as exc:
        logger.warning("Failed to claim active session slot: %s", exc)
        # Fail CLOSED: an errored claim has NOT proven the session unowned; lease-less = silent double-writer hole.
        # Fail CLOSED regardless of surface: per-session exclusivity is a correctness guarantee (see
        # PER_SESSION_EXCLUSIVE_SUBMIT), and a claim that errors out has NOT proven the session is unowned.
        # Proceeding without a lease here is the silent double-writer hole flagged in the #94595 review
        # (blocker 2).
        return (None, _SESSION_OWNERSHIP_UNAVAILABLE)


def _yield_local_desktop_owner(session_key: str) -> bool:
    """Auto-yield (#auto-yield patch): when a cross-surface send (Relay) is fenced out by a
    lease THIS process holds for the desktop UI, and that desktop session is idle, close it
    locally so the other surface can take over. Same teardown the WS-orphan reaper performs;
    the desktop tab receives ``session.reclaimed`` and reloads from the DB on next use.
    Returns True when a local owner existed and was closed."""
    wanted = str(session_key or "")
    if not wanted:
        return False
    with _sessions_lock:
        candidates = [
            (sid, sess) for sid, sess in _sessions.items()
            if str(sess.get("session_key") or "") == wanted
            and (sess.get("active_session_lease") is not None)
            and not sess.get("running")
        ]
        if not candidates:
            return False
        sid, _ = candidates[0]
    return _close_session_by_id(
        sid, end_reason="ws_orphan_reap",
        predicate=lambda sess: (
            str(sess.get("session_key") or "") == wanted
            and not sess.get("running")))


# Preemptible leases (#auto-yield -> preemptible leases): requester-side bounded retry.
# A SESSION_BUSY refusal retries ONLY while the fresh holder entry shows work that is about
# to become stealable (auto inside its grace window, or a user turn approaching the stall
# threshold); a user-fresh or legacy refusal returns immediately — ~50-150ms instead of the
# retired 8s/1.0s yield-request poll that produced the 1.08s x9 refusal bursts.
_LEASE_BUSY_RETRY_INTERVAL_S = 0.25
_LEASE_BUSY_RETRY_WINDOW_S = 2.0


def _busy_refusal_worth_retry(refusal) -> bool:
    """Whether a SESSION_BUSY refusal describes a holder whose busy state is ABOUT to
    lapse, making a short bounded retry cheaper for the user than an immediate error."""
    from hermes_cli import active_sessions as _as
    holder = getattr(refusal, "holder_entry", None)
    if not isinstance(holder, dict) or not holder.get("busy"):
        return False  # legacy/idle holder: nothing is about to change
    now = time.time()
    kind = holder.get("busy_kind")
    if kind == "auto":
        since = _as._optional_float(holder.get("busy_since"))
        return since is None or (now - since) <= _as.HERMES_LEASE_AUTO_BUSY_GRACE_S
    if kind == "user":
        activity = _as._optional_float(holder.get("activity_at"))
        stall = _as.HERMES_LEASE_STALL_ACTIVITY_S
        # Only when one more retry window crosses the stall threshold: a fresh user turn
        # must never burn retries (it may legitimately run for minutes).
        return (activity is not None and stall > 0
                and (now - activity) + _LEASE_BUSY_RETRY_WINDOW_S > stall)
    return False


def _lease_admission_check(sid: str, session: dict) -> str | None:
    """Epoch-fenced admission mark for a session that already holds its lease: one
    revalidate-under-flock that both proves the lease is still ours and publishes
    busy_kind='user' (the lock-serialized point making steal-vs-admit mutually
    exclusive). None when admitted; on displacement the session is gracefully closed
    (end_reason 'lease_preempted' → session.reclaimed broadcast, DB row preserved via
    _other_runtime_lease_guard) and the taken-over refusal is returned so the
    submitting surface learns immediately."""
    lease = session.get("active_session_lease")
    if lease is None:
        return None  # unleased (empty-key no-op lease path): nothing to fence
    from hermes_cli.active_sessions import revalidate_active_session, SESSION_DISPLACED
    refreshed, refusal = revalidate_active_session(lease, mark_busy=True, busy_kind="user")
    if refusal is None:
        _install_lease_busy_hook(sid, session)
        return None
    reason = getattr(refusal, "reason", None)
    if reason == SESSION_DISPLACED:
        logger.info(
            "lease_preempted sid=%s: admission fenced (lease taken by pid=%s); closing",
            sid, (getattr(refusal, "holder_entry", None) or {}).get("pid"))
        with contextlib.suppress(Exception):
            _close_session_by_id(sid, end_reason="lease_preempted")
        from hermes_cli.active_sessions import ActiveSessionRefusal
        return ActiveSessionRefusal(
            "Session taken over by another surface; reload to continue",
            SESSION_DISPLACED,
            holder_entry=getattr(refusal, "holder_entry", None))
    return refusal  # e.g. SESSION_COORDINATION_UNAVAILABLE: fail closed, keep the session


def _session_lease_busy(sid: str, session: dict, kind: str | None, detail: str) -> None:
    """Holder-side auto-busy token bookkeeping (bg-review): registers invisible work on
    the session's lease. ``kind='auto'`` marks, ``kind=None`` clears. The registry write
    is epoch-fenced and identity-matched inside revalidate_active_session — a displaced
    holder can never mutate the new owner's entry."""
    from hermes_cli.active_sessions import revalidate_active_session
    lease = session.get("active_session_lease")
    if lease is None:
        return
    tokens = session.get("_lease_auto_busy_tokens")
    with contextlib.suppress(Exception), session["history_lock"]:
        if kind == "auto":
            tokens = (set(tokens) if isinstance(tokens, set) else set()) | {str(detail or "auto")}
            session["_lease_auto_busy_tokens"] = tokens
        else:
            tokens = (set(tokens) if isinstance(tokens, set) else set()) - {str(detail or "auto")}
            session["_lease_auto_busy_tokens"] = tokens
    remaining = sorted(session.get("_lease_auto_busy_tokens") or ())
    with contextlib.suppress(Exception):
        if remaining:
            revalidate_active_session(
                lease, mark_busy=True, busy_kind="auto", busy_detail=remaining[0])
        else:
            revalidate_active_session(lease, mark_busy=False, busy_kind="auto")


def _install_lease_busy_hook(sid: str, session: dict) -> None:
    """Expose the session's lease-busy channel to its agent so agent-level invisible work
    (bg-review) can publish busy_kind='auto' for its lifetime. The closure binds the
    SESSION dict (not the agent object), so it survives agent rebuilds mid-session."""
    agent = session.get("agent")
    if agent is None:
        return
    with contextlib.suppress(Exception):
        agent._session_lease_busy_hook = (
            lambda kind, detail: _session_lease_busy(sid, session, kind, detail))


def _lease_turn_settled(session: dict) -> None:
    """Turn-thread finally: clear the published busy mark beside running=False, BEFORE
    the finished log and post-turn followups — the registry reads idle in the
    inter-turn gap, so a steal there breaks a followup chain at its next admission
    instead of stranding the phone. An active auto token (bg-review) re-marks 'auto'
    rather than clearing: the review's grace window must survive the turn that spawned it."""
    from hermes_cli.active_sessions import revalidate_active_session
    lease = session.get("active_session_lease")
    if lease is None:
        return
    tokens = session.get("_lease_auto_busy_tokens") or ()
    with contextlib.suppress(Exception):
        if tokens:
            refreshed, refusal = revalidate_active_session(
                lease, mark_busy=True, busy_kind="auto", busy_detail=sorted(tokens)[0])
        else:
            refreshed, refusal = revalidate_active_session(lease, mark_busy=False)
        if refusal is not None and getattr(refusal, "reason", None) == "SESSION_DISPLACED":
            # The turn's tail raced a steal: the new owner's registry state is untouched
            # by this write (identity-fenced inside revalidate) — nothing to do here.
            logger.debug("lease settled after displacement; registry left to the new owner")


def _touch_lease_activity(session: dict) -> None:
    """Turn-path activity piggyback (streaming/tool touch): throttled, non-blocking,
    epoch-fenced refresh of activity_at + heartbeat_at so a visibly streaming turn keeps
    itself protected EVEN when the maintenance watcher thread is dead. Skips on flock
    contention — the next touch retries."""
    lease = session.get("active_session_lease")
    if lease is None:
        return
    from hermes_cli.active_sessions import HERMES_LEASE_ACTIVITY_REFRESH_MIN_S
    now = time.monotonic()
    if now - float(session.get("_lease_activity_refresh_monotonic") or 0.0) < HERMES_LEASE_ACTIVITY_REFRESH_MIN_S:
        return
    session["_lease_activity_refresh_monotonic"] = now
    with contextlib.suppress(Exception):
        from hermes_cli.active_sessions import refresh_lease_activity
        refresh_lease_activity(lease, activity_at=time.time())


def _ensure_active_session_slot(sid: str, session: dict) -> str | None:
    """Claim this session's cap slot on its first real turn; None when ok. session.create/resume deliberately
    do NOT claim: tile paints, reconnect-resumes and abandoned drafts would hold invisible slots (no DB row)
    that starve the messaging gateway sharing the cap. Anything holding a slot must be user-visible.

    Preemptible leases: a cached lease is epoch-fenced-revalidated (displacement closes the
    session as 'lease_preempted'); a fresh claim either STEALS atomically inside the
    registry flock or refuses fast — the cooperative yield-request dance is retired on the
    requester side (new requesters never write request files to acquire)."""
    if session.get("active_session_lease") is not None:
        return _lease_admission_check(sid, session)

    def _claim():
        return _claim_active_session_slot(
            str(session.get("session_key") or ""), live_session_id=sid,
            surface=_session_source(session), profile_home=session.get("profile_home"))

    lease, limit_message = _claim()
    if limit_message is None:
        session["active_session_lease"] = lease
        _install_lease_busy_hook(sid, session)
        _clear_stolen_turn_marker(sid, session, lease)
        return None
    refusal_reason = getattr(limit_message, "reason", None)
    # #auto-yield patch (kept): a same-process desktop tab holds the lease and is idle ->
    # close it and retry, so a phone send takes over in single-process topologies. Under
    # preemption the registry steal usually wins first; this stays as the local fast path.
    from hermes_cli.active_sessions import SESSION_NOT_OWNED, SESSION_BUSY
    if refusal_reason == SESSION_NOT_OWNED and _yield_local_desktop_owner(
            str(session.get("session_key") or "")):
        lease, limit_message = _claim()
        if limit_message is None:
            logger.info(
                "Auto-yield: closed local desktop owner for session %s; cross-surface turn admitted",
                session.get("session_key") or sid)
            session["active_session_lease"] = lease
            _install_lease_busy_hook(sid, session)
            _clear_stolen_turn_marker(sid, session, lease)
            return None
        refusal_reason = getattr(limit_message, "reason", None)
    # Bounded busy-retry: only while the holder's published state is about to lapse (auto
    # inside grace / user approaching stall). User-fresh and legacy refusals return
    # immediately — worst case ~2.3s, user-fresh ~50-150ms.
    if refusal_reason in (SESSION_BUSY, SESSION_NOT_OWNED):
        deadline = time.monotonic() + _LEASE_BUSY_RETRY_WINDOW_S
        while _busy_refusal_worth_retry(limit_message) and time.monotonic() < deadline:
            time.sleep(min(_LEASE_BUSY_RETRY_INTERVAL_S, max(0.0, deadline - time.monotonic())))
            lease, limit_message = _claim()
            if limit_message is None:
                logger.info(
                    "Preemptible lease: holder for %s became stealable; turn admitted after bounded retry",
                    session.get("session_key") or sid)
                session["active_session_lease"] = lease
                _install_lease_busy_hook(sid, session)
                _clear_stolen_turn_marker(sid, session, lease)
                return None
            refusal_reason = getattr(limit_message, "reason", None)
            if refusal_reason not in (SESSION_BUSY, SESSION_NOT_OWNED):
                break  # a different problem appeared; surface it
    return limit_message


def _clear_stolen_turn_marker(sid: str, session: dict, lease) -> None:
    """After a successful STEAL (epoch > 1), force-clear any durable turn marker for this
    session key in its home BEFORE the new owner's first admission — the displaced
    surface's in-flight prompt must not auto-continue on the new owner's surface (the
    observed 01:32:10 double-submit hazard)."""
    if int(getattr(lease, "epoch", 1) or 1) <= 1:
        return
    with contextlib.suppress(Exception):
        from tui_gateway.turn_marker import clear_turn_marker
        key = str(session.get("session_key") or "")
        if key:
            clear_turn_marker(_session_home(session), key, force=True)


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
        if transfer_active_session(lease, session_id=new_session_id, metadata={
                "live_session_id": sid, "bot_live_delivery_consumer": True}):
            return True
    except Exception:
        logger.debug("Failed to transfer active session slot", exc_info=True)
    if getattr(lease, "track_liveness", False):
        return False
    # Fallback (entry pruned / pid-check transiently failed): reserve the new slot BEFORE releasing the old one so
    # a gateway at the cap can't grab the freed slot and leave this session lease-less; on failure KEEP the old lease.
    # See #49041. mark_busy=True: compression re-anchors MID-TURN, so the new lease publishes the
    # same busy state the continuing turn already owns (its finally clears it).
    new_lease, limit_message = _claim_active_session_slot(
        new_session_id, live_session_id=sid, surface=_session_source(session),
        profile_home=session.get("profile_home"), mark_busy=True)
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


def _lock_vault_managers(session: dict) -> None:
    """A per-session unlock ends with the session that made it; siblings in the same profile keep theirs."""
    try:
        from agent.vault_backends import unlock

        if sid := session.get("_sid"):
            unlock.release_session(sid)
    except Exception:
        logging.getLogger(__name__).debug("vault manager lock on session end failed", exc_info=True)


def _finalize_session(session: dict | None, end_reason: str = "tui_close", *, skip_persist: bool = False) -> None:
    """Best-effort finalize hook + memory commit; mirrors the CLI exit path so a force-quit mid-turn (double
    Ctrl-C, terminal close, SIGHUP) loses nothing. ``skip_persist`` (preemptible leases): a
    lease_preempted teardown whose displaced run thread survived the settle join skips the
    transcript persist AND the memory commit — the new owner owns the row, and a lost
    displaced tail beats a corrupted row."""
    if not session or session.get("_finalized"):
        return
    session["_finalized"] = True
    _lock_vault_managers(session)
    if (history_ready := session.get("resume_history_ready")) is not None and not history_ready.is_set():
        session["resume_history_error"] = "session resume cancelled"
        history_ready.set()
    _desktop_automatic_cleanup = (
        end_reason in _AUTOMATIC_SESSION_END_REASONS and _session_source(session).strip().lower() == "desktop")
    # Automatic Desktop cleanup releases its lease inside the lifecycle guard below; other paths keep force/end semantics.
    if not _desktop_automatic_cleanup:
        _release_active_session_slot(session)
    if (stop_event := session.get("_notif_stop")) is not None:
        stop_event.set()
    agent = session.get("agent")
    with (session.get("history_lock") or contextlib.nullcontext()):
        history = list(session.get("history", []))
    # Persist via ``_persist_session``'s marker-based dedup (gateway-shutdown flush contract). Do NOT pass
    # ``conversation_history``: ``session["history"]`` and ``_session_messages`` alias the SAME list after a turn, so
    # the flush would treat every message as durable and skip it — data loss when finalize is the sole persist path.
    if not skip_persist and hasattr(agent, "_persist_session") and (snapshot := getattr(agent, "_session_messages", None)):
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
    if not skip_persist and agent is not None and history and hasattr(agent, "commit_memory_session"):
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
    # The lifecycle guard applies to every automatic DESKTOP cleanup, AND to every
    # lease_preempted close REGARDLESS of surface (preemptible leases): a displaced
    # phone/webui session must not end the state.db row the new desktop owner now holds
    # (C4 symmetry — the row is shared state, the surface is not).
    _guard_warranted = bool(session_id) and (_desktop_automatic_cleanup or end_reason == "lease_preempted")
    _lifecycle_guard = (_other_runtime_lease_guard(session_id, session)
                        if _guard_warranted else contextlib.nullcontext(False))
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
_RECLAIM_END_REASONS = frozenset({
    "idle_timeout", "lru_evict", "ws_orphan_reap",
    # Preemptible leases: another surface stole this session — the displaced client must
    # reload from the DB instead of continuing against its stale in-memory copy.
    "lease_preempted"})


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


def _teardown_session(session: dict | None, *, end_reason: str = "tui_close", skip_persist: bool = False) -> None:
    """Fully tear down a session: finalize, unregister notifier, close agent (``session.close`` + WS reaper). The
    slash-worker is closed in ``_finalize_session`` (the single chokepoint), NOT here. Idempotent via ``_finalized``."""
    if not session:
        return
    _finalize_session(session, end_reason=end_reason, skip_persist=skip_persist)
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


# Wall-clock timestamps, like session last_active; retained after close/reap.
_closed_session_activity: dict[str, float] = {}


def _pop_session_by_id(sid: str) -> dict | None:
    """Atomically detach one live session from the registry — the ownership claim for teardown (a concurrent
    close/reaper no-ops). Separate from ``_teardown_session``: slow finalization must not run under the resume lock."""
    with _sessions_lock:
        session = _sessions.pop(sid, None)
        if session is not None:
            from hermes_constants import get_hermes_home

            home = str(Path(session.get("profile_home") or get_hermes_home()).resolve())
            last_active = time.time() if session.get("running") else float(session.get("last_active") or 0)
            _closed_session_activity[home] = max(_closed_session_activity.get(home, 0), last_active)
            session["_closing"] = True
            session["_sid"] = sid  # out of _sessions now, so teardown can't recover the live id by scanning
    return session


def _teardown_popped_session(session: dict | None, *, end_reason: str = "tui_close") -> bool:
    """Finish a close after the caller has atomically detached the session."""
    if session is None:
        return False
    run_thread = session.get("_run_thread")
    thread_alive_after_settle = False
    if end_reason != "tui_shutdown" and run_thread is not None and run_thread is not threading.current_thread():
        try:
            if run_thread.is_alive():
                run_thread.join(timeout=_TURN_SETTLE_BEFORE_CLOSE_SECONDS)
            thread_alive_after_settle = run_thread.is_alive()
            if thread_alive_after_settle:
                logger.warning(
                    "session turn thread still alive after %.1fs teardown grace", _TURN_SETTLE_BEFORE_CLOSE_SECONDS)
        except Exception:
            logger.debug("failed waiting for session turn thread", exc_info=True)
    # Preemptible leases: a lease_preempted teardown whose displaced run thread SURVIVED the
    # settle join skips finalize's persist + memory commit — the new owner owns the row now,
    # and a lost displaced tail beats a corrupted row (the displaced turn's writes stopped at
    # the interrupt and must never reach the DB through teardown).
    skip_persist = end_reason == "lease_preempted" and thread_alive_after_settle
    if skip_persist:
        logger.warning(
            "lease_preempted teardown: run thread alive after settle; skipping persist (new owner owns the row)")
    _teardown_session(session, end_reason=end_reason, skip_persist=skip_persist)
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
    if should_interrupt:
        # Sibling of gateway/run_agent_cache.py::_interrupt_and_clear_session: a user-initiated stop of a
        # live TUI/desktop turn is the same "loop is gone" event for plugins holding per-turn external
        # resources. Observer-only; dispatch failures never break the interrupt.
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "agent_loop_stopped", session_key=session.get("session_key", ""), platform="tui",
                reason="user_stop", invalidation_reason="session_interrupt",
            )
        except Exception:
            logger.debug("agent_loop_stopped hook dispatch failed", exc_info=True)
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
        # The row lives in the session's OWN store (a named-profile session's row is invisible to
        # the launch handle, which would leave this guard permanently dead).
        with contextlib.suppress(Exception), _session_db(session) as db:
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


def _reattach_refusal(rid, sid: str, session: dict) -> dict | None:
    """Under ``_session_resume_lock``: why a reattaching RPC (resume/activate/prompt.submit) must NOT rebind
    ``session`` — it is stale, or a client-gone interrupt is still settling and the reap Timer must keep
    polling. None when the reattach may proceed."""
    if _sessions.get(sid) is not session:
        return _err(rid, 4007, "session no longer live; retry resume")
    if session.get("_client_gone_interrupt_requested"):
        return _err(rid, 4009, "session disconnect interrupt settling")
    return None


def _rebind_live_transport(sid: str, session: dict, transport: Transport) -> None:
    """Attach a live peer without displacing existing subscribers (caller holds ``history_lock``).
    Subagent control authority needs no bookkeeping here: it resolves against ``session["transport"]``
    at RPC time (``tools.delegate_tool_registry._subagent_transport_matches``)."""
    _attach_session_transport(session, transport)
    # Every transport that showed this session (pop-outs resume the same sid); on disconnect the last
    # viewer becomes the transport instead of the drop sentinel.
    session.setdefault("viewers", {})[transport] = time.time()
    # See #83716.
    if transport is not _detached_ws_transport:
        _cancel_ws_orphan_reap(sid)  # the client is back — a pending ws-orphan reap must not fire


def _ws_orphan_turn_activity_is_fresh(session: dict) -> bool:
    """Whether a detached RUNNING turn's activity clock (``_touch_activity``) is still fresh — the reaper must NOT
    interrupt healthy detached work (closed laptop). Conservative: disabled threshold, missing/opaque agent, unreadable
    summary or never-stamped clock all report NOT fresh (eligible for interrupt-at-grace) to keep the wedged-turn net.

    Reuses the agent's existing activity summary (``_touch_activity`` is stamped by API waits, stream
    tokens, and tool heartbeats — the same clock the turn-liveness watchdog samples; see
    agent/turn_liveness.py). See #100325, #98028.
    Isolated turns mirror that clock from the child under a unique dispatch token;
    their monotonic samples keep aging even if the child or its pipe stalls.
    """
    if _WS_ORPHAN_ACTIVITY_STALE_S <= 0:
        return False
    if session.get("_compute_host_turn_id"):
        with session["history_lock"]:
            stamp = session.get("_compute_host_activity_ns")
            return (session.get("running", False) and isinstance(stamp, int)
                    and 0 <= (time.perf_counter_ns() - stamp) / 1_000_000_000 < _WS_ORPHAN_ACTIVITY_STALE_S)
    if not callable(summary_fn := getattr(session.get("agent"), "get_activity_summary", None)):
        return False
    try:
        elapsed = summary_fn().get("seconds_since_activity")
        return elapsed is not None and float(elapsed) < _WS_ORPHAN_ACTIVITY_STALE_S
    except Exception:
        return False


def _schedule_ws_orphan_reap(
    sid: str, *, delay_s: float | None = None, _expected_timer: threading.Timer | None = None,
) -> None:
    """After a grace window, reap session ``sid`` iff it's still orphaned. Called from the WS-disconnect path; a
    reconnect or ``session.resume`` cancels the reap by re-binding a live transport. Disabled when grace is 0."""
    if _WS_ORPHAN_REAP_GRACE_S <= 0:
        return

    def _reap() -> None:
        # Serialize the re-check against session.resume (rebinds under _session_resume_lock). Claim teardown by popping
        # under both locks, then release the resume lock before slow finalization. Order: resume_lock -> sessions_lock.
        reschedule_delay = interrupt_session = session = None
        with _session_resume_lock, _sessions_lock:
            # Keep ownership through interrupt I/O and continuation registration. A cancelled
            # callback may already be dispatched, but cannot act on a later detachment.
            if _pending_ws_reaps.get(sid) is not timer:
                return
            current = _sessions.get(sid)
            if current is None:
                _pending_ws_reaps.pop(sid, None)
                return
            if not _ws_session_is_detached(current):
                # This Timer is abandoning the interrupt claim because another
                # writer moved the live record off the detached transport.
                # Do not leave reattach RPCs fenced with 4009, or let this
                # generation's settlement polls shorten a later detachment.
                current.pop("_client_gone_interrupt_requested", None)
                current.pop("_client_gone_interrupt_polls", None)
                _pending_ws_reaps.pop(sid, None)
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
            if reschedule_delay is None:
                _pending_ws_reaps.pop(sid, None)
        if interrupt_session is not None:
            try:
                isolated = _interrupt_session_turn(sid, interrupt_session, request_id=f"client-gone-{sid}")
                logger.info("client_gone sid=%s action=interrupt turn_isolation=%s", sid, isolated)
            except Exception:
                logger.exception("client_gone interrupt failed sid=%s", sid)
                with _sessions_lock:
                    if (_sessions.get(sid) is interrupt_session
                            and _pending_ws_reaps.get(sid) is timer):
                        interrupt_session.pop("_client_gone_interrupt_requested", None)
        if reschedule_delay is not None:
            _schedule_ws_orphan_reap(sid, delay_s=reschedule_delay, _expected_timer=timer)
            return
        if session is not None and session.get("_client_gone_interrupt_requested"):
            logger.info("client_gone sid=%s action=reap", sid)
        _teardown_popped_session(session, end_reason="ws_orphan_reap")

    with _sessions_lock:
        if _expected_timer is not None and _pending_ws_reaps.get(sid) is not _expected_timer:
            return
        timer = threading.Timer(_WS_ORPHAN_REAP_GRACE_S if delay_s is None else max(0.0, delay_s), _reap)
        timer.daemon = True
        prior = _pending_ws_reaps.pop(sid, None)
        _pending_ws_reaps[sid] = timer
    if prior is not None:
        with contextlib.suppress(Exception):
            prior.cancel()
    timer.start()


def _close_sessions_for_transport(transport, *, end_reason: str = "ws_disconnect") -> tuple[int, int]:
    """Single WS-disconnect teardown entry point: reap close_on_disconnect sessions (sidecar/dashboard) immediately;
    re-point the rest at the detached transport (later emits miss the dead socket) for the grace-windowed WS-orphan
    reaper. Returns ``(reaped, detached)`` counts.

    Multi-client fan-out: the departing transport is DETACHED from every session first. A session that still has
    another client attached keeps streaming and is neither parked nor reaped — a watcher leaving must not end the
    turn the remaining client is reading. Only the sessions left clientless take the historical
    close_on_disconnect / park-sentinel path, so a single-client disconnect behaves exactly as it always has."""
    clientless = _detach_transport_from_sessions(transport)
    reaped = detached = 0
    for sid, session in clientless:
        claimed_for_teardown = None
        should_schedule_reap = False
        # session.resume fast-path attaches under _session_resume_lock: take it so a reconnect can't attach
        # between the detach above and the claim.
        with _session_resume_lock, _sessions_lock:
            current = _sessions.get(sid)
            if current is not session:
                continue
            # Prune the departing viewer registration in every branch; it must not affect the new owner.
            (current.get("viewers") or {}).pop(transport, None)
            # Revalidate before claiming (#77129, kept under fan-out): between _detach_transport_from_sessions
            # returning this session as clientless and this claim, a concurrent session.resume can attach a NEW
            # live transport. Tearing the session down, or parking the sentinel over it, would knock an attached
            # client into detached state and arm an orphan reap against a session that has a live owner. Attach
            # and detach both serialize on _session_transport_lock, so this check is race-free against them; that
            # lock is a leaf, so taking it under _sessions_lock is safe and _session_has_live_transport does not
            # re-acquire it.
            with _session_transport_lock:
                if _session_has_live_transport(current, excluding=transport):
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
                    _rebind_live_transport(sid, current, live[-1])
                else:
                    current["transport"] = _detached_ws_transport
                    current.pop("_client_gone_interrupt_requested", None)
                    current.pop("_client_gone_interrupt_polls", None)
                    should_schedule_reap = True
                    # Register before releasing the detachment claim: an old disconnect
                    # must not arm its first timer over a reconnect's newer detachment.
                    with contextlib.suppress(Exception):
                        _schedule_ws_orphan_reap(sid)
        if claimed_for_teardown is not None:
            reaped += _teardown_popped_session(claimed_for_teardown, end_reason=end_reason)
        elif should_schedule_reap:
            detached += 1
    return reaped, detached


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
