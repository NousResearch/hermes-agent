"""Session flush / reaping / orphan sweep / cross-backend heartbeat: exit-flush signal handlers, idle + LRU
eviction (compacting a cache-expiry-due chat before release), the cache-expiry compaction ticker, orphaned
session-row sweep, backend heartbeat refresher. Bodies are rebound onto server.py's
globals at install time (method_ctx.bind_module), so they reference server.py globals bare — including the
knobs _SESSION_TTL_S, _REAPER_SCAN_S, _EXIT_FLUSH_BUDGET_S and _INCREMENTAL_FLUSH_INTERVAL_S.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import queue
import secrets
import threading
import time
from typing import Any, Callable

from tui_gateway._env import env_float

from .method_ctx import bind_module


# ── Flush-on-kill + periodic incremental flush ───────────────────────────
# (a) SIGTERM/SIGINT run a bounded flush to state.db BEFORE normal shutdown, chained to the prior handler;
# (b) the idle-reaper scan piggybacks an incremental flush so a SIGKILL loses at most one interval.


def _session_has_pending_flush(session: dict | None) -> bool:
    """Would :func:`_flush_session_messages` write anything for this session? (The exit flush counts
    these to tell a complete flush from a budget overrun.)"""
    agent = session.get("agent") if isinstance(session, dict) else None
    return bool(hasattr(agent, "_persist_session") and getattr(agent, "_session_messages", None))


def _flush_session_messages(session: dict | None) -> bool:
    """Best-effort durable flush of one session's transcript via ``agent._persist_session`` (same marker-deduped
    contract as ``_finalize_session``: repeated calls never duplicate rows).

    See #13121.
    """
    agent = session.get("agent") if session else None
    snapshot = getattr(agent, "_session_messages", None) if hasattr(agent, "_persist_session") else None
    if not snapshot:
        return False
    # config.yaml, the token ledger and the memory/provider lookups ``_persist_session`` touches are
    # resolved at CALL time, and every caller of this helper is an unscoped reaper / exit-flush thread
    # with no turn on the stack — so a SERVED profile's flush resolved them against the LAUNCH home.
    # (The transcript ROWS were never at risk: a profile session gets its own SessionDB whose db_path
    # is frozen at construction, tui_gateway/server.py::_open_profile_session_db). Same binding
    # _finalize_session makes at the identical chokepoint (session_lifecycle.py).
    #
    # hydrate_secrets=False: persisting a transcript needs no external credential, and hydration
    # shells out to the operator's secret command (30s CLI budget, process-global lock) inside a
    # worker whose whole budget is 5s — one slow source silently lost the transcript.
    try:
        with _session_profile_runtime_scope(session or {}, hydrate_secrets=False):
            agent._persist_session(snapshot)
        return True
    except Exception:
        logger.debug("incremental session flush failed", exc_info=True)
        return False


def _reaper_session_snapshot() -> list:
    with _sessions_lock:
        return list(_sessions.values())


def _flush_dirty_sessions(now: float | None = None) -> int:
    """Periodic incremental flush, driven by the idle-reaper scan. Skips ``running`` sessions: the turn thread
    owns mid-turn persistence and mutates the live message list, so racing it from the reaper thread is never
    safe. Idle sessions flush at most once per ``_INCREMENTAL_FLUSH_INTERVAL_S``; ``now`` (monotonic) is
    injectable for tests."""
    if _INCREMENTAL_FLUSH_INTERVAL_S <= 0:
        return 0
    now = time.monotonic() if now is None else now
    flushed = 0
    for session in _reaper_session_snapshot():
        if not isinstance(session, dict) or session.get("running"):
            continue
        last = float(session.get("_last_incremental_flush") or 0.0)
        if last and (now - last) < _INCREMENTAL_FLUSH_INTERVAL_S:
            continue
        flushed += _flush_session_messages(session)
        session["_last_incremental_flush"] = now
    return flushed


def _flush_sessions_before_exit(budget_s: float | None = None) -> int:
    """Bounded flush of ALL in-memory sessions on the way out, on a daemon worker joined with the budget so a
    hung SQLite write can't block exit past ``HERMES_TUI_EXIT_FLUSH_BUDGET_S`` (default 5s). Running sessions
    are included — the process is dying, a partial transcript beats loss."""
    budget = _EXIT_FLUSH_BUDGET_S if budget_s is None else max(0.0, budget_s)
    if budget <= 0:
        return 0
    result = {"flushed": 0}
    sessions = _reaper_session_snapshot()
    flushable = sum(1 for s in sessions if _session_has_pending_flush(s))

    def _run() -> None:
        deadline = time.monotonic() + budget
        for session in sessions:
            if time.monotonic() >= deadline:
                break
            result["flushed"] += _flush_session_messages(session)

    worker = threading.Thread(target=_run, daemon=True, name="hermes-exit-flush")
    worker.start()
    worker.join(budget)
    # Silent loss is the failure mode this guard exists to prevent: both callers discard the return
    # value, so a budget overrun (a slow state.db write, a scope binding that shells out) looks
    # exactly like a clean exit.
    if result["flushed"] < flushable:
        logger.warning(
            "Exit flush persisted %d of %d in-memory session transcript(s) within %.1fs; the rest "
            "may have been lost (HERMES_TUI_EXIT_FLUSH_BUDGET_S)",
            result["flushed"], flushable, budget)
    return result["flushed"]


# Bounded wait for interrupted turns on the way out; the SIGTERM path hard-exits after a ~1s grace.
_EXIT_TURN_SETTLE_S = 0.5


def _stop_turns_before_exit(budget_s: float | None = None) -> None:
    """Interrupt every in-flight turn and give it ``budget_s`` to settle, so a running tool call ends
    with a result the teardown's final persist records. A foreground command runs in its own process
    group and would outlive the gateway, reparented to init. One still alive halfway through the budget
    ignored the interrupt's SIGTERM: SIGKILL it then, early enough for its result to land as well (the
    interrupt's own TERM, 1s, KILL outlasts the SIGTERM path's ~1s grace)."""
    with _sessions_lock:
        running = [(sid, s) for sid, s in _sessions.items() if s.get("running")]
    threads = []
    for sid, session in running:
        with contextlib.suppress(Exception):
            _interrupt_session_turn(sid, session)
        if (t := session.get("_run_thread")) is not None and t is not threading.current_thread():
            threads.append(t)
    budget = _EXIT_TURN_SETTLE_S if budget_s is None else max(0.0, budget_s)
    deadline = time.monotonic() + budget

    def _join(until: float) -> None:
        for t in threads:
            t.join(max(0.0, until - time.monotonic()))

    _join(deadline - budget / 2)
    from tools.environments.base import kill_live_foreground_processes
    kill_live_foreground_processes(now=True)
    _join(deadline)


_exit_flush_prev_handlers: dict[int, Any] = {}
_exit_flush_handlers_installed = False


def _handle_exit_flush_signal(signum, frame) -> None:
    """Flush in-memory sessions, then hand off to the prior handler (uvicorn's graceful shutdown, a supervisor's
    handler, or the default disposition) — this only *prepends* a bounded flush."""
    import signal as _signal
    prev = _exit_flush_prev_handlers.get(signum)
    if prev is _signal.SIG_IGN:
        # An inherited ignore (`cmd &` from a non-interactive shell) ends nothing: stopping turns here
        # would raise the one-way exit fence in a process that keeps running and refuses every command.
        return
    with contextlib.suppress(Exception):
        _flush_sessions_before_exit()
    # The group signal that stopped us never reaches a command in its own session: reap it now,
    # before a supervisor's SIGKILL can cut the graceful shutdown (and its atexit) short.
    with contextlib.suppress(Exception):
        _stop_turns_before_exit()
    if callable(prev):
        prev(signum, frame)
    else:
        # Default disposition: restore it and re-raise so the process dies with the correct signal (exit status
        # visible to supervisors).
        try:
            _signal.signal(signum, _signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except Exception:
            raise SystemExit(128 + int(signum)) from None


def install_exit_flush_signal_handlers() -> bool:
    """Install chaining SIGTERM/SIGINT flush handlers (main thread only). Called before uvicorn takes over
    signals: its ``capture_signals()`` saves these as the "original" handlers and re-raises into them after
    graceful shutdown, so the flush also covers terminations outside uvicorn's serve window. Idempotent; False
    off-main-thread/on failure."""
    global _exit_flush_handlers_installed
    if _exit_flush_handlers_installed:
        return True
    if threading.current_thread() is not threading.main_thread():
        return False
    import signal as _signal
    installed = False
    for signum in (_signal.SIGTERM, _signal.SIGINT):
        with contextlib.suppress(ValueError, OSError, RuntimeError):
            prev = _signal.getsignal(signum)
            _signal.signal(signum, _handle_exit_flush_signal)
            _exit_flush_prev_handlers[signum] = prev
            installed = True
    _exit_flush_handlers_installed = installed
    return installed


def _transport_is_dead(transport) -> bool:
    # _detached_ws_transport is the post-disconnect drop sentinel. _stdio_transport is the REAL transport for
    # standalone `hermes --tui` and must NOT count as dead.
    if transport is _detached_ws_transport:
        return True
    if isinstance(transport, FanoutTransport):
        # A fan-out is never the sentinel and has no ``_closed`` of its own, so without this arm every
        # multi-client session reads as alive forever — the TTL reaper, the LRU cap and the #77129 disconnect
        # revalidation all gate on this predicate. A fan-out can legitimately end up empty, or holding nothing
        # but closed sockets, with no disconnect passing through _close_sessions_for_transport: a failed write
        # prunes the peer that failed. It is dead exactly when no peer of its own is alive, and an empty one is
        # dead. Peers are always leaf transports (attach flattens a fan-out argument instead of nesting it), so
        # this recurses one level at most.
        return all(_transport_is_dead(peer) for peer in transport.transports())
    return getattr(transport, "_closed", None) is True


def _session_is_lru_evictable(sid: str, session: dict) -> bool:
    """Shared hard exemptions for both reapers (the LRU cap applies them WITHOUT the age gate: eligible the moment
    it loses its client): never evict a session mid-turn, awaiting input, still building, owning live delegated
    work, or on a live transport. Lazy watch sessions never start a build, so their unset agent_ready must not
    make them immortal."""
    if session.get("running") or _session_pending_kind(sid) or _session_has_active_delegations(sid, session):
        return False
    ready = session.get("agent_ready")
    if ready is not None and not ready.is_set() and not session.get("lazy"):
        return False
    return _transport_is_dead(session.get("transport"))


def _sessions_quiescent(exclude: str | None = None) -> bool:
    """No session but ``exclude`` is mid-turn, building, awaiting input, holding live delegations, or on a live
    transport. A non-forced memory trim holds the GIL (gc.collect) and every glibc arena lock (malloc_trim) for
    its whole duration — 20-50 s on multi-GB heaps — which stalls the event loop, drops WS clients past the
    write deadline and interrupts their turns (#58576); this is the moment it costs no other session. The
    predicate is advisory (a turn can start right after), so the per-session checks — one may read state.db —
    run outside ``_sessions_lock``."""
    with _sessions_lock:
        others = [(sid, s) for sid, s in _sessions.items() if sid != exclude]
    return all(_session_is_lru_evictable(sid, s) for sid, s in others)


def _session_is_reapable(sid: str, session: dict) -> bool:
    """Reaper eligibility: the LRU exemptions, and not the sole receiver of a pending wake-up — an active /loop or
    /heartbeat, or a running background process that will notify it (closing the session stops the notification
    poller that delivers both, and ``agent_close`` kills the process). Kept out of ``_sessions_quiescent``: an idle
    session waiting on a wake-up does not block a memory trim."""
    return (_session_is_lru_evictable(sid, session) and not _session_owns_live_wakeup_schedule(session)
            and not _session_owns_pending_process_notification(session))


def _session_is_evictable(sid: str, session: dict, now: float) -> bool:
    """TTL eviction: idle-for-TTL AND older-than-TTL, plus the reap exemptions (the cheap clock check first: the
    exemptions read the session store, and this runs for every session on each reaper pass)."""
    last_active = float(session.get("last_active") or 0.0)
    created_at = float(session.get("created_at") or 0.0)
    if (now - last_active) <= _SESSION_TTL_S or (now - created_at) <= _SESSION_TTL_S:
        return False
    return _session_is_reapable(sid, session)


def _reap_idle_sessions() -> None:
    now = time.time()
    try:  # piggyback the incremental flush on the reaper tick — no new timer subsystem
        _flush_dirty_sessions()
    except Exception:
        logger.debug("periodic incremental session flush failed", exc_info=True)
    with _sessions_lock:
        victims = [sid for sid, s in _sessions.items() if _session_is_evictable(sid, s, now)]
    for sid in victims:
        with _sessions_lock:
            session = _sessions.get(sid)
        if session is not None and _compact_then_close(sid, session, end_reason="idle_timeout") is not None:
            continue  # step 2 at release: that job compacts, then closes it
        _close_session_by_id(
            sid, end_reason="idle_timeout",
            predicate=lambda session, vs=sid: _session_is_evictable(vs, session, time.time()))
    _repair_missing_ws_orphan_reaps()
    _enforce_session_cap()
    try:
        if released := _release_idle_session_leases(now):
            logger.info("Released %d idle Desktop active-session lease(s)", released)
    except Exception:
        logger.debug("idle lease release failed", exc_info=True)
    _reclaim_orphaned_leases()
    # Long-lived processes: gen2 GC rarely runs at steady state and glibc retains freed pages as RSS, so trim
    # every quiescent scan to prevent unbounded RSS growth over days/weeks. Forced trims (agent close, cache
    # pressure) are unaffected.
    if not _sessions_quiescent():
        logger.debug("idle reaper periodic trim deferred: a session is busy or attached")
        return
    try:
        from hermes_cli.mem_trim import trim_memory_when_idle
        trim_memory_when_idle(reason="idle reaper periodic trim")
    except Exception as exc:  # debug, not warning — a persistent failure would repeat every scan.
        logger.debug("idle reaper memory trim failed: %s: %s", type(exc).__name__, exc)


def _repair_missing_ws_orphan_reaps() -> None:
    """Re-arm detached sessions whose disconnect path lost its teardown timer.

    A resident record otherwise vouches for its lease during every orphan sweep,
    while the live process prevents PID pruning. Reusing the normal WS grace
    path preserves reconnect and in-flight-work protections instead of stealing
    the lease directly.
    """
    if _WS_ORPHAN_REAP_GRACE_S <= 0:
        return
    # A socket can be closed before its disconnect cleanup reaches the sentinel.
    # Reuse that cleanup (including surviving viewers), never revoke a fence from
    # a stale liveness snapshot.
    with _sessions_lock:
        closed_transports = [session.get("transport") for session in _sessions.values()
                             if session.get("transport") is not _detached_ws_transport
                             and _transport_is_dead(session.get("transport"))]
    for transport in closed_transports:
        _close_sessions_for_transport(transport)
    with _sessions_lock:
        missing = [
            sid for sid, session in _sessions.items()
            if _ws_session_is_detached(session) and sid not in _pending_ws_reaps
        ]
        for sid in missing:
            _schedule_ws_orphan_reap(sid)


# An idle Desktop record keeps its runtime (warm tab reopen) but not its active-session lease. Desktop's tab
# close sends no RPC and the app's live socket exempts the record from TTL/LRU reaping, while
# _own_live_lease_ids vouches for every resident record, so a lease claimed on turn 1 fenced Telegram, the
# dashboard and message_agent out of that chat until the app quit (#146735). The next turn re-claims it
# through _admit_prompt_turn; a surface that took the chat meanwhile then refuses it, as exclusivity requires.
_IDLE_LEASE_RELEASE_S = 600.0


def _session_lease_is_idle(sid: str, session: dict, now: float) -> bool:
    """A liveness-tracked (Desktop) lease on a record with no turn, build, input wait or delegation in flight,
    untouched for ``_IDLE_LEASE_RELEASE_S``. A borrowed compute-host token (``enabled=False``) is not ours."""
    lease = session.get("active_session_lease")
    if lease is None or not getattr(lease, "track_liveness", False) or not getattr(lease, "enabled", True):
        return False
    if now - float(session.get("last_active") or 0.0) <= _IDLE_LEASE_RELEASE_S:
        return False
    if session.get("running") or session.get("_compute_host_turn_id") or _session_holds_transcript_claim(session, now):
        return False
    ready = session.get("agent_ready")
    if ready is not None and not ready.is_set():
        return False
    return not (_session_pending_kind(sid) or _session_has_active_delegations(sid, session)
                or _session_owns_live_wakeup_schedule(session) or _session_owns_pending_process_notification(session))


def _release_idle_session_leases(now: float | None = None) -> int:
    """Release the active-session lease of every idle resident Desktop record; returns the count."""
    now = time.time() if now is None else now
    with _sessions_lock:
        candidates = [(sid, s) for sid, s in _sessions.items() if s.get("active_session_lease") is not None]
    released = 0
    for sid, session in candidates:
        if not _session_lease_is_idle(sid, session, now):
            continue
        # Turns mark ``running`` and off-turn writers refresh ``last_active`` / hold a transcript claim under
        # history_lock: re-evaluate the WHOLE predicate there, or a writer that passed its ownership check after
        # the unlocked probe above loses the lease mid-write. ``_lease_idle_released`` is set under the same lock:
        # a submit that checked the lease before this release, and every rewind, refuses to write the transcript
        # until the lease is re-claimed.
        with session.get("history_lock") or contextlib.nullcontext():
            if session.get("_finalized") or not _session_lease_is_idle(sid, session, now):
                continue
            if _release_active_session_slot(session):
                session["_lease_idle_released"] = sid  # the writer identity a re-claim must present
                released += 1
    return released


def _reclaim_orphaned_leases() -> None:
    """Hand the registry the lease ids we still own so it can drop the rest."""
    try:
        # Stale deferred leases first: a settlement callback that never arrived must not
        # keep vouching for a zombie slot (#62823). Released leases leave _own_live_lease_ids.
        _reap_stale_deferred_leases()
    except Exception:
        logger.debug("stale deferred lease sweep failed", exc_info=True)
    try:
        from hermes_cli.active_sessions import release_orphaned_leases
        if dropped := release_orphaned_leases(_own_live_lease_ids()):
            logger.info("Reclaimed %d orphaned active-session lease(s)", dropped)
    except Exception:
        logger.debug("orphaned lease reclaim failed", exc_info=True)


# Soft LRU cap on in-memory sessions: the TTL reaper only frees sessions idle for hours, so a heavy reconnecting
# user accumulates resident detached agents. The cap evicts the least-recently-active DETACHED sessions sooner —
# never a running / pending / mid-build / live-transport one (reopening re-resumes from the DB). 0/null disables.
def _max_live_sessions() -> int:
    try:
        from hermes_cli.active_sessions import coerce_max_concurrent_sessions
        cfg = _load_cfg() or {}
        raw = cfg.get("max_live_sessions")
        if raw is None and isinstance(gateway_cfg := cfg.get("gateway"), dict):
            raw = gateway_cfg.get("max_live_sessions")
        coerced = coerce_max_concurrent_sessions(raw, key="max_live_sessions")
        return int(coerced) if coerced else 0
    except Exception:
        return 0


def _enforce_session_cap() -> None:
    cap = _max_live_sessions()
    if cap <= 0:
        return
    with _sessions_lock:
        if len(_sessions) <= cap:
            return
        evictable = [(sid, s) for sid, s in _sessions.items() if _session_is_reapable(sid, s)]
    # Oldest-touched first; evict only down to the cap (may stop short: live sessions are never eligible).
    # A close deferred behind its release-time compaction still sits in ``_sessions``: it counts as gone.
    evictable.sort(key=lambda kv: float(kv[1].get("last_active") or 0.0))

    def over_cap() -> bool:
        with _sessions_lock:
            current_cap = _max_live_sessions()  # re-read: a cap disabled mid-sweep evicts nothing more
            return current_cap > 0 and len(_sessions) - _pending_release_closes() > current_cap

    for sid, _s in evictable:
        if not over_cap():
            break
        if _compact_then_close(sid, _s, end_reason="lru_evict", still_due=over_cap) is not None:
            continue  # step 2 at release: that job compacts, then closes it if the cap still needs it
        _close_session_by_id(
            sid, end_reason="lru_evict", predicate=lambda session, vs=sid: _session_is_reapable(vs, session))


def _reaper_daemon_timer(delay: float, fn, fail_log: str, level: str = "debug") -> None:
    """Run ``fn`` once on a daemon Timer after ``delay``; log (never raise) on failure."""
    def _run() -> None:
        try:
            fn()
        except Exception:
            getattr(logger, level)(fail_log, exc_info=True)

    timer = threading.Timer(delay, _run)
    timer.daemon = True
    timer.start()


def _schedule_session_cap_enforcement() -> None:
    """Run the LRU sweep off the response path (eviction can call agent.close)."""
    _reaper_daemon_timer(0.1, _enforce_session_cap, "session cap enforcement failed")


# ── Startup sweep for orphaned session rows ──────────────────────────────
# The WS-orphan reaper is an in-process Timer: a gateway restart kills it before it fires, leaving the row
# `ended_at IS NULL` forever. Scheduled once per process from both gateway entry points (stdio `entry.main`, WS
# sidecar `handle_ws`). state.db is shared by sibling processes on the same profile, so eligibility is
# conservative. Disable via `dashboard.startup_orphan_sweep: false`.
# This is the startup complement every other resource type already has (docker_orphan_reaper, compression
# orphans). See #65194.
_ORPHAN_SWEEP_SOURCES = ("tui", "desktop", "subagent", "unknown")
_startup_orphan_sweep_ran = False
_startup_orphan_sweep_lock = threading.Lock()


def _session_orphan_reaper_enabled() -> bool:
    """``dashboard.startup_orphan_sweep`` (default on). Fail-open on errors and on a missing key (raw yaml, no
    DEFAULT_CONFIG merge on this loader)."""
    try:
        dashboard_cfg = (_load_cfg() or {}).get("dashboard") or {}
        if isinstance(dashboard_cfg, dict) and "startup_orphan_sweep" in dashboard_cfg:
            return is_truthy_value(dashboard_cfg.get("startup_orphan_sweep"), default=True)
    except Exception:
        pass
    return True


def _sweep_orphaned_session_rows() -> list[str]:
    """End orphaned tui/desktop/subagent rows left by a dead process. "Provably orphaned" is inferred
    conservatively: the row must have been created AND last messaged at least the session TTL ago (a fresh row
    that copied an old transcript is protected by its own ``started_at``). Rows held in memory (e.g. a
    ``session.resume`` in the startup grace window) are excluded. Cross-backend: the sweep refuses to close a
    row any live backend (heartbeat within ``2 * TTL``) could own — see ``maintenance.sweep_orphaned_sessions``."""
    db = _get_db()
    if db is None or _SESSION_TTL_S <= 0:
        return []
    live_ids: set[str] = set()  # every id this process holds in memory: live sid, agent session_id, session_key
    with _sessions_lock:
        for sid, session in _sessions.items():
            candidates = [sid]
            if isinstance(session, dict):
                candidates += [getattr(session.get("agent"), "session_id", None), session.get("session_key")]
            live_ids.update(str(c) for c in candidates if c)
    with _persistence_for(db) as persistence:
        swept = list(persistence.maintenance.sweep_orphaned_sessions(
            max_idle_seconds=_SESSION_TTL_S, sources=_ORPHAN_SWEEP_SOURCES,
            exclude_ids=tuple(sorted(live_ids)), exclude_pinned=True))
    if swept:
        logger.info(
            "Closed %d orphaned session row(s) from a previous gateway process (startup_orphan_reap): %s",
            len(swept), ", ".join(swept))
    return swept


# ── Cross-backend heartbeat ──────────────────────────────────────────────
# Each serve / gateway process registers a heartbeat row in ``gateway_heartbeats`` so the startup sweep can tell
# "owned by a live but idle backend" from "truly orphaned" (else the first process to restart reaped every
# inactive row of the other N−1). Refresh 60s default — far shorter than the 6h TTL so a refresh always lands
# inside the staleness window. Removed at exit; a crashed row ages out.
_HEARTBEAT_REFRESH_S = max(0.0, env_float("HERMES_GATEWAY_HEARTBEAT_REFRESH_S", 60.0))
_heartbeat_refresher_started = False
_heartbeat_refresher_lock = threading.Lock()
_BACKEND_NONCE = secrets.token_hex(4)


def _reaper_hostname() -> str:
    return os.uname().nodename if hasattr(os, "uname") else "host"


def _backend_id_for_this_process() -> str:
    """Stable identity for this process's heartbeat row: pid (readability) AND a startup nonce so a PID-reuse
    respawn cannot inherit the dead predecessor's heartbeat."""
    return f"{_current_profile_name()}@{_reaper_hostname()}:{os.getpid()}:{_BACKEND_NONCE}"


def _gateway_started_at() -> float:
    """Wall-clock time this process started (first-call time is a good-enough proxy: the heartbeat refresher
    runs after the gateway is fully wired up)."""
    if getattr(_gateway_started_at, "_t", None) is None:
        _gateway_started_at._t = time.time()
    return _gateway_started_at._t


def _refresh_backend_heartbeat() -> None:
    """Refresh this backend's heartbeat row. No-op when DB unavailable."""
    db = _get_db()
    if db is None:
        return
    try:
        with _persistence_for(db) as persistence:
            persistence.maintenance.register_backend_heartbeat(
                backend_id=_backend_id_for_this_process(), pid=os.getpid(), started_at=_gateway_started_at(),
                profile=_current_profile_name(), host=_reaper_hostname())
    except Exception:
        logger.debug("backend heartbeat refresh failed", exc_info=True)


def _start_backend_heartbeat_refresher() -> None:
    """Register this backend and start the refresher thread (once per process). The first refresh writes the row
    synchronously so this process's own sweep sees itself in the heartbeat table. ``_HEARTBEAT_REFRESH_S <= 0``
    means "register once, never refresh"."""
    global _heartbeat_refresher_started
    with _heartbeat_refresher_lock:
        if _heartbeat_refresher_started:
            return
        _heartbeat_refresher_started = True
    try:
        _refresh_backend_heartbeat()
    except Exception:
        logger.debug("initial backend heartbeat write failed", exc_info=True)
    if _HEARTBEAT_REFRESH_S <= 0:
        return
    stop_event = threading.Event()

    def _loop() -> None:
        while not stop_event.is_set():
            try:
                _refresh_backend_heartbeat()
            except Exception:
                logger.debug("heartbeat refresh loop iteration failed", exc_info=True)
            stop_event.wait(_HEARTBEAT_REFRESH_S)

    def _atexit_clear():
        stop_event.set()
        with contextlib.suppress(Exception):
            if (db := _get_db()) is not None:
                with _persistence_for(db) as persistence:
                    persistence.maintenance.clear_backend_heartbeat(_backend_id_for_this_process())

    atexit.register(_atexit_clear)
    threading.Thread(target=_loop, name="hermes-gateway-heartbeat", daemon=True).start()


def _schedule_startup_orphan_sweep() -> None:
    """Schedule the once-per-process startup orphan sweep, delayed by the WS-orphan grace window so a client
    reconnecting right after a restart can ``session.resume`` its row first. Grace 0 (park forever), TTL 0 and
    ``dashboard.startup_orphan_sweep: false`` all suppress the sweep.

    See #65194.
    """
    global _startup_orphan_sweep_ran
    if _WS_ORPHAN_REAP_GRACE_S <= 0 or _SESSION_TTL_S <= 0 or not _session_orphan_reaper_enabled():
        return
    with _startup_orphan_sweep_lock:
        if _startup_orphan_sweep_ran:
            return
        _startup_orphan_sweep_ran = True
    _reaper_daemon_timer(
        _WS_ORPHAN_REAP_GRACE_S, _sweep_orphaned_session_rows, "startup orphan session sweep failed", level="warning")

# ── Cache-edge compaction and offload (steps 2-3 of agent.prompt_cache_lifecycle) ──
# A daemon ticker compacts an idle TUI / Desktop session at the end of its keep-alive budget (nothing is
# sent after it until the next user message), or offloads old tool output once the cache is over (step 3).
# It runs the manual-compress choke point
# (busy claim, snapshot fence, lease) with an automatic trigger, so a prompt arriving meanwhile queues
# behind it as it does behind ``/compress``, and a job still queued when its chat closes (``_closing``)
# is refused at that claim. The idle reapers above release through :func:`_compact_then_close`: a budget
# ends at its keep-alives or at the release, whichever comes first, so a due chat is compacted (or, past its
# window, offloaded) before it is closed.

_cache_expiry_started = False
_cache_expiry_lock = threading.Lock()
# Summaries running at once: one slow summary must not hold every other due chat past its window
# (the ticker used to run them serially), nor may a burst of due chats fan out unbounded. The pool's
# workers are the bound: a due chat past them waits in the pool's queue, not on a thread of its own.
_CACHE_EXPIRY_CONCURRENCY = 3
_cache_expiry_queue: queue.SimpleQueue = queue.SimpleQueue()
_cache_expiry_workers: list[threading.Thread] = []
_cache_expiry_in_flight: set[str] = set()
# Reaper closes waiting behind a ticker job already queued for that sid: run once it finishes, so the
# close never lands first (the job would see ``_closing`` and drop its compaction or offload).
_cache_expiry_after: dict[str, list[Callable[[], None]]] = {}
# Sessions a reaper is releasing behind their release-time compaction: the session cap counts them as gone.
_cache_expiry_releasing: set[str] = set()


def _cache_expiry_worker() -> None:
    while True:
        future, job, args = _cache_expiry_queue.get()
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(job(*args))
            except BaseException as exc:  # health: allow BLE001 -- handed to the future's owner, which re-raises it
                future.set_exception(exc)


def _cache_expiry_submit(job: Callable[..., Any], *args: Any) -> concurrent.futures.Future:
    """Queue *job* for the cache-edge workers, under the caller's contextvars (profile scope). At most
    ``_CACHE_EXPIRY_CONCURRENCY`` daemon workers exist: a burst of due chats waits in the queue, not on
    a thread each, and a summary still running never holds the process open at exit."""
    from agent.memory_provider import ctx_bound, spawn_context_thread
    future: concurrent.futures.Future = concurrent.futures.Future()
    _cache_expiry_queue.put((future, ctx_bound(job), args))
    with _cache_expiry_lock:
        while len(_cache_expiry_workers) < _CACHE_EXPIRY_CONCURRENCY:
            worker = spawn_context_thread(_cache_expiry_worker, name=f"cache-expiry-{len(_cache_expiry_workers)}")
            _cache_expiry_workers.append(worker)
            worker.start()
    return future


def _cache_expiry_candidates() -> list[tuple[str, dict, str]]:
    """Due sessions as ``(sid, session, action)``: the claim (``PromptCacheWarmer.claim_edge``) latched
    each one's stretch and says whether it is compacted or offloaded."""
    with _sessions_lock:
        live = list(_sessions.items())
    with _cache_expiry_lock:
        busy = set(_cache_expiry_in_flight)
    picked = []
    for sid, session in live:
        agent = session.get("agent")
        warmer = getattr(agent, "_prompt_cache_warmer", None)
        if warmer is None or sid in busy or session.get("running") or session.get("_closing"):
            continue
        if _session_uses_compute_host(session):
            continue  # the compute host owns that transcript; it is not compacted from here
        with _session_profile_runtime_scope(session):
            action = warmer.claim_edge()
        if action:
            picked.append((sid, session, action))
    return picked


def _offload_expired_cache(sid: str, session: dict) -> None:
    """Step 3: the cache is over and nothing compacted it, so old tool output goes to disk now (the cold
    prune), under the same manual-compress claim as a compaction, so the person returns to a small chat."""
    from agent.turn_proactive_prune import offload_cold_history
    agent = session["agent"]
    with _session_profile_runtime_scope(session), _manual_compress_turn(sid, session):
        with session["history_lock"]:
            before, version = list(session.get("history", [])), int(session.get("history_version", 0))
        after = offload_cold_history(agent, before, session.get("session_key") or sid)
        if after is before:
            return
        with session["history_lock"]:
            if int(session.get("history_version", 0)) != version:
                return  # history moved meanwhile: the next turn's at-return check covers it
            session["history"], session["history_version"] = after, version + 1
    logger.info("Cache-expiry offload: sid=%s", sid)


def _compact_before_cache_expiry(sid: str, session: dict, action: str = "compact") -> None:
    """Run the claimed cache-edge *action* on *session*: ``offload`` (step 3) or ``compact`` at the end
    of its keep-alive budget (step 2). Nothing is sent afterwards: the next user message starts the
    compacted chat's cache (and its keep-alives)."""
    from agent.conversation_compression import finalize_context_engine_compression_notification
    from agent.manual_compression_feedback import summarize_manual_compression
    from agent.prompt_cache_lifecycle import TRIGGER
    agent = session["agent"]
    if action == "offload":
        _offload_expired_cache(sid, session)
        return
    with _session_profile_runtime_scope(session), _manual_compress_turn(sid, session):
        with session["history_lock"]:
            before, version = list(session.get("history", [])), int(session.get("history_version", 0))
        try:
            removed, _usage = _compress_session_history(
                session, trigger=TRIGGER, before_messages=before, history_version=version)
        except CompressionLockHeld:
            return
        except Exception:
            finalize_context_engine_compression_notification(agent, committed=False)
            raise
        _sync_session_key_after_compress(sid, session, clear_pending_title=False)
        finalize_context_engine_compression_notification(agent, committed=True)
        with session["history_lock"]:
            compacted = list(session.get("history", []))
        summarize_manual_compression(before, compacted, 0, 0,
                                     compression_state=getattr(agent, "context_compressor", None))
    logger.info("Cache-expiry compaction: sid=%s removed=%d messages", sid, removed)


def _cache_expiry_job(sid: str, session: dict, action: str = "compact") -> None:
    try:
        _compact_before_cache_expiry(sid, session, action)
    except (CompressionBusy, SessionNotOwned):
        pass  # a turn, a close or another owner got there first: nothing to do this window
    except Exception:
        logger.warning("cache-expiry compaction failed sid=%s", sid, exc_info=True)
    finally:
        with _cache_expiry_lock:
            _cache_expiry_in_flight.discard(sid)
            waiting = _cache_expiry_after.pop(sid, [])
        for then in waiting:
            then()


def _cache_expiry_tick() -> list[concurrent.futures.Future]:
    """Queue one job per due session on the bounded pool (contextvars carried, so the profile scope
    survives the hop); a sid already in flight is never claimed again. Returns the queued futures."""
    futures = []
    for sid, session, action in _cache_expiry_candidates():
        with _cache_expiry_lock:
            _cache_expiry_in_flight.add(sid)
        futures.append(_cache_expiry_submit(_cache_expiry_job, sid, session, action))
    return futures


def _pending_release_closes() -> int:
    """Sessions whose reaper close waits behind its release-time compaction (still in ``_sessions``)."""
    with _cache_expiry_lock:
        return len(_cache_expiry_releasing)


def _compact_then_close(sid: str, session: dict, *, end_reason: str, still_due=None):
    """A reaper's release: when the cache edge is due at release (``claim_edge(releasing=True)`` ends the
    budget now, so nothing is warmed after it), compact or offload on the bounded pool first and close
    afterwards if the session is still reapable then and ``still_due()`` (the session cap: other closes
    may have brought it under) still holds. Returns that job's future, or None when the caller closes
    the session now as before. A ticker job already queued for *sid* is waited for, never pre-empted:
    closing first would make it see ``_closing`` and drop the compaction or offload it claimed."""
    warmer = getattr(session.get("agent"), "_prompt_cache_warmer", None)
    if warmer is None or _session_uses_compute_host(session):
        return None

    def _closable(current: dict) -> bool:
        with _cache_expiry_lock:
            _cache_expiry_releasing.discard(sid)  # under _sessions_lock: no longer counted as pending
        return _session_is_reapable(sid, current) and (still_due is None or still_due())

    def _close() -> None:
        if not _close_session_by_id(sid, end_reason=end_reason, predicate=_closable):
            with _cache_expiry_lock:
                _cache_expiry_releasing.discard(sid)

    with _cache_expiry_lock:
        if sid in _cache_expiry_releasing:  # a close already waits behind this sid's job
            done: concurrent.futures.Future = concurrent.futures.Future()
            done.set_result(None)
            return done
        if sid in _cache_expiry_in_flight:  # the ticker's job is queued or running: close after it
            closed: concurrent.futures.Future = concurrent.futures.Future()

            def _after_queued() -> None:
                try:
                    closed.set_result(_close())
                except BaseException as exc:  # health: allow BLE001 -- handed to the future's owner
                    closed.set_exception(exc)

            _cache_expiry_releasing.add(sid)
            _cache_expiry_after.setdefault(sid, []).append(_after_queued)
            return closed
    with _session_profile_runtime_scope(session):
        action = warmer.claim_edge(releasing=True)
    if action is None:
        return None
    with _cache_expiry_lock:
        _cache_expiry_in_flight.add(sid)
        _cache_expiry_releasing.add(sid)

    def _job() -> None:
        try:
            _cache_expiry_job(sid, session, action)
        finally:
            _close()

    return _cache_expiry_submit(_job)


def _start_cache_expiry_compactor() -> None:
    """Start the ticker once per process (stdio entry and WS sidecar both call this)."""
    from agent.prompt_cache_lifecycle import CHECK_INTERVAL_SECONDS
    global _cache_expiry_started
    with _cache_expiry_lock:
        if _cache_expiry_started:
            return
        _cache_expiry_started = True

    def _loop() -> None:
        while True:
            time.sleep(CHECK_INTERVAL_SECONDS)
            try:
                _cache_expiry_tick()
            except Exception:
                logger.debug("cache-expiry tick failed", exc_info=True)
    from agent.memory_provider import spawn_context_thread
    spawn_context_thread(_loop, name="cache-expiry-compactor").start()


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
