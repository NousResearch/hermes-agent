"""Session flush / reaping / orphan sweep / cross-backend heartbeat: exit-flush signal handlers, idle + LRU
eviction, orphaned session-row sweep, backend heartbeat refresher. Bodies are rebound onto server.py's
globals at install time (method_ctx.bind_module), so they reference server.py globals bare — including the
knobs _SESSION_TTL_S, _REAPER_SCAN_S, _EXIT_FLUSH_BUDGET_S and _INCREMENTAL_FLUSH_INTERVAL_S.
"""

from __future__ import annotations

import contextlib
import secrets
import threading

from tui_gateway._env import env_float

from .method_ctx import bind_module


# ── Flush-on-kill + periodic incremental flush ───────────────────────────
# (a) SIGTERM/SIGINT run a bounded flush to state.db BEFORE normal shutdown, chained to the prior handler;
# (b) the idle-reaper scan piggybacks an incremental flush so a SIGKILL loses at most one interval.


def _flush_session_messages(session: dict | None) -> bool:
    """Best-effort durable flush of one session's transcript via ``agent._persist_session`` (same marker-deduped
    contract as ``_finalize_session``: repeated calls never duplicate rows).

    See #13121.
    """
    agent = session.get("agent") if session else None
    snapshot = getattr(agent, "_session_messages", None) if hasattr(agent, "_persist_session") else None
    if not snapshot:
        return False
    try:
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

    def _run() -> None:
        deadline = time.monotonic() + budget
        for session in _reaper_session_snapshot():
            if time.monotonic() >= deadline:
                break
            result["flushed"] += _flush_session_messages(session)

    worker = threading.Thread(target=_run, daemon=True, name="hermes-exit-flush")
    worker.start()
    worker.join(budget)
    return result["flushed"]


_exit_flush_prev_handlers: dict[int, Any] = {}
_exit_flush_handlers_installed = False


def _handle_exit_flush_signal(signum, frame) -> None:
    """Flush in-memory sessions, then hand off to the prior handler (uvicorn's graceful shutdown, a supervisor's
    handler, or the default disposition) — this only *prepends* a bounded flush."""
    with contextlib.suppress(Exception):
        _flush_sessions_before_exit()
    import signal as _signal
    prev = _exit_flush_prev_handlers.get(signum)
    if callable(prev):
        prev(signum, frame)
    elif prev is not _signal.SIG_IGN:
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
    return transport is _detached_ws_transport or getattr(transport, "_closed", None) is True


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


def _session_is_evictable(sid: str, session: dict, now: float) -> bool:
    """TTL eviction: the LRU exemptions plus idle-for-TTL AND older-than-TTL."""
    if not _session_is_lru_evictable(sid, session):
        return False
    last_active = float(session.get("last_active") or 0.0)
    created_at = float(session.get("created_at") or 0.0)
    return (now - last_active) > _SESSION_TTL_S and (now - created_at) > _SESSION_TTL_S


def _reap_idle_sessions() -> None:
    now = time.time()
    try:  # piggyback the incremental flush on the reaper tick — no new timer subsystem
        _flush_dirty_sessions()
    except Exception:
        logger.debug("periodic incremental session flush failed", exc_info=True)
    with _sessions_lock:
        victims = [sid for sid, s in _sessions.items() if _session_is_evictable(sid, s, now)]
    for sid in victims:
        _close_session_by_id(
            sid, end_reason="idle_timeout",
            predicate=lambda session, vs=sid: _session_is_evictable(vs, session, time.time()))
    _enforce_session_cap()
    _reclaim_orphaned_leases()
    # Long-lived processes: gen2 GC rarely runs at steady state and glibc retains freed pages as RSS, so trim
    # every scan to prevent unbounded RSS growth over days/weeks.
    try:
        from hermes_cli.mem_trim import trim_memory
        trim_memory(reason="idle reaper periodic trim")
    except Exception as exc:  # debug, not warning — a persistent failure would repeat every scan.
        logger.debug("idle reaper memory trim failed: %s: %s", type(exc).__name__, exc)


def _reclaim_orphaned_leases() -> None:
    """Hand the registry the lease ids we still own so it can drop the rest."""
    try:
        from hermes_cli.active_sessions import release_orphaned_leases
        with _sessions_lock:
            live = _own_live_lease_ids()
            # Blocker 1b (#99719): a lease DETACHED for deferred release is no
            # longer in _sessions but its child may still be writing, so its
            # registry entry must stay protected. Union the detached-table
            # lease_ids under the SAME _sessions_lock hold so the reaper SKIPS
            # them (never sweeps a detached lease). Held under one lock so the
            # reaper cannot observe a lease mid-detach in neither map.
            live |= {
                getattr(lease, "lease_id", "")
                for lease in _detached_leases.values()
            }
        if dropped := release_orphaned_leases(live):
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
        evictable = [(sid, s) for sid, s in _sessions.items() if _session_is_lru_evictable(sid, s)]
    # Oldest-touched first; evict only down to the cap (may stop short: live sessions are never eligible).
    evictable.sort(key=lambda kv: float(kv[1].get("last_active") or 0.0))
    for sid, _s in evictable:
        with _sessions_lock:
            if len(_sessions) <= cap:
                break
        _close_session_by_id(
            sid, end_reason="lru_evict", predicate=lambda session, vs=sid: _session_is_lru_evictable(vs, session))


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
_ORPHAN_SWEEP_SOURCES = ("tui", "desktop", "subagent")
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
    row any live backend (heartbeat within ``2 * TTL``) could own — see ``SessionDB.sweep_orphaned_sessions``."""
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
    swept = db.sweep_orphaned_sessions(
        max_idle_seconds=_SESSION_TTL_S, sources=_ORPHAN_SWEEP_SOURCES, exclude_ids=tuple(sorted(live_ids)))
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
        db.register_backend_heartbeat(
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
                db.clear_backend_heartbeat(_backend_id_for_this_process())

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


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
