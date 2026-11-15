"""Cache-expiry compaction owner for TUI / Desktop sessions: a daemon ticker that compacts an idle
session while its prompt cache is still warm (decision: ``agent.cache_expiry_compaction``). It runs
the manual-compress choke point (busy claim, snapshot fence, lease) with an automatic trigger, so a
prompt arriving meanwhile queues behind it exactly as it does behind ``/compress``. Bodies are
rebound onto server.py's globals (method_ctx.bind_module) and reference them bare."""

from __future__ import annotations

import threading
import time

from .method_ctx import bind_module

_cache_expiry_started = False
_cache_expiry_lock = threading.Lock()


def _cache_expiry_candidates() -> list[tuple[str, dict]]:
    from agent.cache_expiry_compaction import cache_expiry_compaction_due
    with _sessions_lock:
        live = list(_sessions.items())
    picked = []
    for sid, session in live:
        agent = session.get("agent")
        if agent is None or session.get("running") or session.get("_closing"):
            continue
        if _session_uses_compute_host(session):
            continue  # the compute host owns that transcript; it is not compacted from here
        with _session_profile_runtime_scope(session):
            due, _reason = cache_expiry_compaction_due(agent)
        if due:
            picked.append((sid, session))
    return picked


def _compact_before_cache_expiry(sid: str, session: dict) -> None:
    from agent.cache_expiry_compaction import TRIGGER
    from agent.conversation_compression import finalize_context_engine_compression_notification
    agent = session["agent"]
    with _session_profile_runtime_scope(session), _manual_compress_turn(sid, session):
        try:
            removed, _usage = _compress_session_history(session, trigger=TRIGGER)
        except CompressionLockHeld:
            return
        except Exception:
            finalize_context_engine_compression_notification(agent, committed=False)
            raise
        _sync_session_key_after_compress(sid, session, clear_pending_title=False)
        finalize_context_engine_compression_notification(agent, committed=True)
    logger.info("Cache-expiry compaction: sid=%s removed=%d messages", sid, removed)


def _cache_expiry_tick() -> None:
    for sid, session in _cache_expiry_candidates():
        try:
            _compact_before_cache_expiry(sid, session)
        except (CompressionBusy, SessionNotOwned):
            continue  # a turn or another owner got there first: nothing to do this window
        except Exception:
            logger.warning("cache-expiry compaction failed sid=%s", sid, exc_info=True)


def _start_cache_expiry_compactor() -> None:
    """Start the ticker once per process (stdio entry and WS sidecar both call this)."""
    from agent.cache_expiry_compaction import CHECK_INTERVAL_SECONDS
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
    bind_module(globals(), server)
