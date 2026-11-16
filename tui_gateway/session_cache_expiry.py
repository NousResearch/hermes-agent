"""Cache-expiry compaction owner for TUI / Desktop sessions: a daemon ticker that compacts an idle
session while its prompt cache is still warm (decision: step 2 of ``agent.prompt_cache_lifecycle``), then primes the compacted chat for one fresh budget of keep-alives. It runs
the manual-compress choke point (busy claim, snapshot fence, lease) with an automatic trigger, so a
prompt arriving meanwhile queues behind it exactly as it does behind ``/compress``. Bodies are
rebound onto server.py's globals (method_ctx.bind_module) and reference them bare."""

from __future__ import annotations

import threading
import time

from .method_ctx import bind_module

_cache_expiry_started = False
_cache_expiry_lock = threading.Lock()
# Summaries running at once: one slow summary must not hold every other due chat past its window
# (the ticker used to run them serially), nor may a burst of due chats fan out unbounded.
_CACHE_EXPIRY_CONCURRENCY = 3
_cache_expiry_slots = threading.BoundedSemaphore(_CACHE_EXPIRY_CONCURRENCY)
_cache_expiry_in_flight: set[str] = set()


def _cache_expiry_candidates() -> list[tuple[str, dict]]:
    from agent.prompt_cache_lifecycle import claim_compaction
    with _sessions_lock:
        live = list(_sessions.items())
    with _cache_expiry_lock:
        busy = set(_cache_expiry_in_flight)
    picked = []
    for sid, session in live:
        agent = session.get("agent")
        if agent is None or sid in busy or session.get("running") or session.get("_closing"):
            continue
        if _session_uses_compute_host(session):
            continue  # the compute host owns that transcript; it is not compacted from here
        with _session_profile_runtime_scope(session):
            due = claim_compaction(agent)
        if due:
            picked.append((sid, session))
    return picked


def _compact_before_cache_expiry(sid: str, session: dict) -> None:
    from agent.prompt_cache_lifecycle import TRIGGER, prime_after_compaction, window_still_warm
    from agent.conversation_compression import finalize_context_engine_compression_notification
    agent = session["agent"]
    with _session_profile_runtime_scope(session):
        if not window_still_warm(agent):  # waited for a slot past the window
            return
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
        with session["history_lock"]:
            compacted = list(session.get("history", []))
        prime_after_compaction(agent, compacted)
    logger.info("Cache-expiry compaction: sid=%s removed=%d messages", sid, removed)


def _cache_expiry_job(sid: str, session: dict) -> None:
    try:
        with _cache_expiry_slots:
            _compact_before_cache_expiry(sid, session)
    except (CompressionBusy, SessionNotOwned):
        pass  # a turn or another owner got there first: nothing to do this window
    except Exception:
        logger.warning("cache-expiry compaction failed sid=%s", sid, exc_info=True)
    finally:
        with _cache_expiry_lock:
            _cache_expiry_in_flight.discard(sid)


def _cache_expiry_tick() -> list:
    """Start one job thread per due session (contextvars carried, so the profile scope survives the
    hop); a sid already in flight is never claimed again. Returns the started threads."""
    from agent.memory_provider import spawn_context_thread
    threads = []
    for sid, session in _cache_expiry_candidates():
        with _cache_expiry_lock:
            _cache_expiry_in_flight.add(sid)
        thread = spawn_context_thread(_cache_expiry_job, name=f"cache-expiry-{sid}", args=(sid, session))
        thread.start()
        threads.append(thread)
    return threads


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
    bind_module(globals(), server)
