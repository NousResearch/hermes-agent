"""Local-STT lifecycle for the desktop voice-input path: warm-up leases.

Desktop voice transcription (``POST /api/audio/transcribe``) blocks until the
configured STT engine answers inside the renderer's timeout floor (180s for a
short clip). With the ``local`` provider that floor also covers the cold cost:
first-use faster-whisper download + model load, or a reload after
``stt.local.unload_after_idle_seconds`` evicted the cache — on CPU-bound hosts
that alone can exceed the floor, so every attempt times out while the engine
itself is healthy (issue #105955).

Every surface that is about to need speech-to-text holds a *lease* here while
it does (the desktop acquires when the mic opens, releases after the
transcript settles). Acquiring pre-loads the configured local model into the
exact singleton slot transcription reads, so the bill is paid while the user
is still speaking instead of inside the transcription request's timeout.

Mirrors :mod:`tools.tts_tool_lifecycle` with one deliberate divergence:
releasing the last lease does NOT unload the model. The cached local model is
shared with the gateway/CLI/Telegram surfaces in the same backend process —
one surface's "mic closed" must not evict an engine another surface is about
to use. Eviction stays governed by ``stt.local.unload_after_idle_seconds``,
and warming touches the idle timer so a just-warmed model is never
instantly eligible for unload.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tools.stt_lease")

_stt_lease_lock = threading.Lock()
_stt_leases: set = set()


def warm_stt_provider(
    stt_config: Optional[Dict[str, Any]] = None, provider: Optional[str] = None
) -> Dict[str, Any]:
    """Pre-load the configured STT engine so the next transcription starts hot.

    Blocking; never raises. Only the ``local`` (faster-whisper) provider has
    anything resident: it fills the same cache slot ``_transcribe_local``
    reads (including first-use download and lazy install). ``local_command``
    shells out to an external binary and cloud providers are remote, so both
    are ``action: "noop"``. The result carries ``provider`` / ``warmed`` /
    ``action`` / ``error``.
    """
    from tools import transcription_tools
    from tools.transcription_local import _normalize_local_model

    if stt_config is None:
        stt_config = transcription_tools._load_stt_config()
    name = (provider or transcription_tools._get_provider(stt_config) or "").lower().strip()
    result: Dict[str, Any] = {"provider": name, "warmed": False, "action": "noop"}
    if name not in {"local", "local_command"}:
        return result
    if name == "local_command":
        result.update(warmed=True)
        return result
    if not transcription_tools._HAS_FASTER_WHISPER and not transcription_tools._try_lazy_install_stt():
        result.update(action="error", error="faster-whisper not installed")
        return result
    local_cfg = stt_config.get("local") or {}
    # Same name transcription itself resolves (config > default, cloud-only
    # names normalized) so warm-up fills the slot the next request reads.
    model_name = _normalize_local_model(local_cfg.get("model"))
    hot_before = (
        transcription_tools._local_model is not None
        and transcription_tools._local_model_name == model_name
    )
    started = time.monotonic()
    try:
        model = transcription_tools._get_or_load_local_model(model_name, local_cfg)
        if model is None:  # defensive: load failed without raising
            result.update(action="error", error="Local whisper model failed to load")
            return result
        # A just-warmed model counts as activity: the idle-unload watcher must
        # not treat the load itself as idle time and evict it immediately.
        transcription_tools._touch_transcription_time()
    except Exception as exc:  # engine missing, download failed, bad device…
        logger.warning("[STT] warm-up for local model '%s' failed: %s", model_name, exc)
        result.update(action="error", error=str(exc))
        return result
    elapsed_ms = int((time.monotonic() - started) * 1000)
    result.update(warmed=True, action="cached" if hot_before else "loaded", elapsed_ms=elapsed_ms)
    logger.info("[STT] warm-up local model '%s': %s in %dms", model_name, result["action"], elapsed_ms)
    return result


def acquire_stt_lease(lease: str, stt_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Register ``lease`` (e.g. ``"desktop:voice-input:abcd1234"``) and warm the engine.

    Re-acquiring is idempotent but still re-warms (cheap on a cache hit; heals
    a cache the idle-unload watcher cleared meanwhile).
    """
    with _stt_lease_lock:
        _stt_leases.add(lease)
        holders = len(_stt_leases)
    return {**warm_stt_provider(stt_config), "leases": holders}


def release_stt_lease(lease: str) -> Dict[str, Any]:
    """Drop ``lease``. A never-acquired lease is a no-op (still reports the
    holder count) so surfaces can call this unconditionally. Never unloads the
    model — see the module docstring for why this diverges from TTS leases."""
    with _stt_lease_lock:
        _stt_leases.discard(lease)
        holders = len(_stt_leases)
    return {"leases": holders}


def stt_lease_holders() -> List[str]:
    """Snapshot of live lease names (diagnostics / tests)."""
    with _stt_lease_lock:
        return sorted(_stt_leases)


def _reset_stt_leases_for_tests() -> None:
    with _stt_lease_lock:
        _stt_leases.clear()
