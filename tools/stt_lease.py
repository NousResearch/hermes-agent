"""Local-STT warm-up for the desktop voice-input path.

Desktop voice transcription (``POST /api/audio/transcribe``) blocks until the
configured STT engine answers inside the renderer's timeout floor (180s for a
short clip). With the ``local`` provider that floor also covers the cold cost:
first-use faster-whisper download + model load, or a reload after
``stt.local.unload_after_idle_seconds`` evicted the cache — on CPU-bound hosts
that alone can exceed the floor, so every attempt times out while the engine
itself is healthy (issue #105955).

When the desktop mic opens, warm-up pre-loads the configured local model into
the exact singleton slot transcription reads, so the bill is paid while the
user is still speaking instead of inside the transcription request's timeout.

Unlike :mod:`tools.tts_tool_lifecycle`, nothing here ever unloads the model.
The cached local model is shared with the gateway/CLI/Telegram surfaces in the
same backend process, so one surface's "mic closed" must not evict an engine
another surface is about to use. Eviction stays governed by
``stt.local.unload_after_idle_seconds``, and warming touches the idle timer so
a just-warmed model is never instantly eligible for unload.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("tools.stt_lease")


def warm_stt_provider(
    stt_config: Optional[dict[str, Any]] = None, provider: Optional[str] = None
) -> dict[str, Any]:
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
    result: dict[str, Any] = {"provider": name, "warmed": False, "action": "noop"}
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
        # Register with the idle-lifetime mechanism: a model loaded ONLY by
        # warm-up (mic open → cancel, no transcription) would otherwise sit
        # resident with no watcher, silently ignoring
        # stt.local.unload_after_idle_seconds until some later successful
        # transcription starts one. Also covers the reload-after-prior-eviction
        # case (the previous watcher exited after unloading).
        idle_timeout = transcription_tools._get_idle_unload_seconds(local_cfg)
        if idle_timeout > 0:
            transcription_tools._start_idle_unload_watcher(idle_timeout)
    except Exception as exc:  # engine missing, download failed, bad device…
        logger.warning("[STT] warm-up for local model '%s' failed: %s", model_name, exc)
        result.update(action="error", error=str(exc))
        return result
    elapsed_ms = int((time.monotonic() - started) * 1000)
    result.update(warmed=True, action="cached" if hot_before else "loaded", elapsed_ms=elapsed_ms)
    logger.info("[STT] warm-up local model '%s': %s in %dms", model_name, result["action"], elapsed_ms)
    return result

