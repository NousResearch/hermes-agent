"""Local live-transcription session: partial transcripts while the user is still speaking.

``stt.streaming`` already gives live text to providers with a realtime wire (openai / xai /
elevenlabs). The local faster-whisper backend has no such wire — ``transcribe()`` reads a fixed
file to the end — so local profiles keep falling back to the file path. This closes that gap with
chunked re-decoding: while the mic is open, a worker snapshots the recent audio tail
(``AudioRecorder.tail_wav_path``) and transcribes just that capped window, feeding each newer
partial into the recorder's existing ``on_live_partial`` seam.

It satisfies the SAME live-session contract ``tools.voice_mode`` already drives (``set_input_rate``
/ ``end_audio`` / ``cancel`` / ``finalize`` / ``provider``), so the recorder's stop and cancel paths
retire it with no new plumbing in the voice drivers or the UIs.

Guarantees:
  * **Preview only.** ``preview_only`` keeps it out of the parked-session table, so the take is
    always transcribed in full and the authoritative transcript never comes from a capped tail.
  * **Serialized decodes.** Every re-decode holds ``_stt_inference_lock``, so a partial and the
    turn's final pass never drive the CT2 model concurrently.
  * **One worker, drop-don't-queue.** A tick arriving while a decode is in flight is skipped, so
    partials can never pile up behind the mic.
  * **No surprise model.** Opens only when the profile's STT actually resolves to local, so a
    provider with its own live wire keeps using it (and no whisper model is loaded behind the
    user's back).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["LocalPartialSession", "open_local_partial_session", "get_partial_state"]

PartialCallback = Callable[[str], None]

# Tuning for the local re-decode loop (voice.partial.*). The on/off switch is stt.streaming:
# this is the local answer to that switch, not a second switch fighting it.
_DEFAULTS = {"tail_seconds": 20.0, "interval_seconds": 2.0, "min_seconds": 1.5}
# Floors: a re-decode is a full whisper pass, so a sub-second cadence or a sub-second window just
# burns CPU on audio too short to transcribe meaningfully. Configured values are clamped, not
# trusted.
_FLOORS = {"tail_seconds": 3.0, "interval_seconds": 0.6, "min_seconds": 1.0}

_state_lock = threading.Lock()
_state: Dict[str, Any] = {"enabled": False, "partial": "", "count": 0, "last_error": ""}


def get_partial_state() -> Dict[str, Any]:
    """Snapshot of the live partial (for surfaces and tests that observe a running transcript)."""
    with _state_lock:
        return dict(_state)


def _partial_cfg() -> Dict[str, Any]:
    """Read ``voice.partial.*`` tuning overrides; fall back to defaults."""
    try:
        from hermes_cli.config import load_config  # lazy: avoid import cycle / startup cost
        cfg = load_config() or {}
    except Exception:  # health: allow BLE001 -- a config read must never gate arming; defaults stand
        return dict(_DEFAULTS)
    vp = (cfg.get("voice") or {}).get("partial") or {}

    def _num(key: str, default: float) -> float:
        raw = vp.get(key)
        try:
            return float(default) if raw is None else float(raw)
        except (TypeError, ValueError):
            return float(default)

    return {key: max(_FLOORS[key], _num(key, _DEFAULTS[key])) for key in _DEFAULTS}


def _stt_is_local(stt_config: Dict[str, Any]) -> bool:
    """Partials exist because the LOCAL backend has no live wire. A provider that streams partials
    itself keeps its own path — arming here would load a whisper model the user never selected —
    so this is local (and the local command wrapper) only."""
    from tools.transcription_tools import _get_provider
    return _get_provider(stt_config) in ("local", "local_command")


def _decode_tail(wav_path: str) -> Optional[str]:
    """Transcribe one tail snapshot with the SAME model + kwargs the final pass uses.

    Serialized with the final transcription via the shared inference lock; the lazy segment
    iterator is consumed inside the lock (the decode happens on first ``next()``).
    """
    from tools import transcription_tools as tt
    from tools.transcription_local import (
        _join_confident_segments,
        _normalize_local_stt_language,
        build_local_transcribe_kwargs,
    )
    from tools.voice_mode_transcript import is_whisper_hallucination

    stt_config = tt._load_stt_config()
    local_cfg = stt_config.get("local") or {}
    model_name = local_cfg.get("model", "base")
    with tt._stt_inference_lock:
        if tt._model_unload_in_progress:
            # The idle watcher is retiring the model — skip this tick instead of reloading it.
            return None
        model = tt._get_or_load_local_model(model_name, local_cfg)
        if model is None:
            return None
        kwargs = build_local_transcribe_kwargs(stt_config)
        normalized = _normalize_local_stt_language(
            tt._resolve_stt_language("local", stt_config), getattr(model, "supported_languages", None))
        if normalized:
            kwargs["language"] = normalized
        else:
            kwargs.pop("language", None)
        segments, _info = model.transcribe(wav_path, **kwargs)
        segments = list(segments)
    transcript = (_join_confident_segments(segments, local_cfg) or "").strip()
    if not transcript or is_whisper_hallucination(transcript):
        return None
    return transcript


class LocalPartialSession:
    """Live-session duck type for the local backend: pulls the capture tail instead of receiving
    pushed audio, and never supplies the take's final transcript."""

    provider = "local"
    preview_only = True

    def __init__(self, recorder: Any, on_partial: PartialCallback, config: Dict[str, Any]) -> None:
        self._recorder = recorder
        self._on_partial = on_partial
        self._config = config
        self._sample_rate = 16000
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # ── live-session contract (tools.voice_mode drives these) ────────────────
    def set_input_rate(self, rate: int) -> None:
        self._sample_rate = int(rate)

    def start(self) -> "LocalPartialSession":
        self._stop.clear()
        with _state_lock:
            _state.update(enabled=True, partial="", count=0, last_error="")
        # Profile scope: the worker re-reads STT config each tick, so bind the arming caller's
        # contextvars (the RPC handler runs under the session's profile) — a multiplexed serve
        # process must answer with the owning profile's config, not the launch profile's.
        from agent.memory_provider import spawn_context_thread
        self._worker = spawn_context_thread(self._run, name="voice-stt-partial")
        self._worker.start()
        return self

    def end_audio(self) -> None:
        """Recording stopped: stop re-decoding; the authoritative pass transcribes the file."""
        self._shutdown()

    def cancel(self) -> None:
        self._shutdown()

    def finalize(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Preview-only: no transcript, so a capped tail can never masquerade as the take's text."""
        self._shutdown()
        return {"success": False, "transcript": "", "provider": self.provider,
                "error": "local partials are preview-only"}

    # ── worker ───────────────────────────────────────────────────────────────
    def _shutdown(self) -> None:
        self._stop.set()
        worker, self._worker = self._worker, None
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        with _state_lock:
            _state["enabled"] = False

    def _run(self) -> None:
        cfg = self._config
        while not self._stop.wait(cfg["interval_seconds"]):
            started = time.monotonic()
            try:
                if not getattr(self._recorder, "is_recording", False):
                    continue
                if float(getattr(self._recorder, "elapsed_seconds", 0.0) or 0.0) < cfg["min_seconds"]:
                    continue
                wav_path = self._recorder.tail_wav_path(cfg["tail_seconds"])
                if not wav_path:
                    continue
                try:
                    text = _decode_tail(wav_path)
                finally:
                    if os.path.exists(wav_path):
                        try:
                            os.unlink(wav_path)
                        except OSError as e:
                            logger.debug("voice partial: tail snapshot cleanup failed: %s", e)
            except Exception as e:  # health: allow BLE001 -- a bad tick must not kill the worker
                with _state_lock:
                    _state["last_error"] = str(e)
                logger.debug("voice partial tick failed: %s", e)
                continue
            if not text:
                continue
            with _state_lock:
                if text == _state["partial"]:
                    continue  # nothing new said yet
                _state["partial"] = text
                _state["count"] += 1
                _state["last_error"] = ""
            try:
                self._on_partial(text)
            except Exception as e:  # health: allow BLE001 -- a surface callback failure must not kill the worker
                logger.debug("voice partial on_partial callback failed: %s", e)
            # Pace from the tick start: a slow decode shouldn't add a whole interval on top.
            remaining = cfg["interval_seconds"] - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)


def open_local_partial_session(recorder: Any, on_partial: Optional[PartialCallback],
                               stt_config: Optional[Dict[str, Any]] = None) -> Optional[LocalPartialSession]:
    """Open (and start) a local live-transcription session, or None when it doesn't apply.

    None when there's no callback to render to, when ``stt.streaming`` is off (the same switch the
    live-wire providers use), when the profile's STT resolves to a provider that streams by itself,
    or when the recorder backend can't snapshot a tail (Termux).
    """
    if on_partial is None or recorder is None or not callable(getattr(recorder, "tail_wav_path", None)):
        return None
    from tools import transcription_tools as tt
    from tools.transcription_streaming import streaming_enabled
    cfg = tt._load_stt_config() if stt_config is None else stt_config
    if not streaming_enabled(cfg) or not _stt_is_local(cfg):
        return None
    return LocalPartialSession(recorder, on_partial, _partial_cfg()).start()
