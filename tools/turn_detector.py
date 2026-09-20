"""Smart Turn v3 — audio-based semantic end-of-turn detector (standalone, CPU).

Given the audio of the current utterance, returns P(the user has finished their turn) so the
converse endpoint can be ADAPTIVE: commit fast when the utterance sounds complete, keep
listening when it trails off — instead of a fixed silence timer that either cuts the user off
mid-thought or over-waits when they're clearly done.

Model: ``pipecat-ai/smart-turn-v3`` (BSD-2), ~8.7 MB int8 ONNX — Whisper-tiny encoder + a small
endpointing head. Runs on CPU via onnxruntime; the only extra dependency is
``transformers.WhisperFeatureExtractor`` for the log-mel front end (numpy path, no torch).

It is loaded LAZILY and DEFENSIVELY: if onnxruntime / transformers / the weights aren't present,
:func:`load_turn_detector` returns ``None`` and the caller falls back to the fixed silence timer.
So the whole feature is opt-in by mere availability — nothing here is on the import path of the
gateway unless the converse loop asks for it.

Note (validation): Smart Turn keys on human prosody (trailing off, hesitation). Synthetic TTS
does not reproduce those cues, so it cannot be validated with TTS — it must be tuned against
real speech. The threshold is therefore configurable.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

_log = logging.getLogger("hermes_cli.web_server")

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8                       # the model's fixed input window (last 8 s of the turn)
_MODEL_REPO = "pipecat-ai/smart-turn-v3"
_MODEL_FILE = "smart-turn-v3.2-cpu.onnx"  # int8 QAT, ~8.7 MB, the CPU build
DEFAULT_THRESHOLD = 0.5                   # P(complete) >= threshold -> the turn is over


class SmartTurnDetector:
    """Wraps the ONNX session + Whisper mel front end. One instance, reused across turns."""

    def __init__(self, session: Any, feature_extractor: Any,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        self._session = session
        self._feat = feature_extractor
        self._input_name = session.get_inputs()[0].name
        self.threshold = float(threshold)

    def turn_complete_probability(self, audio_f32: Any) -> float:
        """P(turn complete) in [0, 1] for a 16 kHz mono float32 numpy array.

        The array is trimmed/zero-padded to the model's 8 s window (audio kept at the END), then
        run through Whisper's log-mel front end and the ONNX model (sigmoid output)."""
        import numpy as np

        n = WINDOW_SECONDS * SAMPLE_RATE
        audio = np.asarray(audio_f32, dtype=np.float32)
        if audio.size > n:
            audio = audio[-n:]
        elif audio.size < n:
            audio = np.pad(audio, (n - audio.size, 0))
        inp = self._feat(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np",
            padding="max_length", max_length=n, truncation=True, do_normalize=True)
        feats = np.expand_dims(inp.input_features.squeeze(0).astype(np.float32), axis=0)
        out = self._session.run(None, {self._input_name: feats})
        return float(np.asarray(out[0]).flatten()[0])

    def is_turn_complete(self, audio_f32: Any) -> bool:
        return self.turn_complete_probability(audio_f32) >= self.threshold


_lock = threading.Lock()
_cached: Optional[SmartTurnDetector] = None
_load_attempted = False


def load_turn_detector(threshold: float = DEFAULT_THRESHOLD) -> Optional[SmartTurnDetector]:
    """Return a cached :class:`SmartTurnDetector`, or ``None`` if unavailable (caller falls back).

    First call loads the ONNX model + feature extractor (a few hundred ms); subsequent calls
    return the cached instance (only updating its threshold). Any failure — missing deps, no
    weights, offline with an empty cache — is logged once and yields ``None``, never raises.
    Honors ``HF_HUB_OFFLINE=1`` so a box with the weights pre-cached needs no runtime network.
    """
    global _cached, _load_attempted
    with _lock:
        if _load_attempted:
            if _cached is not None:
                _cached.threshold = float(threshold)
            return _cached
        _load_attempted = True
        try:
            # First-enable install (non-interactive; relies on the config gate). If lazy installs
            # are disabled or unavailable, this raises and we fall back to the fixed timer.
            try:
                from tools import lazy_deps

                lazy_deps.ensure("voice.endpoint", prompt=False)
            except Exception:  # noqa: BLE001 - deps may already be present, or install may be off
                pass
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import WhisperFeatureExtractor

            path = hf_hub_download(_MODEL_REPO, _MODEL_FILE)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, os.cpu_count() or 2)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(path, sess_options=opts)
            feat = WhisperFeatureExtractor(chunk_length=WINDOW_SECONDS)
            _cached = SmartTurnDetector(session, feat, threshold)
            _log.info("smart-turn v3 endpointer loaded (%s)", _MODEL_FILE)
        except Exception as exc:  # noqa: BLE001 - unavailable -> fall back to the fixed timer
            _log.warning("smart-turn endpointer unavailable (%s); using fixed silence endpoint", exc)
            _cached = None
        return _cached
