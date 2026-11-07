"""Silero VAD (v6, 16 kHz ONNX) for the converse voice loop.

Speech/non-speech is decided by a 1.3 MB model instead of signal-level thresholds, so a soft
voice is heard and a loud non-speech noise floor (fan, hum, broadband TV hiss) is not, without
calibrating to the room. Runs on onnxruntime, which the ``voice`` extra already brings in via
faster-whisper; no torch.

The weights come from the upstream MIT-licensed repo at a pinned tag, verified by sha256, and are
cached under ``<HERMES_HOME>/cache/voice``. :func:`load_silero_session` returns ``None`` (with the
reason) when onnxruntime or the weights are unavailable, so the caller can fall back.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_MODEL_FILE = "silero_vad_16k_op15.onnx"
_MODEL_URL = ("https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/"
              f"src/silero_vad/data/{_MODEL_FILE}")
_MODEL_SHA256 = "7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49"

SAMPLE_RATE = 16000
# The model's input contract at 16 kHz: 512 new samples per call, prefixed by the last 64 samples
# of the previous call, with the recurrent state threaded through.
WINDOW = 512
_CONTEXT = 64

_lock = threading.Lock()
_session: Any = None
_failure: Optional[str] = None


def _model_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "voice" / _MODEL_FILE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_model() -> Path:
    """Return the verified model path, downloading it once."""
    path = _model_path()
    if path.is_file() and _sha256(path) == _MODEL_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    logger.info("silero vad: downloading %s (one-time, ~1.3 MB)", _MODEL_FILE)
    urllib.request.urlretrieve(_MODEL_URL, tmp)
    digest = _sha256(tmp)
    if digest != _MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"silero vad model checksum mismatch ({digest[:12]}…)")
    os.replace(tmp, path)
    return path


def load_silero_session() -> tuple[Any, Optional[str]]:
    """Return ``(session, None)``, or ``(None, reason)`` when Silero can't run here. Loaded once per
    process; a failure is remembered so every connection doesn't retry the download."""
    global _session, _failure
    with _lock:
        if _session is not None or _failure is not None:
            return _session, _failure
        try:
            import onnxruntime as ort
        except ImportError:
            _failure = "onnxruntime is not installed (it ships with the `voice` extra)"
            return None, _failure
        try:
            opts = ort.SessionOptions()
            # A 32 ms window costs ~0.2 ms on one core; more threads only add overhead, and
            # explicit counts avoid onnxruntime's CPU-affinity pinning, which fails in containers.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            _session = ort.InferenceSession(str(_ensure_model()), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        except Exception as exc:
            _failure = f"model unavailable ({exc})"
            logger.debug("silero vad load failed", exc_info=True)
        return _session, _failure


class SileroStream:
    """Per-connection streaming state: the recurrent state, the 64-sample context, and a carry
    buffer that reframes arbitrary-size blocks into the model's 512-sample windows."""

    def __init__(self, session: Any, np: Any, *, input_rate: int = SAMPLE_RATE) -> None:
        self._session = session
        self._np = np
        self._input_rate = int(input_rate)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.probability = 0.0

    def feed(self, block: Any) -> float:
        """Consume one int16 block; return the speech probability of the latest full window."""
        np = self._np
        audio = block.astype(np.float32) / 32768.0
        if self._input_rate != SAMPLE_RATE and audio.size:
            n_out = round(audio.size * SAMPLE_RATE / self._input_rate)
            audio = np.interp(np.linspace(0.0, 1.0, n_out, endpoint=False),
                              np.linspace(0.0, 1.0, audio.size, endpoint=False),
                              audio).astype(np.float32)
        self._carry = np.concatenate([self._carry, audio])
        while self._carry.size >= WINDOW:
            window, self._carry = self._carry[:WINDOW], self._carry[WINDOW:]
            chunk = np.concatenate([self._context, window])[None, :]
            prob, self._state = self._session.run(
                None, {"input": chunk, "state": self._state, "sr": self._sr})
            self._context = chunk[0, -_CONTEXT:]
            self.probability = float(prob[0, 0])
        return self.probability
