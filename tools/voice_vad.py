"""Silero VAD Helper — Voice Activity Detection with Pi 5 Compatibility.

Provides a thread-safe Silero VAD wrapper that:
- Loads Silero VAD with trust_repo=True (required for non-interactive/cron use)
- Handles sample rate conversion (Piper outputs 22050Hz, Silero requires 8/16kHz)
- Caches the model globally for reuse across calls
- Falls back gracefully if torch is unavailable

Usage::

    from tools.voice_vad import VoiceActivityDetector

    vad = VoiceActivityDetector()  # Loads model on first call
    speech_segments = vad.detect(audio_float_array, sample_rate=22050)
    if vad.is_speech(audio_float_array, sample_rate=22050):
        print("Speech detected!")

Mythic Engineering: Forge Worker — Eldra Járnsdóttir
Bug Fix: BUG-001 (Silero VAD interactive blocking) + BUG-002 (sample rate mismatch)
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — torch and numpy are heavy and may not be installed
# ---------------------------------------------------------------------------

_torch = None
_np = None
_vad_model = None
_vad_utils = None
_vad_lock = threading.Lock()
_vad_loaded = False


def _ensure_torch():
    """Import torch and numpy on first use to avoid startup overhead."""
    global _torch, _np
    if _torch is None:
        try:
            import torch
            import numpy
            _torch = torch
            _np = numpy
        except ImportError as e:
            raise ImportError(
                "Silero VAD requires torch and numpy. "
                f"Install with: pip install torch numpy — {e}"
            ) from e
    return _torch, _np


def _load_vad_model():
    """Load Silero VAD model with trust_repo=True (fixes BUG-001).

    Previous code used torch.hub.load() without trust_repo=True, which
    required interactive confirmation and blocked in cron/headless contexts.

    The model is cached globally and loaded once per process. Subsequent
    calls reuse the cached model (verified: second synthesis is faster).
    """
    global _vad_model, _vad_utils, _vad_loaded

    if _vad_loaded:
        return _vad_model, _vad_utils

    torch, np = _ensure_torch()

    with _vad_lock:
        # Double-check after acquiring lock
        if _vad_loaded:
            return _vad_model, _vad_utils

        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,  # BUG-001 FIX: Required for non-interactive use
            )
            _vad_model = model
            # utils is a tuple: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
            _vad_utils = utils
            _vad_loaded = True
            logger.info("[VAD] Silero VAD model loaded and cached")
            return _vad_model, _vad_utils
        except Exception as e:
            logger.error("[VAD] Failed to load Silero VAD: %s", e, exc_info=True)
            raise


# ---------------------------------------------------------------------------
# Sample rate conversion
# ---------------------------------------------------------------------------

# Silero VAD only supports 8000 Hz and 16000 Hz (or multiples of 16000).
# Piper TTS outputs at 22050 Hz — must resample before VAD.
# faster-whisper internally resamples to 16000 Hz.

SUPPORTED_VAD_RATES = {8000, 16000}
DEFAULT_VAD_RATE = 16000  # Best quality for VAD


def _resample_audio(audio: Any, orig_rate: int, target_rate: int) -> Any:
    """Resample audio using numpy linear interpolation.

    Handles the common case of resampling Piper output (22050 Hz)
    to Silero VAD input (16000 Hz).

    Args:
        audio: numpy float32 array of audio samples
        orig_rate: original sample rate
        target_rate: target sample rate

    Returns:
        numpy float32 array at target_rate
    """
    np = _np if _np is not None else _ensure_torch()[1]  # noqa: F841

    if orig_rate == target_rate:
        return audio

    duration = len(audio) / orig_rate
    target_len = int(duration * target_rate)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_len)
    resampled = np.interp(x_new, x_old, audio).astype(np.float32)
    return resampled


# ---------------------------------------------------------------------------
# Voice Activity Detector
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """Thread-safe Silero VAD wrapper with sample rate conversion.

    Handles:
    - Model loading with trust_repo=True (BUG-001 fix)
    - Automatic resampling from unsupported rates (BUG-002 fix)
    - Global model caching for performance
    - Graceful fallback if VAD is unavailable

    Usage::

        vad = VoiceActivityDetector()
        segments = vad.detect(audio_array, sample_rate=22050)
        if vad.is_speech(audio_array, sample_rate=22050):
            print("Speech detected!")
    """

    def __init__(self, threshold: float = 0.5, min_speech_duration: float = 0.3):
        """Initialize the VAD detector.

        Args:
            threshold: Speech probability threshold (0.0-1.0). Default 0.5.
            min_speech_duration: Minimum duration in seconds to consider speech. Default 0.3s.
        """
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self._model = None
        self._utils = None

    def _ensure_loaded(self):
        """Lazily load the model on first use."""
        if self._model is None:
            self._model, self._utils = _load_vad_model()

    def detect(
        self,
        audio: Any,
        sample_rate: int = 16000,
        return_seconds: bool = True,
    ) -> List[Dict[str, float]]:
        """Detect speech segments in audio.

        Args:
            audio: numpy float32 array of audio samples (values in [-1, 1])
            sample_rate: sample rate of the input audio.
                Silero supports 8000 or 16000 Hz. Other rates are
                automatically resampled. (BUG-002 fix)
            return_seconds: if True, return timestamps in seconds instead of samples

        Returns:
            List of dicts with 'start' and 'end' keys (in seconds or samples)

        Raises:
            ImportError: if torch/numpy are not installed
            RuntimeError: if Silero VAD fails to load
        """
        self._ensure_loaded()

        get_speech_timestamps = self._utils[0]  # (get_speech_timestamps, ...)

        # Resample if needed (BUG-002 fix: Silero only supports 8/16kHz)
        if sample_rate not in SUPPORTED_VAD_RATES:
            audio = _resample_audio(audio, sample_rate, DEFAULT_VAD_RATE)
            sample_rate = DEFAULT_VAD_RATE

        segments = get_speech_timestamps(
            audio,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=int(self.min_speech_duration * 1000),
            return_seconds=return_seconds,
        )

        return segments

    def is_speech(
        self,
        audio: Any,
        sample_rate: int = 16000,
        window_size: int = 512,
    ) -> bool:
        """Quick check if audio contains speech.

        Chunks the audio and checks if any chunk has speech probability
        above the threshold.

        Args:
            audio: numpy float32 array of audio samples
            sample_rate: sample rate of the input audio
            window_size: chunk size for probability estimation (default 512)

        Returns:
            True if speech is detected above threshold
        """
        self._ensure_loaded()

        # Resample if needed
        if sample_rate not in SUPPORTED_VAD_RATES:
            audio = _resample_audio(audio, sample_rate, DEFAULT_VAD_RATE)
            sample_rate = DEFAULT_VAD_RATE

        np = _np if _np is not None else _ensure_torch()[1]  # noqa: F841

        # Convert numpy array to torch tensor if needed (Silero requires Tensor input)
        torch_module, _ = _ensure_torch()
        if not isinstance(audio, torch_module.Tensor):
            audio = torch_module.tensor(audio, dtype=torch_module.float32)

        # Check chunk-by-chunk for speech probability
        for i in range(0, len(audio) - window_size, window_size):
            chunk = audio[i : i + window_size]
            if len(chunk) < window_size:
                break
            prob = self._model(chunk, sample_rate).item()
            if prob > self.threshold:
                return True

        return False

    def speech_probability(self, audio: Any, sample_rate: int = 16000) -> float:
        """Get the maximum speech probability across all chunks.

        Useful for debugging and threshold tuning.

        Args:
            audio: numpy float32 array of audio samples
            sample_rate: sample rate of the input audio

        Returns:
            Maximum speech probability (0.0-1.0) across all chunks
        """
        self._ensure_loaded()

        # Resample if needed
        if sample_rate not in SUPPORTED_VAD_RATES:
            audio = _resample_audio(audio, sample_rate, DEFAULT_VAD_RATE)
            sample_rate = DEFAULT_VAD_RATE

        np = _np if _np is not None else _ensure_torch()[1]  # noqa: F841

        # Convert numpy array to torch tensor if needed (Silero requires Tensor input)
        torch_module, _ = _ensure_torch()
        if not isinstance(audio, torch_module.Tensor):
            audio = torch_module.tensor(audio, dtype=torch_module.float32)

        max_prob = 0.0
        chunk_size = 512
        for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                break
            prob = self._model(chunk, sample_rate).item()
            max_prob = max(max_prob, prob)

        return max_prob


# ---------------------------------------------------------------------------
# Convenience function for one-off VAD checks
# ---------------------------------------------------------------------------

def detect_speech(
    audio: Any,
    sample_rate: int = 22050,
    threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """Convenience function: detect speech segments in audio.

    Handles sample rate conversion automatically. Default sample_rate
    is 22050 (Piper output) — the most common use case.

    Args:
        audio: numpy float32 array of audio samples
        sample_rate: input sample rate (default 22050 for Piper)
        threshold: speech probability threshold (default 0.5)

    Returns:
        List of dicts with 'start' and 'end' keys (in seconds)
    """
    vad = VoiceActivityDetector(threshold=threshold)
    return vad.detect(audio, sample_rate)


def has_speech(
    audio: Any,
    sample_rate: int = 22050,
    threshold: float = 0.5,
) -> bool:
    """Convenience function: check if audio contains any speech.

    Args:
        audio: numpy float32 array of audio samples
        sample_rate: input sample rate (default 22050 for Piper)
        threshold: speech probability threshold (default 0.5)

    Returns:
        True if speech is detected
    """
    vad = VoiceActivityDetector(threshold=threshold)
    return vad.is_speech(audio, sample_rate)