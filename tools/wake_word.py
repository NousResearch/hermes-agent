"""Wake-word runtime for Hermes CLI voice mode.

This module intentionally stays thin around the existing voice pipeline:
it only listens for a local wake-word model, records one WAV utterance with
pre-roll, and delegates transcription to ``tools.voice_mode``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hermes_constants import get_hermes_home
from tools.voice_mode import (
    CHANNELS,
    DTYPE,
    SAMPLE_RATE,
    SILENCE_DURATION_SECONDS,
    SILENCE_RMS_THRESHOLD,
    AudioRecorder,
    _audio_available,
    _import_audio,
)

logger = logging.getLogger(__name__)

FRAME_SAMPLES = 1280
FRAME_MS = int(FRAME_SAMPLES / SAMPLE_RATE * 1000)
OPENWAKEWORD_RELEASE_TAG = "v0.5.1"
OPENWAKEWORD_DEFAULT_FRAMEWORK = "onnx"
WAKE_MIN_SILENCE_DURATION_SECONDS = 3.0


@dataclass
class WakeTrainingConfig:
    positive_samples: int = 50
    negative_samples: int = 30
    ambient_seconds: int = 60
    command: str = ""


@dataclass
class WakeWordConfig:
    provider: str = "openwakeword"
    phrase: str = "Hermes"
    model_path: str = ""
    threshold: float = 0.5
    patience_frames: int = 2
    vad_threshold: float = 0.25
    pre_roll_ms: int = 1200
    dialog_timeout_seconds: float = 45.0
    max_utterance_seconds: float = 30.0
    silence_threshold: int = SILENCE_RMS_THRESHOLD
    silence_duration: float = SILENCE_DURATION_SECONDS
    frame_samples: int = FRAME_SAMPLES
    training: WakeTrainingConfig = field(default_factory=WakeTrainingConfig)

    @property
    def pre_roll_frames(self) -> int:
        return max(1, math.ceil(self.pre_roll_ms / FRAME_MS))


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _float_value(raw: dict, key: str, default: float) -> float:
    try:
        return float(raw.get(key, default))
    except (TypeError, ValueError):
        return default


def _int_value(raw: dict, key: str, default: int) -> int:
    try:
        return int(raw.get(key, default))
    except (TypeError, ValueError):
        return default


def load_wake_config(config: Optional[dict] = None) -> WakeWordConfig:
    """Load ``voice.wake`` from the Hermes config with safe defaults."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}

    voice_cfg = _as_dict(_as_dict(config).get("voice"))
    wake_cfg = _as_dict(voice_cfg.get("wake"))
    training_cfg = _as_dict(wake_cfg.get("training"))
    voice_silence_duration = _float_value(voice_cfg, "silence_duration", SILENCE_DURATION_SECONDS)
    wake_silence_duration = _float_value(wake_cfg, "silence_duration", voice_silence_duration)
    voice_silence_threshold = max(
        0,
        _int_value(voice_cfg, "silence_threshold", SILENCE_RMS_THRESHOLD),
    )
    wake_silence_threshold = max(
        0,
        _int_value(wake_cfg, "silence_threshold", voice_silence_threshold),
    )
    # Dialog/wake recording is still normal voice capture after the wake event.
    # A stale or overly high wake-specific threshold can make resumed dialog
    # mode look ready while quiet speech never starts/extends recording.
    effective_silence_threshold = min(wake_silence_threshold, voice_silence_threshold)

    return WakeWordConfig(
        provider=str(wake_cfg.get("provider", "openwakeword") or "openwakeword"),
        phrase=str(wake_cfg.get("phrase", "Hermes") or "Hermes"),
        model_path=str(wake_cfg.get("model_path", "") or ""),
        threshold=_float_value(wake_cfg, "threshold", 0.5),
        patience_frames=max(1, _int_value(wake_cfg, "patience_frames", 2)),
        vad_threshold=_float_value(wake_cfg, "vad_threshold", 0.25),
        pre_roll_ms=max(0, _int_value(wake_cfg, "pre_roll_ms", 1200)),
        dialog_timeout_seconds=max(1.0, _float_value(wake_cfg, "dialog_timeout_seconds", 45.0)),
        max_utterance_seconds=max(1.0, _float_value(wake_cfg, "max_utterance_seconds", 30.0)),
        silence_threshold=effective_silence_threshold,
        silence_duration=max(WAKE_MIN_SILENCE_DURATION_SECONDS, wake_silence_duration),
        frame_samples=max(160, _int_value(wake_cfg, "frame_samples", FRAME_SAMPLES)),
        training=WakeTrainingConfig(
            positive_samples=max(1, _int_value(training_cfg, "positive_samples", 50)),
            negative_samples=max(0, _int_value(training_cfg, "negative_samples", 30)),
            ambient_seconds=max(0, _int_value(training_cfg, "ambient_seconds", 60)),
            command=str(training_cfg.get("command", "") or "").strip(),
        ),
    )


def _openwakeword_model_available() -> bool:
    try:
        from openwakeword.model import Model  # noqa: F401

        return True
    except Exception:
        return False


def _canonical_openwakeword_name(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    stem = Path(value).stem
    stem = re.sub(r"_v\d+(?:\.\d+)*$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def _preferred_openwakeword_framework(config: WakeWordConfig) -> str:
    suffix = Path(config.model_path).suffix.lower().lstrip(".")
    if suffix in {"onnx", "tflite"}:
        return suffix
    return OPENWAKEWORD_DEFAULT_FRAMEWORK


def _openwakeword_model_specs() -> dict[str, dict]:
    try:
        import openwakeword

        specs = getattr(openwakeword, "MODELS", {})
        return specs if isinstance(specs, dict) else {}
    except Exception:
        return {}


def _openwakeword_feature_specs() -> dict[str, dict]:
    try:
        import openwakeword

        specs = getattr(openwakeword, "FEATURE_MODELS", {})
        return specs if isinstance(specs, dict) else {}
    except Exception:
        return {}


def _openwakeword_pretrained_model_name(config: WakeWordConfig) -> str:
    specs = _openwakeword_model_specs()
    if not specs:
        return ""

    candidates = [
        _canonical_openwakeword_name(config.model_path),
        _canonical_openwakeword_name(config.phrase),
    ]
    for candidate in candidates:
        if candidate in specs:
            return candidate
    return ""


def _existing_file(path: str | Path) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path).expanduser()
    try:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    except OSError:
        return None
    return None


def _openwakeword_cache_dir(framework: str = OPENWAKEWORD_DEFAULT_FRAMEWORK) -> Path:
    return get_hermes_home() / "wake_words" / "openwakeword" / OPENWAKEWORD_RELEASE_TAG / framework


def _framework_url(url: str, framework: str) -> str:
    if framework == "onnx":
        return re.sub(r"\.tflite$", ".onnx", url)
    if framework == "tflite":
        return re.sub(r"\.onnx$", ".tflite", url)
    return url


def _framework_filename(path_or_url: str, framework: str) -> str:
    path = Path(str(path_or_url))
    suffix = f".{framework}"
    if path.suffix.lower() in {".onnx", ".tflite"}:
        return path.with_suffix(suffix).name
    return path.name


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(target_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, tmp)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty wake-word asset from {url}")
    tmp_path.replace(target_path)


def _ensure_openwakeword_asset(url: str, filename: str, framework: str, *, download_missing: bool) -> Optional[Path]:
    target_path = _openwakeword_cache_dir(framework) / filename
    existing = _existing_file(target_path)
    if existing is not None:
        return existing
    if not download_missing:
        return None
    _download_file(url, target_path)
    return _existing_file(target_path)


def resolve_openwakeword_feature_paths(
    framework: str = OPENWAKEWORD_DEFAULT_FRAMEWORK,
    *,
    model_dir: Optional[Path] = None,
    download_missing: bool = False,
) -> dict[str, Path]:
    """Resolve the openWakeWord audio feature models needed by ONNX/TFLite."""
    names = {
        "melspectrogram": f"melspectrogram.{framework}",
        "embedding": f"embedding_model.{framework}",
    }
    resolved: dict[str, Path] = {}

    if model_dir is not None:
        for key, filename in names.items():
            sibling = _existing_file(model_dir / filename)
            if sibling is not None:
                resolved[key] = sibling

    missing = [key for key in names if key not in resolved]
    if missing:
        specs = _openwakeword_feature_specs()
        for key in missing:
            spec = specs.get(key)
            if not isinstance(spec, dict):
                continue
            url = spec.get("download_url", "")
            if not url:
                continue
            path = _ensure_openwakeword_asset(
                _framework_url(str(url), framework),
                names[key],
                framework,
                download_missing=download_missing,
            )
            if path is not None:
                resolved[key] = path

    return resolved


def resolve_wake_model_path(
    config: WakeWordConfig,
    *,
    download_missing: bool = False,
) -> Optional[Path]:
    """Resolve a configured or known openWakeWord model to a durable file path."""
    existing = _existing_file(config.model_path)
    if existing is not None:
        return existing

    model_name = _openwakeword_pretrained_model_name(config)
    if not model_name:
        return None

    framework = _preferred_openwakeword_framework(config)
    specs = _openwakeword_model_specs()
    spec = specs.get(model_name, {})
    if not isinstance(spec, dict):
        return None

    model_path = str(spec.get("model_path", ""))
    download_url = str(spec.get("download_url", ""))
    if not model_path or not download_url:
        return None

    filename = _framework_filename(model_path, framework)
    resolved = _ensure_openwakeword_asset(
        _framework_url(download_url, framework),
        filename,
        framework,
        download_missing=download_missing,
    )
    if resolved is None:
        return None

    if framework in {"onnx", "tflite"}:
        features = resolve_openwakeword_feature_paths(framework, download_missing=download_missing)
        if not {"melspectrogram", "embedding"}.issubset(features):
            return None

    return resolved


def check_wake_requirements(
    config: Optional[WakeWordConfig] = None,
    *,
    download_missing: bool = False,
) -> Dict[str, Any]:
    """Check whether wake-word listening can start."""
    cfg = config or load_wake_config()
    missing: list[str] = []
    details: list[str] = []

    audio_ok = _audio_available()
    if audio_ok:
        details.append("Audio capture: OK")
    else:
        details.append("Audio capture: MISSING (pip install sounddevice numpy)")
        missing.extend(["sounddevice", "numpy"])

    provider_ok = cfg.provider == "openwakeword"
    if provider_ok:
        details.append("Wake provider: OK (openWakeWord)")
    else:
        details.append(f"Wake provider: MISSING (unsupported provider: {cfg.provider})")

    runtime_ok = _openwakeword_model_available()
    if runtime_ok:
        details.append("Wake runtime: OK")
    else:
        details.append("Wake runtime: MISSING (pip install openwakeword onnxruntime)")
        missing.append("openwakeword")

    resolved_model_path = None
    resolved_features: dict[str, Path] = {}
    model_error = ""
    if runtime_ok:
        try:
            resolved_model_path = resolve_wake_model_path(cfg, download_missing=download_missing)
            if resolved_model_path is not None and resolved_model_path.suffix == ".onnx":
                resolved_features = resolve_openwakeword_feature_paths(
                    "onnx",
                    model_dir=resolved_model_path.parent,
                    download_missing=download_missing,
                )
        except Exception as exc:
            model_error = str(exc)

    feature_assets_ok = True
    if resolved_model_path is not None and resolved_model_path.suffix == ".onnx":
        feature_assets_ok = {"melspectrogram", "embedding"}.issubset(resolved_features)

    model_ok = resolved_model_path is not None and feature_assets_ok
    if model_ok:
        configured = Path(cfg.model_path).expanduser() if cfg.model_path else None
        if configured and configured != resolved_model_path:
            details.append(f"Wake model: OK ({resolved_model_path}; resolved from {configured})")
        else:
            details.append(f"Wake model: OK ({resolved_model_path})")
    else:
        pretrained_name = _openwakeword_pretrained_model_name(cfg) if runtime_ok else ""
        if model_error:
            details.append(f"Wake model: MISSING (auto-download failed: {model_error})")
        elif resolved_model_path is not None and not feature_assets_ok:
            details.append(
                "Wake model: MISSING (openWakeWord ONNX feature assets are not cached; "
                "run /voice wake on with network access or set voice.wake.model_path to a model directory with "
                "melspectrogram.onnx and embedding_model.onnx)"
            )
        elif pretrained_name:
            details.append(
                f"Wake model: MISSING (openWakeWord model '{pretrained_name}' is not cached; "
                "run /voice wake on with network access, /voice wake train, or set voice.wake.model_path)"
            )
        else:
            details.append("Wake model: MISSING (run /voice wake train --phrase \"Hermes\" or set voice.wake.model_path)")

    return {
        "available": bool(audio_ok and provider_ok and runtime_ok and model_ok),
        "audio_available": audio_ok,
        "runtime_available": runtime_ok,
        "model_available": model_ok,
        "resolved_model_path": str(resolved_model_path) if resolved_model_path else "",
        "resolved_feature_paths": {key: str(path) for key, path in resolved_features.items()},
        "missing_packages": missing,
        "details": "\n".join(details),
    }


def _default_model_factory(config: WakeWordConfig):
    import openwakeword
    from openwakeword.model import Model

    kwargs: dict[str, Any] = {"vad_threshold": config.vad_threshold}
    model_path = resolve_wake_model_path(config, download_missing=True)
    if model_path is not None:
        kwargs["wakeword_models"] = [str(model_path)]
        if model_path.suffix == ".onnx":
            kwargs["inference_framework"] = "onnx"
            features = resolve_openwakeword_feature_paths(
                "onnx",
                model_dir=model_path.parent,
                download_missing=True,
            )
            if features.get("melspectrogram"):
                kwargs["melspec_model_path"] = str(features["melspectrogram"])
            if features.get("embedding"):
                kwargs["embedding_model_path"] = str(features["embedding"])

    if kwargs["vad_threshold"] > 0:
        vad_path = (
            Path(openwakeword.__file__).resolve().parent
            / "resources"
            / "models"
            / "silero_vad.onnx"
        )
        if not vad_path.is_file():
            kwargs["vad_threshold"] = 0.0
    return Model(**kwargs)


def _copy_frame(frame):
    return frame.copy() if hasattr(frame, "copy") else frame


def _flatten_frame(frame):
    if hasattr(frame, "reshape"):
        return frame.reshape(-1)
    return frame


def _rms(frame) -> int:
    try:
        _, np = _import_audio()
        arr = frame.astype(np.float64) if hasattr(frame, "astype") else np.asarray(frame, dtype=np.float64)
        if arr.size == 0:
            return 0
        return int(np.sqrt(np.mean(arr ** 2)))
    except Exception:
        try:
            values = [float(v) for row in frame for v in (row if isinstance(row, (list, tuple)) else [row])]
            if not values:
                return 0
            return int(math.sqrt(sum(v * v for v in values) / len(values)))
        except Exception:
            return 0


class OpenWakeWordDetector:
    """Small adapter around ``openwakeword.Model.predict``."""

    def __init__(
        self,
        config: WakeWordConfig,
        *,
        model_factory: Optional[Callable[[WakeWordConfig], Any]] = None,
    ) -> None:
        self.config = config
        self._model_factory = model_factory or _default_model_factory
        self._model = None
        self._consecutive = 0
        self.last_score = 0.0
        self.last_label = ""

    @property
    def model(self):
        if self._model is None:
            self._model = self._model_factory(self.config)
        return self._model

    def reset(self) -> None:
        self._consecutive = 0
        if self._model is not None and hasattr(self._model, "reset"):
            try:
                self._model.reset()
            except Exception:
                logger.debug("openWakeWord reset failed", exc_info=True)

    def process_frame(self, frame) -> bool:
        predictions = self._predict(frame)
        score, label = self._extract_score(predictions)
        self.last_score = score
        self.last_label = label

        if score >= self.config.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        if self._consecutive >= self.config.patience_frames:
            self._consecutive = 0
            return True
        return False

    def _predict(self, frame) -> dict:
        return self.model.predict(_flatten_frame(frame))

    @staticmethod
    def _extract_score(predictions: Any) -> tuple[float, str]:
        if not isinstance(predictions, dict):
            try:
                return float(predictions), ""
            except (TypeError, ValueError):
                return 0.0, ""

        best_score = 0.0
        best_label = ""
        for label, value in predictions.items():
            if isinstance(value, dict):
                values = value.values()
            else:
                values = [value]
            for candidate in values:
                try:
                    score = float(candidate)
                except (TypeError, ValueError):
                    continue
                if score >= best_score:
                    best_score = score
                    best_label = str(label)
        return best_score, best_label


class WakeWordListener:
    """Continuous wake-word listener with passive and dialog states."""

    def __init__(
        self,
        config: WakeWordConfig,
        *,
        on_transcript: Callable[[str], None],
        on_status: Optional[Callable[[dict], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        transcribe_func: Optional[Callable[[str], dict]] = None,
        model_factory: Optional[Callable[[WakeWordConfig], Any]] = None,
        time_fn: Callable[[], float] = time.monotonic,
        async_transcribe: bool = True,
    ) -> None:
        self.config = config
        self.on_transcript = on_transcript
        self.on_status = on_status
        self.on_error = on_error
        self.transcribe_func = transcribe_func
        self.time_fn = time_fn
        self.async_transcribe = async_transcribe
        self.detector = OpenWakeWordDetector(config, model_factory=model_factory)
        self.state = "passive"
        self._lock = threading.RLock()
        self._stream = None
        self._running = False
        self._paused = False
        self._resume_state = "passive"
        self._pre_roll = deque(maxlen=config.pre_roll_frames)
        self._recording_frames: list[Any] = []
        self._recording_started_at = 0.0
        self._last_speech_at = 0.0
        self._has_speech = False
        self._dialog_deadline = 0.0
        self._last_wav_path: Optional[str] = None
        self._recording_finish_reason = ""

    @property
    def recording_frames(self) -> list[Any]:
        with self._lock:
            return list(self._recording_frames)

    @property
    def last_score(self) -> float:
        return self.detector.last_score

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "phrase": self.config.phrase,
                "model_path": self.config.model_path,
                "threshold": self.config.threshold,
                "last_score": self.detector.last_score,
                "paused": self._paused,
            }

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._paused = False
            self._set_state_locked("passive")

        sd, _np = _import_audio()

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("wake-word sounddevice status: %s", status)
            try:
                self.process_frame(indata)
            except Exception as exc:
                self._handle_error(exc)

        stream = None
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=self.config.frame_samples,
                callback=_callback,
            )
            stream.start()
        except Exception:
            with self._lock:
                self._running = False
                self._set_state_locked("stopped")
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            raise

        with self._lock:
            self._stream = stream

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._paused = False
            stream = self._stream
            self._stream = None
            self._recording_frames = []
            self._pre_roll.clear()
            self._set_state_locked("stopped")

        if stream is None:
            return

        def _close_stream() -> None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.debug("wake-word stream close failed", exc_info=True)

        closer = threading.Thread(target=_close_stream, daemon=True)
        closer.start()
        closer.join(timeout=3.0)
        if closer.is_alive():
            logger.warning("Wake-word audio stream close timed out")

    def pause(self, reason: str = "") -> None:
        del reason
        with self._lock:
            if self.state != "paused":
                self._resume_state = self.state if self.state != "stopped" else "passive"
            self._paused = True
            self._set_state_locked("paused")

    def resume(self) -> None:
        with self._lock:
            if not self._paused:
                return
            self._paused = False
            target = self._resume_state or "passive"
            if target == "dialog" and self._dialog_deadline and self.time_fn() >= self._dialog_deadline:
                target = "passive"
            self._set_state_locked(target)

    def enter_dialog(self) -> None:
        with self._lock:
            self._dialog_deadline = self.time_fn() + self.config.dialog_timeout_seconds
            self._set_state_locked("dialog")

    def process_frame(self, frame) -> None:
        frame_copy = _copy_frame(frame)
        with self._lock:
            if self.state == "stopped":
                return
            if self._paused:
                self._set_state_locked("paused")
                return

            if self.state == "dialog" and self._dialog_deadline and self.time_fn() >= self._dialog_deadline:
                self._set_state_locked("passive")

            if self.state == "recording":
                self._record_frame_locked(frame_copy)
                finish_reason = self._finish_reason_locked()
                if finish_reason:
                    frames = self._finish_recording_locked(finish_reason)
                else:
                    frames = None
            else:
                frames = None
                self._pre_roll.append(frame_copy)
                if self.state == "dialog":
                    if _rms(frame_copy) > self.config.silence_threshold:
                        self._begin_recording_locked()
                elif self.state == "passive" and self.detector.process_frame(frame_copy):
                    self._begin_recording_locked()

        if frames is not None:
            self._transcribe_frames(frames)

    def _begin_recording_locked(self) -> None:
        self._recording_frames = [_copy_frame(frame) for frame in self._pre_roll]
        now = self.time_fn()
        self._recording_started_at = now
        self._last_speech_at = now
        self._has_speech = any(_rms(frame) > self.config.silence_threshold for frame in self._recording_frames)
        if not self._has_speech:
            self._last_speech_at = 0.0
        self._set_state_locked("recording")
        logger.info(
            "Wake recording started (silence_threshold=%d, silence_duration=%.1fs, max_utterance=%.1fs)",
            self.config.silence_threshold,
            self.config.silence_duration,
            self.config.max_utterance_seconds,
        )

    def _record_frame_locked(self, frame) -> None:
        self._recording_frames.append(frame)
        if _rms(frame) > self.config.silence_threshold:
            self._has_speech = True
            self._last_speech_at = self.time_fn()

    def _finish_reason_locked(self) -> str:
        now = self.time_fn()
        if now - self._recording_started_at >= self.config.max_utterance_seconds:
            return "max_utterance"
        if self._has_speech and self._last_speech_at and now - self._last_speech_at >= self.config.silence_duration:
            return "silence"
        return ""

    def _finish_recording_locked(self, reason: str) -> list[Any]:
        frames = list(self._recording_frames)
        self._recording_frames = []
        self._pre_roll.clear()
        elapsed = max(0.0, self.time_fn() - self._recording_started_at)
        self._recording_finish_reason = reason
        logger.info(
            "Wake recording finished (reason=%s, elapsed=%.1fs, frames=%d)",
            reason,
            elapsed,
            len(frames),
        )
        self._set_state_locked("transcribing")
        return frames

    def _transcribe_frames(self, frames: list[Any]) -> None:
        if self.async_transcribe:
            threading.Thread(target=self._transcribe_frames_sync, args=(frames,), daemon=True).start()
        else:
            self._transcribe_frames_sync(frames)

    def _transcribe_frames_sync(self, frames: list[Any]) -> None:
        wav_path = None
        try:
            if not frames:
                self._return_to_passive()
                return
            _, np = _import_audio()
            audio = np.concatenate(frames, axis=0)
            wav_path = AudioRecorder._write_wav(audio)
            self._last_wav_path = wav_path
            transcribe = self.transcribe_func
            if transcribe is None:
                from tools.voice_mode import transcribe_recording

                transcribe = transcribe_recording
            result = transcribe(wav_path)
            if result.get("success") and result.get("transcript", "").strip():
                transcript = strip_wake_phrase(result["transcript"].strip(), self.config.phrase)
                if transcript:
                    self.enter_dialog()
                    self.on_transcript(transcript)
                else:
                    self._return_to_passive()
            else:
                self._return_to_passive()
        except Exception as exc:
            self._handle_error(exc)
            self._return_to_passive()
        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _return_to_passive(self) -> None:
        with self._lock:
            if self.state != "stopped":
                self._set_state_locked("passive")

    def _set_state_locked(self, state: str) -> None:
        self.state = state
        if self.on_status:
            try:
                self.on_status(self.status())
            except Exception:
                logger.debug("wake-word status callback failed", exc_info=True)

    def _handle_error(self, exc: Exception) -> None:
        if self.on_error:
            try:
                self.on_error(exc)
            except Exception:
                logger.debug("wake-word error callback failed", exc_info=True)
        else:
            logger.warning("Wake-word listener error: %s", exc, exc_info=True)


def strip_wake_phrase(transcript: str, phrase: str) -> str:
    """Remove the configured wake phrase from the beginning of a transcript."""
    text = transcript.strip()
    phrase = phrase.strip()
    if not text or not phrase:
        return text
    pattern = rf"^\s*{re.escape(phrase)}[\s,;:!\-—]*"
    cleaned = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
    return cleaned or text


def _slugify_phrase(phrase: str) -> str:
    slug = re.sub(r"[^\w]+", "-", phrase.lower(), flags=re.UNICODE).strip("-")
    return slug or "wake-word"


class WakeWordTrainer:
    """Collect wake samples and export a local wake-word model artifact.

    The default trainer supports an external command hook because openWakeWord
    custom training pipelines vary heavily by environment. Tests can inject a
    pure-Python backend, and local setups can set ``voice.wake.training.command``.
    """

    def __init__(
        self,
        config: WakeWordConfig,
        *,
        output_root: Optional[Path] = None,
        sample_collector: Optional[Callable[[WakeWordConfig, Path], dict]] = None,
        training_backend: Optional[Callable[[WakeWordConfig, Path, Path], None]] = None,
    ) -> None:
        self.config = config
        self.output_root = Path(output_root or (get_hermes_home() / "wake_words")).expanduser()
        self.sample_collector = sample_collector or self._collect_samples_interactively
        self.training_backend = training_backend or self._run_training_command_backend

    def train(self) -> dict:
        phrase_slug = _slugify_phrase(self.config.phrase)
        model_dir = self.output_root / phrase_slug
        dataset_dir = model_dir / "samples"
        model_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        samples = self.sample_collector(self.config, dataset_dir)
        model_path = model_dir / "model.onnx"
        self.training_backend(self.config, dataset_dir, model_path)
        if not model_path.is_file() or model_path.stat().st_size <= 0:
            raise RuntimeError(f"Wake-word training did not create a model at {model_path}")

        counts = {key: len(value or []) for key, value in samples.items()}
        metadata = {
            "phrase": self.config.phrase,
            "provider": self.config.provider,
            "model_path": str(model_path),
            "threshold": self.config.threshold,
            "patience_frames": self.config.patience_frames,
            "counts": counts,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        metadata_path = model_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "success": True,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "counts": counts,
            "config_update": {
                "voice.wake.provider": self.config.provider,
                "voice.wake.phrase": self.config.phrase,
                "voice.wake.model_path": str(model_path),
                "voice.wake.threshold": self.config.threshold,
                "voice.wake.patience_frames": self.config.patience_frames,
            },
        }

    def _collect_samples_interactively(self, config: WakeWordConfig, dataset_dir: Path) -> dict:
        positive = []
        negative = []
        ambient = []
        positive_dir = dataset_dir / "positive"
        negative_dir = dataset_dir / "negative"
        ambient_dir = dataset_dir / "ambient"
        positive_dir.mkdir(parents=True, exist_ok=True)
        negative_dir.mkdir(parents=True, exist_ok=True)
        ambient_dir.mkdir(parents=True, exist_ok=True)

        print(f"Collecting {config.training.positive_samples} positive samples for: {config.phrase}")
        for index in range(config.training.positive_samples):
            positive.append(self._record_sample(positive_dir, f"positive_{index:03d}.wav", f"Say wake phrase #{index + 1}, then pause. Press Enter when ready."))

        print(f"Collecting {config.training.negative_samples} negative voice samples.")
        for index in range(config.training.negative_samples):
            negative.append(self._record_sample(negative_dir, f"negative_{index:03d}.wav", f"Say any non-wake phrase #{index + 1}, then pause. Press Enter when ready."))

        if config.training.ambient_seconds > 0:
            ambient.append(self._record_fixed_sample(ambient_dir, "ambient_000.wav", config.training.ambient_seconds))

        return {"positive": positive, "negative": negative, "ambient": ambient}

    def _record_sample(self, target_dir: Path, filename: str, prompt: str) -> Path:
        input(prompt)
        from tools.voice_mode import create_audio_recorder

        done = threading.Event()
        recorder = create_audio_recorder()
        try:
            recorder.start(on_silence_stop=done.set)
            done.wait(timeout=15.0)
            wav_path = recorder.stop()
        finally:
            try:
                recorder.shutdown()
            except Exception:
                pass
        if not wav_path:
            raise RuntimeError("No audio captured for wake-word training sample")
        target_path = target_dir / filename
        shutil.move(wav_path, target_path)
        return target_path

    def _record_fixed_sample(self, target_dir: Path, filename: str, seconds: int) -> Path:
        print(f"Recording {seconds}s of ambient noise. Stay quiet and press Enter when ready.")
        input()
        from tools.voice_mode import create_audio_recorder

        recorder = create_audio_recorder()
        try:
            recorder.start(on_silence_stop=None)
            time.sleep(seconds)
            wav_path = recorder.stop()
        finally:
            try:
                recorder.shutdown()
            except Exception:
                pass
        if not wav_path:
            raise RuntimeError("No ambient audio captured for wake-word training")
        target_path = target_dir / filename
        shutil.move(wav_path, target_path)
        return target_path

    def _run_training_command_backend(self, config: WakeWordConfig, dataset_dir: Path, output_path: Path) -> None:
        command = config.training.command.strip()
        if not command:
            raise RuntimeError(
                "Wake-word samples were collected, but no training backend is configured. "
                "Install wake training deps and set voice.wake.training.command to a command "
                "that writes HERMES_WAKE_OUTPUT_PATH, or provide an existing ONNX model via "
                "voice.wake.model_path."
            )
        env = os.environ.copy()
        env.update(
            {
                "HERMES_WAKE_PHRASE": config.phrase,
                "HERMES_WAKE_DATASET_DIR": str(dataset_dir),
                "HERMES_WAKE_OUTPUT_PATH": str(output_path),
            }
        )
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(dataset_dir.parent),
            env=env,
            text=True,
            capture_output=True,
            timeout=3600,
            check=False,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Wake-word training command failed: {details}")


def wake_install_hint(training: bool = False) -> str:
    packages = "openwakeword onnxruntime"
    if training:
        packages += " torch scipy tqdm torchmetrics torchinfo"
    return f"{sys.executable} -m pip install {packages}"


def write_existing_model_config(model_path: str, phrase: str, threshold: float = 0.5) -> dict:
    """Return config key updates for a pre-trained openWakeWord model."""
    return {
        "voice.wake.provider": "openwakeword",
        "voice.wake.phrase": phrase,
        "voice.wake.model_path": str(Path(model_path).expanduser()),
        "voice.wake.threshold": threshold,
    }


__all__ = [
    "FRAME_SAMPLES",
    "WAKE_MIN_SILENCE_DURATION_SECONDS",
    "WakeTrainingConfig",
    "WakeWordConfig",
    "OpenWakeWordDetector",
    "WakeWordListener",
    "WakeWordTrainer",
    "check_wake_requirements",
    "load_wake_config",
    "resolve_openwakeword_feature_paths",
    "resolve_wake_model_path",
    "strip_wake_phrase",
    "wake_install_hint",
    "write_existing_model_config",
]
