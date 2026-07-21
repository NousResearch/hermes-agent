"""Wake-word ("Hey Hermes") detection — hands-free session trigger.

A lightweight, always-on hotword listener that fires a callback when a wake
phrase is spoken — the "Hey Siri" / "Alexa" pattern. Shared by the CLI, TUI, and
desktop GUI (one of them owns it, gated by ``wake_surface_enabled``): say the
wake word, Hermes opens a fresh session and captures voice via the existing
pipeline, then answers.

Two engines, both fully on-device (no audio leaves the machine for detection):

* **openwakeword** (default, free, no API key) — loads a pretrained or custom
  ONNX model. Ships with ``hey_jarvis``, ``alexa``, ``hey_mycroft``, … ; point
  ``wake_word.openwakeword.model`` at a custom ``.onnx`` to detect a real
  "hey hermes" (training guide in the wake-word docs).
* **porcupine** (premium) — Picovoice's engine. Needs ``PORCUPINE_ACCESS_KEY``;
  supports built-in keywords and custom ``.ppn`` files from the Picovoice
  Console.

Audio capture reuses the same 16 kHz mono int16 ``sounddevice`` path as voice
mode. The detector runs on its own daemon thread; callers ``pause()`` it while a
voice turn holds the microphone and ``resume()`` it once the system is idle
again (two input streams on one device is unreliable cross-platform).

Nothing here mutates agent context or the prompt cache — on wake we hand a plain
string to the caller, exactly like a voice transcript.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# 16 kHz mono int16 — Whisper-native and what both engines expect.
SAMPLE_RATE = 16000

# Minimum gap between two consecutive wake fires, so one "hey hermes" can't
# retrigger across several frames while the caller is still reacting.
_FIRE_COOLDOWN_SECONDS = 2.0
_START_TIMEOUT_SECONDS = 5.0


class WakeWordInUse(RuntimeError):
    """Raised when another surface or process owns the wake-word listener."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "surface": "auto",
    "provider": "openwakeword",
    "phrase": "hey jarvis",
    "sensitivity": 0.5,
    "start_new_session": True,
}


def load_wake_word_config() -> Dict[str, Any]:
    """Return the ``wake_word`` config section, shape-guarded to a dict."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("wake_word")
    except Exception:
        cfg = None
    return cfg if isinstance(cfg, dict) else {}


def _get(cfg: Dict[str, Any], key: str) -> Any:
    val = cfg.get(key, _DEFAULTS.get(key))
    return _DEFAULTS.get(key) if val is None else val


def _provider(cfg: Dict[str, Any]) -> str:
    return str(_get(cfg, "provider")).strip().lower() or "openwakeword"


def _sensitivity(cfg: Dict[str, Any]) -> float:
    raw = _get(cfg, "sensitivity")
    try:
        s = float(raw)
    except (TypeError, ValueError):
        s = 0.5
    return min(max(s, 0.0), 1.0)


def wake_phrase(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Human-facing wake phrase label (purely cosmetic; engine keys detection)."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    return str(_get(cfg, "phrase")) or "hey jarvis"


def wake_surface_enabled(surface: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Should ``surface`` (``cli`` / ``tui`` / ``gui``) host the listener?

    True when the wake word is enabled and the configured ``surface`` is either
    ``auto`` or this exact surface.  ``auto`` makes a surface eligible; the
    process/machine ownership lock still permits only the first claimant.
    """
    cfg = cfg if cfg is not None else load_wake_word_config()
    if not cfg.get("enabled"):
        return False
    want = str(_get(cfg, "surface")).strip().lower() or "auto"
    return want == "auto" or want == surface.strip().lower()


# ---------------------------------------------------------------------------
# Audio capture (lazy — never import sounddevice at module load)
# ---------------------------------------------------------------------------

def _import_audio():
    import numpy as np
    import sounddevice as sd

    return sd, np


def _audio_available() -> bool:
    try:
        _import_audio()
        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class _Engine:
    """Minimal hotword-engine contract: feed int16 frames, get a bool."""

    frame_length: int = 1280  # 80 ms at 16 kHz

    def process(self, frame) -> bool:  # frame: 1-D int16 ndarray
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any internal audio/feature buffer (called on every (re)start)."""
        pass

    def close(self) -> None:
        pass


def _looks_like_path(value: str) -> bool:
    return (
        os.sep in value
        or value.endswith((".onnx", ".tflite", ".ppn"))
        or os.path.exists(value)
    )


class _OpenWakeWordEngine(_Engine):
    """openWakeWord — free, local ONNX hotword detection."""

    # openWakeWord recommends 80 ms frames (1280 samples) for efficiency.
    frame_length = 1280

    def __init__(self, cfg: Dict[str, Any]):
        from tools import lazy_deps

        lazy_deps.ensure("wake.openwakeword", prompt=False)

        import openwakeword
        from openwakeword.model import Model

        sub = cfg.get("openwakeword") if isinstance(cfg.get("openwakeword"), dict) else {}
        model_ref = str(sub.get("model") or "hey_jarvis").strip()
        framework = str(sub.get("inference_framework") or "onnx").strip().lower()
        self._threshold = _sensitivity(cfg)

        if _looks_like_path(model_ref):
            models = [model_ref]
        else:
            # Pretrained name (e.g. "hey_jarvis"). Best-effort one-time fetch
            # of the bundled models; harmless if already present / offline.
            try:
                openwakeword.utils.download_models([model_ref])
            except Exception as e:  # pragma: no cover - network/path dependent
                logger.debug("openwakeword model download skipped: %s", e)
            models = [model_ref]

        self._model = Model(wakeword_models=models, inference_framework=framework)
        self._labels = list(self._model.models.keys())

    def process(self, frame) -> bool:
        scores = self._model.predict(frame)
        return any(score >= self._threshold for score in scores.values())

    def reset(self) -> None:
        # Clears openWakeWord's rolling feature/prediction buffer so stale audio
        # captured before a pause can't re-fire the moment we resume.
        try:
            self._model.reset()
        except Exception:
            pass

    def close(self) -> None:
        self.reset()


class _PorcupineEngine(_Engine):
    """Picovoice Porcupine — premium, on-device, needs an access key."""

    def __init__(self, cfg: Dict[str, Any]):
        from tools import lazy_deps

        lazy_deps.ensure("wake.porcupine", prompt=False)

        import pvporcupine

        access_key = (os.getenv("PORCUPINE_ACCESS_KEY") or "").strip()
        if not access_key:
            raise RuntimeError(
                "Porcupine wake word requires PORCUPINE_ACCESS_KEY "
                "(get a free key at https://console.picovoice.ai)."
            )

        sub = cfg.get("porcupine") if isinstance(cfg.get("porcupine"), dict) else {}
        keyword = str(sub.get("keyword") or "jarvis").strip()
        sensitivity = _sensitivity(cfg)

        kwargs: Dict[str, Any] = {"access_key": access_key, "sensitivities": [sensitivity]}
        if _looks_like_path(keyword):
            kwargs["keyword_paths"] = [keyword]
        else:
            kwargs["keywords"] = [keyword]

        self._porcupine = pvporcupine.create(**kwargs)
        self.frame_length = self._porcupine.frame_length

    def process(self, frame) -> bool:
        # pvporcupine wants a plain list/sequence of int16 samples.
        return self._porcupine.process(frame) >= 0

    def close(self) -> None:
        try:
            self._porcupine.delete()
        except Exception:
            pass


def _build_engine(cfg: Dict[str, Any]) -> _Engine:
    provider = _provider(cfg)
    if provider == "porcupine":
        return _PorcupineEngine(cfg)
    if provider in ("openwakeword", "oww", "local"):
        return _OpenWakeWordEngine(cfg)
    raise ValueError(f"Unknown wake_word provider: {provider!r}")


# ---------------------------------------------------------------------------
# Requirements probe (for /wake status + enable path)
# ---------------------------------------------------------------------------

def check_wake_word_requirements(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Report whether wake-word detection can run, with a remediation hint."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    provider = _provider(cfg)
    from tools import lazy_deps

    feature = "wake.porcupine" if provider == "porcupine" else "wake.openwakeword"
    deps_ok = lazy_deps.is_available(feature)
    audio_ok = _audio_available()
    key_ok = True
    hint = ""

    if provider == "porcupine" and not (os.getenv("PORCUPINE_ACCESS_KEY") or "").strip():
        key_ok = False
        hint = "Set PORCUPINE_ACCESS_KEY (free key at https://console.picovoice.ai)."
    elif not deps_ok:
        hint = lazy_deps.feature_install_command(feature) or ""
    elif not audio_ok:
        hint = "Microphone capture needs sounddevice + numpy and a working audio device."

    return {
        "available": audio_ok and (deps_ok or lazy_deps._allow_lazy_installs()) and key_ok,
        "provider": provider,
        "deps_available": deps_ok,
        "audio_available": audio_ok,
        "access_key_set": key_ok,
        "phrase": wake_phrase(cfg),
        "hint": hint,
    }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """Background hotword listener. Fires ``on_wake()`` when the phrase is heard.

    The engine is built once and kept alive across pause/resume; only the audio
    stream + reader thread cycle, so toggling the mic for a voice turn is cheap.
    """

    def __init__(self, engine: _Engine, on_wake: Callable[[], None],
                 cooldown: float = _FIRE_COOLDOWN_SECONDS,
                 on_failure: Optional[Callable[["WakeWordDetector"], None]] = None):
        self.engine = engine
        self.on_wake = on_wake
        self.cooldown = cooldown
        self.on_failure = on_failure
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._callback_inflight = threading.Event()
        self._last_fire = 0.0
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def start(self) -> None:
        """Open the mic and begin listening. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            ready = threading.Event()
            startup_errors: list[BaseException] = []
            self._thread = threading.Thread(
                target=self._run,
                args=(ready, startup_errors),
                daemon=True,
                name="wake-word",
            )
            self._thread.start()
        if not ready.wait(_START_TIMEOUT_SECONDS):
            self._halt_thread()
            raise TimeoutError("Timed out while opening the wake-word microphone.")
        if startup_errors:
            self._halt_thread()
            raise RuntimeError("Failed to open the wake-word microphone.") from startup_errors[0]

    # pause/resume keep the engine; stop tears it down.
    def pause(self) -> None:
        self._halt_thread()

    def resume(self) -> None:
        self.start()

    def stop(self) -> None:
        self._halt_thread()
        self.engine.close()

    def _halt_thread(self) -> None:
        with self._lock:
            self._stop.set()
            t = self._thread
            if t is not None and t is not threading.current_thread():
                t.join(timeout=2.0)
            if self._thread is t:
                self._thread = None

    def _dispatch_wake(self) -> None:
        try:
            self.on_wake()
        except Exception as e:
            logger.warning("wake word callback failed: %s", e)
        finally:
            self._callback_inflight.clear()

    def _run(self, ready: threading.Event,
             startup_errors: list[BaseException]) -> None:
        try:
            sd, _ = _import_audio()
        except (ImportError, OSError) as e:
            logger.error("wake word: audio libraries unavailable: %s", e)
            startup_errors.append(e)
            ready.set()
            return

        frame_length = self.engine.frame_length
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=frame_length,
            )
            stream.start()
        except Exception as e:
            logger.error("wake word: failed to open microphone: %s", e)
            startup_errors.append(e)
            ready.set()
            return

        # Drop any buffered audio/feature state so a resume right after a voice
        # turn can't immediately re-fire on audio captured before the pause (the
        # wake → voice → resume → wake runaway loop).
        try:
            self.engine.reset()
        except Exception:
            pass

        logger.info("wake word: listening (frame=%d, rate=%d)", frame_length, SAMPLE_RATE)
        ready.set()
        failed = False
        try:
            while not self._stop.is_set():
                try:
                    data, _overflow = stream.read(frame_length)
                except Exception as e:
                    logger.warning("wake word: stream read error: %s", e)
                    failed = not self._stop.is_set()
                    break
                frame = data[:, 0] if getattr(data, "ndim", 1) == 2 else data
                try:
                    fired = self.engine.process(frame)
                except Exception as e:
                    logger.debug("wake word: engine error: %s", e)
                    continue
                if fired:
                    now = time.monotonic()
                    if now - self._last_fire >= self.cooldown:
                        self._last_fire = now
                        logger.info("wake word: phrase detected — firing callback")
                        if not self._callback_inflight.is_set():
                            self._callback_inflight.set()
                            threading.Thread(
                                target=self._dispatch_wake,
                                daemon=True,
                                name="wake-word-callback",
                            ).start()
                    else:
                        logger.debug("wake word: detection within cooldown — ignored")
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            logger.info("wake word: stream closed")
            if failed and self.on_failure is not None:
                self.on_failure(self)


# ---------------------------------------------------------------------------
# Process-wide singleton (mirrors hermes_cli.voice's continuous API)
# ---------------------------------------------------------------------------

_detector: Optional[WakeWordDetector] = None
_detector_owner: object | None = None
_detector_file_lock = None
_detector_lock = threading.Lock()


def _lock_path() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "runtime" / "wake-word.lock"


def _acquire_machine_lock(path: Optional[Path] = None):
    """Acquire the cross-process microphone lease, or raise WakeWordInUse."""
    lock_path = path or _lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as e:
        handle.close()
        raise WakeWordInUse("Wake-word microphone is already owned.") from e
    return handle


def _release_machine_lock(handle) -> None:
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        handle.close()


def _detector_failed(detector: WakeWordDetector) -> None:
    """Release ownership if the active microphone stream dies unexpectedly."""
    global _detector, _detector_owner, _detector_file_lock
    with _detector_lock:
        if _detector is not detector:
            return
        lock_handle = _detector_file_lock
        _detector = None
        _detector_owner = None
        _detector_file_lock = None
        try:
            detector.engine.close()
        finally:
            _release_machine_lock(lock_handle)


def start_listening(
    on_wake: Callable[[], None],
    *,
    owner: object,
    config: Optional[Dict[str, Any]] = None,
) -> WakeWordDetector:
    """Claim, build, and start the detector. Idempotent for the same owner.

    Raises if engine construction fails (missing deps / access key / model);
    callers should probe :func:`check_wake_word_requirements` first. A different
    owner, including another process, receives :class:`WakeWordInUse`.
    """
    if owner is None:
        raise ValueError("wake-word owner must not be None")

    global _detector, _detector_owner, _detector_file_lock
    with _detector_lock:
        if _detector is not None:
            if _detector_owner is not owner:
                raise WakeWordInUse("Wake-word microphone is already owned.")
            _detector.on_wake = on_wake
            _detector.resume()
            return _detector
        lock_handle = _acquire_machine_lock()
        try:
            cfg = config if config is not None else load_wake_word_config()
            engine = _build_engine(cfg)
            detector = WakeWordDetector(engine, on_wake, on_failure=_detector_failed)
            _detector = detector
            _detector_owner = owner
            _detector_file_lock = lock_handle
            detector.start()
            return detector
        except Exception:
            if _detector is not None:
                try:
                    _detector.stop()
                except Exception:
                    pass
            _detector = None
            _detector_owner = None
            _detector_file_lock = None
            _release_machine_lock(lock_handle)
            raise


def owns_listener(owner: object) -> bool:
    with _detector_lock:
        return _detector is not None and _detector_owner is owner


def pause_listening(*, owner: object) -> bool:
    """Release the microphone only when ``owner`` holds the lease."""
    with _detector_lock:
        if _detector is None or _detector_owner is not owner:
            return False
        _detector.pause()
        return True


def resume_listening(*, owner: object) -> bool:
    """Re-open the microphone only when ``owner`` holds the lease."""
    with _detector_lock:
        if _detector is None or _detector_owner is not owner:
            return False
        _detector.resume()
        return True


def stop_listening(*, owner: object) -> bool:
    """Fully stop the detector only when ``owner`` holds the lease."""
    global _detector, _detector_owner, _detector_file_lock
    with _detector_lock:
        if _detector is None or _detector_owner is not owner:
            return False
        det = _detector
        lock_handle = _detector_file_lock
        _detector = None
        _detector_owner = None
        _detector_file_lock = None
        try:
            det.stop()
        finally:
            _release_machine_lock(lock_handle)
        return True


def is_listening() -> bool:
    with _detector_lock:
        det = _detector
    return det is not None and det.running
