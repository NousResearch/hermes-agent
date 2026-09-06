"""Wake-word ("Hey Hermes") detection — hands-free session trigger.

One always-on hotword listener shared by CLI, TUI and desktop GUI (a single owner,
gated by ``wake_surface_enabled``). Engines live in :mod:`tools.wake_word_engines`;
this module owns config, the capture loop and the process-wide listener singleton.
Capture reuses voice mode's 16 kHz mono int16 ``sounddevice`` path on a daemon
thread; callers ``pause()`` while a voice turn holds the mic and ``resume()`` once
idle (two input streams on one device is unreliable cross-platform).
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from tools.wake_word_engines import _Engine, _OpenWakeWordEngine, _PorcupineEngine, _SherpaKwsEngine, _sub

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # 16 kHz mono int16 — Whisper-native and what every engine expects.

# Minimum gap between two wake fires, so one "hey hermes" can't retrigger across
# several frames while the caller is still reacting.
_FIRE_COOLDOWN_SECONDS = 2.0
_START_TIMEOUT_SECONDS = 5.0

# Ambient-speech rejection: N consecutive over-threshold frames before firing
# (a stray phoneme spikes one frame; a real phrase holds several).
_DEFAULT_CONFIRMATION_FRAMES = 3

# Dead-mic detection: an int16 stream whose peak stays at/below _SILENCE_PEAK for
# this many consecutive seconds is flagged silent (desktop push-to-talk and the
# backend listener capture differently, so one can work while the other is all zeros).
_SILENCE_PEAK = 10
_SILENCE_ALERT_SECONDS = 10

# provider alias -> (engine class name on this module, lazy_deps feature).
# Unknown providers probe as openwakeword but fail to build.
_PROVIDERS: Dict[str, tuple[str, str]] = {
    "porcupine": ("_PorcupineEngine", "wake.porcupine"),
    **{k: ("_SherpaKwsEngine", "wake.sherpa") for k in ("sherpa", "sherpa-onnx", "kws", "open")},
    **{k: ("_OpenWakeWordEngine", "wake.openwakeword") for k in ("openwakeword", "oww", "local")},
}


class WakeWordInUse(RuntimeError):
    """Raised when another surface or process owns the wake-word listener."""


# ── Config ──

# capture: "local" (PortAudio on the backend host), "client" (desktop/TUI streams int16
# frames via wake.feed), or "auto" (local when a device exists, else client).
_DEFAULTS: Dict[str, Any] = {
    "enabled": False, "surface": "auto", "input_device": None, "capture": "auto",
    "provider": "openwakeword", "phrase": "hey hermes", "sensitivity": 0.6,
    "confirmation_frames": _DEFAULT_CONFIRMATION_FRAMES, "start_new_session": True,
}

# Bundled "hey hermes" model (tools/wakewords/) — the default; alias names resolve
# to it, not to an openWakeWord built-in.
_BUNDLED_MODEL_NAME = "hey_hermes"
_BUNDLED_MODEL_ALIASES = frozenset({"", "hey_hermes", "hey hermes", "hermes"})


def _bundled_wakeword_path(framework: str = "onnx") -> str:
    """Path to the shipped hey_hermes model (.onnx/.tflite) for ``framework``."""
    ext = "tflite" if str(framework).strip().lower() == "tflite" else "onnx"
    return os.path.join(os.path.dirname(__file__), "wakewords", f"{_BUNDLED_MODEL_NAME}.{ext}")


def _is_macos_arm64() -> bool:
    import platform
    return sys.platform == "darwin" and platform.machine() == "arm64"


def default_inference_framework() -> str:
    """tflite on macOS ARM64, onnx elsewhere: openWakeWord's ONNX embedding model
    scores near-zero on Apple Silicon — the detector arms but never fires."""
    return "tflite" if _is_macos_arm64() else "onnx"


_warned_onnx_coerced = False


def resolve_inference_framework(cfg: Dict[str, Any]) -> str:
    """Effective openWakeWord backend: explicit ``openwakeword.inference_framework`` or
    the platform default. Explicit ``onnx`` on macOS ARM64 is provably dead, so it is
    coerced to tflite with a one-time warning (a pre-fix pin must not stay deaf)."""
    global _warned_onnx_coerced

    framework = str(_sub(cfg, "openwakeword").get("inference_framework") or "").strip().lower()
    if not framework:
        return default_inference_framework()
    if framework == "onnx" and _is_macos_arm64():
        if not _warned_onnx_coerced:
            _warned_onnx_coerced = True
            logger.warning("wake: openwakeword.inference_framework='onnx' is set but ONNX's "
                           "embedding model never fires on macOS ARM64 (openWakeWord #336) — "
                           "using tflite instead. Set inference_framework to '' (auto) or "
                           "'tflite' in config.yaml to silence this.")
        return "tflite"
    return framework


def ensure_tflite_runtime() -> bool:
    """Make ``import tflite_runtime.interpreter`` resolve, returning success. openWakeWord hardcodes
    that import but only declares ``tflite-runtime`` on Linux; on macOS the wheel is ``ai-edge-litert``,
    so alias it in-process (site-packages untouched)."""
    try:
        import tflite_runtime.interpreter  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from ai_edge_litert import interpreter as _litert  # type: ignore[import-not-found]
    except ImportError:
        return False
    import types
    pkg = types.ModuleType("tflite_runtime")
    pkg.__path__ = []  # type: ignore[attr-defined]  # mark as package
    sys.modules.setdefault("tflite_runtime", pkg)
    sys.modules["tflite_runtime.interpreter"] = _litert
    logger.debug("wake word: bridged tflite_runtime -> ai_edge_litert")
    return True


def load_wake_word_config() -> Dict[str, Any]:
    """Return the ``wake_word`` config section, shape-guarded to a dict."""
    cfg = None
    with suppress(Exception):
        from hermes_cli.config import load_config
        cfg = load_config().get("wake_word")
    return cfg if isinstance(cfg, dict) else {}


def _get(cfg: Dict[str, Any], key: str) -> Any:
    val = cfg.get(key)
    return _DEFAULTS.get(key) if val is None else val


def _clamped(cfg: Dict[str, Any], key: str, cast, lo, hi):
    """Numeric config value via ``cast``, defaulting on junk, clamped to lo..hi."""
    try:
        n = cast(_get(cfg, key))
    except (TypeError, ValueError):
        n = cast(_DEFAULTS[key])
    return min(max(n, lo), hi)


def _provider(cfg: Dict[str, Any]) -> str:
    return str(_get(cfg, "provider")).strip().lower() or "openwakeword"


def _input_device(cfg: Dict[str, Any]) -> int | str | None:
    """Configured PortAudio input selector, preserving indices and names."""
    raw = _get(cfg, "input_device")
    return None if isinstance(raw, bool) else raw if raw is None or isinstance(raw, int) else (str(raw).strip() or None)


def _sensitivity(cfg: Dict[str, Any]) -> float:
    return _clamped(cfg, "sensitivity", float, 0.0, 1.0)


def _confirmation_frames(cfg: Dict[str, Any]) -> int:
    """Consecutive over-threshold frames required to fire, clamped 1..10 (1 = single-frame)."""
    return _clamped(cfg, "confirmation_frames", int, 1, 10)


def wake_phrase(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Human-facing wake phrase label (purely cosmetic; engine keys detection)."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    return str(_get(cfg, "phrase")) or "hey hermes"


def resolve_capture_mode(cfg: Optional[Dict[str, Any]] = None, *, prefer_client: bool = False,
                         force_local: bool = False) -> str:
    """Return ``local`` or ``client`` capture mode for this arm. ``prefer_client`` is set by remote
    desktop; ``force_local`` keeps CLI/TUI on the process mic. Under ``auto`` a working backend input
    always wins; client is the fallback only for a preferring surface with no usable backend mic —
    CLI/TUI stay local so status reports the real requirement rather than a path nothing will feed."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    if force_local:
        return "local"
    raw = str(_get(cfg, "capture") or "auto").strip().lower()
    if raw in ("client", "remote", "external"):
        return "client"
    return "client" if raw != "local" and prefer_client and not _local_input_device_ready() else "local"


def _input_channels(info: Any) -> int:
    ch = info.get("max_input_channels") if isinstance(info, dict) else getattr(info, "max_input_channels", 0)
    return int(ch or 0)


def _local_input_device_ready() -> bool:
    """True when PortAudio is importable and at least one input device exists."""
    try:
        sd, _ = _import_audio()
        devices = sd.query_devices()
        if isinstance(devices, dict):
            return _input_channels(devices) > 0
        # Also accept a resolvable default input (some hosts list devices oddly).
        return (any(_input_channels(d) > 0 for d in devices)
                or _input_channels(sd.query_devices(None, "input")) > 0)
    except Exception:
        return False


def wake_surface_enabled(surface: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Should ``surface`` (cli/tui/gui) host the listener? True when enabled and the configured
    surface is ``auto`` or this one; ``auto`` only makes it eligible — the lock admits one claimant."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    want = str(_get(cfg, "surface")).strip().lower() or "auto"
    return bool(cfg.get("enabled")) and want in ("auto", surface.strip().lower())


# ── Multi-profile phrase enrollment (open-vocabulary routing) ──

def _active_profile_name() -> str:
    with suppress(Exception):
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    return "default"


def enrolled_profile_phrases() -> Dict[str, str]:
    """Map ``profile name -> wake phrase`` for every wake-enabled profile, reading each profile's own
    ``config.yaml`` raw (``load_config()`` targets only the ACTIVE profile). Phrase defaults to
    ``"hey <profile>"``; the sherpa engine listens for all and routes to the match. Unreadable → skipped."""
    phrases: Dict[str, str] = {}
    with suppress(Exception):
        from hermes_cli.config import read_user_config_raw
        from hermes_cli.profiles import get_profile_dir, list_profiles
        for info in list_profiles():
            name = getattr(info, "name", None) or str(info)
            with suppress(Exception):
                wc = read_user_config_raw(Path(get_profile_dir(name)) / "config.yaml").get("wake_word") or {}
                if isinstance(wc, dict) and wc.get("enabled"):
                    phrase = str(wc.get("phrase") or f"hey {name}").strip()
                    if phrase:
                        phrases[name] = phrase
    return phrases


# ── Audio capture (lazy — never import sounddevice at module load) ──

def _import_audio():
    import numpy as np
    import sounddevice as sd
    return sd, np


def _audio_available() -> bool:
    with suppress(ImportError, OSError):
        return bool(_import_audio())
    return False


def _describe_input_device(selector: int | str | None, sd=None) -> Dict[str, Any]:
    """Resolve a PortAudio selector into JSON-safe diagnostics (``InputStream`` stays the
    authority on whether the device actually opens). Imports sounddevice unless ``sd`` is given."""
    details: Dict[str, Any] = {"selector": selector}
    try:
        sd = sd or _import_audio()[0]
        info = sd.query_devices(selector, "input")
    except Exception as e:
        details["error"] = str(e)
        return details
    if not isinstance(info, dict):
        return details
    if info.get("name"):
        details["name"] = str(info["name"])
    for key, out_key, cast in (("max_input_channels", "max_input_channels", int),
                               ("default_samplerate", "default_samplerate", float), ("hostapi", "hostapi_index", int)):
        if isinstance(info.get(key), (int, float)):
            details[out_key] = cast(info[key])
    if "hostapi_index" in details:
        with suppress(Exception):
            hostapi = sd.query_hostapis(details["hostapi_index"])
            if isinstance(hostapi, dict) and hostapi.get("name"):
                details["hostapi"] = str(hostapi["name"])
    return details


def _resample_audio_frame(np, frame, output_length: int):
    """Convert one native-rate capture block to an exact engine frame."""
    source = np.asarray(frame, dtype=np.float64).reshape(-1)
    if source.size == output_length:
        return np.asarray(frame, dtype=np.int16).reshape(-1)
    if source.size == 0:
        return np.zeros(output_length, dtype=np.int16)
    if source.size > output_length:
        # Average each source window when reducing (matches the desktop wake capture
        # path) so speech energy is retained instead of decimated.
        edges = np.linspace(0, source.size, output_length + 1, dtype=np.int64)
        values = np.add.reduceat(source, edges[:-1]) / np.diff(edges)
    else:
        # Unusual low-rate devices: interpolate up to the 16 kHz frame size.
        source_positions = np.arange(source.size, dtype=np.float64)
        values = np.interp(np.linspace(0, source.size - 1, output_length), source_positions, source)
    return np.rint(values).clip(-32768, 32767).astype(np.int16)


def silent_audio_hint(details: Dict[str, Any]) -> str:
    """Platform-specific remediation for an armed stream delivering silence."""
    if sys.platform == "darwin":
        return ("Microphone delivers only silence. Grant the Hermes backend "
                "microphone access in System Settings > Privacy & Security > "
                "Microphone, then toggle the wake word.")
    fix = ("Set wake_word.input_device to a different PortAudio input device"
           if sys.platform == "win32" else "Check the selected input device")
    selector = details.get("selector")
    label = str(details.get("name") or "").strip() or ("system default" if selector is None else str(selector))
    hostapi = str(details.get("hostapi") or "").strip()
    label = f"{label} ({hostapi})" if hostapi else label
    return f"Microphone delivers only silence from {label}. {fix}, then toggle the wake word."


# ── Engines (implementations live in tools.wake_word_engines) ──

def _build_engine(cfg: Dict[str, Any]) -> _Engine:
    provider = _provider(cfg)
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown wake_word provider: {provider!r}")
    return globals()[_PROVIDERS[provider][0]](cfg)


# ── Requirements probe (for /wake status + enable path) ──

def _stt_ready() -> bool:
    """Is a speech-to-text provider configured and enabled? (A wake without STT arms the
    mic but every utterance dies at transcription — same bar as ``check_voice_requirements``.)"""
    with suppress(Exception):
        from tools.transcription_tools import _get_provider, _load_stt_config, is_stt_enabled
        stt_config = _load_stt_config()
        return is_stt_enabled(stt_config) and _get_provider(stt_config) != "none"
    return False


def _tts_ready() -> bool:
    """Can the configured TTS provider run (or install at first use)? PROBE, not an installer:
    ``check_tts_requirements`` lazily pip-installs the SDK, which froze wake.status polls for a whole
    pip run. Uninstalled deps count as ready iff lazy installs are allowed; pip is never touched here."""
    try:
        from tools.tts_tool import _get_provider, _load_tts_config, check_tts_requirements
        provider = _get_provider(_load_tts_config())
        feature = f"tts.{provider}" if provider in ("edge", "elevenlabs", "mistral") else None
        if feature is not None:
            from tools import lazy_deps
            if not lazy_deps.is_available(feature):
                return lazy_deps._allow_lazy_installs()
        return bool(check_tts_requirements())
    except Exception:
        return False


def check_wake_word_requirements(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Report whether wake-word detection can run, with a remediation hint."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    provider = _provider(cfg)
    from tools import lazy_deps
    feature = _PROVIDERS.get(provider, ("", "wake.openwakeword"))[1]
    deps_ok = lazy_deps.is_available(feature)
    lazy_ok = lazy_deps._allow_lazy_installs()
    # The audio probe imports sounddevice + numpy — packages the lazy installer would
    # fetch — so only trust it once deps are installed; on a fresh install the engine
    # constructors' ``lazy_deps.ensure()`` + stream-open surface any real audio problem.
    audio_ok = _audio_available() if deps_ok else False
    # Loop is wake → record → STT → agent → TTS; without either end the mic hears you
    # and nothing perceptible happens — refuse with a hint.
    stt_ok, tts_ok = _stt_ready(), _tts_ready()
    # tflite needs a runtime openWakeWord doesn't declare off Linux; report it as a
    # remediation instead of arming a detector that can't fire.
    tflite_ok = (feature != "wake.openwakeword" or resolve_inference_framework(cfg) != "tflite"
                 or ensure_tflite_runtime() or lazy_deps.is_available("wake.openwakeword.tflite") or lazy_ok)
    key_ok = provider != "porcupine" or bool((os.getenv("PORCUPINE_ACCESS_KEY") or "").strip())
    capture_mode = resolve_capture_mode(cfg)
    missing = " and ".join(n for n, ok in (("speech-to-text", stt_ok), ("text-to-speech", tts_ok)) if not ok)

    # Ordered remediation ladder: first true predicate wins.
    ladder = (
        (not key_ok, lambda: "Set PORCUPINE_ACCESS_KEY (free key at https://console.picovoice.ai)."),
        (not deps_ok and not lazy_ok, lambda: lazy_deps.feature_install_command(feature) or ""),
        (not tflite_ok,
         lambda: "The wake word needs the tflite runtime on this Mac: pip install ai-edge-litert"),
        (deps_ok and not audio_ok and capture_mode == "local",
         lambda: "Microphone capture needs sounddevice + numpy and a working audio device."),
        (bool(missing), lambda: (f"Wake word needs {missing} configured — run `hermes tools` "
                                 f"(Voice section) or see the voice-mode docs.")),
    )
    hint = next((make() for cond, make in ladder if cond), "")

    # Client capture needs deps (engine) but not a server-side PortAudio device.
    if capture_mode == "client":
        mic_ok = deps_ok or lazy_ok
    else:
        mic_ok = (deps_ok and audio_ok) or (not deps_ok and lazy_ok)
        if deps_ok and not audio_ok and not hint:
            hint = ("No local microphone on this backend. Remote desktop can stream "
                    "the client mic — set wake_word.capture: client or use a desktop "
                    "build with client-capture wake support.")

    return {
        "available": key_ok and stt_ok and tts_ok and tflite_ok and mic_ok, "provider": provider,
        "deps_available": deps_ok, "audio_available": audio_ok,
        "local_input_available": _local_input_device_ready() if deps_ok else False,
        "capture": capture_mode, "access_key_set": key_ok, "stt_available": stt_ok, "tts_available": tts_ok,
        "phrase": wake_phrase(cfg), "hint": hint,
    }


# ── Detector ──

@dataclass
class _Capture:
    """One armed audio source: a PortAudio stream (local) or the feed queue (client)."""

    stream: Any = None  # sounddevice.InputStream, None in client mode
    queue: Any = None  # client-capture frame queue, None in local mode
    np: Any = None
    rate: int = SAMPLE_RATE
    frame_length: int = 1280  # samples per read at ``rate``

    def read(self):
        """One raw block; None when no client frame arrived within 250 ms. Stream errors propagate."""
        if self.stream is not None:
            return self.stream.read(self.frame_length)[0]
        with suppress(Exception):
            return self.queue.get(timeout=0.25)
        return None

    def close(self) -> None:
        with suppress(Exception):
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()


class WakeWordDetector:
    """Background hotword listener; fires ``on_wake()`` when the phrase is heard. The engine is built
    once and kept across pause/resume — only the stream + reader thread cycle, so mic toggles are cheap."""

    def __init__(self, engine: _Engine, on_wake: Callable[[], None], cooldown: float = _FIRE_COOLDOWN_SECONDS,
                 on_failure: Optional[Callable[["WakeWordDetector"], None]] = None,
                 input_device: int | str | None = None, external_audio: bool = False):
        self.engine, self.on_wake, self.cooldown, self.on_failure = engine, on_wake, cooldown, on_failure
        self.input_device, self.external_audio = input_device, bool(external_audio)
        self.input_device_details: Dict[str, Any] = (
            {"selector": "client", "name": "client capture", "hostapi": "remote"}
            if self.external_audio else {"selector": input_device})
        self._thread: Optional[threading.Thread] = None
        self._stop, self._callback_inflight = threading.Event(), threading.Event()
        self._lock, self._last_fire = threading.Lock(), 0.0
        # Client-capture PCM queue (int16 mono frames). Local mode ignores this.
        self._audio_q: "queue.Queue[Any]" = queue.Queue(maxsize=64)
        # True when the stream is open but every frame is (near-)silence, so status
        # surfaces can tell "armed" from "deaf".
        self.audio_silent, self._silent_frames = False, 0

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def feed(self, pcm_int16) -> None:
        """Enqueue one int16 mono frame (or raw bytes) for client capture. Short frames are
        zero-padded to ``engine.frame_length``, long ones split; on overflow the oldest is dropped."""
        if not self.external_audio:
            return
        try:
            import numpy as np
        except Exception:
            return
        if isinstance(pcm_int16, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(pcm_int16, dtype=np.int16)
        else:
            arr = np.asarray(pcm_int16, dtype=np.int16).reshape(-1)
        fl = int(self.engine.frame_length)
        if fl <= 0:
            return
        for offset in range(0, int(arr.shape[0]), fl):
            chunk = arr[offset : offset + fl]
            if chunk.shape[0] < fl:
                chunk = np.pad(chunk, (0, fl - chunk.shape[0]))
            try:
                self._audio_q.put_nowait(chunk)
            except Exception:
                # Drop oldest on overflow so we stay real-time
                with suppress(Exception):
                    self._audio_q.get_nowait()
                with suppress(Exception):
                    self._audio_q.put_nowait(chunk)

    def start(self) -> None:
        """Open the mic (or client feeder) and begin listening. Idempotent."""
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            ready = threading.Event()
            startup_errors: list[BaseException] = []
            self._thread = threading.Thread(target=self._run, args=(ready, startup_errors),
                                            daemon=True, name="wake-word")
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

    def _open_capture(self, frame_length: int) -> _Capture:
        """Open the audio source; raises on any local-mic failure."""
        if self.external_audio:
            with suppress(Exception):  # drain stale frames from a previous arm
                while True:
                    self._audio_q.get_nowait()
            logger.info("wake word: client-capture mode (frame=%d, rate=%d) — waiting for wake.feed",
                        frame_length, SAMPLE_RATE)
            return _Capture(queue=self._audio_q, frame_length=frame_length)

        try:
            sd, np = _import_audio()
        except (ImportError, OSError) as e:
            logger.error("wake word: audio libraries unavailable: %s", e)
            raise
        details = self.input_device_details = _describe_input_device(self.input_device, sd)
        # Capture at the device's native rate when PortAudio reports one; frames are resampled to the engine.
        cap, rate = _Capture(np=np), details.get("default_samplerate")
        if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
            with suppress(OverflowError, ValueError):
                cap.rate = int(round(rate))
        cap.frame_length = max(1, int(round(frame_length * cap.rate / SAMPLE_RATE)))
        logger.info("wake word: opening microphone device=%s selector=%r hostapi=%s "
                    "default_rate=%s capture_rate=%d engine_rate=%d", details.get("name") or "system default",
                    self.input_device, details.get("hostapi") or "unknown",
                    details.get("default_samplerate") or "unknown", cap.rate, SAMPLE_RATE)
        try:
            cap.stream = sd.InputStream(device=self.input_device, samplerate=cap.rate, channels=1,
                                        dtype="int16", blocksize=cap.frame_length)
            cap.stream.start()
        except Exception as e:
            logger.error("wake word: failed to open microphone: %s", e)
            raise
        return cap

    def _note_silence(self, frame, silent_alert_frames: int) -> None:
        """Track consecutive near-zero frames; flag/unflag ``audio_silent``. ``frame`` is None when
        no client frame arrived (counts as silence for status, but is never logged as a dead mic)."""
        try:
            peak = 0 if frame is None or not len(frame) else int(abs(frame).max())
        except Exception:
            peak = _SILENCE_PEAK + 1
        if peak <= _SILENCE_PEAK:
            self._silent_frames += 1
            if self._silent_frames == silent_alert_frames:
                self.audio_silent = True
                if frame is not None:
                    logger.warning("wake word: mic delivers only silence (peak<=%d for %ds); %s",
                                   _SILENCE_PEAK, _SILENCE_ALERT_SECONDS,
                                   silent_audio_hint(self.input_device_details))
        elif self._silent_frames:
            if self.audio_silent:
                logger.info("wake word: mic audio detected — stream healthy")
            self._silent_frames, self.audio_silent = 0, False

    def _fire(self) -> None:
        """Honor the cooldown, then run ``on_wake`` on its own thread (once)."""
        now = time.monotonic()
        if now - self._last_fire < self.cooldown:
            logger.debug("wake word: detection within cooldown — ignored")
            return
        self._last_fire = now
        logger.info("wake word: phrase detected — firing callback")
        if not self._callback_inflight.is_set():
            self._callback_inflight.set()
            threading.Thread(target=self._dispatch_wake, daemon=True, name="wake-word-callback").start()

    def _run(self, ready: threading.Event, startup_errors: list[BaseException]) -> None:
        frame_length = self.engine.frame_length
        try:
            cap = self._open_capture(frame_length)
        except Exception as e:
            startup_errors.append(e)
            ready.set()
            return
        # Drop buffered audio/feature state so a resume right after a voice turn can't
        # re-fire on audio captured before the pause (wake → voice → resume → wake loop).
        with suppress(Exception):
            self.engine.reset()
        logger.info("wake word: listening (frame=%d, rate=%d, external=%s)",
                    frame_length, SAMPLE_RATE, self.external_audio)
        ready.set()
        failed = False
        silent_alert_frames = max(1, int(_SILENCE_ALERT_SECONDS * SAMPLE_RATE / max(1, frame_length)))
        try:
            while not self._stop.is_set():
                try:
                    data = cap.read()
                except Exception as e:
                    logger.warning("wake word: stream read error: %s", e)
                    failed = not self._stop.is_set()
                    break
                if data is None:  # no client frames yet — counts as silence for status
                    self._note_silence(None, silent_alert_frames)
                    continue
                frame = data[:, 0] if getattr(data, "ndim", 1) == 2 else data
                if cap.rate != SAMPLE_RATE:
                    frame = _resample_audio_frame(cap.np, frame, frame_length)
                self._note_silence(frame, silent_alert_frames)
                try:
                    if self.engine.process(frame):
                        self._fire()
                except Exception as e:
                    logger.debug("wake word: engine error: %s", e)
        finally:
            cap.close()
            logger.info("wake word: stream closed")
            if failed and self.on_failure is not None:
                self.on_failure(self)


# ── Process-wide singleton (mirrors hermes_cli.voice's continuous API) ──

_detector: Optional[WakeWordDetector] = None
_detector_owner: object | None = None
_detector_file_lock = None
_detector_lock = threading.Lock()


def _lock_path() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "runtime" / "wake-word.lock"


def _flock(handle, acquire: bool) -> None:
    """Non-blocking exclusive lock (or unlock) of one byte / whole file, per OS."""
    if os.name == "nt":
        import msvcrt
        if acquire:  # msvcrt needs at least one byte to lock
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)


def _acquire_machine_lock(path: Optional[Path] = None):
    """Acquire the cross-process microphone lease, or raise WakeWordInUse."""
    lock_path = path or _lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        _flock(handle, True)
    except OSError as e:  # BlockingIOError is an OSError: lock held elsewhere
        handle.close()
        raise WakeWordInUse("Wake-word microphone is already owned.") from e
    return handle


def _release_machine_lock(handle) -> None:
    if handle is None:
        return
    try:
        _flock(handle, False)
    except OSError:
        pass
    finally:
        handle.close()


def _teardown_locked(close: Callable[[], None]) -> None:
    """Forget the singleton and run ``close``, always releasing the machine lease (caller holds the lock)."""
    global _detector, _detector_owner, _detector_file_lock
    lock_handle = _detector_file_lock
    _detector = _detector_owner = _detector_file_lock = None
    try:
        close()
    finally:
        _release_machine_lock(lock_handle)


def _owned_detector(owner: object) -> Optional[WakeWordDetector]:
    """The armed detector iff ``owner`` holds the lease (caller holds the lock)."""
    return _detector if _detector is not None and _detector_owner is owner else None


def _detector_failed(detector: WakeWordDetector) -> None:
    """Release ownership if the active microphone stream dies unexpectedly."""
    with _detector_lock:
        if _detector is detector:
            _teardown_locked(detector.engine.close)


def start_listening(on_wake: Callable[[], None], *, owner: object, config: Optional[Dict[str, Any]] = None,
                    external_audio: bool = False) -> WakeWordDetector:
    """Claim, build, and start the detector. Idempotent for the same owner; a different owner
    (or process) gets :class:`WakeWordInUse`. Raises if engine construction fails (missing deps /
    access key / model) — callers should probe :func:`check_wake_word_requirements` first."""
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
        _detector_file_lock = _acquire_machine_lock()
        try:
            cfg = config if config is not None else load_wake_word_config()
            _detector = WakeWordDetector(_build_engine(cfg), on_wake, on_failure=_detector_failed,
                                         input_device=_input_device(cfg), external_audio=external_audio)
            _detector_owner = owner
            _detector.start()
            return _detector
        except Exception:
            with suppress(Exception):
                _teardown_locked(_detector.stop if _detector is not None else lambda: None)
            raise


def _owned_call(owner: object, action: Optional[Callable[[WakeWordDetector], None]] = None) -> bool:
    """Under the lock, True iff ``owner`` holds the lease; also runs ``action(detector)`` when given."""
    with _detector_lock:
        det = _owned_detector(owner)
        if det is None:
            return False
        if action is not None:
            action(det)
        return True


def owns_listener(owner: object) -> bool:
    return _owned_call(owner)


def pause_listening(*, owner: object) -> bool:
    """Release the microphone only when ``owner`` holds the lease."""
    return _owned_call(owner, WakeWordDetector.pause)


def resume_listening(*, owner: object) -> bool:
    """Re-open the microphone only when ``owner`` holds the lease."""
    return _owned_call(owner, WakeWordDetector.resume)


def stop_listening(*, owner: object) -> bool:
    """Fully stop the detector only when ``owner`` holds the lease."""
    return _owned_call(owner, lambda det: _teardown_locked(det.stop))


def _current_detector() -> Optional[WakeWordDetector]:
    with _detector_lock:
        return _detector


def is_listening() -> bool:
    return (det := _current_detector()) is not None and det.running


def audio_is_silent() -> bool:
    """True when the armed stream opens fine but delivers only silence (dead mic), so
    detection can never fire; status shows "listening but the microphone appears silent"."""
    return (det := _current_detector()) is not None and det.audio_silent


def get_input_device_status(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return configured/active PortAudio input diagnostics for status UIs."""
    if (det := _current_detector()) is not None:
        return dict(det.input_device_details)
    return _describe_input_device(_input_device(cfg if cfg is not None else load_wake_word_config()))


def get_last_match() -> Optional[tuple[str, str]]:
    """(matched phrase, profile) of the most recent wake fire when the engine reports
    per-phrase matches (sherpa multi-profile routing); None otherwise."""
    return None if (det := _current_detector()) is None else getattr(det.engine, "last_match", None)


def feed_audio(*, owner: object, pcm_int16) -> bool:
    """Push client-captured PCM into ``owner``'s armed detector; True when accepted."""
    with _detector_lock:
        det = _owned_detector(owner)
        if det is None or not det.external_audio:
            return False
    det.feed(pcm_int16)
    return True


def detector_frame_info() -> Dict[str, Any]:
    """Sample rate + frame length for client capture streamers."""
    if (det := _current_detector()) is None:
        return {"sample_rate": SAMPLE_RATE, "frame_length": 1280}
    return {"sample_rate": SAMPLE_RATE, "external_audio": bool(det.external_audio),
            "frame_length": int(getattr(det.engine, "frame_length", 1280) or 1280)}
