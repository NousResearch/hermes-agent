"""Wake-word ("Hey Hermes") detection — hands-free session trigger.

A lightweight, always-on hotword listener that fires a callback when a wake
phrase is spoken — the "Hey Siri" / "Alexa" pattern. Shared by the CLI, TUI, and
desktop GUI (one of them owns it, gated by ``wake_surface_enabled``): say the
wake word, Hermes opens a fresh session and captures voice via the existing
pipeline, then answers.

Three engines, all fully on-device (no audio leaves the machine for detection):

* **openwakeword** (default, free, no API key) — loads an ONNX model. Defaults
  to the bundled "hey hermes" model (``tools/wakewords/``) so the wake word
  works out of the box; or point ``wake_word.openwakeword.model`` at a built-in
  name (``hey_jarvis``, ``alexa``, …) or a custom ``.onnx`` for another phrase.
* **sherpa** (free, no API key, open vocabulary) — sherpa-onnx keyword
  spotting. Detects ANY typed phrase with no training: set
  ``wake_word.phrase`` and the phrase is tokenized at runtime against a small
  streaming zipformer model (~13 MB English model, one-time download).
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
import sys
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

# Post-wake capture: after a wake fires, the same mic stream keeps reading
# frames so the user's *next* utterance (the command after "amos") is captured
# on a healthy, already-routed device. Without this, the wake→voice handoff
# would have to close the wake stream and open a second one on the same
# physical mic, which on macOS races the CoreAudio AUHAL handle-release window
# and delivers attenuated/empty audio. Keeping the same stream open eliminates
# that race entirely.
_POST_WAKE_SILENCE_RMS = 250           # int16 RMS below this counts as silence
_POST_WAKE_SILENCE_DURATION = 1.5       # seconds of silence to auto-stop
_POST_WAKE_MAX_DURATION = 30.0          # hard cap (matches voice default)
_POST_WAKE_MIN_SPEECH_DURATION = 0.3   # seconds above threshold to confirm speech

# Ambient-speech rejection: openWakeWord scores one ~80ms frame at a time, and a
# stray phoneme in background conversation can spike a single frame over the
# threshold. A real utterance of the phrase holds the score high across several
# consecutive frames, so we require N-in-a-row above threshold before firing.
# This is the primary lever against unintended triggers on ambient talk.
_DEFAULT_CONFIRMATION_FRAMES = 3

# Dead-mic detection: an int16 stream whose peak stays at/below this for this
# many consecutive seconds is flagged as silent. Desktop push-to-talk and the
# backend listener use different capture paths, so one can work while the
# backend-selected stream is all zeros.
_SILENCE_PEAK = 10
_SILENCE_ALERT_SECONDS = 10


class WakeWordInUse(RuntimeError):
    """Raised when another surface or process owns the wake-word listener."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "surface": "auto",
    "input_device": None,
    # Where PCM is captured:
    #   "local"  — PortAudio on the backend host (historic default)
    #   "client" — desktop/TUI streams int16 frames via wake.feed
    #   "auto"   — local when a device exists, else client capture
    "capture": "auto",
    "provider": "openwakeword",
    "phrase": "hey hermes",
    "sensitivity": 0.6,
    "confirmation_frames": _DEFAULT_CONFIRMATION_FRAMES,
    "start_new_session": True,
}

# Bundled "hey hermes" model (tools/wakewords/) — the default, so the wake word
# works out of the box. Config names in _ALIASES resolve to it, not a built-in.
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
    """The openWakeWord backend to use on this platform.

    openWakeWord's ONNX backend produces near-zero scores on macOS ARM64 — its
    shared *embedding* model is the broken stage (the melspectrogram front-end
    and the wake classifier both match tflite exactly). The detector arms, the
    microphone works, and no phrase can ever cross the threshold. Prefer the
    tflite backend there; ONNX stays the default everywhere else.

    Upstream: https://github.com/dscripka/openWakeWord/issues/336
    """
    return "tflite" if _is_macos_arm64() else "onnx"


_warned_onnx_coerced = False


def resolve_inference_framework(cfg: Dict[str, Any]) -> str:
    """Resolve the effective openWakeWord backend from config.

    Honors an explicit ``openwakeword.inference_framework`` — EXCEPT the one
    combination that is provably dead: an explicit ``onnx`` on macOS ARM64,
    where ONNX's embedding model never lets a phrase cross threshold (upstream
    #336). Existing macOS users who pinned ``onnx`` before the tflite fix landed
    would otherwise keep a wake word that arms but never fires. Coerce that one
    case to tflite (with a one-time warning) instead of silently shipping a dead
    ear. Every other explicit value is respected as-is; empty falls back to the
    platform default.
    """
    global _warned_onnx_coerced

    sub = cfg.get("openwakeword") if isinstance(cfg.get("openwakeword"), dict) else {}
    framework = str(sub.get("inference_framework") or "").strip().lower()

    if not framework:
        return default_inference_framework()

    if framework == "onnx" and _is_macos_arm64():
        if not _warned_onnx_coerced:
            _warned_onnx_coerced = True
            logger.warning(
                "wake: openwakeword.inference_framework='onnx' is set but ONNX's "
                "embedding model never fires on macOS ARM64 (openWakeWord #336) — "
                "using tflite instead. Set inference_framework to '' (auto) or "
                "'tflite' in config.yaml to silence this."
            )
        return "tflite"

    return framework



def ensure_tflite_runtime() -> bool:
    """Make ``import tflite_runtime.interpreter`` resolve, returning success.

    openWakeWord hardcodes that import but only declares ``tflite-runtime`` for
    ``platform_system == "Linux"``; on macOS the equivalent wheel is
    ``ai-edge-litert``. Alias the module so the upstream import succeeds. The
    alias is process-local — nothing is written to site-packages.
    """
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


def _input_device(cfg: Dict[str, Any]) -> int | str | None:
    """Configured PortAudio input selector, preserving indices and names."""
    raw = _get(cfg, "input_device")
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    value = str(raw).strip()
    return value or None


def _sensitivity(cfg: Dict[str, Any]) -> float:
    raw = _get(cfg, "sensitivity")
    try:
        s = float(raw)
    except (TypeError, ValueError):
        s = float(_DEFAULTS["sensitivity"])
    return min(max(s, 0.0), 1.0)


def _confirmation_frames(cfg: Dict[str, Any]) -> int:
    """How many consecutive over-threshold frames are required to fire.

    ``1`` restores the old single-frame behaviour; higher values reject
    ambient-speech blips at the cost of a few tens of ms of extra latency.
    Clamped to a sane 1..10.
    """
    raw = _get(cfg, "confirmation_frames")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = _DEFAULT_CONFIRMATION_FRAMES
    return min(max(n, 1), 10)


def wake_phrase(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Human-facing wake phrase label (purely cosmetic; engine keys detection)."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    return str(_get(cfg, "phrase")) or "hey hermes"


def resolve_capture_mode(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    prefer_client: bool = False,
    force_local: bool = False,
) -> str:
    """Return ``local`` or ``client`` capture mode for this arm.

    ``prefer_client`` is set by remote desktop (Mac mic, headless backend).
    ``force_local`` keeps CLI/TUI on the process mic. Config ``capture`` is
    ``auto`` | ``local`` | ``client``.
    """
    cfg = cfg if cfg is not None else load_wake_word_config()
    if force_local:
        return "local"
    raw = str(_get(cfg, "capture") or "auto").strip().lower()
    if raw in ("client", "remote", "external"):
        return "client"
    if raw == "local":
        return "local"
    # auto: a working backend input always wins so local desktops keep
    # PortAudio and the configured ``input_device`` selection. Client capture
    # is the fallback for a preferring surface (desktop remote) on a backend
    # with no usable mic — the headless VPS/Cloud case.
    if _local_input_device_ready():
        return "local"
    if prefer_client:
        return "client"
    # No local mic and no client preference (CLI/TUI): stay local so status
    # reports the real requirement instead of advertising a capture path
    # nothing will feed.
    return "local"


def _local_input_device_ready() -> bool:
    """True when PortAudio is importable and at least one input device exists."""
    try:
        sd, _ = _import_audio()
    except (ImportError, OSError):
        return False
    try:
        devices = sd.query_devices()
    except Exception:
        return False
    if isinstance(devices, dict):
        return int(devices.get("max_input_channels") or 0) > 0
    try:
        for dev in devices:
            channels = dev.get("max_input_channels") if isinstance(dev, dict) else None
            if channels is None:
                channels = getattr(dev, "max_input_channels", 0)
            if int(channels or 0) > 0:
                return True
    except Exception:
        return False
    # Also accept a resolvable default input (some hosts list devices oddly).
    try:
        info = sd.query_devices(None, "input")
        channels = info.get("max_input_channels") if isinstance(info, dict) else 0
        return int(channels or 0) > 0
    except Exception:
        return False


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
# Multi-profile phrase enrollment (open-vocabulary routing)
# ---------------------------------------------------------------------------

def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def enrolled_profile_phrases() -> Dict[str, str]:
    """Map ``profile name -> wake phrase`` for every wake-enabled profile.

    Reads each profile's own ``config.yaml`` raw (cheap, no full config merge).
    A profile is enrolled when its ``wake_word.enabled`` is truthy; its phrase
    defaults to ``"hey <profile>"`` when unset. Used by the sherpa engine to
    listen for every enrolled profile's phrase at once and route the wake to
    the matching profile. Best-effort: unreadable profiles are skipped.
    """
    phrases: Dict[str, str] = {}
    try:
        from hermes_cli.config import read_user_config_raw
        from hermes_cli.profiles import get_profile_dir, list_profiles

        for info in list_profiles():
            name = getattr(info, "name", None) or str(info)
            try:
                # Multi-profile read: load_config() targets the ACTIVE
                # profile's home, so read each profile's file directly.
                cfg_path = Path(get_profile_dir(name)) / "config.yaml"
                raw = read_user_config_raw(cfg_path)
                wc = raw.get("wake_word") or {}
                if not isinstance(wc, dict) or not wc.get("enabled"):
                    continue
                phrase = str(wc.get("phrase") or f"hey {name}").strip()
                if phrase:
                    phrases[name] = phrase
            except Exception:
                continue
    except Exception:
        pass
    return phrases


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


def _describe_input_device(sd, selector: int | str | None) -> Dict[str, Any]:
    """Resolve a PortAudio selector into JSON-safe diagnostics.

    Device discovery is diagnostic only. ``InputStream`` remains the authority
    on whether the selected device can actually open at the requested format.
    """
    details: Dict[str, Any] = {"selector": selector}
    try:
        info = sd.query_devices(selector, "input")
    except Exception as e:
        details["error"] = str(e)
        return details

    if isinstance(info, dict):
        name = info.get("name")
        if name:
            details["name"] = str(name)
        channels = info.get("max_input_channels")
        if isinstance(channels, (int, float)):
            details["max_input_channels"] = int(channels)
        rate = info.get("default_samplerate")
        if isinstance(rate, (int, float)):
            details["default_samplerate"] = float(rate)
        hostapi_index = info.get("hostapi")
        if isinstance(hostapi_index, (int, float)):
            details["hostapi_index"] = int(hostapi_index)
            try:
                hostapi = sd.query_hostapis(int(hostapi_index))
                hostapi_name = hostapi.get("name") if isinstance(hostapi, dict) else None
                if hostapi_name:
                    details["hostapi"] = str(hostapi_name)
            except Exception:
                pass

    return details


def _device_label(details: Dict[str, Any]) -> str:
    name = str(details.get("name") or "").strip()
    selector = details.get("selector")
    label = name or ("system default" if selector is None else str(selector))
    hostapi = str(details.get("hostapi") or "").strip()
    return f"{label} ({hostapi})" if hostapi else label


def _capture_sample_rate(details: Dict[str, Any]) -> int:
    """Use the selected device's native rate when PortAudio reports one."""
    rate = details.get("default_samplerate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
        try:
            return int(round(rate))
        except (OverflowError, ValueError):
            pass
    return SAMPLE_RATE


def _resample_audio_frame(np, frame, output_length: int):
    """Convert one native-rate capture block to an exact engine frame."""
    source = np.asarray(frame, dtype=np.float64).reshape(-1)
    if source.size == output_length:
        return np.asarray(frame, dtype=np.int16).reshape(-1)
    if source.size == 0:
        return np.zeros(output_length, dtype=np.int16)

    if source.size > output_length:
        # Match the desktop wake capture path: average each source window when
        # reducing the rate so speech energy is retained instead of decimating.
        edges = np.linspace(0, source.size, output_length + 1, dtype=np.int64)
        values = np.add.reduceat(source, edges[:-1]) / np.diff(edges)
    else:
        # Unusual low-rate devices need interpolation to reach the 16 kHz
        # frame size expected by every wake-word engine.
        source_positions = np.arange(source.size, dtype=np.float64)
        target_positions = np.linspace(0, source.size - 1, output_length)
        values = np.interp(target_positions, source_positions, source)

    return np.rint(values).clip(-32768, 32767).astype(np.int16)


def silent_audio_hint(details: Dict[str, Any]) -> str:
    """Platform-specific remediation for an armed stream delivering silence."""
    if sys.platform == "darwin":
        return (
            "Microphone delivers only silence. Grant the Hermes backend "
            "microphone access in System Settings > Privacy & Security > "
            "Microphone, then toggle the wake word."
        )
    if sys.platform == "win32":
        return (
            f"Microphone delivers only silence from {_device_label(details)}. "
            "Set wake_word.input_device to a different PortAudio input device, "
            "then toggle the wake word."
        )
    return (
        f"Microphone delivers only silence from {_device_label(details)}. "
        "Check the selected input device, then toggle the wake word."
    )


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class _Engine:
    """Minimal hotword-engine contract: feed int16 frames, get a bool."""

    frame_length: int = 1280  # 80 ms at 16 kHz

    #: Optional (matched phrase, profile name) of the most recent fire.
    #: Multi-phrase engines (sherpa) set this for profile routing; the
    #: single-phrase engines leave it None (callers fall back to the
    #: configured phrase / active profile).
    last_match: Optional[tuple[str, str]] = None

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
        model_ref = str(sub.get("model") or _BUNDLED_MODEL_NAME).strip()
        framework = resolve_inference_framework(cfg)
        # openWakeWord returns a 0..1 score per frame; sensitivity IS the raw
        # threshold a score must clear. Higher = stricter (fewer false fires).
        # Default 0.6 sits above openWakeWord's permissive 0.5 baseline, which
        # let near-misses like "hey hor" through.
        self._threshold = _sensitivity(cfg)
        self._confirm_needed = _confirmation_frames(cfg)
        self._confirm_streak = 0

        # openWakeWord silently downgrades tflite -> onnx when no tflite runtime
        # imports (model.py). On macOS ARM64 that lands on the backend whose
        # embedding model is broken, so the listener would arm and never fire.
        # Install + bridge the runtime first, and refuse the downgrade rather
        # than ship a dead ear.
        if framework == "tflite" and not ensure_tflite_runtime():
            # Same lazy-install contract as every other backend; the platform
            # gate lives here because dep specs can't carry PEP 508 markers.
            try:
                lazy_deps.ensure("wake.openwakeword.tflite", prompt=False)
            except Exception as e:
                logger.debug("wake word: tflite runtime install failed: %s", e)
            if not ensure_tflite_runtime():
                if _is_macos_arm64():
                    raise RuntimeError(
                        "The wake word needs the tflite backend on this Mac, but its "
                        "runtime is missing. Install it with: pip install ai-edge-litert"
                    )
                logger.warning("wake word: no tflite runtime available — falling back to onnx")
                framework = "onnx"

        # Default (or explicit "hey_hermes") → the bundled model; a built-in name
        # or custom path is used as-is.
        if model_ref.lower() in _BUNDLED_MODEL_ALIASES:
            model_ref = _bundled_wakeword_path(framework)

        # openWakeWord needs its shared feature models (melspectrogram + embedding)
        # for ANY model — download_models() fetches those first on every call, so a
        # custom path must call it too, else a fresh install crashes on a missing
        # melspectrogram.onnx. A built-in name additionally pulls that pretrained
        # model; a path matches nothing in the catalog and is a no-op beyond base.
        try:
            openwakeword.utils.download_models([model_ref])
        except Exception as e:  # pragma: no cover - network/path dependent
            logger.debug("openwakeword model download skipped: %s", e)
        models = [model_ref]

        self._model = Model(wakeword_models=models, inference_framework=framework)
        self._labels = list(self._model.models.keys())

    def process(self, frame) -> bool:
        scores = self._model.predict(frame)
        over = any(score >= self._threshold for score in scores.values())
        # Require N consecutive over-threshold frames: a real phrase holds the
        # score high across frames, a stray ambient phoneme spikes just one.
        if over:
            self._confirm_streak += 1
            if self._confirm_streak >= self._confirm_needed:
                self._confirm_streak = 0
                return True
            return False
        self._confirm_streak = 0
        return False

    def reset(self) -> None:
        # Clears openWakeWord's rolling feature/prediction buffer so stale audio
        # captured before a pause can't re-fire the moment we resume.
        self._confirm_streak = 0
        try:
            self._model.reset()
        except Exception:
            pass

    def close(self) -> None:
        self.reset()


# sherpa-onnx open-vocabulary KWS model: a small streaming zipformer
# transducer. English (GigaSpeech); one-time download, cached under
# HERMES_HOME. Keywords are typed phrases tokenized at RUNTIME — no
# training step, unlike openWakeWord/Porcupine custom models.
_SHERPA_KWS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
)
_SHERPA_KWS_MODEL_DIR = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"


def _sherpa_model_root() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "wakewords"


def _ensure_sherpa_model(root: Optional[Path] = None) -> Path:
    """Download + unpack the sherpa KWS model once; return its directory."""
    root = root or _sherpa_model_root()
    target = root / _SHERPA_KWS_MODEL_DIR
    if (target / "tokens.txt").exists():
        return target
    import tarfile
    import urllib.request

    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{_SHERPA_KWS_MODEL_DIR}.tar.bz2"
    logger.info("wake word: downloading sherpa KWS model (one-time, ~13 MB)")
    urllib.request.urlretrieve(_SHERPA_KWS_MODEL_URL, archive)  # noqa: S310
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(root, filter="data")
    archive.unlink(missing_ok=True)
    if not (target / "tokens.txt").exists():
        raise RuntimeError(f"sherpa KWS model unpack failed: {target}")
    return target


class _SherpaKwsEngine(_Engine):
    """sherpa-onnx open-vocabulary keyword spotting — any typed phrase, zero training.

    The configured ``wake_word.phrase`` is BPE-tokenized at runtime against the
    model's vocabulary, so "hey hermes", "hey coder", or any other phrase works
    immediately. Here ``phrase`` is DETECTION config, not a cosmetic label.
    """

    # sherpa's streaming zipformer consumes arbitrary chunk sizes; 1280
    # samples (80 ms) matches the shared capture path.
    frame_length = 1280

    def __init__(self, cfg: Dict[str, Any]):
        from tools import lazy_deps

        lazy_deps.ensure("wake.sherpa", prompt=False)

        import sherpa_onnx
        from sherpa_onnx import text2token

        sub = cfg.get("sherpa") if isinstance(cfg.get("sherpa"), dict) else {}
        model_dir = str(sub.get("model_dir") or "").strip()
        d = Path(model_dir) if model_dir else _ensure_sherpa_model()
        if not (d / "tokens.txt").exists():
            raise RuntimeError(f"sherpa KWS model not found at {d}")

        # Phrase set: this profile's own phrase, plus — when profile routing is
        # on — every other wake-enabled profile's phrase, so ONE listener can
        # wake any profile ("hey hermes" / "hey coder" / ...). display-name →
        # profile is kept for routing the match back.
        phrase = str(_get(cfg, "phrase") or "hey hermes").strip()
        own_profile = _active_profile_name()
        phrase_map: Dict[str, str] = {phrase: own_profile}
        if bool(cfg.get("profile_routing", True)):
            for prof, p in enrolled_profile_phrases().items():
                phrase_map.setdefault(p.strip(), prof)

        phrases = list(phrase_map)
        # Runtime tokenization of the arbitrary phrases — the open-vocab core.
        tokens = text2token(
            [p.upper() for p in phrases],
            tokens=str(d / "tokens.txt"),
            tokens_type="bpe",
            bpe_model=str(d / "bpe.model"),
        )
        import tempfile

        # sherpa keyword entries reject spaces in the @display-name; underscore
        # them and map display → profile for match routing.
        self._display_to_profile: Dict[str, str] = {}
        kw = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="hermes-kws-", delete=False, encoding="utf-8"
        )
        for p, toks in zip(phrases, tokens):
            display = p.upper().replace(" ", "_")
            self._display_to_profile[display] = phrase_map[p]
            kw.write(" ".join(toks) + f" @{display}\n")
        kw.close()
        self._keywords_file = kw.name
        #: (phrase display name, profile) of the most recent fire, for routing.
        self.last_match: Optional[tuple[str, str]] = None

        # Map the shared 0..1 sensitivity onto sherpa's keywords_threshold.
        # 0.5 lands exactly on sherpa's recommended default (0.25); live TTS
        # matrix testing showed our previous stricter mapping (0.35) missed
        # ~12% of true positives while 0.25 held zero false fires.
        threshold = 0.05 + 0.4 * _sensitivity(cfg)

        def _model_file(pattern: str) -> str:
            hits = sorted(d.glob(pattern))
            if not hits:
                raise RuntimeError(f"sherpa KWS model file missing: {d}/{pattern}")
            return str(hits[0])

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(d / "tokens.txt"),
            encoder=_model_file("encoder-*[!8].onnx"),
            decoder=_model_file("decoder-*[!8].onnx"),
            joiner=_model_file("joiner-*[!8].onnx"),
            keywords_file=self._keywords_file,
            keywords_threshold=threshold,
            num_threads=1,
        )
        self._stream = self._spotter.create_stream()

    def process(self, frame) -> bool:
        import numpy as np

        samples = np.asarray(frame, dtype=np.float32) / 32768.0
        self._stream.accept_waveform(SAMPLE_RATE, samples)
        fired = False
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                fired = True
                display = str(result)
                self.last_match = (
                    display.replace("_", " ").lower(),
                    self._display_to_profile.get(display, ""),
                )
                # Reset decoder state so one utterance can't fire repeatedly.
                self._spotter.reset_stream(self._stream)
        return fired

    def reset(self) -> None:
        # Fresh stream drops all buffered audio/decoder state (pause → resume
        # must not re-fire on stale audio).
        try:
            self._stream = self._spotter.create_stream()
        except Exception:
            pass

    def close(self) -> None:
        try:
            os.unlink(self._keywords_file)
        except OSError:
            pass


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
        # Porcupine's `sensitivities` runs the OPPOSITE way to our shared knob:
        # per Picovoice, higher = more true positives AND more false alarms
        # (looser). Our config contract is "higher = stricter" everywhere, so
        # invert it here to keep one consistent meaning across all engines.
        porcupine_sensitivity = 1.0 - _sensitivity(cfg)

        kwargs: Dict[str, Any] = {"access_key": access_key, "sensitivities": [porcupine_sensitivity]}
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
    if provider in ("sherpa", "sherpa-onnx", "kws", "open"):
        return _SherpaKwsEngine(cfg)
    if provider in ("openwakeword", "oww", "local"):
        return _OpenWakeWordEngine(cfg)
    raise ValueError(f"Unknown wake_word provider: {provider!r}")


# ---------------------------------------------------------------------------
# Requirements probe (for /wake status + enable path)
# ---------------------------------------------------------------------------

def _stt_ready() -> bool:
    """Is a speech-to-text provider configured and enabled?

    A wake without STT arms the mic but every captured utterance dies at
    transcription — a useless (and confusing) experience. Same standard as
    voice mode's ``check_voice_requirements``: enabled + a real provider.
    """
    try:
        from tools.transcription_tools import _get_provider, _load_stt_config, is_stt_enabled

        stt_config = _load_stt_config()
        return is_stt_enabled(stt_config) and _get_provider(stt_config) != "none"
    except Exception:
        return False


def _tts_ready() -> bool:
    """Can the configured text-to-speech provider run (or install at first use)?

    The wake flow is fully hands-free (wake → speak → hear the reply); without
    TTS the reply is silent and the loop is pointless.

    PROBE, not an installer: ``check_tts_requirements`` lazily pip-installs the
    provider SDK via ``_import_*`` → ``lazy_deps.ensure`` — running that inside
    a status poll froze wake.status for the length of a pip install (and a
    failed install marked the wake word unavailable, unmounting the desktop
    ear). When the provider's deps aren't installed yet, "installable at first
    use" counts as ready and we never touch pip from here.
    """
    try:
        from tools.tts_tool import _get_provider, _load_tts_config

        provider = _get_provider(_load_tts_config())
    except Exception:
        return False

    _LAZY_TTS_FEATURES = {
        "edge": "tts.edge",
        "elevenlabs": "tts.elevenlabs",
        "mistral": "tts.mistral",
    }
    feature = _LAZY_TTS_FEATURES.get(provider)
    if feature is not None:
        try:
            from tools import lazy_deps

            if not lazy_deps.is_available(feature):
                # Not installed: ready iff it can install at first speak.
                return lazy_deps._allow_lazy_installs()
        except Exception:
            return False

    try:
        from tools.tts_tool import check_tts_requirements

        return bool(check_tts_requirements())
    except Exception:
        return False


def check_wake_word_requirements(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Report whether wake-word detection can run, with a remediation hint."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    provider = _provider(cfg)
    from tools import lazy_deps

    if provider == "porcupine":
        feature = "wake.porcupine"
    elif provider in ("sherpa", "sherpa-onnx", "kws", "open"):
        feature = "wake.sherpa"
    else:
        feature = "wake.openwakeword"
    deps_ok = lazy_deps.is_available(feature)
    lazy_ok = lazy_deps._allow_lazy_installs()
    # The audio probe imports sounddevice + numpy — two of the very packages
    # the lazy installer would fetch — so it can only be trusted once the
    # feature's deps are installed. On a fresh install (deps missing, lazy
    # installs allowed) we defer the mic check: the engine constructors call
    # ``lazy_deps.ensure()`` and the stream-open surfaces any real audio
    # problem. Gating ``available`` on the probe here made the lazy-install
    # path unreachable (the probe always failed before ensure() could run).
    audio_ok = _audio_available() if deps_ok else False
    key_ok = True
    # The full wake loop is wake → record → STT → agent → TTS. Arming without
    # either end configured gives a mic that hears you and then does nothing
    # the user can perceive — refuse with a pointer instead.
    stt_ok = _stt_ready()
    tts_ok = _tts_ready()
    hint = ""

    # The tflite backend needs a runtime openWakeWord doesn't declare off Linux.
    # Report it as a real remediation instead of arming a detector that can't fire.
    tflite_ok = True
    if provider not in ("porcupine", "sherpa", "sherpa-onnx", "kws", "open"):
        framework = resolve_inference_framework(cfg)
        if framework == "tflite":
            tflite_ok = ensure_tflite_runtime() or lazy_deps.is_available("wake.openwakeword.tflite") or lazy_ok

    if provider == "porcupine" and not (os.getenv("PORCUPINE_ACCESS_KEY") or "").strip():
        key_ok = False
        hint = "Set PORCUPINE_ACCESS_KEY (free key at https://console.picovoice.ai)."
    elif not deps_ok and not lazy_ok:
        hint = lazy_deps.feature_install_command(feature) or ""
    elif not tflite_ok:
        hint = "The wake word needs the tflite runtime on this Mac: pip install ai-edge-litert"
    elif deps_ok and not audio_ok and resolve_capture_mode(cfg) == "local":
        hint = "Microphone capture needs sounddevice + numpy and a working audio device."
    elif not stt_ok or not tts_ok:
        missing = " and ".join(
            name for name, ok in (("speech-to-text", stt_ok), ("text-to-speech", tts_ok)) if not ok
        )
        hint = (f"Wake word needs {missing} configured — run `hermes tools` "
                f"(Voice section) or see the voice-mode docs.")

    capture_mode = resolve_capture_mode(cfg)
    local_input_ok = _local_input_device_ready() if deps_ok else False
    # Client capture needs deps (engine) but not a server-side PortAudio device.
    if capture_mode == "client":
        mic_ok = deps_ok or (not deps_ok and lazy_ok)
        if deps_ok and not hint:
            # No server mic required; clear the local-device hint if that was set.
            if hint.startswith("Microphone capture needs"):
                hint = ""
    else:
        mic_ok = (deps_ok and audio_ok) or (not deps_ok and lazy_ok)
        if deps_ok and not audio_ok and not hint:
            hint = (
                "No local microphone on this backend. Remote desktop can stream "
                "the client mic — set wake_word.capture: client or use a desktop "
                "build with client-capture wake support."
            )

    return {
        "available": key_ok and stt_ok and tts_ok and tflite_ok and mic_ok,
        "provider": provider,
        "deps_available": deps_ok,
        "audio_available": audio_ok,
        "local_input_available": local_input_ok,
        "capture": capture_mode,
        "access_key_set": key_ok,
        "stt_available": stt_ok,
        "tts_available": tts_ok,
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
                 on_failure: Optional[Callable[["WakeWordDetector"], None]] = None,
                 input_device: int | str | None = None,
                 external_audio: bool = False):
        self.engine = engine
        self.on_wake = on_wake
        self.cooldown = cooldown
        self.on_failure = on_failure
        self.input_device = input_device
        self.external_audio = bool(external_audio)
        self.input_device_details: Dict[str, Any] = (
            {"selector": "client", "name": "client capture", "hostapi": "remote"}
            if self.external_audio
            else {"selector": input_device}
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._callback_inflight = threading.Event()
        self._last_fire = 0.0
        self._lock = threading.Lock()
        # Client-capture PCM queue (int16 mono frames). Local mode ignores this.
        import queue as _queue

        self._audio_q: "_queue.Queue[Any]" = _queue.Queue(maxsize=64)
        # True when the stream is open but every frame is (near-)silence.
        # Surfaced via wake.status / /wake status so users can tell "armed"
        # from "deaf".
        self.audio_silent = False
        self._silent_frames = 0

        # Post-wake capture state. When set, the read loop transitions out
        # of wake-detection mode and into capture mode: it stops calling
        # engine.process(frame), instead appending frames to a buffer, and
        # fires ``_capture_done_callback(audio_array, sample_rate)`` when
        # silence is detected or the duration cap elapses. The detector's
        # mic stream stays open the whole time — no close/reopen, no
        # CoreAudio AUHAL race.
        self._capture_active = False
        self._capture_done_callback: Optional[Any] = None
        self._capture_buffer: list = []

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def feed(self, pcm_int16) -> None:
        """Enqueue one int16 mono frame (or raw bytes) for client capture.

        Frame length should match ``engine.frame_length`` (typically 1280 samples
        at 16 kHz). Short frames are zero-padded; long frames are split.
        """
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
        # Split / pad into engine frames
        offset = 0
        n = int(arr.shape[0])
        while offset < n:
            chunk = arr[offset : offset + fl]
            offset += fl
            if chunk.shape[0] < fl:
                pad = np.zeros(fl, dtype=np.int16)
                pad[: chunk.shape[0]] = chunk
                chunk = pad
            try:
                self._audio_q.put_nowait(chunk)
            except Exception:
                # Drop oldest on overflow so we stay real-time
                try:
                    self._audio_q.get_nowait()
                except Exception:
                    pass
                try:
                    self._audio_q.put_nowait(chunk)
                except Exception:
                    pass

    def start(self) -> None:
        """Open the mic (or client feeder) and begin listening. Idempotent."""
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

    def start_post_wake_capture(
        self,
        on_done: Callable[["Any", int], None],
        *,
        silence_rms: int = _POST_WAKE_SILENCE_RMS,
        silence_duration: float = _POST_WAKE_SILENCE_DURATION,
        max_duration: float = _POST_WAKE_MAX_DURATION,
        min_speech_duration: float = _POST_WAKE_MIN_SPEECH_DURATION,
    ) -> bool:
        """After a wake fires, switch the read loop into capture mode.

        The detector's mic stream stays open. Instead of forwarding frames to
        the wake engine, frames are appended to an internal buffer. When
        silence (RMS < ``silence_rms``) is sustained for ``silence_duration``
        seconds — or ``max_duration`` elapses — ``on_done(audio_array,
        sample_rate)`` is called from the wake thread with the captured
        int16 mono audio and its sample rate (16 kHz).

        Must be called from the wake callback thread (the ``on_wake``
        callback runs on a daemon thread started by ``_dispatch_wake``). If
        the detector is not currently running, returns ``False`` so the
        caller can fall back to opening its own mic.

        Returns ``True`` if capture was armed.
        """
        if not self.running:
            return False
        with self._lock:
            self._capture_active = True
            self._capture_done_callback = on_done
            self._capture_buffer = []
            self._capture_silence_rms = silence_rms
            self._capture_silence_duration = silence_duration
            self._capture_max_duration = max_duration
            self._capture_min_speech_duration = min_speech_duration
            self._capture_speech_start = 0.0
            self._capture_has_spoken = False
            self._capture_silence_start = 0.0
            self._capture_start_time = time.monotonic()
            self._capture_dip_start = 0.0
            # Re-arm wake cooldown so a residual phoneme tail of "amos"
            # can't re-fire wake detection before the user finishes their
            # command. The reader thread sees _capture_active=True and
            # stops calling engine.process(frame) entirely.
            self._last_fire = time.monotonic()
        logger.info(
            "wake word: post-wake capture armed (silence_rms=%d, silence_duration=%.1fs, max=%.1fs)",
            silence_rms, silence_duration, max_duration,
        )
        return True

    def cancel_post_wake_capture(self) -> None:
        """Drop any in-flight post-wake capture without firing the callback."""
        with self._lock:
            if not self._capture_active:
                return
            self._capture_active = False
            self._capture_done_callback = None
            self._capture_buffer = []
            logger.debug("wake word: post-wake capture cancelled")

    def _capture_tick(self, frame: "Any", np: "Any"):
        """Per-frame capture logic. Appends the frame, runs VAD, returns
        ``(audio_array, sample_rate)`` when capture should end, else None.
        """
        # Phase 1: under lock, append + decide whether to end.
        should_finalize = False
        try:
            with self._lock:
                if not self._capture_active:
                    return None
                self._capture_buffer.append(frame)
                now = time.monotonic()
                elapsed = now - self._capture_start_time

                # Hard cap.
                if elapsed >= self._capture_max_duration:
                    logger.info(
                        "wake word: post-wake capture reached max duration (%.1fs)",
                        self._capture_max_duration,
                    )
                    should_finalize = True

                if not should_finalize:
                    # Compute RMS for VAD. int16 mono, frame is already
                    # resampled to SAMPLE_RATE / engine.frame_length.
                    rms_sq = float(np.mean(frame.astype(np.float64) ** 2))
                    rms = int(rms_sq ** 0.5)
                    threshold = self._capture_silence_rms

                    if rms > threshold:
                        # Audio above silence threshold — track speech start.
                        self._capture_dip_start = 0.0
                        if self._capture_speech_start == 0.0:
                            self._capture_speech_start = now
                        elif (
                            not self._capture_has_spoken
                            and now - self._capture_speech_start >= self._capture_min_speech_duration
                        ):
                            self._capture_has_spoken = True
                        if self._capture_has_spoken:
                            self._capture_silence_start = 0.0
                    elif self._capture_has_spoken:
                        # Below threshold after speech was confirmed — start
                        # the silence timer. Brief dips during speech don't
                        # count, mirroring AudioRecorder's VAD behavior.
                        if self._capture_silence_start == 0.0:
                            self._capture_silence_start = now
                        elif now - self._capture_silence_start >= self._capture_silence_duration:
                            logger.info(
                                "wake word: post-wake capture silence detected (%.1fs)",
                                self._capture_silence_duration,
                            )
                            should_finalize = True
                    elif self._capture_speech_start > 0:
                        # We were in a speech attempt but dipped.
                        if self._capture_dip_start == 0.0:
                            self._capture_dip_start = now
                        elif now - self._capture_dip_start >= 0.3:
                            # Tolerable dip window passed; reset speech attempt.
                            self._capture_speech_start = 0.0
                            self._capture_dip_start = 0.0
        except Exception as e:
            logger.warning("wake word: post-wake capture tick error: %s", e)
            return None

        # Phase 2: finalize OUTSIDE the lock (which is re-acquired inside
        # _finalize_capture). This avoids the deadlock where _capture_tick
        # holds self._lock and tries to re-acquire it in _finalize_capture.
        if should_finalize:
            return self._finalize_capture(np)
        return None

    def _finalize_capture(self, np: "Any"):
        """Stop capturing, concatenate the buffer, return (audio, sample_rate).

        Caller must NOT hold ``self._lock`` — this acquires it.
        """
        with self._lock:
            if not self._capture_active:
                return None
            self._capture_active = False
            buf = self._capture_buffer
            self._capture_buffer = []
        if not buf:
            return np.zeros(0, dtype=np.int16), SAMPLE_RATE
        try:
            audio = np.concatenate(buf).astype(np.int16, copy=False)
        except Exception as e:
            logger.warning("wake word: capture concatenate failed: %s", e)
            return np.zeros(0, dtype=np.int16), SAMPLE_RATE
        return audio, SAMPLE_RATE

    @staticmethod
    def _invoke_capture_callback(cb: Callable[["Any", int], None],
                                 audio: "Any", sample_rate: int) -> None:
        """Run the capture-done callback in a daemon thread with full error guarding."""
        try:
            cb(audio, sample_rate)
        except Exception as e:
            logger.error("wake word: post-wake capture callback failed: %s", e, exc_info=True)

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
        frame_length = self.engine.frame_length
        capture_frame_length = frame_length
        capture_rate = SAMPLE_RATE
        np = None
        stream = None

        if self.external_audio:
            # Drain any stale frames from a previous arm.
            try:
                while True:
                    self._audio_q.get_nowait()
            except Exception:
                pass
            logger.info(
                "wake word: client-capture mode (frame=%d, rate=%d) — waiting for wake.feed",
                frame_length, SAMPLE_RATE,
            )
        else:
            try:
                sd, np = _import_audio()
            except (ImportError, OSError) as e:
                logger.error("wake word: audio libraries unavailable: %s", e)
                startup_errors.append(e)
                ready.set()
                return

            self.input_device_details = _describe_input_device(sd, self.input_device)
            capture_rate = _capture_sample_rate(self.input_device_details)
            capture_frame_length = max(
                1, int(round(frame_length * capture_rate / SAMPLE_RATE))
            )
            logger.info(
                "wake word: opening microphone device=%s selector=%r hostapi=%s "
                "default_rate=%s capture_rate=%d engine_rate=%d",
                self.input_device_details.get("name") or "system default",
                self.input_device,
                self.input_device_details.get("hostapi") or "unknown",
                self.input_device_details.get("default_samplerate") or "unknown",
                capture_rate,
                SAMPLE_RATE,
            )
            try:
                stream = sd.InputStream(
                    device=self.input_device,
                    samplerate=capture_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=capture_frame_length,
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

        logger.info("wake word: listening (frame=%d, rate=%d, external=%s)",
                    frame_length, SAMPLE_RATE, self.external_audio)
        ready.set()
        failed = False
        # ~seconds of consecutive near-zero frames before we flag the stream
        # as silent.
        silent_alert_frames = max(1, int(_SILENCE_ALERT_SECONDS * SAMPLE_RATE / max(1, frame_length)))
        try:
            while not self._stop.is_set():
                try:
                    if self.external_audio:
                        try:
                            frame = self._audio_q.get(timeout=0.25)
                        except Exception:
                            # No client frames yet — count as silence for status.
                            self._silent_frames += 1
                            if self._silent_frames == silent_alert_frames:
                                self.audio_silent = True
                            continue
                        data = frame
                    else:
                        data, _overflow = stream.read(capture_frame_length)
                except Exception as e:
                    logger.warning("wake word: stream read error: %s", e)
                    failed = not self._stop.is_set()
                    break
                frame = data[:, 0] if getattr(data, "ndim", 1) == 2 else data
                if capture_rate != SAMPLE_RATE:
                    frame = _resample_audio_frame(np, frame, frame_length)
                try:
                    peak = int(abs(frame).max()) if len(frame) else 0
                except Exception:
                    peak = _SILENCE_PEAK + 1
                if peak <= _SILENCE_PEAK:
                    self._silent_frames += 1
                    if self._silent_frames == silent_alert_frames:
                        self.audio_silent = True
                        logger.warning(
                            "wake word: mic delivers only silence (peak<=%d for %ds); %s",
                            _SILENCE_PEAK, _SILENCE_ALERT_SECONDS,
                            silent_audio_hint(self.input_device_details),
                        )
                elif self._silent_frames:
                    if self.audio_silent:
                        logger.info("wake word: mic audio detected — stream healthy")
                    self._silent_frames = 0
                    self.audio_silent = False

                # Post-wake capture branch: when armed, the read loop stays
                # alive on the SAME mic stream (no close/reopen → no AUHAL
                # race). Frames accumulate into a buffer; silence or a hard
                # duration cap fires the done-callback with the captured
                # audio. The wake engine is paused for the duration so a
                # residual phoneme tail of "amos" can't immediately refire.
                if self._capture_active:
                    done = self._capture_tick(frame, np)
                    if done is not None:
                        # Callback is fired on a daemon thread so the read
                        # loop is free to immediately re-arm for the next
                        # wake — the user can keep issuing voice commands
                        # without a 200ms thread-spin penalty.
                        audio, sr = done
                        cb = self._capture_done_callback
                        self._capture_done_callback = None
                        if cb is not None:
                            threading.Thread(
                                target=self._invoke_capture_callback,
                                args=(cb, audio, sr),
                                daemon=True,
                                name="wake-word-capture-done",
                            ).start()
                        # Continue the read loop; do not exit on capture done.
                    continue

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
            if stream is not None:
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
    external_audio: bool = False,
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
            detector = WakeWordDetector(
                engine,
                on_wake,
                on_failure=_detector_failed,
                input_device=_input_device(cfg),
                external_audio=external_audio,
            )
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


def start_post_wake_capture(
    on_done: Callable[["Any", int], None],
    *,
    owner: object,
    **kwargs,
) -> bool:
    """Arm post-wake capture on the singleton detector.

    If the wake listener was started by ``owner``, switches its read loop
    into capture mode on the SAME mic stream (no close/reopen, so no
    CoreAudio AUHAL race). The detector's wake engine is paused for the
    duration; once capture ends, the engine resumes automatically on the
    next frame.

    Returns True on success, False if the detector isn't owned by ``owner``
    or isn't running.
    """
    with _detector_lock:
        if _detector is None or _detector_owner is not owner:
            return False
        det = _detector
    return det.start_post_wake_capture(on_done, **kwargs)


def cancel_post_wake_capture(*, owner: object) -> bool:
    """Cancel any in-flight post-wake capture without firing the callback."""
    with _detector_lock:
        if _detector is None or _detector_owner is not owner:
            return False
        _detector.cancel_post_wake_capture()
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


def audio_is_silent() -> bool:
    """True when the armed stream has delivered only silence (dead mic).

    The stream opens fine but every frame is zeros, so detection can never
    fire. Lets status surfaces show "listening but the microphone appears
    silent" instead of a healthy state.
    """
    with _detector_lock:
        det = _detector
    return det is not None and det.audio_silent


def get_input_device_status(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return configured/active PortAudio input diagnostics for status UIs."""
    with _detector_lock:
        det = _detector
    if det is not None:
        return dict(det.input_device_details)

    cfg = cfg if cfg is not None else load_wake_word_config()
    selector = _input_device(cfg)
    try:
        sd, _ = _import_audio()
    except (ImportError, OSError) as e:
        return {"selector": selector, "error": str(e)}
    return _describe_input_device(sd, selector)


def get_last_match() -> Optional[tuple[str, str]]:
    """(matched phrase, profile) of the most recent wake fire, if the engine
    reports per-phrase matches (sherpa multi-profile routing). None otherwise."""
    with _detector_lock:
        det = _detector
    if det is None:
        return None
    return getattr(det.engine, "last_match", None)


def feed_audio(*, owner: object, pcm_int16) -> bool:
    """Push client-captured PCM into the armed detector (client capture mode).

    Returns True when the frame was accepted for ``owner``'s armed detector.
    """
    with _detector_lock:
        if _detector is None or _detector_owner is not owner:
            return False
        if not _detector.external_audio:
            return False
        det = _detector
    det.feed(pcm_int16)
    return True


def detector_frame_info() -> Dict[str, Any]:
    """Sample rate + frame length for client capture streamers."""
    with _detector_lock:
        det = _detector
    if det is None:
        return {"sample_rate": SAMPLE_RATE, "frame_length": 1280}
    return {
        "sample_rate": SAMPLE_RATE,
        "frame_length": int(getattr(det.engine, "frame_length", 1280) or 1280),
        "external_audio": bool(det.external_audio),
    }
