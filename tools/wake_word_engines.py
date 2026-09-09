"""Wake-word hotword engines (pyopen-wakeword / sherpa-onnx KWS / Porcupine).

All three run fully on-device. Config, platform probes and sensitivity accessors
live in :mod:`tools.wake_word`; engines read them lazily through that module (import cycle).
This module is the SINGLE owner of the engine implementations — ``tools.wake_word``
imports these classes and must not shadow them with copies.

Dependency admission: constructing an engine ensures its ``wake-*`` extra. The
``audio-io`` extra (sounddevice + numpy) is ensured only when the resolved
capture mode is ``local`` — client capture (desktop/TUI streaming PCM via
``wake.feed``) never needs, and must never trigger installation of, local
audio libraries.
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("tools.wake_word")


def _ww():
    from tools import wake_word
    return wake_word


def _ensure_dep(feature: str, cfg: Dict[str, Any]) -> None:
    import pm

    pm.ensure_import(feature)
    # Only local capture needs the microphone dependencies.
    if _ww().resolve_capture_mode(cfg) == "local" and not pm.available("audio-io"):
        pm.ensure_import("audio-io")


class _Engine:
    """Minimal hotword-engine contract: feed int16 frames, get a bool. Subclasses set ``feature``
    (the pm extra ensured before ``_build``) and their own ``cfg`` sub-section ``section``."""

    feature: str = ""
    section: str = ""
    frame_length: int = 1280  # 80 ms at 16 kHz

    #: (matched phrase, profile name) of the most recent fire. Multi-phrase engines
    #: (sherpa) set this for profile routing; single-phrase engines leave it None.
    last_match: Optional[tuple[str, str]] = None

    def __init__(self, cfg: Dict[str, Any]):
        _ensure_dep(self.feature, cfg)
        self._build(cfg, _sub(cfg, self.section), _ww())

    def _build(self, cfg: Dict[str, Any], sub: Dict[str, Any], ww) -> None:
        raise NotImplementedError

    def process(self, frame) -> bool:  # frame: 1-D int16 ndarray
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any internal audio/feature buffer (called on every (re)start)."""

    def close(self) -> None:
        """Release engine resources (called once on stop)."""


def _looks_like_path(value: str) -> bool:
    return os.sep in value or value.endswith((".onnx", ".tflite", ".ppn")) or os.path.exists(value)


def _sub(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    sub = cfg.get(key)
    return sub if isinstance(sub, dict) else {}


class _OpenWakeWordEngine(_Engine):
    """pyopen-wakeword — free, local hotword detection (TFLite via a bundled
    tensorflowlite_c lib; no runtime download, no framework choice). Scores one
    ~80 ms frame at a time; ``sensitivity`` IS the raw 0..1 threshold (higher =
    stricter). A real utterance holds the score high across frames while a stray
    phoneme spikes one, so ``confirmation_frames`` hits are required."""

    feature, section = "wake-openwakeword", "openwakeword"

    def _build(self, cfg, sub, ww) -> None:
        from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures

        model_ref = str(sub.get("model") or ww._BUNDLED_MODEL_NAME).strip()
        # Default (or explicit "hey_hermes") → the bundled model; a custom path
        # is used as-is. pyopen-wakeword bundles the shared feature models
        # (melspectrogram + embedding — byte-identical to the openWakeWord
        # v0.5.1 files) inside its wheel, so there is no download_models step.
        if model_ref.lower() in ww._BUNDLED_MODEL_ALIASES:
            model_ref = ww._bundled_wakeword_path()
        # pyopen-wakeword returns a 0..1 score per completed window; sensitivity
        # IS the raw threshold a score must clear. Higher = stricter (fewer
        # false fires). Default 0.6 sits above openWakeWord's permissive 0.5
        # baseline, which let near-misses like "hey hor" through.
        self._threshold = ww._sensitivity(cfg)
        self._confirm_needed = ww._confirmation_frames(cfg)
        self._confirm_streak = 0
        self._features = OpenWakeWordFeatures.from_builtin()
        self._model = OpenWakeWord.from_model(model_ref)
        self._labels = [self._model.id]

    def process(self, frame) -> bool:
        # frame is a 1-D int16 ndarray; the features pipeline consumes int16
        # bytes. process_streaming() yields embeddings as the window fills and
        # the model yields one 0..1 score per completed window.
        over = False
        for emb in self._features.process_streaming(frame.tobytes()):
            for score in self._model.process_streaming(emb):
                if score >= self._threshold:
                    over = True
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
        # Clears pyopen-wakeword's rolling feature/prediction buffer so stale
        # audio captured before a pause can't re-fire the moment we resume.
        self._confirm_streak = 0
        with suppress(Exception):
            self._features.reset()
            self._model.reset()

    def close(self) -> None:
        self.reset()
        with suppress(Exception):
            self._features.close()
            self._model.close()


# sherpa-onnx open-vocabulary KWS model: small streaming zipformer transducer (English,
# GigaSpeech), downloaded once under HERMES_HOME. Keywords are tokenized at RUNTIME.
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
    """sherpa-onnx open-vocabulary keyword spotting — any typed phrase, zero training. ``wake_word.phrase``
    is BPE-tokenized at runtime against the model's vocabulary: DETECTION config, not a cosmetic label."""

    feature, section = "wake-sherpa", "sherpa"
    frame_length = 1280  # streaming zipformer accepts any chunk; match capture path.

    def _build(self, cfg, sub, ww) -> None:
        import sherpa_onnx
        import tempfile
        from sherpa_onnx import text2token
        model_dir = str(sub.get("model_dir") or "").strip()
        d = Path(model_dir) if model_dir else _ensure_sherpa_model()
        if not (d / "tokens.txt").exists():
            raise RuntimeError(f"sherpa KWS model not found at {d}")

        # Phrase set: this profile's phrase plus — with profile routing on — every other
        # wake-enabled profile's phrase, so ONE listener can wake any profile.
        phrase = str(ww._get(cfg, "phrase") or "hey hermes").strip()
        phrase_map: Dict[str, str] = {phrase: ww._active_profile_name()}
        if bool(cfg.get("profile_routing", True)):
            for prof, p in ww.enrolled_profile_phrases().items():
                phrase_map.setdefault(p.strip(), prof)
        phrases = list(phrase_map)
        tokens = text2token([p.upper() for p in phrases], tokens=str(d / "tokens.txt"), tokens_type="bpe",
                            bpe_model=str(d / "bpe.model"))
        # sherpa keyword entries reject spaces in the @display-name; underscore them and
        # map display → profile for match routing.
        self._display_to_profile: Dict[str, str] = {}
        kw = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="hermes-kws-", delete=False,
                                         encoding="utf-8")
        for p, toks in zip(phrases, tokens):
            display = p.upper().replace(" ", "_")
            self._display_to_profile[display] = phrase_map[p]
            kw.write(" ".join(toks) + f" @{display}\n")
        kw.close()
        self._keywords_file = kw.name

        # Shared 0..1 sensitivity → sherpa keywords_threshold. 0.5 lands on sherpa's
        # recommended 0.25; a stricter 0.35 missed ~12% of true positives in live TTS
        # matrix tests while 0.25 held zero false fires.
        threshold = 0.05 + 0.4 * ww._sensitivity(cfg)

        def _model_file(part: str) -> str:
            hits = sorted(d.glob(f"{part}-*[!8].onnx"))
            if not hits:
                raise RuntimeError(f"sherpa KWS model file missing: {d}/{part}-*[!8].onnx")
            return str(hits[0])

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(d / "tokens.txt"), encoder=_model_file("encoder"), decoder=_model_file("decoder"),
            joiner=_model_file("joiner"), keywords_file=self._keywords_file, keywords_threshold=threshold,
            num_threads=1,
        )
        self._stream = self._spotter.create_stream()

    def process(self, frame) -> bool:
        import numpy as np
        self._stream.accept_waveform(_ww().SAMPLE_RATE, np.asarray(frame, dtype=np.float32) / 32768.0)
        fired = False
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                fired, display = True, str(result)
                self.last_match = (display.replace("_", " ").lower(),
                                   self._display_to_profile.get(display, ""))
                self._spotter.reset_stream(self._stream)  # one utterance must not fire repeatedly
        return fired

    def reset(self) -> None:
        # Fresh stream drops buffered audio/decoder state (pause → resume must not re-fire).
        with suppress(Exception):
            self._stream = self._spotter.create_stream()

    def close(self) -> None:
        with suppress(OSError):
            os.unlink(self._keywords_file)


class _PorcupineEngine(_Engine):
    """Picovoice Porcupine — premium, on-device, needs an access key."""

    feature, section = "wake-porcupine", "porcupine"

    def _build(self, cfg, sub, ww) -> None:
        import pvporcupine
        access_key = (os.getenv("PORCUPINE_ACCESS_KEY") or "").strip()
        if not access_key:
            raise RuntimeError("Porcupine wake word requires PORCUPINE_ACCESS_KEY "
                               "(get a free key at https://console.picovoice.ai).")
        keyword = str(sub.get("keyword") or "jarvis").strip()
        # Porcupine's `sensitivities` runs the OPPOSITE way to our shared knob (higher =
        # looser); invert so "higher = stricter" holds for every engine.
        kwargs: Dict[str, Any] = {"access_key": access_key, "sensitivities": [1.0 - ww._sensitivity(cfg)]}
        kwargs["keyword_paths" if _looks_like_path(keyword) else "keywords"] = [keyword]
        self._porcupine = pvporcupine.create(**kwargs)
        self.frame_length = self._porcupine.frame_length

    def process(self, frame) -> bool:
        return self._porcupine.process(frame) >= 0  # pvporcupine wants a plain sequence of int16

    def close(self) -> None:
        with suppress(Exception):
            self._porcupine.delete()
