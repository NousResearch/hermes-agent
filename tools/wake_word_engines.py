"""Wake-word hotword engines (openWakeWord / sherpa-onnx KWS / Porcupine).

All three run fully on-device. Config, platform probes and sensitivity accessors
live in :mod:`tools.wake_word`; engines read them lazily through that module (import cycle).
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("tools.wake_word")


def _ww():
    from tools import wake_word
    return wake_word


def _ensure_dep(feature: str) -> None:
    from tools import lazy_deps
    lazy_deps.ensure(feature, prompt=False)


class _Engine:
    """Minimal hotword-engine contract: feed int16 frames, get a bool. Subclasses set ``feature``
    (lazy_deps name, ensured before ``_build``) and their own ``cfg`` sub-section ``section``."""

    feature: str = ""
    section: str = ""
    frame_length: int = 1280  # 80 ms at 16 kHz

    #: (matched phrase, profile name) of the most recent fire. Multi-phrase engines
    #: (sherpa) set this for profile routing; single-phrase engines leave it None.
    last_match: Optional[tuple[str, str]] = None

    def __init__(self, cfg: Dict[str, Any]):
        _ensure_dep(self.feature)
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
    """openWakeWord — free, local ONNX/tflite hotword detection. Scores one ~80 ms frame at a time;
    ``sensitivity`` IS the raw 0..1 threshold (higher = stricter). A real utterance holds the score
    high across frames while a stray phoneme spikes one, so ``confirmation_frames`` hits are required."""

    feature, section = "wake.openwakeword", "openwakeword"
    frame_length = 1280  # openWakeWord recommends 80 ms frames.

    def _build(self, cfg, sub, ww) -> None:
        import openwakeword
        from openwakeword.model import Model
        model_ref = str(sub.get("model") or ww._BUNDLED_MODEL_NAME).strip()
        framework = self._usable_framework(ww.resolve_inference_framework(cfg))
        self._threshold = ww._sensitivity(cfg)
        self._confirm_needed = ww._confirmation_frames(cfg)
        self._confirm_streak = 0
        # Default (or explicit "hey_hermes") → the bundled model; built-in names / paths as-is.
        if model_ref.lower() in ww._BUNDLED_MODEL_ALIASES:
            model_ref = ww._bundled_wakeword_path(framework)
        # download_models() also fetches the shared feature models (melspectrogram +
        # embedding) needed for ANY model, so a custom path must call it too.
        try:
            openwakeword.utils.download_models([model_ref])
        except Exception as e:  # pragma: no cover - network/path dependent
            logger.debug("openwakeword model download skipped: %s", e)
        self._model = Model(wakeword_models=[model_ref], inference_framework=framework)
        self._labels = list(self._model.models.keys())

    @staticmethod
    def _usable_framework(framework: str) -> str:
        """Refuse openWakeWord's silent tflite→onnx downgrade: without a tflite runtime it falls back
        to onnx, which on macOS ARM64 never fires (armed but deaf). Install + bridge the runtime first
        (gate lives here because dep specs can't carry PEP 508 markers); on that Mac raise instead."""
        ww = _ww()
        if framework != "tflite" or ww.ensure_tflite_runtime():
            return framework
        try:
            _ensure_dep("wake.openwakeword.tflite")
        except Exception as e:
            logger.debug("wake word: tflite runtime install failed: %s", e)
        if ww.ensure_tflite_runtime():
            return framework
        if ww._is_macos_arm64():
            raise RuntimeError("The wake word needs the tflite backend on this Mac, but its "
                               "runtime is missing. Install it with: pip install ai-edge-litert")
        logger.warning("wake word: no tflite runtime available — falling back to onnx")
        return "onnx"

    def process(self, frame) -> bool:
        hit = any(score >= self._threshold for score in self._model.predict(frame).values())
        self._confirm_streak = self._confirm_streak + 1 if hit else 0
        if self._confirm_streak < self._confirm_needed:
            return False
        self._confirm_streak = 0
        return True

    def reset(self) -> None:
        # Clears openWakeWord's rolling feature buffer so stale audio captured before a
        # pause can't re-fire the moment we resume.
        self._confirm_streak = 0
        with suppress(Exception):
            self._model.reset()

    def close(self) -> None:
        self.reset()


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

    feature, section = "wake.sherpa", "sherpa"
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

    feature, section = "wake.porcupine", "porcupine"

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


# ── Whisper (VAD-gated speech-to-text, any language) ─────────────────────────────────────
# The hotword engines above are English-only (openWakeWord/sherpa models) or need a vendor
# console (Porcupine). This one reuses the local faster-whisper STT stack: an RMS gate collects
# one utterance, Whisper transcribes it in the configured language, and the text is matched
# against ``wake_word.phrase``. Idle CPU is ~zero (nothing runs on silence); a short utterance
# costs one tiny/base decode (~0.2-0.4 s on CPU). Any language Whisper knows works.

_WHISPER_PAD_SECONDS = 0.3


def _normalize_speech(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace — so "Ei, Juca!" == "ei juca"."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text).split())


def _phrase_similarity(phrase: str, text: str) -> float:
    """Best match (0..1) of a normalized phrase against any window of the normalized text.

    Exact containment scores 1.0; otherwise every window of ``n-1..n+1`` words (Whisper may
    split or merge a name) is compared without spaces via ``difflib`` so "ei jucá" ≈ "ei juca".
    """
    from difflib import SequenceMatcher
    if not phrase or not text:
        return 0.0
    if phrase in text:
        return 1.0
    words, n = text.split(), len(phrase.split())
    target = phrase.replace(" ", "")
    best = 0.0
    for size in range(max(1, n - 1), n + 2):
        for i in range(0, max(1, len(words) - size + 1)):
            window = "".join(words[i:i + size])
            if window:
                best = max(best, SequenceMatcher(None, target, window).ratio())
    return best


class _WhisperEngine(_Engine):
    """faster-whisper keyword spotting: gate on speech energy, transcribe the utterance, match the
    phrase text. ``wake_word.phrase`` is the DETECTION key (any language Whisper supports)."""

    feature, section = "wake.whisper", "whisper"

    def _build(self, cfg, sub, ww) -> None:
        import numpy as np
        self._np = np
        self._model_name = str(sub.get("model") or "").strip() or _stt_local_model_name() or "tiny"
        self._language = str(sub.get("language") or "").strip() or _stt_language() or None
        self._silence_threshold = float(_num(sub, "silence_threshold", 200.0))
        self._silence_duration = float(_num(sub, "silence_duration", 0.5))
        self._min_speech = float(_num(sub, "min_speech_seconds", 0.3))
        self._max_speech = float(_num(sub, "max_speech_seconds", 3.0))
        # When a window is cut at max_speech_seconds without a match, its last second is carried
        # into the next window so a phrase straddling the boundary is still heard whole.
        self._overlap = float(_num(sub, "window_overlap_seconds", 1.0))
        # 0..1 sensitivity → minimum text similarity (0.6 → 0.79; 1.0 → 0.95; 0.0 → 0.55).
        self._min_similarity = 0.55 + 0.4 * ww._sensitivity(cfg)
        phrase = str(ww._get(cfg, "phrase") or "hey hermes").strip()
        phrase_map: Dict[str, str] = {phrase: ww._active_profile_name()}
        if ww._get(cfg, "profile_routing"):
            for prof, p in ww.enrolled_profile_phrases().items():
                phrase_map.setdefault(p.strip(), prof)
        self._phrases = {_normalize_speech(p): (p, prof) for p, prof in phrase_map.items() if p.strip()}
        self._prompt = ", ".join(p for p, _ in self._phrases.values())
        self._model = _load_whisper_model(self._model_name)
        self.reset()

    # ── audio gate ──
    def process(self, frame) -> bool:
        np = self._np
        frame = np.asarray(frame, dtype=np.int16)
        seconds = len(frame) / _ww().SAMPLE_RATE
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) if len(frame) else 0.0
        loud = rms > self._silence_threshold
        if not loud and not self._buffer:
            return False  # idle silence: nothing to do (and nothing for Whisper to hallucinate on)
        self._buffer.append(frame)
        self._buffered += seconds
        if loud:
            self._speech += seconds
            self._trailing = 0.0
        else:
            self._trailing += seconds
        ended_by_silence = self._trailing >= self._silence_duration
        if not ended_by_silence and self._buffered < self._max_speech:
            return False
        audio, speech = self._buffer, self._speech
        self.reset()
        if speech < self._min_speech:
            return False
        if self._match(self._transcribe(np.concatenate(audio))):
            return True
        if not ended_by_silence and self._overlap > 0:
            # Cut mid-speech with no match: keep the tail so the next decode sees a phrase that
            # straddles the boundary from its start.
            keep = max(1, int(round(self._overlap / seconds))) if seconds > 0 else 0
            tail = audio[-keep:]
            self._buffer = list(tail)
            self._buffered = self._speech = len(tail) * seconds
        return False

    def _transcribe(self, pcm) -> str:
        np = self._np
        pad = np.zeros(int(_WHISPER_PAD_SECONDS * _ww().SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([pad, pcm.astype(np.float32) / 32768.0, pad])
        # The wake phrase is the initial_prompt on purpose: names outside Whisper's vocabulary
        # ("Juca") are otherwise not transcribed at all — ablation with the `base` model on pt-BR
        # TTS: 6/6 positives with the prompt vs 0/6 without; false accepts 4/32 vs 3/32, all on
        # phonetic near-misses ("Ei Luca", sentences containing "Juca"). `sensitivity` tunes that.
        segments, _info = self._model.transcribe(
            audio, language=self._language, beam_size=1, initial_prompt=self._prompt,
            condition_on_previous_text=False, vad_filter=False, without_timestamps=True)
        return " ".join(str(getattr(seg, "text", seg)).strip() for seg in segments)

    def _match(self, text: str) -> bool:
        heard = _normalize_speech(text)
        best, hit = 0.0, None
        for norm, (display, prof) in self._phrases.items():
            score = _phrase_similarity(norm, heard)
            if score > best:
                best, hit = score, (display.lower(), prof)
        logger.debug("wake word: whisper heard %r (best %.2f for %s)", heard, best, hit and hit[0])
        if hit is None or best < self._min_similarity:
            return False
        self.last_match = hit
        return True

    def reset(self) -> None:
        self._buffer: list = []
        self._buffered = self._speech = self._trailing = 0.0

    def close(self) -> None:
        self._model = None


def _num(sub: Dict[str, Any], key: str, default: float) -> float:
    value = sub.get(key)
    return default if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value)


def _stt_config() -> Dict[str, Any]:
    with suppress(Exception):
        from tools.transcription_tools import _load_stt_config
        cfg = _load_stt_config()
        return cfg if isinstance(cfg, dict) else {}
    return {}


def _stt_language() -> str:
    return str(_stt_config().get("language") or "").strip()


def _stt_local_model_name() -> str:
    local = _stt_config().get("local")
    return str(local.get("model") or "").strip() if isinstance(local, dict) else ""


def _load_whisper_model(model_name: str):
    """Reuse the local-STT loader (Hub cache first, Apple Silicon → CPU int8, CUDA fallback)."""
    from tools.transcription_local import _load_local_whisper_model
    return _load_local_whisper_model(model_name)
