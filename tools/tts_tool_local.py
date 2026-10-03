"""Local on-device TTS engines for ``tools.tts_tool``: NeuTTS, Piper, KittenTTS, Kokoro.

All four synthesize WAV natively; :func:`_finalize_wav_output` converts/renames to the requested
container. Piper, KittenTTS and Kokoro keep loaded models in small LRU caches registered in
``_LOCAL_TTS_MODEL_CACHES`` so warm/release can pre-load or drop them. ``_import_piper`` /
``_import_kittentts`` / ``_import_kokoro_onnx`` are resolved through the origin module at call
time (test monkeypatches).
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from tools.tts_tool_delivery import _finalize_wav_output, _origin, _section, _wav_sidecar_path

logger = logging.getLogger("tools.tts_tool")

DEFAULT_KITTENTTS_MODEL = "KittenML/kitten-tts-nano-0.8-int8"  # 25MB
DEFAULT_KITTENTTS_VOICE = "Jasper"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"  # balanced size/quality
_NEUTTS_SAMPLES = Path(__file__).parent / "neutts_samples"

# --- Bounded model caches ---
# Each entry is a whole loaded model (tens of MB); unbounded, one would be pinned per distinct
# voice for the process lifetime. Most sessions use one or two voices; a cold reload is cheap.
_TTS_MODEL_CACHE_MAX = 3

# Provider name -> the cache it populates (warm/release in tts_tool_lifecycle; a new local engine
# adds a row here plus a loader in _local_tts_warmers()). Piper keyed on absolute .onnx path
# (+cuda flag); KittenTTS on model name; Kokoro on (model, voices) path pair.
_piper_voice_cache: Dict[str, Any] = {}
_kittentts_model_cache: Dict[str, Any] = {}
_kokoro_model_cache: Dict[str, Any] = {}
_LOCAL_TTS_MODEL_CACHES: Dict[str, Dict[str, Any]] = {
    "piper": _piper_voice_cache, "kittentts": _kittentts_model_cache, "kokoro": _kokoro_model_cache}


def _tts_cache_get_or_load(cache: Dict[str, Any], key: str, load: Callable[[], Any]) -> Any:
    """Get ``key`` from ``cache`` or load it, LRU-bounded at ``_TTS_MODEL_CACHE_MAX`` (a hit refreshes
    recency via pop + reinsert; eviction only releases the slot, not live references)."""
    if key in cache:
        cache[key] = cache.pop(key)
        return cache[key]
    value = load()
    cache[key] = value
    while len(cache) > _TTS_MODEL_CACHE_MAX:
        cache.pop(next(iter(cache)), None)
    return value


def _run_helper(cmd: list, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout, stdin=subprocess.DEVNULL,
    )


# --- NeuTTS (subprocess via tools/neutts_synth.py so the ~500MB model exits after use) ---
def _generate_neutts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    neutts_config = tts_config.get("neutts") or {}
    wav_path = _wav_sidecar_path(output_path)
    cmd = [
        sys.executable, str(Path(__file__).parent / "neutts_synth.py"),
        "--text", text,
        "--out", wav_path,
        "--ref-audio", neutts_config.get("ref_audio", "") or str(_NEUTTS_SAMPLES / "jo.wav"),
        "--ref-text", neutts_config.get("ref_text", "") or str(_NEUTTS_SAMPLES / "jo.txt"),
        "--model", neutts_config.get("model", "neuphonic/neutts-air-q4-gguf"),
        "--device", neutts_config.get("device", "cpu")]
    result = _run_helper(cmd, 120)
    if result.returncode != 0:  # the synth script reports success lines as "OK:" on stderr too
        error_lines = [l for l in result.stderr.strip().splitlines() if not l.startswith("OK:")]
        raise RuntimeError(f"NeuTTS synthesis failed: {chr(10).join(error_lines) or 'unknown error'}")
    return _finalize_wav_output(wav_path, output_path)


# --- Piper (local neural VITS, 44 languages) ---
def _get_piper_voices_dir() -> Path:
    """``<HERMES_HOME>/cache/piper-voices/`` so voice downloads follow profile boundaries."""
    from hermes_constants import get_hermes_dir
    root = Path(get_hermes_dir("cache/piper-voices", "piper_voices_cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_piper_voice_path(voice: str, download_dir: Path) -> str:
    """Resolve *voice* (an .onnx path or a name like ``en_US-lessac-medium``, downloaded into
    *download_dir* on first use) to a concrete .onnx file; RuntimeError when it can't be."""
    voice = voice or DEFAULT_PIPER_VOICE
    candidate = Path(voice).expanduser()
    if candidate.suffix.lower() == ".onnx" and candidate.exists():
        return str(candidate)
    cached = download_dir / f"{voice}.onnx"
    if cached.exists() and (download_dir / f"{voice}.onnx.json").exists():
        return str(cached)
    logger.info("[Piper] Downloading voice '%s' to %s (first use)", voice, download_dir)
    try:
        result = _run_helper(
            [sys.executable, "-m", "piper.download_voices", voice, "--download-dir", str(download_dir)], 300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Piper voice download timed out after 300s for '{voice}'") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "no stderr output"
        raise RuntimeError(f"Piper voice download failed for '{voice}': {stderr[:400]}")
    if not cached.exists():
        raise RuntimeError(
            f"Piper voice download completed but {cached} is missing — "
            f"check voice name (see: https://github.com/OHF-Voice/piper1-gpl/"
            f"blob/main/docs/VOICES.md)")
    return str(cached)


def _load_piper_voice_for_config(tts_config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Resolve + load (or fetch from cache) the selected Piper voice -> ``(voice, piper_config)``.
    Shared by synthesis and ``warm_tts_provider`` so a warm-up fills exactly the slot synthesis hits."""
    PiperVoice = _origin()._import_piper()
    piper_config = _section(tts_config, "piper")
    voice_name = piper_config.get("voice") or DEFAULT_PIPER_VOICE
    download_dir = Path(piper_config.get("voices_dir") or _get_piper_voices_dir()).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = bool(piper_config.get("use_cuda", False))
    model_path = _resolve_piper_voice_path(voice_name, download_dir)

    def _load_piper_voice():
        logger.info("[Piper] Loading voice: %s", model_path)
        v = PiperVoice.load(model_path, use_cuda=use_cuda)
        logger.info("[Piper] Voice loaded")
        return v

    # speaker_id is applied per call via syn_config, so one instance serves every speaker.
    cache_key = f"{model_path}::cuda={use_cuda}"
    return _tts_cache_get_or_load(_piper_voice_cache, cache_key, _load_piper_voice), piper_config


_PIPER_ADVANCED_KNOBS = ("length_scale", "noise_scale", "noise_w_scale", "volume", "normalize_audio", "speaker_id")


def _generate_piper_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    import wave
    voice, piper_config = _load_piper_voice_for_config(tts_config)
    # Bad speaker_id drops to 0 (Piper's default); bools are rejected (they'd coerce to 1/0).
    _raw_speaker = piper_config.get("speaker_id", 0)
    speaker_id = _raw_speaker if type(_raw_speaker) is int else 0
    # Only build a SynthesisConfig when an advanced knob is configured, so we don't depend on a
    # newer piper-tts than the user's unless we must.
    syn_config = None
    if any(k in piper_config for k in _PIPER_ADVANCED_KNOBS):
        try:
            from piper import SynthesisConfig  # type: ignore
            syn_config = SynthesisConfig(
                length_scale=float(piper_config.get("length_scale", 1.0)),
                noise_scale=float(piper_config.get("noise_scale", 0.667)),
                noise_w_scale=float(piper_config.get("noise_w_scale", 0.8)),
                volume=float(piper_config.get("volume", 1.0)),
                normalize_audio=bool(piper_config.get("normalize_audio", True)),
                speaker_id=speaker_id)
        except ImportError:
            logger.warning("[Piper] SynthesisConfig not available in this piper-tts version — advanced knobs ignored")
    wav_path = _wav_sidecar_path(output_path)
    with wave.open(wav_path, "wb") as wav_file:
        if syn_config is not None:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        else:
            voice.synthesize_wav(text, wav_file)
    return _finalize_wav_output(wav_path, output_path)


# --- KittenTTS (local ONNX, 25-80MB models, CPU only) ---
def _load_kittentts_model_for_config(tts_config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Load (or fetch from cache) the KittenTTS model; returns ``(model, kittentts_config)``."""
    KittenTTS = _origin()._import_kittentts()
    kt_config = _section(tts_config, "kittentts")
    model_name = kt_config.get("model", DEFAULT_KITTENTTS_MODEL)

    def _load_kittentts_model():
        logger.info("[KittenTTS] Loading model: %s", model_name)
        m = KittenTTS(model_name)
        logger.info("[KittenTTS] Model loaded successfully")
        return m

    return _tts_cache_get_or_load(_kittentts_model_cache, model_name, _load_kittentts_model), kt_config


def _generate_kittentts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    model, kt_config = _load_kittentts_model_for_config(tts_config)
    audio = model.generate(  # numpy array at 24kHz
        text, voice=kt_config.get("voice", DEFAULT_KITTENTTS_VOICE),
        speed=kt_config.get("speed", 1.0), clean_text=kt_config.get("clean_text", True))
    import soundfile as sf
    wav_path = _wav_sidecar_path(output_path)
    sf.write(wav_path, audio, 24000)
    return _finalize_wav_output(wav_path, output_path)


# --- Kokoro (local ONNX, 82M params, CPU via onnxruntime, Apache-2.0) ---
DEFAULT_KOKORO_VOICE = "af_heart"
_KOKORO_MODEL_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                     "model-files-v1.1/kokoro-v1.0.onnx")
_KOKORO_VOICES_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                      "model-files-v1.1/voices-v1.0.bin")
_KOKORO_VOICE_CATALOG_URL = "https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md"


def _get_kokoro_dir() -> Path:
    """``<HERMES_HOME>/share/kokoro/`` — profile-bound model-asset store. The files are shared
    model assets rather than a per-voice download dir; resolved straight off the home because
    Kokoro has no legacy layout for :func:`get_hermes_dir` to honour."""
    from hermes_constants import get_hermes_home
    root = get_hermes_home() / "share" / "kokoro"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _kokoro_download(url: str, dest: Path) -> None:
    """Stream *url* to *dest*. Concurrent first-use callers (a warm-up lease racing the first
    synthesis, two sessions sharing a HERMES_HOME) single-flight on a lock file so the loser
    waits instead of duplicating the fetch; installation is a unique temp file + ``os.replace``
    so writers can never interleave and an interrupted fetch never leaves a truncated model
    that later reads as present. urllib keeps the dependency set at stdlib; GitHub releases
    follow redirects. The rename only happens when the response carried a usable body — a
    clean early EOF (short read against Content-Length, or zero bytes) is a failed download,
    never installed."""
    logger.info("[Kokoro] Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # Windows: the unique tempname already makes install atomic;
        fcntl = None     # the lock only saves a duplicate download, it is not correctness.
    lock_fh = None
    try:
        if fcntl is not None:
            lock_fh = dest.with_name(dest.name + ".lock").open("a+b")
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            if dest.is_file() and dest.stat().st_size > 0:
                return  # another caller finished this exact fetch while we waited
        _kokoro_fetch(url, dest)
    finally:
        if lock_fh is not None:
            lock_fh.close()
        # The lock file is deliberately left in place: unlinking it re-opens the classic
        # flock race (waiter on the old inode vs latecomer creating a fresh one) for zero
        # real gain, and a zero-byte marker per model artifact costs nothing. Sweep any
        # .part fragments an earlier crash left behind, though — those are real litter.
        for stale in dest.parent.glob(dest.name + ".*.part"):
            with suppress(OSError):
                stale.unlink()


def _kokoro_fetch(url: str, dest: Path) -> None:
    """One fetch attempt: stream to a unique temp file beside *dest*, validate the body, then
    install atomically. The unique name (not a fixed ``.part``) is what keeps two racing
    writers from corrupting each other even without the lock."""
    import tempfile
    import urllib.request
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, \
                urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hermes-agent"}), timeout=60) as resp:
            expected = resp.headers.get("Content-Length")
            written = 0
            while chunk := resp.read(1 << 20):
                written += len(chunk)
                out.write(chunk)
        # A truncated-but-clean close writes a short file that would otherwise pass the
        # size>0 cache check forever; the Content-Length comparison catches it.
        if not written:
            raise RuntimeError("server returned an empty body")
        if expected and written < int(expected):
            raise RuntimeError(f"truncated download ({written} of {expected} bytes)")
        if tmp.stat().st_size != written:
            raise RuntimeError("downloaded bytes did not land completely")
        os.replace(tmp, dest)
    except BaseException:
        with suppress(OSError):
            tmp.unlink()
        raise
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Kokoro download produced no data at {dest}")


def _resolve_kokoro_paths(kokoro_config: Dict[str, Any]) -> tuple:
    """Model + voices file paths: explicit config paths > cached downloads (fetched on first
    use) > RuntimeError with the manual-fetch recipe (offline boxes). Explicit paths must be
    absolute (``~`` expanded) — a relative path would silently download the weights into
    whatever directory Hermes was launched from — and are used verbatim: a *configured* path
    that is missing is an installation mistake, so it raises the manual-install recipe instead
    of quietly downloading ~350MB into it. The two keys are independent: configuring one never
    blocks the other from using the shared download cache."""
    configured_keys = {key for key in ("model_path", "voices_path")
                       if str(kokoro_config.get(key) or "").strip()}
    paths = []
    for key, default_name in (("model_path", "kokoro-v1.0.onnx"), ("voices_path", "voices-v1.0.bin")):
        configured = str(kokoro_config.get(key) or "").strip()
        if configured:
            path = Path(os.path.expanduser(configured))
            if not path.is_absolute():
                raise RuntimeError(
                    f"tts.kokoro.{key} must be an absolute path (got {configured!r}); "
                    "leave it unset to use the shared download cache")
        else:
            path = _get_kokoro_dir() / default_name
        paths.append(path)
    model_path, voices_path = paths
    for key, path, url, label in (
            ("model_path", model_path, _KOKORO_MODEL_URL, "model"),
            ("voices_path", voices_path, _KOKORO_VOICES_URL, "voices")):
        if path.is_file() and path.stat().st_size > 0:
            continue
        if key in configured_keys:
            raise RuntimeError(
                f"Kokoro {label} not found at the configured path {path}. Fix the path or "
                f"remove tts.kokoro.{key} to use the shared download cache — "
                f"sources: {_KOKORO_MODEL_URL} , {_KOKORO_VOICES_URL}")
        logger.info("[Kokoro] %s not cached at %s — downloading (first use)", label, path)
        try:
            _kokoro_download(url, path)
        except Exception as exc:
            raise RuntimeError(
                f"Kokoro {label} download failed ({exc}). Manual install: place the files at "
                f"{model_path} and {voices_path} — sources: {_KOKORO_MODEL_URL} , "
                f"{_KOKORO_VOICES_URL} (or a Kokoro-82M-v1.0-ONNX mirror).") from exc
    return model_path, voices_path


def _load_kokoro_model_for_config(tts_config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Load (or fetch from cache) the Kokoro engine; returns ``(model, kokoro_config)``.
    Shared by synthesis and ``warm_tts_provider`` so a warm-up fills exactly the slot
    synthesis hits. The SDK import is checked before path resolution so a missing
    ``kokoro-onnx`` extra surfaces as an ImportError (which the warmer reports as
    not-installed) instead of after a needless ~350MB download."""
    Kokoro = _origin()._import_kokoro_onnx()
    kokoro_config = _section(tts_config, "kokoro")
    model_path, voices_path = _resolve_kokoro_paths(kokoro_config)
    cache_key = f"{model_path}::{voices_path}"

    def _load_kokoro():
        logger.info("[Kokoro] Loading model: %s", model_path)
        m = Kokoro(str(model_path), str(voices_path))
        logger.info("[Kokoro] Model loaded")
        return m

    return _tts_cache_get_or_load(_kokoro_model_cache, cache_key, _load_kokoro), kokoro_config


# Voice-id prefix -> espeak-ng language code for the phonemizer. Kokoro voice ids are
# ``<lang><gender>_<name>``; the code must match the voice's language or pronunciation
# degrades (verified against the espeak backend kokoro-onnx 0.6.1 bundles).
_KOKORO_VOICE_LANG = {
    "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi",
    "i": "it", "j": "ja", "p": "pt-br", "z": "cmn"}


def _kokoro_voice_lang(voice: str) -> str:
    prefix = str(voice)[:1]
    lang = _KOKORO_VOICE_LANG.get(prefix)
    if lang is None:
        if str(voice).strip():
            logger.warning("[Kokoro] Unknown voice prefix %r on voice %r — phonemizing en-us",
                           prefix, voice)
        lang = "en-us"
    return lang


def _generate_kokoro(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    model, kokoro_config = _load_kokoro_model_for_config(tts_config)
    voice = kokoro_config.get("voice") or DEFAULT_KOKORO_VOICE
    # Per-call speed (the tool's ``speed`` param, staged under ``_call_speed`` by
    # _apply_call_overrides) wins over the provider-specific ``tts.kokoro.speed``, which wins
    # over the global ``tts.speed`` multiplier — the documented precedence. tts.kokoro.speed
    # has NO registered default precisely so a user who only sets the global tts.speed
    # (DEFAULT_CONFIG carries no tts.speed) reaches the third rung; absence is the unset
    # signal, and an explicit null in YAML means the same thing here. Non-finite or
    # non-numeric values reset to 1.0 (a YAML ``.nan`` would sail through min/max); the
    # final clamp is kokoro-onnx's sane range.
    speed = tts_config.get("_call_speed", kokoro_config.get("speed", tts_config.get("speed", 1.0)))
    if speed is None:
        speed = 1.0
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 1.0
    if not math.isfinite(speed):
        speed = 1.0
    speed = min(max(speed, 0.5), 2.0)
    try:
        samples, sample_rate = model.create(text, voice=voice, speed=speed,
                                            lang=_kokoro_voice_lang(voice))
    except Exception as exc:
        get_voices = getattr(model, "get_voices", None)
        try:
            known = sorted(get_voices()) if callable(get_voices) else []
        except Exception:
            known = []
        hint = f"Voice ids: {', '.join(known[:40])} — " if known else ""
        raise RuntimeError(
            f"Kokoro synthesis failed: {exc}. {hint}catalog: {_KOKORO_VOICE_CATALOG_URL}") from exc
    import soundfile as sf
    wav_path = _wav_sidecar_path(output_path)
    sf.write(wav_path, samples, sample_rate)
    return _finalize_wav_output(wav_path, output_path)
