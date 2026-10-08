"""Local on-device TTS engines for ``tools.tts_tool``: NeuTTS, Piper, KittenTTS.

All three synthesize WAV natively; :func:`_finalize_wav_output` converts/renames to the requested
container. Piper and KittenTTS keep loaded models in small LRU caches registered in
``_LOCAL_TTS_MODEL_CACHES`` so warm/release can pre-load or drop them. ``_import_piper`` /
``_import_kittentts`` are resolved through the origin module at call time (test monkeypatches).
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.tts_tool_delivery import (
    _ffmpeg_run, _finalize_wav_output, _origin, _remove_quietly, _section, _wav_sidecar_path,
)

logger = logging.getLogger("tools.tts_tool")

DEFAULT_KITTENTTS_MODEL = "KittenML/kitten-tts-nano-0.8-int8"  # 25MB
DEFAULT_KITTENTTS_VOICE = "Jasper"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"  # balanced size/quality
DEFAULT_NEUTTS_MODEL = "neuphonic/neutts-air-q4-gguf"
DEFAULT_NEUTTS_CODEC_REPO = "neuphonic/neucodec"
DEFAULT_NEUTTS_IDLE_UNLOAD_SECONDS = 1800
NEUTTS_OUTPUT_FORMATS = frozenset({"mp3", "m4a", "aac", "wav", "ogg", "flac"})
_NEUTTS_SAMPLES = Path(__file__).parent / "neutts_samples"

_neutts_cache_lock = threading.RLock()
_neutts_cache: Dict[str, Any] = {}
_neutts_idle_timer: threading.Timer | None = None
_neutts_idle_generation = 0
_make_neutts_idle_timer = threading.Timer  # test seam for deterministic idle-unload checks

# --- Bounded model caches ---
# Each entry is a whole loaded model (tens of MB); unbounded, one would be pinned per distinct
# voice for the process lifetime. Most sessions use one or two voices; a cold reload is cheap.
_TTS_MODEL_CACHE_MAX = 3

# Provider name -> the cache it populates (warm/release in tts_tool_lifecycle; a new local engine
# adds a row here plus a loader in _local_tts_warmers()). Piper keyed on absolute .onnx path
# (+cuda flag); KittenTTS on model name.
_piper_voice_cache: Dict[str, Any] = {}
_kittentts_model_cache: Dict[str, Any] = {}
_LOCAL_TTS_MODEL_CACHES: Dict[str, Dict[str, Any]] = {
    "neutts": _neutts_cache, "piper": _piper_voice_cache, "kittentts": _kittentts_model_cache,
}


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
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(),
    )


# --- NeuTTS (subprocess by default; optional in-process cache) ---
def _bool_config(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _neutts_warm_cache_enabled(neutts_config: Dict[str, Any]) -> bool:
    return _bool_config(neutts_config.get("warm_cache"), default=False)


def _neutts_idle_unload_seconds(neutts_config: Dict[str, Any]) -> float:
    raw = neutts_config.get("idle_unload_seconds", DEFAULT_NEUTTS_IDLE_UNLOAD_SECONDS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_NEUTTS_IDLE_UNLOAD_SECONDS)
    if not math.isfinite(value) or value <= 0:
        return float(DEFAULT_NEUTTS_IDLE_UNLOAD_SECONDS)
    return value


def _neutts_output_format(neutts_config: Dict[str, Any], default: str = "mp3") -> str:
    raw = neutts_config.get("output_format") or neutts_config.get("format") or default
    fmt = str(raw).lower().strip().lstrip(".")
    if fmt == "aac":
        return "m4a"
    return fmt if fmt in NEUTTS_OUTPUT_FORMATS else default


def _neutts_resolved_config(tts_config: Dict[str, Any]) -> Dict[str, str]:
    neutts_config = _section(tts_config, "neutts")
    return {
        "ref_audio": str(Path(neutts_config.get("ref_audio") or str(_NEUTTS_SAMPLES / "jo.wav")).expanduser()),
        "ref_text": str(Path(neutts_config.get("ref_text") or str(_NEUTTS_SAMPLES / "jo.txt")).expanduser()),
        "model": str(neutts_config.get("model") or DEFAULT_NEUTTS_MODEL),
        "device": str(neutts_config.get("device") or "cpu"),
        "codec_repo": str(neutts_config.get("codec_repo") or DEFAULT_NEUTTS_CODEC_REPO),
    }


def _neutts_cache_key(resolved: Dict[str, str]) -> str:
    def stamp(path_value: str) -> tuple[str, int, int]:
        path = Path(path_value)
        try:
            stat = path.stat()
            return str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return str(path), 0, 0

    return json.dumps({
        "model": resolved["model"],
        "device": resolved["device"],
        "codec_repo": resolved["codec_repo"],
        "ref_audio": stamp(resolved["ref_audio"]),
        "ref_text": stamp(resolved["ref_text"]),
    }, sort_keys=True)


def _write_neutts_wav(path: str, samples, sample_rate: int = 24000) -> None:
    """Write mono 16-bit WAV samples; soundfile/numpy are optional test/runtime helpers."""
    try:
        sf = importlib.import_module("soundfile")
        sf.write(path, samples, sample_rate)
        return
    except ImportError:
        pass

    try:
        np = importlib.import_module("numpy")
    except ImportError:
        values = samples.tolist() if hasattr(samples, "tolist") else samples
        pcm_bytes = b"".join(
            struct.pack("<h", max(-32768, min(32767, round(float(value) * 32767))))
            for value in values
        )
    else:
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float32)
        pcm_bytes = (np.clip(samples.flatten(), -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    data_size = len(pcm_bytes)
    byte_rate = sample_rate * 2
    with open(path, "wb") as wav_file:
        wav_file.write(b"RIFF")
        wav_file.write(struct.pack("<I", 36 + data_size))
        wav_file.write(b"WAVEfmt ")
        wav_file.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16))
        wav_file.write(b"data")
        wav_file.write(struct.pack("<I", data_size))
        wav_file.write(pcm_bytes)


def _remove_neutts_partial_output(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("[NeuTTS] failed to remove partial output: %s", path, exc_info=True)


def _clear_neutts_cache(reason: str = "manual") -> int:
    global _neutts_idle_timer, _neutts_idle_generation
    with _neutts_cache_lock:
        released = len(_neutts_cache)
        _neutts_idle_generation += 1
        if _neutts_idle_timer is not None:
            _neutts_idle_timer.cancel()
            _neutts_idle_timer = None
        if released:
            logger.info("[NeuTTS] clearing warm cache (%s)", reason)
        _neutts_cache.clear()
    if released:
        gc.collect()
    return released


def _schedule_neutts_idle_unload(idle_seconds: float) -> None:
    global _neutts_idle_timer, _neutts_idle_generation
    with _neutts_cache_lock:
        if _neutts_idle_timer is not None:
            _neutts_idle_timer.cancel()
        _neutts_idle_generation += 1
        generation = _neutts_idle_generation

        def unload_if_idle(expected_generation: int) -> None:
            global _neutts_idle_timer
            with _neutts_cache_lock:
                if expected_generation != _neutts_idle_generation:
                    return
                if not _neutts_cache:
                    _neutts_idle_timer = None
                    return
                last_used = max(float(entry.get("last_used_at", 0.0)) for entry in _neutts_cache.values())
                remaining = idle_seconds - (time.monotonic() - last_used)
                if remaining > 0:
                    _schedule_neutts_idle_unload(remaining)
                    return
                _clear_neutts_cache("idle")

        timer = _make_neutts_idle_timer(idle_seconds, unload_if_idle, args=(generation,))
        timer.daemon = True
        _neutts_idle_timer = timer
        timer.start()


def _load_neutts_cache_entry(tts_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load one configuration's model/reference pair; replace stale state before loading."""
    resolved = _neutts_resolved_config(tts_config)
    key = _neutts_cache_key(resolved)
    with _neutts_cache_lock:
        entry = _neutts_cache.get(key)
        if entry is not None:
            entry["last_used_at"] = time.monotonic()
            return entry
        if _neutts_cache or _neutts_idle_timer is not None:
            _clear_neutts_cache("config_changed")
        try:
            ref_audio = Path(resolved["ref_audio"])
            ref_text_path = Path(resolved["ref_text"])
            if not ref_audio.exists():
                raise RuntimeError(f"NeuTTS reference audio not found: {ref_audio}")
            if not ref_text_path.exists():
                raise RuntimeError(f"NeuTTS reference text not found: {ref_text_path}")
            NeuTTS = getattr(importlib.import_module("neutts"), "NeuTTS")
            # llama.cpp offloads only for literal "gpu"; torch expects "cuda".
            backbone_device = "gpu" if resolved["device"] == "cuda" else resolved["device"]
            codec_device = resolved["device"]
            model = NeuTTS(
                backbone_repo=resolved["model"],
                backbone_device=backbone_device,
                codec_repo=resolved["codec_repo"],
                codec_device=codec_device,
            )
            ref_codes = model.encode_reference(str(ref_audio))
            ref_text = ref_text_path.read_text(encoding="utf-8-sig").strip()
            now = time.monotonic()
            entry = {
                "tts": model,
                "ref_codes": ref_codes,
                "ref_text": ref_text,
                "last_used_at": now,
                "resolved": resolved,
            }
            _neutts_cache[key] = entry
            logger.info("[NeuTTS] warm cache loaded: model=%s device=%s", resolved["model"], resolved["device"])
            return entry
        except Exception:
            _clear_neutts_cache("load_error")
            raise


def _warm_neutts_cache_for_config(tts_config: Dict[str, Any]) -> Any:
    """Preload only when the user explicitly opted in to the in-process cache."""
    neutts_config = _section(tts_config, "neutts")
    if not _neutts_warm_cache_enabled(neutts_config):
        return None
    entry = _load_neutts_cache_entry(tts_config)
    _schedule_neutts_idle_unload(_neutts_idle_unload_seconds(neutts_config))
    return entry["tts"]


def _new_neutts_wav_path(output_path: str) -> str:
    """Create an owned, unique WAV path beside the requested output."""
    parent = Path(output_path).expanduser().parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="hermes-neutts-", suffix=".wav", dir=parent, delete=False
    ) as wav_file:
        return wav_file.name


def _convert_neutts_wav(wav_path: str, output_path: str) -> str:
    """Convert a private WAV sidecar into the requested format, publishing only on success."""
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if wav_path == str(output):
        return str(output)
    if output.suffix.lower() == ".wav":
        os.replace(wav_path, output)
        return str(output)

    with tempfile.TemporaryDirectory(prefix=".hermes-neutts-convert-", dir=output.parent) as temp_dir:
        temporary_output = Path(temp_dir) / f"output{output.suffix}"
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg and output.suffix.lower() in {".m4a", ".aac"}:
            _ffmpeg_run(
                ffmpeg,
                ["-i", wav_path, "-c:a", "aac", "-b:a", "128k", "-y", "-loglevel", "error",
                 str(temporary_output)],
                check=True,
                capture=False,
            )
        else:
            _finalize_wav_output(wav_path, str(temporary_output))
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError("NeuTTS audio conversion produced no output")
        os.replace(temporary_output, output)
    _remove_quietly(wav_path)
    return str(output)


def _generate_neutts_warm(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate with an opt-in in-process model/reference cache."""
    neutts_config = _section(tts_config, "neutts")
    wav_path = _new_neutts_wav_path(output_path)
    with _neutts_cache_lock:
        try:
            entry = _load_neutts_cache_entry(tts_config)
            wav = entry["tts"].infer(text, entry["ref_codes"], entry["ref_text"])
            _write_neutts_wav(wav_path, wav, 24000)
            output = _convert_neutts_wav(wav_path, output_path)
            entry["last_used_at"] = time.monotonic()
            _schedule_neutts_idle_unload(_neutts_idle_unload_seconds(neutts_config))
            return output
        except Exception:
            # Keep conversion under the same lock as inference so another request cannot
            # start a new load while a failed call is clearing the previous model.
            _clear_neutts_cache("synthesis_or_conversion_error")
            raise
        finally:
            _remove_neutts_partial_output(wav_path)


def _generate_neutts_subprocess(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate via a one-shot process to preserve the default memory-lifetime contract."""
    resolved = _neutts_resolved_config(tts_config)
    wav_path = _new_neutts_wav_path(output_path)
    cmd = [
        sys.executable, str(Path(__file__).parent / "neutts_synth.py"),
        "--text", text, "--out", wav_path,
        "--ref-audio", resolved["ref_audio"], "--ref-text", resolved["ref_text"],
        "--model", resolved["model"], "--codec-repo", resolved["codec_repo"],
        "--device", resolved["device"],
    ]
    try:
        result = _run_helper(cmd, 120)
        if result.returncode != 0:
            error_lines = [line for line in (result.stderr or "").strip().splitlines() if not line.startswith("OK:")]
            raise RuntimeError(f"NeuTTS synthesis failed: {chr(10).join(error_lines) or 'unknown error'}")
        return _convert_neutts_wav(wav_path, output_path)
    finally:
        _remove_neutts_partial_output(wav_path)


def _generate_neutts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Use in-process caching only with explicit opt-in; otherwise retain one-shot subprocess behavior."""
    neutts_config = _section(tts_config, "neutts")
    if _neutts_warm_cache_enabled(neutts_config):
        return _generate_neutts_warm(text, output_path, tts_config)
    with _neutts_cache_lock:
        cache_is_active = bool(_neutts_cache) or _neutts_idle_timer is not None
    if cache_is_active:
        _clear_neutts_cache("warm_cache_disabled")
    return _generate_neutts_subprocess(text, output_path, tts_config)


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
