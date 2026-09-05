"""Local on-device TTS engines for ``tools.tts_tool``: NeuTTS, Piper, KittenTTS.

All three synthesize WAV natively; :func:`_finalize_wav_output` converts/renames to the requested
container. Piper and KittenTTS keep loaded models in small LRU caches registered in
``_LOCAL_TTS_MODEL_CACHES`` so warm/release can pre-load or drop them. ``_import_piper`` /
``_import_kittentts`` are resolved through the origin module at call time (test monkeypatches).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from tools.tts_tool_delivery import _finalize_wav_output, _origin, _section, _wav_sidecar_path

logger = logging.getLogger("tools.tts_tool")

DEFAULT_KITTENTTS_MODEL = "KittenML/kitten-tts-nano-0.8-int8"  # 25MB
DEFAULT_KITTENTTS_VOICE = "Jasper"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"  # balanced size/quality
_NEUTTS_SAMPLES = Path(__file__).parent / "neutts_samples"

# --- Paragraph-pause support (issue #103103) ---
PARAGRAPH_MARKER = "\u200B¶\u200B"


def _split_paragraphs(text: str) -> List[str]:
    """Split text on the zero-width-space paragraph marker, stripping whitespace around each."""
    if PARAGRAPH_MARKER not in text:
        return [text.strip()] if text.strip() else []
    return [para.strip() for para in text.split(PARAGRAPH_MARKER) if para.strip()]


def _generate_silence_pcm(duration_ms: int, sample_rate: int, channels: int = 1) -> bytes:
    """Return 16-bit PCM data for *duration_ms* milliseconds of silence."""
    frames = int(sample_rate * duration_ms / 1000.0)
    # 16-bit signed zero samples: b"\x00\x00" per sample per channel
    return b"\x00\x00" * frames * channels

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
    "piper": _piper_voice_cache, "kittentts": _kittentts_model_cache}


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
    import wave
    import tempfile
    neutts_config = tts_config.get("neutts") or {}
    paragraphs = _split_paragraphs(text)
    pause_ms = neutts_config.get("paragraph_pause_ms", 0)
    ref_audio = neutts_config.get("ref_audio", "") or str(_NEUTTS_SAMPLES / "jo.wav")
    ref_text = neutts_config.get("ref_text", "") or str(_NEUTTS_SAMPLES / "jo.txt")
    model = neutts_config.get("model", "neuphonic/neutts-air-q4-gguf")
    device = neutts_config.get("device", "cpu")
    wav_path = _wav_sidecar_path(output_path)
    if len(paragraphs) > 1 and pause_ms > 0:
        # Multi-paragraph: synthesize each, concatenate with silence
        tmp_wavs: List[str] = []
        try:
            for para_text in paragraphs:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp_path = tmp.name
                    tmp_wavs.append(tmp_path)
                cmd = [
                    sys.executable, str(Path(__file__).parent / "neutts_synth.py"),
                    "--text", para_text,
                    "--out", tmp_path,
                    "--ref-audio", ref_audio,
                    "--ref-text", ref_text,
                    "--model", model,
                    "--device", device]
                result = _run_helper(cmd, 120)
                if result.returncode != 0:
                    error_lines = [l for l in result.stderr.strip().splitlines() if not l.startswith("OK:")]
                    raise RuntimeError(f"NeuTTS synthesis failed: {chr(10).join(error_lines) or 'unknown error'}")
            # Concatenate paragraph WAVs with silence
            all_frames = []
            sample_rate = None
            for i, tmp_wav in enumerate(tmp_wavs):
                with wave.open(tmp_wav, "rb") as r:
                    if sample_rate is None:
                        sample_rate = r.getframerate()
                    all_frames.append(r.readframes(r.getnframes()))
                if i < len(tmp_wavs) - 1:
                    all_frames.append(_generate_silence_pcm(pause_ms, sample_rate, 1))
            assert sample_rate is not None, "No paragraphs synthesized"
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b"".join(all_frames))
        finally:
            for tmp_wav in tmp_wavs:
                try:
                    Path(tmp_wav).unlink()
                except Exception:
                    pass
    else:
        # Single paragraph or no pause: legacy behavior
        cmd = [
            sys.executable, str(Path(__file__).parent / "neutts_synth.py"),
            "--text", text,
            "--out", wav_path,
            "--ref-audio", ref_audio,
            "--ref-text", ref_text,
            "--model", model,
            "--device", device]
        result = _run_helper(cmd, 120)
        if result.returncode != 0:
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
    # Paragraph-pause logic: if the text contains the marker and paragraph_pause_ms is set,
    # synthesize each paragraph and insert silence between them.
    paragraphs = _split_paragraphs(text)
    pause_ms = piper_config.get("paragraph_pause_ms", 0)
    wav_path = _wav_sidecar_path(output_path)
    if len(paragraphs) > 1 and pause_ms > 0:
        # Multi-paragraph synthesis: build one PCM buffer, then write it
        import io
        all_frames = []
        sample_rate = voice.config.sample_rate  # Piper voice config has sample_rate
        for i, para_text in enumerate(paragraphs):
            # Synthesize paragraph into a memory buffer
            buf = io.BytesIO()
            with wave.open(buf, "wb") as tmp_wav:
                if syn_config is not None:
                    voice.synthesize_wav(para_text, tmp_wav, syn_config=syn_config)
                else:
                    voice.synthesize_wav(para_text, tmp_wav)
            buf.seek(0)
            with wave.open(buf, "rb") as r:
                all_frames.append(r.readframes(r.getnframes()))
            # Insert silence between paragraphs (but not after the last one)
            if i < len(paragraphs) - 1:
                all_frames.append(_generate_silence_pcm(pause_ms, sample_rate, 1))
        # Write the concatenated audio
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(all_frames))
    else:
        # Single paragraph or no pause configured: legacy behavior
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
    paragraphs = _split_paragraphs(text)
    pause_ms = kt_config.get("paragraph_pause_ms", 0)
    if len(paragraphs) > 1 and pause_ms > 0:
        # Multi-paragraph: synthesize each, concatenate with silence
        import numpy as np
        import soundfile as sf
        sample_rate = 24000
        all_audio = []
        for i, para_text in enumerate(paragraphs):
            para_audio = model.generate(
                para_text, voice=kt_config.get("voice", DEFAULT_KITTENTTS_VOICE),
                speed=kt_config.get("speed", 1.0), clean_text=kt_config.get("clean_text", True))
            all_audio.append(para_audio)
            if i < len(paragraphs) - 1:
                silence_frames = int(sample_rate * pause_ms / 1000.0)
                all_audio.append(np.zeros(silence_frames, dtype=para_audio.dtype))
        audio = np.concatenate(all_audio)
        wav_path = _wav_sidecar_path(output_path)
        sf.write(wav_path, audio, sample_rate)
    else:
        # Single paragraph or no pause: legacy behavior
        audio = model.generate(
            text, voice=kt_config.get("voice", DEFAULT_KITTENTTS_VOICE),
            speed=kt_config.get("speed", 1.0), clean_text=kt_config.get("clean_text", True))
        import soundfile as sf
        wav_path = _wav_sidecar_path(output_path)
        sf.write(wav_path, audio, 24000)
    return _finalize_wav_output(wav_path, output_path)
