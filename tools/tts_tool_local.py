"""Local on-device TTS engines for ``tools.tts_tool``: NeuTTS, Piper, KittenTTS, Phonon.

All of them synthesize WAV natively; :func:`_finalize_wav_output` converts/renames to the requested
container. Piper, KittenTTS and Phonon keep loaded models in small LRU caches registered in
``_LOCAL_TTS_MODEL_CACHES`` so warm/release can pre-load or drop them. ``_import_piper`` /
``_import_kittentts`` / ``_import_ptts`` are resolved through the origin module at call time (test
monkeypatches).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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
# (+cuda flag); KittenTTS on model name.
_piper_voice_cache: dict[str, Any] = {}
_kittentts_model_cache: dict[str, Any] = {}
_phonon_model_cache: dict[str, Any] = {}
_LOCAL_TTS_MODEL_CACHES: dict[str, dict[str, Any]] = {
    "piper": _piper_voice_cache, "kittentts": _kittentts_model_cache,
    "phonon": _phonon_model_cache}


def _tts_cache_get_or_load(cache: dict[str, Any], key: str, load: Callable[[], Any]) -> Any:
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
        check=False,
    )


# --- NeuTTS (subprocess via tools/neutts_synth.py so the ~500MB model exits after use) ---
def _generate_neutts(text: str, output_path: str, tts_config: dict[str, Any]) -> str:
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


def _load_piper_voice_for_config(tts_config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
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


def _generate_piper_tts(text: str, output_path: str, tts_config: dict[str, Any]) -> str:
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
def _load_kittentts_model_for_config(tts_config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
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


def _generate_kittentts(text: str, output_path: str, tts_config: dict[str, Any]) -> str:
    model, kt_config = _load_kittentts_model_for_config(tts_config)
    audio = model.generate(  # numpy array at 24kHz
        text, voice=kt_config.get("voice", DEFAULT_KITTENTTS_VOICE),
        speed=kt_config.get("speed", 1.0), clean_text=kt_config.get("clean_text", True))
    import soundfile as sf
    wav_path = _wav_sidecar_path(output_path)
    sf.write(wav_path, audio, 24000)
    return _finalize_wav_output(wav_path, output_path)


# --- Phonon (local, Gradium 31M/40M via the ptts Rust runtime, CPU) ---
# ptts ships abi3 wheels for win32/linux/macOS with numpy as the only dependency, so unlike
# NeuTTS/KittenTTS there is no torch, no ONNX runtime and no espeak-ng. Weights (64MB q8) download
# from the Hugging Face Hub on first use and are cached by the runtime itself.
DEFAULT_PHONON_MODEL = "hf://gradium/phonon-31m-en"
DEFAULT_PHONON_VOICE = "default"  # the checkpoint's own voice, not a catalogue name
DEFAULT_PHONON_QUANT = "q8"

# Checkpoint -> the ``lang`` its text normalization needs. ``None`` means the checkpoint carries
# every language (the multilingual one), so the selected voice's locale decides.
#
# The checkpoint wins over the voice on purpose: a French *voice* on an English *checkpoint* still
# speaks with an English model, so normalizing its text as French would spell numbers in a language
# the model cannot pronounce — ptts says as much with "text is normalized as fr, but the model was
# trained for en". Only the multilingual checkpoint has no single answer, hence the locale fallback.
PHONON_MODELS: dict[str, Optional[str]] = {
    "hf://gradium/phonon-31m-en": "en",
    "hf://gradium/phonon-40m-en": "en",
    "hf://gradium/phonon-40m-multilingual": None,
    "hf://gradium/phonon-40m-fr": "fr",
    "hf://gradium/phonon-40m-de": "de",
    "hf://gradium/phonon-40m-es": "es",
    "hf://gradium/phonon-40m-pt": "pt",
}


# Short tags (mirroring the Ollama library) -> Hub repo id, so the settings dropdown can stay
# readable. A full hf:// id, a bare owner/name or a local path passes through untouched.
PHONON_MODEL_ALIASES: dict[str, str] = {
    "31m-en": "hf://gradium/phonon-31m-en", "en": "hf://gradium/phonon-31m-en",
    "40m-en": "hf://gradium/phonon-40m-en",
    "40m-fr": "hf://gradium/phonon-40m-fr", "fr": "hf://gradium/phonon-40m-fr",
    "40m-de": "hf://gradium/phonon-40m-de", "de": "hf://gradium/phonon-40m-de",
    "40m-es": "hf://gradium/phonon-40m-es", "es": "hf://gradium/phonon-40m-es",
    "40m-pt": "hf://gradium/phonon-40m-pt", "pt": "hf://gradium/phonon-40m-pt",
    "40m-multilingual": "hf://gradium/phonon-40m-multilingual",
    "multilingual": "hf://gradium/phonon-40m-multilingual",
}


def _phonon_model_id(raw: Any) -> str:
    """Resolve ``tts.phonon.model`` to something ptts accepts.

    Full ``hf://owner/name[@revision]`` ids, bare ``owner/name`` and local directories (``.``,
    ``/`` or ``~``) pass through unchanged; a short tag resolves through
    :data:`PHONON_MODEL_ALIASES`, falling back to ``hf://gradium/phonon-<tag>``.
    """
    value = str(raw or "").strip()
    if not value:
        return DEFAULT_PHONON_MODEL
    if value.startswith("hf://") or "/" in value or value.startswith((".", "/", "~")):
        return value
    alias = PHONON_MODEL_ALIASES.get(value.lower())
    if alias:
        return alias
    return f"hf://{value}" if value.count("/") == 1 else f"hf://gradium/phonon-{value}"


def _phonon_voice_arg(raw: Any) -> Any:
    """ptts ``voice`` argument: None (checkpoint default) for empty/``default``, else the name as-is.

    A name is left for ptts to resolve (case/accent-insensitive, or a catalogue ``voice_id``), so a
    voice Gradium adds later works without a Hermes release.
    """
    value = str(raw or "").strip()
    return None if value.lower() in ("", "default") else value


def _phonon_load_error(exc: BaseException) -> str:
    """Turn a load failure into something the user can act on.

    Two cases dominate: the 40M checkpoints are **gated** on the Hub (CC BY-NC 4.0), so ptts raises
    ``GatedRepo``/403 until the user accepts the terms and authenticates; and a machine with no
    network gets a plain fetch error. Everything else keeps ptts's own text.
    """
    import re
    text = str(exc)
    lowered = text.lower()
    if "gatedrepo" in lowered or "restricted" in lowered or "403" in text or "401" in text:
        match = re.search(r"gradium/phonon-[\w.-]+", text)
        repo = match.group(0) if match else "gradium/phonon-40m-en"
        return (f"{text}\n\nThe 40M Phonon checkpoints are gated (CC BY-NC 4.0). Accept the terms at "
                f"https://huggingface.co/{repo}, then authenticate with `hf auth login` or by setting "
                "HF_TOKEN, and retry. The 31m-en checkpoint is ungated and needs no account.")
    return text


def _load_phonon_model_for_config(tts_config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Load (or fetch from cache) the Phonon model; returns ``(model, phonon_config)``."""
    ptts = _origin()._import_ptts()
    ph_config = _section(tts_config, "phonon")
    model_id = _phonon_model_id(ph_config.get("model"))
    voice = _phonon_voice_arg(ph_config.get("voice"))
    # Normalization language: explicit config > the checkpoint's own language > (multilingual
    # checkpoint only) the selected voice's locale > English.
    from tools.phonon_voices import phonon_lang_for_voice
    checkpoint_lang = PHONON_MODELS.get(model_id)
    lang = (str(ph_config.get("lang") or "").strip()
            or checkpoint_lang
            or (phonon_lang_for_voice(voice) if voice else None)
            or "en")
    quant = str(ph_config.get("quant") or DEFAULT_PHONON_QUANT).strip()
    device = str(ph_config.get("device") or "").strip() or None
    catalog = str(ph_config.get("voice_catalog") or "").strip() or None
    # 0 / absent = ptts's own thread default; set_num_threads only has an effect before the first
    # load, so it is applied inside the loader (a cache hit already sized the pool).
    raw_threads = ph_config.get("threads")
    threads = raw_threads if type(raw_threads) is int and raw_threads > 0 else 0

    def _load_phonon_model():
        if threads:
            ptts.set_num_threads(threads)
        logger.info("[Phonon] Loading model: %s (lang=%s, quant=%s)", model_id, lang, quant)
        try:
            model = ptts.TTS(model_id, lang=lang, quant=quant, device=device, voice_catalog=catalog)
        except OSError as exc:  # gated repo, offline box, bad revision
            raise RuntimeError(_phonon_load_error(exc)) from exc
        logger.info("[Phonon] Model loaded (%d Hz)", getattr(model, "sample_rate", 24000))
        return model

    cache_key = f"{model_id}::lang={lang}::quant={quant}::device={device}::catalog={catalog}"
    return _tts_cache_get_or_load(_phonon_model_cache, cache_key, _load_phonon_model), ph_config


def _generate_phonon(text: str, output_path: str, tts_config: dict[str, Any]) -> str:
    model, ph_config = _load_phonon_model_for_config(tts_config)
    # Sampling knobs are passed only when configured: ptts's own defaults (temperature 0.3, a fixed
    # seed) are the reproducible settings its release notes were tuned against.
    knobs: dict[str, Any] = {}
    for key in ("temperature", "seed", "cfg_coef"):
        value = ph_config.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            knobs[key] = value
    wav_path = _wav_sidecar_path(output_path)
    seconds = model.save(wav_path, text, voice=_phonon_voice_arg(ph_config.get("voice")), **knobs)
    logger.info("[Phonon] Synthesized %.2fs of audio", seconds or 0.0)
    return _finalize_wav_output(wav_path, output_path)
