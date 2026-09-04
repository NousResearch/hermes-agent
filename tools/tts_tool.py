#!/usr/bin/env python3
"""Text-to-speech tool: config resolution, built-in provider dispatch, output policy, registration.

Built-ins: Edge (free default), ElevenLabs, OpenAI, DeepInfra, MiniMax, Mistral, Gemini, xAI,
local NeuTTS / KittenTTS / Piper; plus ``type: command`` providers under ``tts.providers.<name>``
and plugin-registered ones. Output is Opus (.ogg) on voice-bubble platforms, MP3 elsewhere.
Sibling ``tts_tool_*`` modules hold backends/delivery/lifecycle; they read the seams defined
here (config, provider resolution, lazy SDK importers) through ``_origin()`` at call time.
"""

import asyncio
import contextlib
import datetime
import importlib.util
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

import copy

from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)


def _resolve_provider_key(env_var: str, provider_id: str) -> str:
    """Resolve a TTS provider API key via the shared voice-key resolver (config > env/.env > pool)."""
    from tools.tool_backend_helpers import resolve_provider_secret
    return resolve_provider_secret(env_var, provider_id)


from tools.tts_command_provider import (
    BUILTIN_TTS_PROVIDERS, _configured_command_tts_output_path, _generate_command_tts,
    _get_command_tts_output_format, _is_command_tts_voice_compatible, _resolve_command_provider_config)
from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER
from tools.tts_tool_delivery import (
    _resolve_max_text_length, _build_audio_delivery_files, _convert_to_opus, _remove_quietly,
    _repair_ogg_container, _resolve_audio_delivery_profile, _split_text_for_tts)
from tools.tts_tool_providers import (
    _generate_edge_tts, _generate_elevenlabs, _generate_gemini_tts, _generate_minimax_tts,
    _generate_mistral_tts, _generate_xai_tts, _resolve_minimax_tts_runtime)
from tools.tts_tool_local import _generate_kittentts, _generate_neutts, _generate_piper_tts
from tools.tts_tool_plugins import (
    _dispatch_to_plugin_provider, _plugin_provider_is_available,
    _plugin_provider_is_voice_compatible)
from tools.tts_tool_openai import _generate_deepinfra_tts, _generate_openai_tts, _has_openai_audio_backend


# --- Lazy SDK importers -- providers import only when used (headless boxes lack PortAudio etc.) ---
def _sdk_importer(module: str, attr: Optional[str] = None, feature: Optional[str] = None) -> Callable[[], Any]:
    """Lazy SDK importer: returns ``module`` (or ``module.attr``), raising ImportError when absent.

    ``feature`` names a ``tools.lazy_deps`` feature to best-effort install first (users who enabled
    a provider in config.yaml never ran the post-setup hook); any failure there falls through so
    the raw import still raises cleanly. sounddevice also raises OSError without PortAudio."""
    def _import():
        if feature:
            with contextlib.suppress(Exception):
                from tools.lazy_deps import ensure
                ensure(feature, prompt=False)
        mod = importlib.import_module(module)
        return getattr(mod, attr) if attr else mod
    _import.__name__ = f"_import_{module.split('.')[0]}"
    return _import


_import_edge_tts = _sdk_importer("edge_tts", feature="tts.edge")
_import_elevenlabs = _sdk_importer("elevenlabs.client", "ElevenLabs", feature="tts.elevenlabs")
_import_openai_client = _sdk_importer("openai", "OpenAI")
_import_mistral_client = _sdk_importer("mistralai.client", "Mistral", feature="tts.mistral")
_import_sounddevice = _sdk_importer("sounddevice")
_import_kittentts = _sdk_importer("kittentts", "KittenTTS")
_import_piper = _sdk_importer("piper", "PiperVoice")  # piper-tts wheels embed espeak-ng


def _importable(importer: Callable[[], Any]) -> bool:
    try:
        from tools.tool_backend_helpers import resolve_provider_secret
    except ImportError:  # pragma: no cover — helpers are in-repo
        return str(get_env_value(env_var) or "").strip()
    return resolve_provider_secret(env_var, provider_id, env_getter=get_env_value)

from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import (
    NOUS_MANAGED_PROVIDER,
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    read_selection,
    resolve_openai_audio_api_key,
    selection_error,
)
from tools.xai_http import hermes_xai_user_agent

# ---------------------------------------------------------------------------
# Lazy imports -- providers are imported only when actually used to avoid
# crashing in headless environments (SSH, Docker, WSL, no PortAudio).
# ---------------------------------------------------------------------------

def _import_edge_tts():
    """Lazy import edge_tts. Returns the module or raises ImportError."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tts.edge", prompt=False)
    except ImportError:
        return False


def _package_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _check_neutts_available() -> bool: return _package_installed("neutts")
def _check_kittentts_available() -> bool: return _package_installed("kittentts")
def _check_piper_available() -> bool: return _package_installed("piper")


# --- Defaults / config ---
DEFAULT_PROVIDER = "edge"


def _get_default_output_dir() -> str:
    from hermes_constants import get_hermes_dir
    return str(get_hermes_dir("cache/audio", "audio_cache"))

DEFAULT_OUTPUT_DIR = _get_default_output_dir()
_DEFAULT_OUTPUT_DIR_AT_IMPORT = DEFAULT_OUTPUT_DIR

def _default_output_dir() -> str:
    """Return the active profile's audio output dir at call time.

    Same bug class as skills_tool (f8723c478) and skills_sync (#65828):
    long-lived multi-profile runtimes (dashboard console, TUI/Desktop backend,
    cron, kanban workers) import this module once under the launch
    HERMES_HOME and later scope requests to a different profile via
    ``hermes_constants.set_hermes_home_override()`` — a frozen module
    constant keeps writing synthesized audio into the launch profile's
    cache instead of the active profile's (#98749). Keep the legacy
    ``DEFAULT_OUTPUT_DIR`` module attribute for tests and external patchers;
    when it has not been patched, re-resolve from the live profile-scoped
    HERMES_HOME on every call.
    """
    configured = DEFAULT_OUTPUT_DIR
    if configured != _DEFAULT_OUTPUT_DIR_AT_IMPORT:
        return configured
    return _get_default_output_dir()

DEFAULT_OUTPUT_DIR = _DEFAULT_OUTPUT_DIR_AT_IMPORT = _get_default_output_dir()


def _default_output_dir() -> str:
    """The active profile's audio output dir at call time (long-lived runtimes switch profiles
    after import); a monkeypatched ``DEFAULT_OUTPUT_DIR`` wins.

    Same bug class as skills_tool (f8723c478) and skills_sync (#65828): long-lived multi-profile runtimes
    (dashboard console, TUI/Desktop backend, cron, kanban workers) import this module once under the launch
    HERMES_HOME and later scope requests to a different profile via
    ``hermes_constants.set_hermes_home_override()`` — a frozen module constant keeps writing synthesized
    audio into the launch profile's cache instead of the active profile's (#98749). Keep the legacy
    ``DEFAULT_OUTPUT_DIR`` module attribute for tests and external patchers; when it has not been patched,
    re-resolve from the live profile-scoped HERMES_HOME on every call.
    """
    if DEFAULT_OUTPUT_DIR != _DEFAULT_OUTPUT_DIR_AT_IMPORT:
        return DEFAULT_OUTPUT_DIR
    return _get_default_output_dir()


def _load_tts_config() -> Dict[str, Any]:
    """Return the ``tts`` config section ({} when unavailable)."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("tts") or {}
    except ImportError:
        logger.debug("hermes_cli.config not available, using default TTS config")
    except Exception as e:
        logger.warning("Failed to load TTS config: %s", e, exc_info=True)
    return {}


def _get_provider(tts_config: Dict[str, Any]) -> str:
    """Get the explicitly configured TTS provider or the free default.

    Inference credentials do not imply consent to paid speech generation.
    Users opt into cloud TTS by setting ``tts.provider`` (normally through
    ``hermes tools``); otherwise the historical Edge backend remains active.

    The managed "Nous Subscription" selection (``tts.provider: nous``) is
    serviced by the OpenAI provider implementation, routed through the
    managed openai-audio gateway by ``_resolve_openai_audio_client_config``.
    """
    provider = (tts_config.get("provider") or DEFAULT_PROVIDER).lower().strip()
    if provider == NOUS_MANAGED_PROVIDER:
        return "openai"
    return provider


# Platforms whose native voice-bubble delivery requires Ogg/Opus (MP3 renders broken there).
OPUS_VOICE_PLATFORMS = frozenset({"telegram", "matrix", "feishu", "whatsapp", "signal"})
# Built-ins that emit Opus natively when asked for .ogg; the rest need ffmpeg for voice bubbles.
_NATIVE_OPUS_PROVIDERS = frozenset({"openai", "elevenlabs", "mistral", "gemini"})
_FFMPEG_OPUS_PROVIDERS = frozenset({"edge", "neutts", "minimax", "xai", "kittentts", "piper"})


# --- Built-in provider dispatch ---
# provider -> (availability predicate or None, log label, generator name, "package missing" error).
# Predicates/generator names resolve module globals at call time so test monkeypatches apply.
_BUILTIN_DISPATCH: Dict[str, tuple] = {
    "elevenlabs": (lambda: _importable(_import_elevenlabs), "ElevenLabs", "_generate_elevenlabs",
                   "ElevenLabs provider selected but 'elevenlabs' package not installed. Run: pip install elevenlabs"),
    "openai": (lambda: _importable(_import_openai_client), "OpenAI TTS", "_generate_openai_tts",
               "OpenAI provider selected but 'openai' package not installed."),
    "deepinfra": (lambda: _importable(_import_openai_client), "DeepInfra TTS", "_generate_deepinfra_tts",
                  "DeepInfra TTS uses the 'openai' SDK but it isn't installed."),
    "minimax": (None, "MiniMax TTS", "_generate_minimax_tts", None),
    "xai": (None, "xAI TTS", "_generate_xai_tts", None),
    "mistral": (lambda: _importable(_import_mistral_client), "Mistral Voxtral TTS", "_generate_mistral_tts",
                "Mistral provider selected but 'mistralai' package not installed. "
                "Run `hermes setup` to install Mistral support."),
    "gemini": (None, "Google Gemini TTS", "_generate_gemini_tts", None),
    "neutts": (lambda: _check_neutts_available(), "NeuTTS (local)", "_generate_neutts",
               "NeuTTS provider selected but neutts is not installed. "
               "Run hermes setup and choose NeuTTS, or install espeak-ng and run python -m pip install -U neutts[all]."),
    "kittentts": (lambda: _importable(_import_kittentts), "KittenTTS (local, ~25MB)", "_generate_kittentts",
                  "KittenTTS provider selected but 'kittentts' package not installed. "
                  "Run 'hermes setup tts' and choose KittenTTS, or install manually: "
                  "pip install https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl"),
    "piper": (lambda: _importable(_import_piper), "Piper (local)", "_generate_piper_tts",
              "Piper provider selected but 'piper-tts' package not installed. "
              "Run 'hermes tools' and select Piper under TTS, or install manually: "
              "pip install piper-tts")}


def _error_json(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _run_edge_tts(text: str, file_str: str, tts_config: Dict[str, Any]) -> None:
    """Run the async Edge generator from sync code (worker thread; direct run if that fails)."""
    run = lambda: asyncio.run(_generate_edge_tts(text, file_str, tts_config))  # noqa: E731
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run).result(timeout=60)
    except RuntimeError:
        run()


def _select_builtin_engine(provider: str) -> tuple:
    """SDK check -> ``(engine, None)`` or ``(provider, error_json)``. Unknown names take the Edge
    default; without edge-tts NeuTTS is the fallback (engine != provider)."""
    entry = _BUILTIN_DISPATCH.get(provider)
    if entry is not None:
        available, _label, _generator, missing_error = entry
        return provider, (_error_json(missing_error) if available is not None and not available() else None)
    if _importable(_import_edge_tts):
        return provider, None  # Edge default; the reported provider stays as configured
    if _check_neutts_available():
        logger.info("Edge TTS not available, falling back to NeuTTS (local)...")
        return "neutts", None
    return provider, _error_json(
        "No TTS provider available. Install edge-tts (pip install edge-tts) "
        "or set up NeuTTS for local synthesis.")


def _synthesize_builtin(engine: str, text: str, file_str: str, tts_config: Dict[str, Any], instructions: Optional[str]) -> None:
    """Run the already-selected built-in *engine*."""
    entry = _BUILTIN_DISPATCH.get(engine)
    logger.info("Generating speech with %s...", entry[1] if entry else "Edge TTS")
    if entry is None:
        _run_edge_tts(text, file_str, tts_config)
    elif engine == "openai":
        _generate_openai_tts(text, file_str, tts_config, instructions=instructions)
    else:
        globals()[entry[2]](text, file_str, tts_config)


def _finalize_voice_delivery(
    file_str: str, provider: str, command_provider_config: Optional[Dict[str, Any]], want_opus: bool,
) -> tuple:
    """Voice-bubble eligibility (Opus-converting when needed) -> ``(path, voice_compatible)``.

    Command/plugin providers are documents unless they opt in via ``voice_compatible``; native-Opus
    built-ins qualify when the platform wants Opus and they wrote .ogg; MP3/WAV built-ins are
    ffmpeg-converted only when the platform needs Opus."""
    if command_provider_config is not None:
        opted_in = _is_command_tts_voice_compatible(command_provider_config)
    elif provider not in BUILTIN_TTS_PROVIDERS:
        opted_in = _plugin_provider_is_voice_compatible(provider)
    elif want_opus and provider in _FFMPEG_OPUS_PROVIDERS and not file_str.endswith(".ogg"):
        opus_path = _convert_to_opus(file_str)
        return (opus_path, True) if opus_path else (file_str, False)
    else:
        native = provider in _NATIVE_OPUS_PROVIDERS
        return file_str, native and want_opus and file_str.endswith(".ogg")
    if not opted_in:
        return file_str, False
    # Plugin-registered provider (issue #30398). Voice-bubble delivery opts in via
    # ``TTSProvider.voice_compatible`` (mirrors the command-provider opt-in). Plugins that already write
    # Opus skip the ffmpeg conversion.
    if not file_str.endswith(".ogg"):
        file_str = _convert_to_opus(file_str) or file_str
    return file_str, file_str.endswith(".ogg")


# --- Main tool function ---
def _apply_call_overrides(tts_config: Dict[str, Any], speed: Optional[float], provider: Optional[str]):
    """Apply per-call ``speed`` (clamped, on a shallow copy so the cached config isn't mutated) and
    resolve the provider name."""
    if speed is not None:
        tts_config = {**tts_config, "speed": max(0.25, min(4.0, float(speed)))}
    return tts_config, provider.lower().strip() if provider else _get_provider(tts_config)


    response = requests.post(
        f"{base_url}/tts",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": hermes_xai_user_agent(),
        },
        json=payload,
        timeout=60,
        stream=True,
    )
    response.raise_for_status()

    _write_tts_response_to_file(response, output_path, label="xAI TTS")

    return output_path


# ===========================================================================
# Provider: MiniMax TTS
# ===========================================================================
def _generate_minimax_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using MiniMax TTS API.

    Supports two endpoints:
    - v1/text_to_speech: simple payload, returns raw audio (Content-Type: audio/mpeg)
    - v1/t2a_v2: nested voice_setting/audio_setting, returns JSON with hex-encoded audio

    Args:
        text: Text to convert (max 10,000 characters).
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    import requests

    runtime = _resolve_minimax_tts_runtime(tts_config)

    mm_config = tts_config.get("minimax", {})
    if not isinstance(mm_config, dict):
        mm_config = {}
    model = mm_config.get("model", DEFAULT_MINIMAX_MODEL)
    voice_id = mm_config.get("voice_id", DEFAULT_MINIMAX_VOICE_ID)
    base_url = runtime.endpoint
    speed = mm_config.get("speed", 1.0)
    vol = mm_config.get("vol", 1.0)
    pitch = mm_config.get("pitch", 0)
    emotion = mm_config.get("emotion", "neutral")
    sample_rate = mm_config.get("sample_rate", 32000)
    bitrate = mm_config.get("bitrate", 128000)

    # MiniMax accounts scope TTS requests by GroupId.  When present, the docs
    # show it as a ?GroupId=<id> query param on the t2a_v2 URL.  Accept it
    # from config or from the MINIMAX_GROUP_ID env var; only attach when the
    # URL doesn't already carry one.
    group_id = (
        str(mm_config.get("group_id") or "").strip()
        or (get_env_value("MINIMAX_GROUP_ID") or "").strip()
    )
    if group_id and "GroupId=" not in base_url:
        sep = "&" if "?" in base_url else "?"
        base_url = f"{base_url}{sep}GroupId={group_id}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {runtime.api_key}",
    }

    # Detect endpoint from URL
    is_t2a_v2 = "t2a_v2" in base_url

    if is_t2a_v2:
        # t2a_v2 endpoint: nested voice_setting/audio_setting structure
        payload = {
            "model": model,
            "text": text,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": vol,
                "pitch": pitch,
                "emotion": emotion,
            },
            "audio_setting": {
                "sample_rate": sample_rate,
                "bitrate": bitrate,
                "format": "mp3",
                "channel": 1,
            },
        }
    else:
        # text_to_speech endpoint: flat payload
        payload = {
            "model": model,
            "text": text,
            "voice_id": voice_id,
        }

    response = requests.post(
        base_url,
        json=payload,
        headers=headers,
        timeout=60,
        stream=True,
    )

    if is_t2a_v2:
        # t2a_v2 returns JSON with hex-encoded audio
        response.raise_for_status()
        result = _read_tts_response_json(response, label="MiniMax TTS")
        base_resp = result.get("base_resp", {})
        status_code = base_resp.get("status_code", -1)

        if status_code != 0:
            status_msg = base_resp.get("status_msg", "unknown error")
            raise RuntimeError(f"MiniMax TTS API error (code {status_code}): {status_msg}")

        hex_audio = result.get("data", {}).get("audio", "")
        if not hex_audio:
            raise RuntimeError("MiniMax TTS returned empty audio data")

        audio_bytes = bytes.fromhex(hex_audio)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return output_path

    else:
        # text_to_speech returns raw audio directly
        content_type = response.headers.get("Content-Type", "")

        if "audio/" in content_type:
            _write_tts_response_to_file(response, output_path, label="MiniMax TTS")
            return output_path

        # Fallback: try parsing as JSON
        try:
            raw_body = _read_tts_response_bytes(response, label="MiniMax TTS")
            result = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            base_resp = result.get("base_resp", {})
            status_code = base_resp.get("status_code", -1)
            if status_code != 0:
                status_msg = base_resp.get("status_msg", "unknown error")
                raise RuntimeError(f"MiniMax TTS API error (code {status_code}): {status_msg}")
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            response.raise_for_status()
            raise RuntimeError(
                f"MiniMax TTS returned unexpected Content-Type '{content_type}' "
                f"({len(raw_body) if 'raw_body' in locals() else 0} bytes)"
            )

        raise RuntimeError("MiniMax TTS returned no audio data")


# ===========================================================================
# Provider: Mistral (Voxtral TTS)
# ===========================================================================
def _generate_mistral_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate audio using Mistral Voxtral TTS API.

    The API returns base64-encoded audio; this function decodes it
    and writes the raw bytes to *output_path*.
    Supports native Opus output for Telegram voice bubbles.
    """
    api_key = (_resolve_provider_key("MISTRAL_API_KEY", "mistral") or "")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set. Get one at https://console.mistral.ai/")

    mi_config = tts_config.get("mistral") or {}
    model = mi_config.get("model", DEFAULT_MISTRAL_TTS_MODEL)
    voice_id = mi_config.get("voice_id") or DEFAULT_MISTRAL_TTS_VOICE_ID
    # Class-level base_url parity: every cloud TTS provider section supports
    # base_url. The Mistral SDK calls it server_url.
    base_url = mi_config.get("base_url")

    if output_path.endswith(".ogg"):
        response_format = "opus"
    elif output_path.endswith(".wav"):
        response_format = "wav"
    elif output_path.endswith(".flac"):
        response_format = "flac"
    else:
        response_format = "mp3"

    Mistral = _import_mistral_client()
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["server_url"] = base_url
    try:
        with Mistral(**client_kwargs) as client:
            response = client.audio.speech.complete(
                model=model,
                input=text,
                voice_id=voice_id,
                response_format=response_format,
            )
            audio_bytes = base64.b64decode(response.audio_data)
    except ValueError:
        raise
    except Exception as e:
        logger.error("Mistral TTS failed: %s", e, exc_info=True)
        raise RuntimeError(f"Mistral TTS failed: {type(e).__name__}") from e

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return output_path


# ===========================================================================
# Provider: Google Gemini TTS
# ===========================================================================
def _wrap_pcm_as_wav(
    pcm_bytes: bytes,
    sample_rate: int = GEMINI_TTS_SAMPLE_RATE,
    channels: int = GEMINI_TTS_CHANNELS,
    sample_width: int = GEMINI_TTS_SAMPLE_WIDTH,
) -> bytes:
    """Wrap raw signed-little-endian PCM with a standard WAV RIFF header.

    Gemini TTS returns audio/L16;codec=pcm;rate=24000 -- raw PCM samples with
    no container. We add a minimal WAV header so the file is playable and
    ffmpeg can re-encode it to MP3/Opus downstream.
    """
    import struct

    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    data_size = len(pcm_bytes)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,             # fmt chunk size (PCM)
        1,              # audio format (PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
    )
    data_chunk_header = struct.pack("<4sI", b"data", data_size)
    riff_size = 4 + len(fmt_chunk) + len(data_chunk_header) + data_size
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    return riff_header + fmt_chunk + data_chunk_header + pcm_bytes


def _resolve_gemini_persona_prompt_path(gemini_config: Dict[str, Any]) -> Optional[Path]:
    """Return the configured persona prompt file path, if any."""
    raw = gemini_config.get("persona_prompt_file")
    if not isinstance(raw, str) or not raw.strip():
        return None

    expanded = os.path.expandvars(raw.strip())
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        try:
            from hermes_constants import get_hermes_home
            path = get_hermes_home() / path
        except Exception:
            path = Path.cwd() / path
    return path


def _read_gemini_persona_prompt(gemini_config: Dict[str, Any]) -> str:
    """Read the Gemini persona prompt file, failing soft on config mistakes."""
    path = _resolve_gemini_persona_prompt_path(gemini_config)
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "Gemini TTS persona prompt file unavailable at %s: %s",
            path,
            exc,
        )
        return ""


def _gemini_model_supports_audio_tags(model: str) -> bool:
    """Return True for Gemini TTS models known to support expressive audio tags."""
    normalized = (model or "").strip().lower().rsplit("/", 1)[-1]
    return "gemini-3.1" in normalized and "tts" in normalized


def _gemini_audio_tags_enabled(gemini_config: Dict[str, Any], model: str) -> bool:
    raw = gemini_config.get("audio_tags")
    if isinstance(raw, dict):
        raw = raw.get("enabled")
    enabled = _config_bool(raw, default=DEFAULT_GEMINI_AUDIO_TAGS)
    if not enabled:
        return False
    if not _gemini_model_supports_audio_tags(model):
        logger.warning(
            "Gemini TTS audio_tags enabled, but model %s is not known to support "
            "Gemini audio tags; skipping hidden tag rewrite",
            model,
        )
        return False
    return True


def _clean_gemini_audio_tag_rewrite(content: str) -> str:
    clean = (content or "").strip()
    fence = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", clean, flags=re.DOTALL)
    if fence:
        clean = fence.group(1).strip()
    return clean


def _extract_auxiliary_message_content(response: Any) -> str:
    try:
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")
    except Exception:
        return ""


def _rewrite_gemini_tts_audio_tags(text: str, persona_prompt: str = "") -> str:
    """Use the configured auxiliary model to insert Gemini audio tags."""
    transcript = text.strip()
    if not transcript:
        return text

    system_prompt = (
        "You rewrite transcripts for Gemini 3.1 Flash TTS by inserting expressive "
        "audio tags.\n\n"
        "Audio tags are inline square-bracket modifiers such as [whispers], "
        "[excitedly], [very slow], [sarcastically], [laughs], [sighs], or [gasp]. "
        "There is no fixed allowlist. Use creative freeform tags generously but "
        "naturally to control tone, pace, emotional vibe, emphasis, section-level "
        "delivery, and non-verbal sounds. Use English audio tags even when the "
        "spoken transcript is not English.\n\n"
        "Rules:\n"
        "- Preserve the spoken words, order, and meaning.\n"
        "- Do not add new spoken sentences or remove existing spoken words.\n"
        "- Use square brackets for every audio tag.\n"
        "- Do not use SSML or XML tags.\n"
        "- Do not explain or comment.\n"
        "- Return only the tagged TTS script."
    )
    context = persona_prompt.strip() or "(none)"
    user_prompt = (
        "PERSONA AND DIRECTOR CONTEXT:\n"
        f"{context}\n\n"
        "TRANSCRIPT TO TAG:\n"
        f"{transcript}"
    )
    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            task=GEMINI_AUDIO_TAG_REWRITE_TASK,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        tagged = _clean_gemini_audio_tag_rewrite(_extract_auxiliary_message_content(response))
        return tagged or text
    except Exception as exc:
        logger.warning("Gemini TTS audio tag rewrite failed; using untagged text: %s", exc)
        return text


def _compose_gemini_tts_prompt(
    text: str,
    gemini_config: Dict[str, Any],
    persona_prompt: Optional[str] = None,
) -> str:
    """Build the Gemini prompt from persona direction plus the live transcript."""
    transcript = text.strip()
    if persona_prompt is None:
        persona_prompt = _read_gemini_persona_prompt(gemini_config)
    if not persona_prompt:
        return transcript

    preamble = (
        "Synthesize speech from the TRANSCRIPT only. Treat AUDIO PROFILE, "
        "SCENE, DIRECTOR'S NOTES, and SAMPLE CONTEXT as performance direction; "
        "do not speak those sections aloud."
    )

    placeholder_patterns = (
        re.compile(r"\{\{\s*transcript\s*\}\}", flags=re.IGNORECASE),
        re.compile(r"\{\s*transcript\s*\}", flags=re.IGNORECASE),
    )
    prompt = persona_prompt
    for pattern in placeholder_patterns:
        if pattern.search(prompt):
            prompt = pattern.sub(transcript, prompt)
            return f"{preamble}\n\n{prompt}".strip()

    return f"{preamble}\n\n{persona_prompt}\n\n#### TRANSCRIPT\n{transcript}".strip()


def _generate_gemini_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate audio using Google Gemini TTS.

    Gemini's generateContent endpoint with responseModalities=["AUDIO"] returns
    raw 24kHz mono 16-bit PCM (L16) as base64. We wrap it with a WAV RIFF
    header to produce a playable file, then ffmpeg-convert to MP3 / Opus if
    the caller requested those formats (same pattern as NeuTTS).

    Args:
        text: Text to convert (prompt-style; supports inline direction like
              "Say cheerfully:" and audio tags like [whispers]).
        output_path: Where to save the audio file (.wav, .mp3, or .ogg).
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    import requests

    api_key = (
        _resolve_provider_key("GEMINI_API_KEY", "gemini")
        or _resolve_provider_key("GOOGLE_API_KEY", "gemini")
    )
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/app/apikey"
        )

    raw_gemini_config = tts_config.get("gemini") or {}
    gemini_config = raw_gemini_config if isinstance(raw_gemini_config, dict) else {}
    model = str(gemini_config.get("model", DEFAULT_GEMINI_TTS_MODEL)).strip() or DEFAULT_GEMINI_TTS_MODEL
    voice = str(gemini_config.get("voice", DEFAULT_GEMINI_TTS_VOICE)).strip() or DEFAULT_GEMINI_TTS_VOICE
    base_url = str(
        gemini_config.get("base_url")
        or get_env_value("GEMINI_BASE_URL")
        or DEFAULT_GEMINI_TTS_BASE_URL
    ).strip().rstrip("/")
    persona_prompt = _read_gemini_persona_prompt(gemini_config)
    tts_script = text
    if _gemini_audio_tags_enabled(gemini_config, model):
        tts_script = _rewrite_gemini_tts_audio_tags(text, persona_prompt=persona_prompt)
    prompt_text = _compose_gemini_tts_prompt(
        tts_script,
        gemini_config,
        persona_prompt=persona_prompt,
    )
    max_len = _resolve_max_text_length("gemini", tts_config)
    if len(prompt_text) > max_len:
        raise ValueError(
            "Gemini TTS composed prompt exceeds the provider request limit "
            f"({len(prompt_text)} > {max_len} chars). Reduce the persona/audio-tag "
            "prompt or lower tts.gemini.max_text_length so long-form text is "
            "split with enough prompt headroom."
        )

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice},
                },
            },
        },
    }

    headers = {"Content-Type": "application/json"}
    if urlparse(base_url).hostname == "generativelanguage.googleapis.com":
        try:
            import hermes_cli as _hermes_cli

            _hermes_version = str(_hermes_cli.__version__)
        except Exception:
            _hermes_version = "0.0.0"
        # Include Hermes client context following Gemini's partner
        # integration guidance:
        # https://ai.google.dev/gemini-api/docs/partner-integration
        headers["X-Goog-Api-Client"] = f"hermes-agent/{_hermes_version}"

    endpoint = f"{base_url}/models/{model}:generateContent"
    response = requests.post(
        endpoint,
        params={"key": api_key},
        headers=headers,
        json=payload,
        timeout=60,
        stream=True,
    )
    if response.status_code != 200:
        # Surface the API error message when present
        raw_body = _read_tts_response_bytes(response, label="Gemini TTS")
        try:
            if raw_body:
                err = json.loads(raw_body.decode("utf-8")).get("error", {})
            elif not _response_has_explicit_stream(response) and callable(getattr(response, "json", None)):
                err = response.json().get("error", {})
            else:
                err = {}
            detail = err.get("message") or raw_body.decode("utf-8", errors="replace")[:300]
        except Exception:
            detail = raw_body.decode("utf-8", errors="replace")[:300]
        raise RuntimeError(
            f"Gemini TTS API error (HTTP {response.status_code}): {detail}"
        )

    try:
        data = _read_tts_response_json(response, label="Gemini TTS")
        parts = data["candidates"][0]["content"]["parts"]
        audio_part = next((p for p in parts if "inlineData" in p or "inline_data" in p), None)
        if audio_part is None:
            raise RuntimeError("Gemini TTS response contained no audio data")
        inline = audio_part.get("inlineData") or audio_part.get("inline_data") or {}
        audio_b64 = inline.get("data", "")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Gemini TTS response was malformed: {e}") from e

    if not audio_b64:
        raise RuntimeError("Gemini TTS returned empty audio data")

    pcm_bytes = base64.b64decode(audio_b64)
    wav_bytes = _wrap_pcm_as_wav(pcm_bytes)

    # Fast path: caller wants WAV directly, just write.
    if output_path.lower().endswith(".wav"):
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        return output_path

    # Otherwise write WAV to a temp file and ffmpeg-convert to the target
    # format (.mp3 or .ogg). If ffmpeg is missing, fall back to renaming the
    # WAV -- this matches the NeuTTS behavior and keeps the tool usable on
    # systems without ffmpeg (audio still plays, just with a misleading
    # extension).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        wav_path = tmp.name

    try:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            # For .ogg output, force libopus encoding (Telegram voice bubbles
            # require Opus specifically; ffmpeg's default for .ogg is Vorbis).
            if output_path.lower().endswith(".ogg"):
                cmd = [
                    ffmpeg, "-i", wav_path,
                    "-acodec", "libopus", "-ac", "1",
                    "-b:a", "48k", "-vbr", "on",
                    "-application", "voip", "-compression_level", "10",
                    "-y", "-loglevel", "error",
                    output_path,
                ]
            else:
                cmd = [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path]
            result = subprocess.run(cmd, capture_output=True, timeout=30, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="ignore")[:300]
                raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
        else:
            logger.warning(
                "ffmpeg not found; writing raw WAV to %s (extension may be misleading)",
                output_path,
            )
            shutil.copyfile(wav_path, output_path)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    return output_path


# ===========================================================================
# NeuTTS (local, on-device TTS via neutts_cli)
# ===========================================================================

def _check_neutts_available() -> bool:
    """Check if the neutts engine is importable (installed locally)."""
    try:
        import importlib.util
        return importlib.util.find_spec("neutts") is not None
    except Exception:
        return False


def _check_kittentts_available() -> bool:
    """Check if the kittentts engine is importable (installed locally)."""
    try:
        import importlib.util
        return importlib.util.find_spec("kittentts") is not None
    except Exception:
        return False


def _default_neutts_ref_audio() -> str:
    """Return path to the bundled default voice reference audio."""
    return str(Path(__file__).parent / "neutts_samples" / "jo.wav")


def _default_neutts_ref_text() -> str:
    """Return path to the bundled default voice reference transcript."""
    return str(Path(__file__).parent / "neutts_samples" / "jo.txt")


def _generate_neutts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate speech using the local NeuTTS engine.

    Runs synthesis in a subprocess via tools/neutts_synth.py to keep the
    ~500MB model in a separate process that exits after synthesis.
    Outputs WAV; the caller handles conversion for Telegram if needed.
    """
    import sys

    neutts_config = tts_config.get("neutts") or {}
    ref_audio = neutts_config.get("ref_audio", "") or _default_neutts_ref_audio()
    ref_text = neutts_config.get("ref_text", "") or _default_neutts_ref_text()
    model = neutts_config.get("model", "neuphonic/neutts-air-q4-gguf")
    device = neutts_config.get("device", "cpu")

    # NeuTTS outputs WAV natively — use a .wav path for generation,
    # let the caller convert to the final format afterward.
    wav_path = output_path
    if not output_path.endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"

    synth_script = str(Path(__file__).parent / "neutts_synth.py")
    cmd = [
        sys.executable, synth_script,
        "--text", text,
        "--out", wav_path,
        "--ref-audio", ref_audio,
        "--ref-text", ref_text,
        "--model", model,
        "--device", device,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Filter out the "OK:" line from stderr
        error_lines = [l for l in stderr.splitlines() if not l.startswith("OK:")]
        raise RuntimeError(f"NeuTTS synthesis failed: {chr(10).join(error_lines) or 'unknown error'}")

    # If the caller wanted .mp3 or .ogg, convert from WAV
    if wav_path != output_path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            conv_cmd = [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path]
            subprocess.run(conv_cmd, check=True, timeout=30, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
            os.remove(wav_path)
        else:
            # No ffmpeg — just rename the WAV to the expected path
            os.rename(wav_path, output_path)

    return output_path


# ===========================================================================
# Provider: Piper (local, neural VITS, 44 languages)
# ===========================================================================

# Each cached entry below is a whole loaded TTS model (tens of MB). An
# unbounded dict pins one model per distinct voice/model for the process
# lifetime, so a surface that sweeps voices grows memory with no ceiling. Cap
# each cache with a small LRU — most sessions use one or two voices, and a
# reload on a cold miss is cheap next to keeping every model resident.
_TTS_MODEL_CACHE_MAX = 3


def _tts_cache_get_or_load(cache: Dict[str, Any], key: str, load: Callable[[], Any]) -> Any:
    """Get ``key`` from ``cache`` or load it, keeping the cache LRU-bounded.

    Refreshes recency on a hit (insertion-ordered dict: pop + reinsert), loads
    on a miss, then evicts least-recently-used entries beyond the cap. An entry
    evicted while a caller still holds its returned reference stays alive for
    that caller; only the cache slot is released.
    """
    if key in cache:
        cache[key] = cache.pop(key)
        return cache[key]
    value = load()
    cache[key] = value
    while len(cache) > _TTS_MODEL_CACHE_MAX:
        cache.pop(next(iter(cache)), None)
    return value


# ===========================================================================
# Local-engine lifecycle: warm-up / release driven by TTS-output toggles
# ===========================================================================
#
# Local engines (Piper, KittenTTS) load their model lazily on the first
# synthesis call, so the first spoken reply after a user turns on "read
# replies aloud" / a voice conversation pays the whole load (plus a voice
# download on a fresh install) as dead air before the first word. And once
# loaded, the model stays resident for the process lifetime even after every
# TTS-output toggle is off again.
#
# The toggles ARE the intent signal. Every surface that flips speech output
# on holds a *lease* here (warming the configured engine as a side effect);
# flipping it off releases the lease, and when the last lease is gone the
# local model caches are dropped. Lease-counting instead of a bare
# on/off keeps one surface's "off" from unloading a model another surface
# (TUI /voice tts, desktop read-aloud, desktop conversation) still needs —
# they share this process's caches.
#
# Cloud providers have no resident model; warming them is a no-op beyond
# making sure the lazily-installed SDK is importable (edge-tts), which is
# also first-use latency users see as silence.

# Provider name → local model cache it populates. The single registry both
# warm_tts_provider() and the release path consult — a new local engine adds
# one row here (at its cache declaration) plus a loader in
# _local_tts_warmers() and gets warm/release for free.
_LOCAL_TTS_MODEL_CACHES: Dict[str, Dict[str, Any]] = {}


def _local_tts_warmers() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    # Resolved lazily: the loader functions are defined later in this module.
    return {
        "piper": lambda cfg: _load_piper_voice_for_config(cfg)[0],
        "kittentts": lambda cfg: _load_kittentts_model_for_config(cfg)[0],
    }


def _lazy_sdk_feature_for_provider(provider: str) -> Optional[str]:
    """tools.lazy_deps feature key for providers whose SDK installs on first use."""
    return {
        "edge": "tts.edge",
        "elevenlabs": "tts.elevenlabs",
        "mistral": "tts.mistral",
    }.get(provider)


_tts_lease_lock = threading.Lock()
_tts_leases: set = set()


def _signal_user_tts_provider(name: str, tts_config: Dict[str, Any], hook: str) -> Optional[str]:
    """Forward a lease ``hook`` (``"warm"`` / ``"release"``) to a user-declared provider.

    Command providers run their optional ``warm_command`` / ``release_command``
    (same template/env/timeout rules as ``command``; output discarded) on a
    background thread so a toggle never waits on a model server. Plugin
    providers get :meth:`TTSProvider.warm` / :meth:`TTSProvider.release`.
    Best-effort: failures are logged at debug. Returns the action taken.
    """
    if not name or name in BUILTIN_TTS_PROVIDERS:
        return None
    cfg = _get_named_provider_config(tts_config, name)
    try:
        if _is_command_provider_config(cfg):
            template = str(cfg.get(f"{hook}_command") or "").strip()
            if not template:
                return None
            command = _render_command_tts_template(template, {
                "voice": str(cfg.get("voice", "")),
                "model": str(cfg.get("model", "")),
                "speed": str(cfg.get("speed", tts_config.get("speed", ""))),
            })

            def _run() -> None:
                try:
                    _run_command_tts(command, _get_command_tts_timeout(cfg),
                                     env_passthrough=_command_provider_env_passthrough(cfg))
                except Exception as exc:  # noqa: BLE001 — best-effort hook
                    logger.debug("[TTS] %s_command for %s failed: %s", hook, name, exc)

            threading.Thread(target=_run, name=f"tts-{hook}-{name}", daemon=True).start()
            return hook
        from agent.tts_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        plugin_provider = get_provider(name)
        if plugin_provider is None:
            return None
        getattr(plugin_provider, hook)()
        return hook
    except Exception as exc:  # noqa: BLE001 — best-effort hook
        logger.debug("[TTS] %s hook for %s failed: %s", hook, name, exc)
        return "error"


def warm_tts_provider(
    tts_config: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Pre-load the configured TTS provider so the next synthesis starts hot.

    * Local engines (Piper, KittenTTS): resolve the configured voice/model
      exactly as synthesis would (including first-use voice download) and
      load it into the same LRU cache slot synthesis reads.
    * Lazily-installed cloud SDKs (edge-tts, ElevenLabs, Mistral): make sure
      the SDK is importable, installing it if lazy installs are allowed.
    * User-declared providers: command providers run ``warm_command`` when
      set; plugin providers get :meth:`TTSProvider.warm`.
    * Everything else: nothing to warm — reported as ``action: "noop"``.

    Never raises; the result dict carries ``warmed`` / ``action`` / ``error``
    so callers on a toggle path can log and move on. Blocking — callers on a
    UI thread should run it in the background.
    """
    if tts_config is None:
        tts_config = _load_tts_config()
    name = (provider or _get_provider(tts_config) or "").lower().strip()
    result: Dict[str, Any] = {"provider": name, "warmed": False, "action": "noop"}

    warmer = _local_tts_warmers().get(name)
    if warmer is not None:
        cache = _LOCAL_TTS_MODEL_CACHES.get(name)
        before = len(cache) if cache is not None else 0
        started = time.monotonic()
        try:
            warmer(tts_config)
        except Exception as exc:  # engine missing, download failed, bad voice…
            logger.warning("[TTS] warm-up for %s failed: %s", name, exc)
            result.update(action="error", error=str(exc))
            return result
        after = len(cache) if cache is not None else 0
        result.update(
            warmed=True,
            action="loaded" if after > before else "cached",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        logger.info("[TTS] warm-up %s: %s in %dms", name, result["action"], result["elapsed_ms"])
        return result

    signalled = _signal_user_tts_provider(name, tts_config, "warm")
    if signalled is not None:
        result.update(warmed=signalled != "error", action="warmed" if signalled != "error" else "error")
        return result

    feature = _lazy_sdk_feature_for_provider(name)
    if feature is not None:
        try:
            from tools.lazy_deps import ensure, is_available

            if is_available(feature):
                result.update(warmed=True, action="cached")
            else:
                ensure(feature, prompt=False)
                result.update(warmed=True, action="installed")
        except Exception as exc:
            logger.debug("[TTS] SDK warm-up for %s skipped: %s", name, exc)
            result.update(action="error", error=str(exc))
    return result


def release_tts_provider(provider: Optional[str] = None) -> Dict[str, Any]:
    """Drop resident local TTS models so their memory is returned.

    With ``provider`` given, only that engine's cache is cleared; otherwise
    every local engine cache is and the configured user-declared provider
    (plugin ``release()`` / command ``release_command``) is signalled.
    Cloud providers hold nothing to release.
    Returns ``{"released": <number of model instances dropped>}``. The next
    synthesis simply reloads (or a warm-up does it ahead of time).
    """
    name = (provider or "").lower().strip()
    if not name:
        tts_config = _load_tts_config()
        _signal_user_tts_provider(_get_provider(tts_config), tts_config, "release")
    released = 0
    for cache_name, cache in _LOCAL_TTS_MODEL_CACHES.items():
        if name and cache_name != name:
            continue
        released += len(cache)
        cache.clear()
    if released:
        logger.info("[TTS] released %d resident local model(s)", released)
    return {"released": released}


def acquire_tts_lease(lease: str, tts_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Register ``lease`` as a live TTS-output consumer and warm the provider.

    ``lease`` names the surface/toggle (e.g. ``"desktop:read-aloud"``,
    ``"tui:voice-tts"``). Re-acquiring an existing lease is idempotent (still
    re-warms — cheap on a cache hit, and heals a cache cleared elsewhere).
    """
    with _tts_lease_lock:
        _tts_leases.add(lease)
        holders = len(_tts_leases)
    result = warm_tts_provider(tts_config)
    result["leases"] = holders
    return result


def release_tts_lease(lease: str) -> Dict[str, Any]:
    """Drop ``lease``; when it was the last one, unload resident local models.

    Releasing a lease that was never acquired is a no-op (still reports the
    live holder count) so surfaces can call it unconditionally on their
    "off" path.
    """
    with _tts_lease_lock:
        _tts_leases.discard(lease)
        holders = len(_tts_leases)
        result: Dict[str, Any] = {"leases": holders, "released": 0}
        if holders == 0:
            result["released"] = release_tts_provider()["released"]
    return result


def tts_lease_holders() -> List[str]:
    """Snapshot of live lease names (diagnostics / tests)."""
    with _tts_lease_lock:
        return sorted(_tts_leases)


def _reset_tts_leases_for_tests() -> None:
    with _tts_lease_lock:
        _tts_leases.clear()


# Module-level cache for Piper voice instances. Voices are keyed on their
# absolute .onnx model path so switching voices doesn't invalidate older
# cached voices.
_piper_voice_cache: Dict[str, Any] = {}
_LOCAL_TTS_MODEL_CACHES["piper"] = _piper_voice_cache


def _check_piper_available() -> bool:
    """Check whether the piper-tts package is importable."""
    try:
        import importlib.util
        return importlib.util.find_spec("piper") is not None
    except Exception:
        return False


def _get_piper_voices_dir() -> Path:
    """Return the directory where Hermes caches Piper voice models.

    Resolves to ``~/.hermes/cache/piper-voices/`` under the active
    HERMES_HOME so voice downloads follow profile boundaries.
    """
    from hermes_constants import get_hermes_dir
    root = Path(get_hermes_dir("cache/piper-voices", "piper_voices_cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_piper_voice_path(voice: str, download_dir: Path) -> str:
    """Resolve *voice* (a model name or path) to a concrete .onnx file path.

    Accepts any of:
      - Absolute / expanded path to an .onnx file the user already has
      - A voice *name* like ``en_US-lessac-medium`` (downloads to
        ``download_dir`` on first use via ``python -m piper.download_voices``)

    Raises RuntimeError if the model can't be located or downloaded.
    """
    if not voice:
        voice = DEFAULT_PIPER_VOICE

    # Case 1: user gave a direct file path.
    candidate = Path(voice).expanduser()
    if candidate.suffix.lower() == ".onnx" and candidate.exists():
        return str(candidate)

    # Case 2: user gave a voice *name*. See if it's already downloaded.
    cached = download_dir / f"{voice}.onnx"
    if cached.exists() and (download_dir / f"{voice}.onnx.json").exists():
        return str(cached)

    # Case 3: download the voice. piper ships a download helper module.
    import sys as _sys
    logger.info("[Piper] Downloading voice '%s' to %s (first use)", voice, download_dir)
    try:
        result = subprocess.run(
            [_sys.executable, "-m", "piper.download_voices", voice,
             "--download-dir", str(download_dir)],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Piper voice download timed out after 300s for '{voice}'"
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "no stderr output"
        raise RuntimeError(
            f"Piper voice download failed for '{voice}': {stderr[:400]}"
        )

    if not cached.exists():
        raise RuntimeError(
            f"Piper voice download completed but {cached} is missing — "
            f"check voice name (see: https://github.com/OHF-Voice/piper1-gpl/"
            f"blob/main/docs/VOICES.md)"
        )
    return str(cached)


def _load_piper_voice_for_config(tts_config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Resolve + load (or fetch from cache) the Piper voice ``tts_config`` selects.

    Shared by synthesis and :func:`warm_tts_provider` so a warm-up populates
    exactly the cache slot the next synthesis call will hit — same voice
    resolution, same download-on-first-use, same cache key.

    Returns ``(voice, piper_config)``.
    """
    PiperVoice = _import_piper()

    piper_config = tts_config.get("piper") or {} if isinstance(tts_config, dict) else {}
    voice_name = piper_config.get("voice") or DEFAULT_PIPER_VOICE
    download_dir = Path(piper_config.get("voices_dir") or _get_piper_voices_dir()).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = bool(piper_config.get("use_cuda", False))

    model_path = _resolve_piper_voice_path(voice_name, download_dir)

    # speaker_id is applied per-call via syn_config.speaker_id — the same
    # PiperVoice instance serves all speakers, so it stays out of the cache
    # key. Multi-speaker workflows share one model load.
    cache_key = f"{model_path}::cuda={use_cuda}"

    def _load_piper_voice():
        logger.info("[Piper] Loading voice: %s", model_path)
        v = PiperVoice.load(model_path, use_cuda=use_cuda)
        logger.info("[Piper] Voice loaded")
        return v

    voice = _tts_cache_get_or_load(_piper_voice_cache, cache_key, _load_piper_voice)
    return voice, piper_config


def _generate_piper_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate speech using the local Piper engine.

    Loads the voice model once per process (cached by absolute path) and
    writes a WAV file. Caller is responsible for converting to MP3/Opus
    via ffmpeg when a different output format is required.
    """
    import wave

    voice, piper_config = _load_piper_voice_for_config(tts_config)

    # Tolerant speaker_id parse: drop bad input (non-int strings, lists, dicts)
    # to 0 (Piper's own default). Booleans are rejected outright — True/False
    # would silently coerce to 1/0 and hide a config mistake.
    _raw_speaker = piper_config.get("speaker_id", 0)
    if isinstance(_raw_speaker, bool) or not isinstance(_raw_speaker, int):
        speaker_id = 0
    else:
        speaker_id = _raw_speaker

    # Optional synthesis knobs — only pass a SynthesisConfig when at least
    # one advanced knob is configured, so we don't depend on a newer Piper
    # version than the user's installed one unless we need to.
    syn_config = None
    has_advanced = any(
        k in piper_config
        for k in (
            "length_scale",
            "noise_scale",
            "noise_w_scale",
            "volume",
            "normalize_audio",
            "speaker_id",
        )
    )
    if has_advanced:
        try:
            from piper import SynthesisConfig  # type: ignore
            syn_config = SynthesisConfig(
                length_scale=float(piper_config.get("length_scale", 1.0)),
                noise_scale=float(piper_config.get("noise_scale", 0.667)),
                noise_w_scale=float(piper_config.get("noise_w_scale", 0.8)),
                volume=float(piper_config.get("volume", 1.0)),
                normalize_audio=bool(piper_config.get("normalize_audio", True)),
                speaker_id=speaker_id,
            )
        except ImportError:
            logger.warning(
                "[Piper] SynthesisConfig not available in this piper-tts "
                "version — advanced knobs ignored"
            )

    # Piper outputs WAV. Caller handles downstream MP3/Opus conversion.
    wav_path = output_path
    if not output_path.endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"

    with wave.open(wav_path, "wb") as wav_file:
        if syn_config is not None:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        else:
            voice.synthesize_wav(text, wav_file)

    # Convert to desired format if caller requested mp3/ogg
    if wav_path != output_path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            conv_cmd = [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path]
            subprocess.run(conv_cmd, check=True, timeout=30, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
            try:
                os.remove(wav_path)
            except OSError:
                pass
        else:
            # No ffmpeg — keep WAV and return that path
            os.rename(wav_path, output_path)

    return output_path


# ===========================================================================
# Provider: KittenTTS (local, lightweight)
# ===========================================================================

# Module-level cache for KittenTTS model instance
_kittentts_model_cache: Dict[str, Any] = {}
_LOCAL_TTS_MODEL_CACHES["kittentts"] = _kittentts_model_cache


def _load_kittentts_model_for_config(tts_config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Load (or fetch from cache) the KittenTTS model ``tts_config`` selects.

    Shared by synthesis and :func:`warm_tts_provider` — same model name,
    same cache key. Returns ``(model, kittentts_config)``.
    """
    KittenTTS = _import_kittentts()
    kt_config = tts_config.get("kittentts", {}) if isinstance(tts_config, dict) else {}
    kt_config = kt_config or {}
    model_name = kt_config.get("model", DEFAULT_KITTENTTS_MODEL)

    def _load_kittentts_model():
        logger.info("[KittenTTS] Loading model: %s", model_name)
        m = KittenTTS(model_name)
        logger.info("[KittenTTS] Model loaded successfully")
        return m

    model = _tts_cache_get_or_load(_kittentts_model_cache, model_name, _load_kittentts_model)
    return model, kt_config


def _generate_kittentts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate speech using KittenTTS local ONNX model.

    KittenTTS is a lightweight TTS engine (25-80MB models) that runs
    entirely on CPU without requiring a GPU or API key.

    Args:
        text: Text to convert to speech.
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    model, kt_config = _load_kittentts_model_for_config(tts_config)
    voice = kt_config.get("voice", DEFAULT_KITTENTTS_VOICE)
    speed = kt_config.get("speed", 1.0)
    clean_text = kt_config.get("clean_text", True)

    # Generate audio (returns numpy array at 24kHz)
    audio = model.generate(text, voice=voice, speed=speed, clean_text=clean_text)

    # Save as WAV
    import soundfile as sf
    wav_path = output_path
    if not output_path.endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"

    sf.write(wav_path, audio, 24000)

    # Convert to desired format if needed
    if wav_path != output_path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            conv_cmd = [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path]
            subprocess.run(conv_cmd, check=True, timeout=30, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
            os.remove(wav_path)
        else:
            # No ffmpeg — rename the WAV to the expected path
            os.rename(wav_path, output_path)

    return output_path


# ===========================================================================
# Main tool function
# ===========================================================================
def _text_to_speech_single(
    text: str,
    output_path: Optional[str] = None,
    *,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
    provider: Optional[str] = None,
    tts_config_override: Optional[Dict[str, Any]] = None,
) -> str:
    """Synthesize one provider-safe text chunk and return one final-encoded file.

    The public :func:`text_to_speech_tool` wrapper owns long-form splitting,
    delivery packing, and post-encoding size enforcement.
    """
    if not text or not text.strip():
        return tool_error("Text is required", success=False)

    # The wrapper already normalizes text via prepare_spoken_text; the inner
    # function should not re-normalize or truncate.
    tts_config = (
        tts_config_override
        if tts_config_override is not None
        else _load_tts_config()
    )

    # When the model supplies a speed parameter, inject it into the config
    # so all downstream provider functions pick it up uniformly.
    if speed is not None:
        clamped = max(0.25, min(4.0, float(speed)))
        tts_config = dict(tts_config)  # shallow copy to avoid mutating the cache
        tts_config["speed"] = clamped

    # Allow per-call provider override; fall back to the configured default.
    if provider:
        provider = provider.lower().strip()
    else:
        provider = _get_provider(tts_config)

    # User-declared command provider (type: command under tts.providers.<name>)
    # resolves BEFORE the built-in dispatch. Built-in names short-circuit here
    # so a user's ``tts.providers.openai.command`` can't override the real
    # OpenAI handler.
    command_provider_config = _resolve_command_provider_config(provider, tts_config)

    # The wrapper splits text into provider-safe chunks before calling this
    # function. If text exceeds the cap here, it means the caller bypassed
    # the wrapper — log a warning but don't silently truncate.
    max_len = _resolve_max_text_length(provider, tts_config)
    if len(text) > max_len:
        logger.warning(
            "TTS text exceeds provider %s cap (%d > %d chars) — "
            "use text_to_speech_tool() for automatic chunking",
            provider, len(text), max_len,
        )

    # Detect platform from gateway env var to choose the best output format.
    # Several platforms deliver native voice bubbles only for Ogg/Opus
    # (Telegram, Matrix, Feishu/Lark, WhatsApp, Signal); OpenAI and
    # ElevenLabs can produce Opus natively (no ffmpeg needed). Edge TTS
    # always outputs MP3 and needs ffmpeg for conversion.
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "").lower()
    return platform, platform in OPUS_VOICE_PLATFORMS


def _resolve_output_base(
    output_path: Optional[str], provider: str, command_provider_config: Optional[Dict[str, Any]], want_opus: bool,
) -> tuple:
    """Pick the output file -> ``(Path, None)`` or ``(None, error_json)``.

    A caller path is rejected on ``..`` traversal (bug or prompt-injection; absolute is fine) and
    on protected credential/system locations. Default ``<audio cache>/tts_<timestamp>.<ext>``: the
    command format, ``.ogg`` for native-Opus providers on Opus platforms, else ``.mp3``."""
    if output_path:
        from tools.path_security import has_traversal_component
        if has_traversal_component(output_path):
            return None, _error_json(
                f"output_path contains '..' traversal component: {output_path}. "
                "Use an absolute path or one relative to the current directory without '..'.")
        file_path = Path(output_path).expanduser()
        if command_provider_config is not None:
            file_path = _configured_command_tts_output_path(file_path, command_provider_config)
        from agent.file_safety import is_write_approval_required, is_write_denied
        if is_write_denied(str(file_path)) or is_write_approval_required(str(file_path)):
            return None, _error_json(
                f"output_path targets a protected credential or system path: "
                f"{file_path}. Choose a normal audio output location.")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path(_default_output_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        if command_provider_config is not None:
            ext = _get_command_tts_output_format(command_provider_config)
        else:
            ext = "ogg" if want_opus and provider in _NATIVE_OPUS_PROVIDERS else "mp3"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = Path(_default_output_dir()) / f"tts_{timestamp}.{ext}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path, None


def _media_tag(paths: List[str], voice_compatible: bool) -> str:
    """``MEDIA:<path>`` lines; the ``[[audio_as_voice]]`` marker asks the platform for a voice bubble."""
    media_tag = "\n".join(f"MEDIA:{path}" for path in paths)
    return f"[[audio_as_voice]]\n{media_tag}" if voice_compatible else media_tag


def _tool_failure(prefix: str, provider: str, exc: BaseException) -> str:
    """Log and wrap a synthesis failure as the standard error envelope (traceback except for config errors)."""
    error_msg = f"{prefix} ({provider}): {exc}"
    logger.error("%s", error_msg, exc_info=not isinstance(exc, ValueError))
    return tool_error(error_msg, success=False)


def _text_to_speech_single(
    text: str, file_str: str, *, provider: str, tts_config: Dict[str, Any],
    command_provider_config: Optional[Dict[str, Any]], want_opus: bool, instructions: Optional[str],
) -> str:
    """Synthesize one provider-safe chunk into *file_str*; returns the result envelope.

    Command providers resolve BEFORE built-in dispatch, but built-in names short-circuit so
    ``tts.providers.openai.command`` can't shadow OpenAI. Plugins fire only for names that are
    neither; a None return falls through to built-in dispatch (unknown -> Edge default)."""
    try:
        if command_provider_config is not None:
            logger.info("Generating speech with command TTS provider '%s'...", provider)
            file_str = _generate_command_tts(
                text, file_str, provider, command_provider_config, tts_config)
        # Plugin-registered TTS backend (issue #30398). Fires when the configured provider is neither a
        # built-in nor a command-type entry, AND a plugin is registered under that name. The walrus binds
        # `_plugin_path` only when the dispatcher returns a path (i.e. a plugin was actually found); a None
        # return falls through to the built-in elif chain so unknown names hit the Edge TTS default at the
        # bottom. The dispatcher itself enforces built-ins-always-win + command-wins-over-plugin
        # defensively.
        elif provider not in BUILTIN_TTS_PROVIDERS and (
            _plugin_path := _dispatch_to_plugin_provider(text, file_str, provider, tts_config)
        ) is not None:
            file_str = _plugin_path
        else:
            provider, error = _select_builtin_engine(provider)
            if error:
                return error
            _synthesize_builtin(provider, text, file_str, tts_config, instructions)
        if not os.path.exists(file_str) or os.path.getsize(file_str) == 0:
            return _error_json(f"TTS generation produced no output (provider: {provider})")

        # Sniff once for every provider: MP3/WAV bytes in a .ogg path render as 0-second bubbles.
        file_str = _repair_ogg_container(file_str)
        file_str, voice_compatible = _finalize_voice_delivery(
            file_str, provider, command_provider_config, want_opus)
        logger.info("TTS audio saved: %s (%s bytes, provider: %s)", file_str, f"{os.path.getsize(file_str):,}", provider)
        return json.dumps({
            "success": True, "file_path": file_str, "media_tag": _media_tag([file_str], voice_compatible),
            "provider": provider, "voice_compatible": voice_compatible,
        }, ensure_ascii=False)
    except ValueError as e:
        return _tool_failure("TTS configuration error", provider, e)
    except FileNotFoundError as e:
        return _tool_failure("TTS dependency missing", provider, e)
    except Exception as e:
        return _tool_failure("TTS generation failed", provider, e)


class _ChunkFailed(Exception):
    """One chunk's synthesis returned an error envelope; message is the final tool error text."""


def _synthesize_chunks(chunks: List[str], base_path: Path, generated_artifacts: set, **single_kwargs) -> tuple:
    """Synthesize chunks into ``<base>.chunkNNN<ext>`` (or ``base`` alone) -> ``(encoded_paths, results)``.

    Every touched path lands in *generated_artifacts* for the caller's sweep. Raises
    :class:`_ChunkFailed` on a reported failure, ``RuntimeError`` on garbage or missing audio."""
    provider = single_kwargs["provider"]
    encoded_paths: List[str] = []
    chunk_results: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_path = base_path
        if len(chunks) > 1:
            chunk_path = base_path.with_name(f"{base_path.stem}.chunk{index:03d}{base_path.suffix}")
        generated_artifacts.add(str(chunk_path))
        raw_result = _text_to_speech_single(chunk, str(chunk_path), **single_kwargs)
        try:
            chunk_result = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            raise RuntimeError(f"TTS chunk {index} returned invalid JSON: {str(raw_result)[:200]}")
        if not chunk_result.get("success"):
            error_msg = chunk_result.get("error", "unknown error")
            raise _ChunkFailed(f"TTS chunk {index} failed ({provider}): {error_msg}")
        actual_path = str(chunk_result.get("file_path") or chunk_path)
        if not os.path.isfile(actual_path) or os.path.getsize(actual_path) <= 0:
            raise RuntimeError(f"TTS chunk {index} produced no final audio: {actual_path}")
        generated_artifacts.add(actual_path)
        encoded_paths.append(actual_path)
        chunk_results.append(chunk_result)
    return encoded_paths, chunk_results


def text_to_speech_tool(
    text: str, output_path: Optional[str] = None, speed: Optional[float] = None,
    instructions: Optional[str] = None, provider: Optional[str] = None) -> str:
    """Convert text to speech with long-form chunking; returns the JSON result envelope.

    Text is normalized, split into provider-safe chunks (never silently truncated), synthesized
    sequentially, then packed against the platform's upload limit: a failed combine keeps the
    separate valid files and no over-limit artifact is ever returned."""
    if not text or not text.strip():
        return tool_error("Text is required", success=False)
    try:  # shared cleaner: markdown, emoji, think blocks, verifier footer, units, newlines
        from tools.tts_text_normalize import prepare_spoken_text
        text = prepare_spoken_text(text, max_chars=None)
    except Exception:
        text = text.strip()
    if not text:
        return tool_error("Text is empty after TTS cleanup", success=False)
    tts_config, provider = _apply_call_overrides(_load_tts_config(), speed, provider)
    command_provider_config = _resolve_command_provider_config(provider, tts_config)
    max_len = _resolve_max_text_length(provider, tts_config)
    chunks = _split_text_for_tts(text, max_len)
    if not chunks:
        return tool_error("Text is required", success=False)
    if len(chunks) > 1:
        logger.info("TTS text for provider %s split into %d chunks (input=%d chars, cap=%d)",
                    provider, len(chunks), len(text), max_len)
    platform, want_opus = _session_platform()
    delivery_profile = _resolve_audio_delivery_profile(platform, tts_config)

    # Determine output path (single-chunk short-circuit uses the final path).
    if output_path:
        from tools.path_security import has_traversal_component
        if has_traversal_component(output_path):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path contains '..' traversal component: {output_path}. "
                    "Use an absolute path or one relative to the current directory "
                    "without '..'."
                ),
            }, ensure_ascii=False)
        base_path = Path(output_path).expanduser()
        if command_provider_config is not None:
            base_path = _configured_command_tts_output_path(
                base_path, command_provider_config,
            )
        from agent.file_safety import is_write_approval_required, is_write_denied
        if is_write_denied(str(base_path)) or is_write_approval_required(str(base_path)):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path targets a protected credential or system path: "
                    f"{base_path}. Choose a normal audio output location."
                ),
            }, ensure_ascii=False)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path(_default_output_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        if command_provider_config is not None:
            fmt = _get_command_tts_output_format(command_provider_config)
            base_path = out_dir / f"tts_{timestamp}.{fmt}"
        elif want_opus and provider in {"openai", "elevenlabs", "mistral", "gemini"}:
            base_path = out_dir / f"tts_{timestamp}.ogg"
        else:
            base_path = out_dir / f"tts_{timestamp}.mp3"
    base_path.parent.mkdir(parents=True, exist_ok=True)

    generated_artifacts: set[str] = set()
    final_paths: List[str] = []
    try:
        encoded_paths, chunk_results = _synthesize_chunks(
            chunks, base_path, generated_artifacts, provider=provider, tts_config=tts_config,
            command_provider_config=command_provider_config, want_opus=want_opus,
            instructions=instructions)
        voice_compatible = bool(chunk_results) and all(bool(r.get("voice_compatible")) for r in chunk_results)
        delivery_base = base_path.with_suffix(Path(encoded_paths[0]).suffix)
        final_paths, combined_chunks = _build_audio_delivery_files(
            encoded_paths, str(delivery_base), delivery_profile, voice_compatible=voice_compatible)
        for path in final_paths:
            logger.info("TTS audio saved: %s (%s bytes, provider: %s)", path, f"{os.path.getsize(path):,}", provider)
        return json.dumps({
            "success": True, "file_path": final_paths[0], "file_paths": final_paths,
            "media_tag": _media_tag(final_paths, voice_compatible),
            "provider": chunk_results[0].get("provider", provider), "voice_compatible": voice_compatible,
            "chunk_count": len(chunks), "delivery_file_count": len(final_paths),
            "combined_chunks": bool(combined_chunks),
            "delivery_profile": {
                "platform": delivery_profile.platform, "max_file_bytes": delivery_profile.max_file_bytes,
                "target_file_bytes": delivery_profile.target_file_bytes},
        }, ensure_ascii=False)
    except _ChunkFailed as exc:
        return tool_error(str(exc), success=False)
    except ValueError as exc:
        return _tool_failure("TTS delivery error", provider, exc)
    except Exception as exc:
        return _tool_failure("TTS long-form generation failed", provider, exc)
    finally:
        final_absolute = {os.path.abspath(path) for path in final_paths}
        for artifact in generated_artifacts:
            if os.path.abspath(artifact) not in final_absolute:
                _remove_quietly(artifact)


# --- check_fn ---
def _minimax_requirements() -> bool:
    try:
        _resolve_minimax_tts_runtime(_load_tts_config())
        return True
    except ValueError:
        return False


def _xai_requirements() -> bool:
    try:
        from tools.xai_http import resolve_xai_http_credentials
        return bool(resolve_xai_http_credentials().get("api_key"))
    except Exception:
        return False


# Must mirror text_to_speech_tool dispatch: unrelated cloud credentials never make the Edge
# default usable, and an explicit provider is checked on its own.
_BUILTIN_REQUIREMENTS: Dict[str, Callable[[], bool]] = {
    "edge": lambda: _importable(_import_edge_tts) or _check_neutts_available(),
    "elevenlabs": lambda: _importable(_import_elevenlabs) and bool(_resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")),
    "openai": lambda: _package_installed("openai") and _has_openai_audio_backend(),
    "deepinfra": lambda: _package_installed("openai") and bool(_resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")),
    "minimax": _minimax_requirements,
    "xai": _xai_requirements,
    "gemini": lambda: bool(_resolve_provider_key("GEMINI_API_KEY", "gemini") or _resolve_provider_key("GOOGLE_API_KEY", "gemini")),
    "mistral": lambda: _importable(_import_mistral_client) and bool(_resolve_provider_key("MISTRAL_API_KEY", "mistral")),
    "neutts": lambda: _check_neutts_available(),
    "kittentts": lambda: _check_kittentts_available(),
    "piper": lambda: _check_piper_available()}

    ``is_managed`` is True when the config resolves to the Nous managed audio
    gateway (a restricted proxy), so callers can coerce the request to what the
    gateway supports.

    Strict selection semantics (switch on the stored ``tts`` provider
    string):
    - ``"nous"`` (or legacy ``use_gateway: true``) → managed gateway ONLY;
      unentitled/unreachable is a selection-naming error.
    - any other stored tts provider → direct credentials ONLY
      (``tts.openai.api_key`` then ``VOICE_TOOLS_OPENAI_KEY``/
      ``OPENAI_API_KEY``); missing credentials is a selection-naming error —
      no silent managed fallback.
    - never-configured tts section → legacy ladder: config key → env key →
      managed gateway.
    """
    tts_config = _load_tts_config()
    openai_cfg = (tts_config.get("openai") if isinstance(tts_config, dict) else None) or {}
    cfg_api_key = openai_cfg.get("api_key") or ""
    cfg_base_url = openai_cfg.get("base_url") or ""

    selected = read_selection("tts")

    if selected == NOUS_MANAGED_PROVIDER:
        managed_gateway = resolve_managed_tool_gateway("openai-audio")
        if managed_gateway is None:
            raise ValueError(selection_error(
                "tts",
                NOUS_MANAGED_PROVIDER,
                "the Nous Tool Gateway is not available (not entitled or "
                "unreachable)",
            ))
        return (
            managed_gateway.nous_user_token,
            urljoin(f"{managed_gateway.gateway_origin.rstrip('/')}/", "v1"),
            True,
        )

    if selected is not None:
        # Stored vendor selection: direct credentials only.
        if cfg_api_key:
            return cfg_api_key, (cfg_base_url or DEFAULT_OPENAI_BASE_URL), False
        direct_api_key = resolve_openai_audio_api_key()
        if direct_api_key:
            return direct_api_key, (cfg_base_url or DEFAULT_OPENAI_BASE_URL), False
        raise ValueError(selection_error(
            "tts",
            selected,
            "neither tts.openai.api_key in config nor "
            "VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set",
        ))

    # Never-configured tts section: legacy credential ladder.
    if cfg_api_key:
        return cfg_api_key, (cfg_base_url or DEFAULT_OPENAI_BASE_URL), False

    direct_api_key = resolve_openai_audio_api_key()
    if direct_api_key:
        return direct_api_key, (cfg_base_url or DEFAULT_OPENAI_BASE_URL), False

    managed_gateway = resolve_managed_tool_gateway("openai-audio")
    if managed_gateway is None:
        message = (
            "Neither tts.openai.api_key in config nor "
            "VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        )
        if managed_nous_tools_enabled():
            message += (
                ". "
                + nous_tool_gateway_unavailable_message(
                    "managed OpenAI audio for TTS",
                )
            )
        raise ValueError(message)

    return (
        managed_gateway.nous_user_token,
        urljoin(f"{managed_gateway.gateway_origin.rstrip('/')}/", "v1"),
        True,
    )


def _has_openai_audio_backend() -> bool:
    """Return True when the selected OpenAI audio route is usable."""
    try:
        _resolve_openai_audio_client_config()
        return True
    except ValueError:
        return False


# ===========================================================================
# Streaming TTS: sentence-by-sentence pipeline
# ===========================================================================
# Markdown stripping patterns (same as cli.py _voice_speak_response)
_MD_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_MD_URL = re.compile(r'https?://\S+')
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'\*(.+?)\*')
_MD_INLINE_CODE = re.compile(r'`(.+?)`')
_MD_HEADER = re.compile(r'^#+\s*', flags=re.MULTILINE)
_MD_LIST_ITEM = re.compile(r'^\s*[-*]\s+', flags=re.MULTILINE)
_MD_HR = re.compile(r'---+')
_MD_EXCESS_NL = re.compile(r'\n{3,}')
# Emoji + variation selectors/ZWJ — TTS providers render these as awkward
# pauses or literal descriptions ("smiling face"), breaking the speech flow.
_EMOJI = re.compile(
    '[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\U000E0020-\U000E007F]+'
)

# Strip <think>...</think> reasoning blocks before TTS — models with
# /reasoning show enabled produce think blocks that shouldn't be spoken.
_THINK_BLOCK = re.compile(r'<think[\s>].*?</think>', flags=re.DOTALL)


def _strip_markdown_for_tts(text: str) -> str:
    """Prepare text for speech via the shared cleaner in tts_text_normalize.

    One cleaner for every TTS path (tool, gateway auto-TTS, voice-mode
    streaming, web dashboard): strips <think> reasoning blocks, the
    file-mutation verifier footer, markdown, and emoji; expands units and
    symbols; and flattens newlines to sentence breaks so newline-sensitive
    providers (Kokoro) speak the whole script.  Falls back to the legacy
    regex pipeline if the normalizer ever fails.
    """
    try:
        from tools.tts_text_normalize import prepare_spoken_text
        return prepare_spoken_text(text, max_chars=None)
    except Exception:
        pass
    text = _THINK_BLOCK.sub(' ', text)
    text = _MD_CODE_BLOCK.sub(' ', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_URL.sub('', text)
    text = _MD_BOLD.sub(r'\1', text)
    text = _MD_ITALIC.sub(r'\1', text)
    text = _MD_INLINE_CODE.sub(r'\1', text)
    text = _MD_HEADER.sub('', text)
    text = _MD_LIST_ITEM.sub('', text)
    text = _MD_HR.sub('', text)
    text = _EMOJI.sub(' ', text)
    text = _MD_EXCESS_NL.sub('\n\n', text)
    return text.strip()


class _SyncSentencePipeline:
    """Overlap per-sentence synthesis with playback for non-streaming providers.

    The universal sync fallback used to run strictly serially per sentence —
    synthesize, play, and only then start synthesizing the next sentence — so
    every sentence boundary added a full synthesis-time of dead air. For local
    model providers that cost dominates the conversation: a provider at
    real-time-factor ~1 spends as long silent between sentences as it does
    speaking. Chunked streamers already avoid this; this closes the same gap
    for everyone else (edge, piper, plugin providers, …) without touching the
    provider contract.

    Shape: one synthesis worker (single-threaded executor, so sentences are
    synthesized FIFO and providers never see concurrent calls from this loop —
    same effective concurrency as the serial path) feeding one playback worker
    through a small bounded queue. While sentence *n* plays, sentence *n+1* is
    already synthesizing. The bound keeps lookahead — and the temp files it
    implies — small, and gives natural backpressure to the caller.

    ``synthesize``/``play`` are resolved late (module global / import inside
    the worker) so tests that monkeypatch ``text_to_speech_tool`` or
    ``tools.voice_mode`` keep working unchanged.
    """

    def __init__(self, stop_event: threading.Event, *, lookahead: int = 2):
        self._stop = stop_event
        self._queue: "queue.Queue[Optional[tuple[str, Future]]]" = queue.Queue(
            maxsize=max(1, lookahead)
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts-sync-synth"
        )
        self._player = threading.Thread(
            target=self._drain, name="tts-sync-play", daemon=True
        )
        self._player.start()

    def speak(self, cleaned: str) -> None:
        """Queue one sentence. Blocks only when the lookahead bound is full."""
        if self._stop.is_set():
            return
        future = self._executor.submit(self._synthesize_to_tmp, cleaned)
        self._queue.put((cleaned, future))

    def close(self) -> None:
        """Flush queued sentences in order (skipped if stopped), then join."""
        self._queue.put(None)
        self._player.join()
        self._executor.shutdown(wait=True)

    def _synthesize_to_tmp(self, cleaned: str) -> Optional[str]:
        if self._stop.is_set():
            return None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            text_to_speech_tool(text=cleaned, output_path=tmp_path)
            return tmp_path
        except Exception as exc:
            logger.warning("Sync per-sentence TTS synthesis failed: %s", exc)
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return None

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            _sentence, future = item
            tmp_path = None
            try:
                tmp_path = future.result()
                if (tmp_path and not self._stop.is_set()
                        and os.path.isfile(tmp_path)
                        and os.path.getsize(tmp_path) > 0):
                    from tools.voice_mode import play_audio_file
                    play_audio_file(tmp_path)
            except Exception as exc:
                logger.warning("Sync per-sentence TTS failed: %s", exc)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass


def stream_tts_to_speaker(
    text_queue: queue.Queue,
    stop_event: threading.Event,
    tts_done_event: threading.Event,
    display_callback: Optional[Callable[[str], None]] = None,
    provider: Optional[str] = None,
):
    """Consume text deltas from *text_queue*, buffer them into sentences, and
    speak each sentence the moment it's ready — the conversational path.

    Provider-agnostic. A registered streaming provider (ElevenLabs, OpenAI, …)
    plays chunked PCM through one sounddevice stream for the lowest latency;
    every other provider (edge, the default) is spoken per-sentence via the sync
    ``text_to_speech_tool`` path, so audio still starts on sentence one instead
    of after the whole reply.

    Protocol:
        * The producer puts ``str`` deltas onto *text_queue*.
        * A ``None`` sentinel signals end-of-text (flush remaining buffer).
        * *stop_event* can be set to abort early (barge-in / user interrupt).
        * *tts_done_event* is **set** in the ``finally`` block so callers
          waiting on it (continuous voice mode) know playback is finished.
    """
    tts_done_event.clear()
    sync_pipeline: Optional[_SyncSentencePipeline] = None

    try:
        output_stream = None
        streamer = None  # type: ignore[assignment]
        _worker_thread = None
        _audio_queue = None  # type: ignore[assignment]
        _prefetch_threads = []
        tts_config = _load_tts_config()

        # Prefer a chunked streamer for low time-to-first-audio; fall back to
        # per-sentence sync synthesis (universal — edge + every non-streamer).
        from tools.tts_streaming import SentenceChunker, resolve_streaming_provider
        streamer = resolve_streaming_provider(tts_config, preferred=provider)

        # No chunked streamer: per-sentence sync synthesis, pipelined so the
        # next sentence synthesizes while the current one plays (closed in the
        # finally block, which flushes anything still queued).
        sync_pipeline = _SyncSentencePipeline(stop_event) if streamer is None else None

        stream_max_len = 0
        if streamer is not None:
            try:
                stream_max_len = _resolve_max_text_length(
                    provider or _get_provider(tts_config), tts_config
                )
            except Exception:
                stream_max_len = 0
            # On macOS, skip the sounddevice OutputStream entirely: PortAudio/
            # CoreAudio init triggers a kTCCServiceMediaLibrary permission
            # prompt even though output needs no media-library access. Leaving
            # output_stream=None routes each sentence through the tempfile
            # -> play_audio_file -> afplay path. See PR #62601 / #13291.
            if platform.system() == "Darwin":
                output_stream = None
            else:
                try:
                    sd = _import_sounddevice()
                    output_stream = sd.OutputStream(
                        samplerate=streamer.sample_rate,
                        channels=streamer.channels,
                        dtype="int16",
                    )
                    output_stream.start()
                except (ImportError, OSError) as exc:
                    logger.debug("sounddevice not available, streamer→tempfile: %s", exc)
                    output_stream = None
                except Exception as exc:
                    logger.warning("sounddevice OutputStream failed: %s", exc)
                    output_stream = None

        chunker = SentenceChunker()
        long_flush_len = 100
        queue_timeout = 0.5
        _spoken_sentences: list[str] = []  # track spoken sentences to skip duplicates

        # --- Per-sentence prefetch pipeline ---
        # Every sentence gets its own streamer.stream() call the moment it's
        # complete. A background prefetch thread fires the HTTP request
        # immediately, buffering PCM chunks into a per-segment queue. The
        # single playback worker drains these queues in FIFO order. This
        # means sentence N+1's HTTP request fires WHILE sentence N is still
        # playing, so by the time the worker reaches it, audio is already
        # arriving — no inter-sentence gap.
        _audio_queue: queue.Queue[Optional[queue.Queue[Optional[bytes]]]] = queue.Queue()
        _prefetch_threads: list[threading.Thread] = []
        _prefetch_sem = threading.Semaphore(3)
        _CHUNK_QUEUE_MAX = 64

        def _create_output_stream():
            """Create and start a fresh PortAudio OutputStream."""
            sd = _import_sounddevice()
            new_stream = sd.OutputStream(
                samplerate=streamer.sample_rate,
                channels=streamer.channels,
                dtype="int16",
            )
            new_stream.start()
            return new_stream

        def _consume_to_queue(
            audio_iter: Iterator[bytes],
            chunk_queue: "queue.Queue[Optional[bytes]]",
        ) -> None:
            """Consume a generator into a thread-safe queue."""
            try:
                for chunk in audio_iter:
                    if stop_event.is_set():
                        logger.info(
                            "TTS CUT: prefetch cancelled (stop_event set "
                            "mid-sentence) — partial audio only"
                        )
                        break
                    chunk_queue.put(chunk, timeout=30.0)
            except Exception as exc:
                logger.warning(
                    "TTS CUT: streaming TTS prefetch failed mid-sentence "
                    "(partial audio only): %s",
                    exc,
                )
            finally:
                chunk_queue.put(None)  # sentinel: no more chunks
                _prefetch_sem.release()  # free a prefetch slot

        def _reinit_output_stream():
            """Close the broken PortAudio stream and try to create a fresh one."""
            nonlocal output_stream
            if output_stream is not None:
                try:
                    output_stream.stop()
                    output_stream.close()
                except Exception:
                    pass
            try:
                new_stream = _create_output_stream()
                output_stream = new_stream
                logger.info(
                    "TTS: PortAudio output stream reinitialized after error"
                )
                return new_stream
            except Exception as exc:
                logger.warning(
                    "TTS: PortAudio stream reinit failed: %s", exc
                )
                output_stream = None
                return None

        def _playback_worker() -> None:
            """Single consumer: play audio segments from the queue in order."""
            assert streamer is not None
            if output_stream is not None:
                import numpy as _np

                try:
                    from tools.voice_mode import mark_audio_output_active
                except Exception:
                    def mark_audio_output_active(_active):
                        return None

                mark_audio_output_active(True)
                try:
                    _max_reinit = 3
                    _reinit_count = 0
                    _current_stream = output_stream
                    while True:
                        chunk_queue = _audio_queue.get()
                        if chunk_queue is None:
                            break
                        if stop_event.is_set():
                            continue
                        if _current_stream is None:
                            _chunks = []
                            while True:
                                chunk = chunk_queue.get()
                                if chunk is None:
                                    break
                                _chunks.append(chunk)
                            _play_via_tempfile(
                                iter(_chunks), stop_event, streamer.sample_rate
                            )
                            continue
                        _pcm_leftover = b""
                        while True:
                            chunk = chunk_queue.get()
                            if chunk is None:
                                break
                            if stop_event.is_set():
                                break
                            _buf = _pcm_leftover + chunk
                            _aligned_len = len(_buf) - (len(_buf) % 2)
                            if _aligned_len >= 2:
                                try:
                                    _current_stream.write(
                                        _np.frombuffer(
                                            _buf[:_aligned_len], dtype="<i2"
                                        ).reshape(-1, 1)
                                    )
                                except Exception as write_exc:
                                    logger.warning(
                                        "PortAudio write failed, attempting "
                                        "stream reinit: %s",
                                        write_exc,
                                    )
                                    if _reinit_count < _max_reinit:
                                        _reinit_count += 1
                                        _current_stream = _reinit_output_stream()
                                        if _current_stream is not None:
                                            try:
                                                _current_stream.write(
                                                    _np.frombuffer(
                                                        _buf[:_aligned_len],
                                                        dtype="<i2",
                                                    ).reshape(-1, 1)
                                                )
                                            except Exception:
                                                pass
                                            _pcm_leftover = (
                                                _buf[_aligned_len:]
                                                if _aligned_len < len(_buf)
                                                else b""
                                            )
                                            continue
                                    else:
                                        logger.warning(
                                            "TTS: PortAudio reinit exhausted "
                                            "after %d attempts, falling back "
                                            "to tempfile for remaining "
                                            "sentences",
                                            _max_reinit,
                                        )
                                        _current_stream = None
                                    break
                            _pcm_leftover = (
                                _buf[_aligned_len:] if _aligned_len < len(_buf) else b""
                            )
                finally:
                    mark_audio_output_active(False)
            else:
                while True:
                    chunk_queue = _audio_queue.get()
                    if chunk_queue is None:
                        break
                    if stop_event.is_set():
                        continue
                    _chunks = []
                    while True:
                        chunk = chunk_queue.get()
                        if chunk is None:
                            break
                        _chunks.append(chunk)
                    _play_via_tempfile(
                        iter(_chunks), stop_event, streamer.sample_rate
                    )

        def _enqueue_audio(text_to_speak: str) -> None:
            """Synthesize *text_to_speak* and start prefetching immediately."""
            assert streamer is not None
            try:
                audio_iter = streamer.stream(text_to_speak)
            except Exception as exc:
                logger.warning("Streaming TTS synthesis failed: %s", exc)
                return
            _prefetch_sem.acquire()
            chunk_queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=_CHUNK_QUEUE_MAX)
            _audio_queue.put(chunk_queue)
            t = threading.Thread(
                target=_consume_to_queue,
                args=(audio_iter, chunk_queue),
                daemon=True,
            )
            _prefetch_threads.append(t)
            t.start()

        _worker_thread: Optional[threading.Thread] = None
        if streamer is not None:
            _worker_thread = threading.Thread(target=_playback_worker, daemon=True)
            _worker_thread.start()

        def _speak_sentence(sentence: str):
            """Display sentence and route to the appropriate audio path."""
            if stop_event.is_set():
                return
            cleaned = _strip_markdown_for_tts(sentence).strip()
            if not cleaned:
                return
            # Skip duplicate/near-duplicate sentences (LLM repetition)
            cleaned_lower = cleaned.lower().rstrip(".!,")
            for prev in _spoken_sentences:
                if prev.lower().rstrip(".!,") == cleaned_lower:
                    return
            _spoken_sentences.append(cleaned)
            # Display raw sentence on screen before TTS processing
            if display_callback is not None:
                display_callback(sentence)
            # No chunked streamer → per-sentence sync synthesis (universal),
            # pipelined: this enqueues and returns, so sentence n+1 is already
            # synthesizing while sentence n is still playing.
            if sync_pipeline is not None:
                sync_pipeline.speak(cleaned)
                return
            # Truncate very long sentences to the provider's per-request cap.
            if stream_max_len and len(cleaned) > stream_max_len:
                cleaned = cleaned[:stream_max_len]
            # Every sentence gets its own prefetch thread — the HTTP request
            # fires the moment the sentence boundary is detected, so audio for
            # sentence N+1 is already buffering while sentence N plays.
            _enqueue_audio(cleaned)

        def _align_int16_chunks(chunks, stop_evt):
            """Yield int16-aligned byte chunks from an iterable."""
            leftover = b""
            for chunk in chunks:
                if stop_evt.is_set():
                    break
                buf = leftover + chunk
                aligned_len = len(buf) - (len(buf) % 2)
                if aligned_len >= 2:
                    yield buf[:aligned_len]
                leftover = buf[aligned_len:] if aligned_len < len(buf) else b""
            if leftover:
                yield b"\x00"

        def _play_via_tempfile(audio_iter, stop_evt, sample_rate=24000):
            """Write PCM chunks to a temp WAV file and play it."""
            tmp = None
            tmp_path = None
            try:
                import wave
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    for aligned in _align_int16_chunks(audio_iter, stop_evt):
                        wf.writeframes(aligned)
                # wave.open() given a file object flushes but does NOT close it
                # (it only closes files it opened itself, by name), so the OS
                # handle to tmp stays open.  On Windows an open write handle
                # blocks the system player from reading the file and blocks the
                # os.unlink() below (WinError 32, swallowed → temp .wav files
                # pile up).  Release the handle before playback and cleanup.
                tmp.close()
                from tools.voice_mode import play_audio_file
                play_audio_file(tmp_path)
            except Exception as exc:
                logger.warning("Temp-file TTS fallback failed: %s", exc)
            finally:
                if tmp is not None:
                    try:
                        tmp.close()  # idempotent; ensures close on early error
                    except Exception:
                        pass
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        while not stop_event.is_set():
            # Read next delta from queue
            try:
                delta = text_queue.get(timeout=queue_timeout)
            except queue.Empty:
                # Idle producer: flush a long buffer instead of sitting on it
                if len(chunker.buf) > long_flush_len:
                    for sentence in chunker.flush():
                        _speak_sentence(sentence)
                continue

            if delta is None:
                # End-of-text sentinel: flush whatever remains
                for sentence in chunker.flush():
                    _speak_sentence(sentence)
                break

            for sentence in chunker.feed(delta):
                _speak_sentence(sentence)

        # Drain any remaining items from the queue
        while True:
            try:
                text_queue.get_nowait()
            except queue.Empty:
                break

        # output_stream is closed in the finally block below

    except Exception as exc:
        logger.warning("Streaming TTS pipeline error: %s", exc)
    finally:
        # Flush the sync pipeline first: queued sentences finish playing (or
        # are skipped when stop_event is set) BEFORE tts_done_event fires, so
        # continuous voice mode never reopens the mic over its own voice.
        if sync_pipeline is not None:
            try:
                sync_pipeline.close()
            except Exception:
                pass
        # Signal the playback worker that no more audio is coming.  This lives
        # in finally: so an exception in the text pump still sends the sentinel.
        if streamer is not None and _worker_thread is not None:
            _audio_queue.put(None)
            _worker_thread.join(timeout=300.0)
        for t in _prefetch_threads:
            t.join(timeout=10.0)
        # Always close the audio output stream to avoid locking the device
        if output_stream is not None:
            try:
                output_stream.stop()
                output_stream.close()
            except Exception:
                pass
        tts_done_event.set()


# ===========================================================================
# Main -- quick diagnostics
# ===========================================================================
if __name__ == "__main__":
    print("🔊 Text-to-Speech Tool Module")
    print("=" * 50)

    def _check(importer, label):
        try:
            importer()
            return True
        except ImportError:
            return False

    print("\nProvider availability:")
    print(f"  Edge TTS:   {'installed' if _check(_import_edge_tts, 'edge') else 'not installed (pip install edge-tts)'}")
    print(f"  ElevenLabs: {'installed' if _check(_import_elevenlabs, 'el') else 'not installed (pip install elevenlabs)'}")
    print(f"    API Key:  {'set' if _resolve_provider_key('ELEVENLABS_API_KEY', 'elevenlabs') else 'not set'}")
    print(f"  OpenAI:     {'installed' if _check(_import_openai_client, 'oai') else 'not installed'}")
    print(
        "    API Key:  "
        f"{'set' if resolve_openai_audio_api_key() else 'not set (VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY)'}"
    )
    config = _load_tts_config()
    try:
        minimax_runtime = _resolve_minimax_tts_runtime(config)
        minimax_status = (
            f"API key set ({minimax_runtime.region}, "
            f"{minimax_runtime.credential_source})"
        )
    except ValueError as exc:
        minimax_status = f"unavailable ({exc})"
    print(f"  MiniMax:    {minimax_status}")
    print(f"  Piper:      {'installed' if _check_piper_available() else 'not installed (pip install piper-tts)'}")
    print(f"  ffmpeg:     {'✅ found' if _has_ffmpeg() else '❌ not found (needed for Telegram Opus)'}")
    print(f"\n  Output dir: {_default_output_dir()}")

    provider = _get_provider(config)
    print(f"  Configured provider: {provider}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

def _output_path_description(home: str) -> str:
    return f"Optional custom file path to save the audio. Defaults to {home}/audio_cache/<timestamp>.mp3"


def _tts_schema_overrides() -> dict:
    """Rebuild the ``output_path`` default hint from the ACTIVE profile at every get_definitions():
    the multiplexed gateway serves every profile from one process, so a path baked in at import
    would name the launch profile's home for everyone else (#95685)."""
    params = copy.deepcopy(TTS_SCHEMA["parameters"])
    params["properties"]["output_path"]["description"] = _output_path_description(display_hermes_home())
    return {"parameters": params}


TTS_SCHEMA = {
    "name": "text_to_speech",
    "description": "Convert text to speech audio. Returns a MEDIA: path that the platform delivers as native audio. Compatible providers render as a voice bubble on Telegram; otherwise audio is sent as a regular attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider are user-configured (built-in providers like edge/openai or custom command providers under tts.providers.<name>), not model-selected.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to convert to speech. Provider-specific per-request character caps apply automatically (OpenAI 4096, xAI 15000, MiniMax 10000, ElevenLabs 5k-40k depending on model); longer input is split into ordered chunks without silent truncation."
            },
            "output_path": {
                "type": "string",
                "description": _output_path_description("the profile HERMES_HOME")
            },
            "speed": {
                "type": "number",
                "description": "Playback speed multiplier. 1.0 = normal, 0.5 = very slow (language learning), 2.0 = fast. Range: 0.25-4.0. Overrides the speed configured in config.yaml."
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional voice-design guidance: tone, emotion, pacing, accent, "
                    "whispering, impressions (e.g. 'Speak in a cheerful, excited whisper'). "
                    "Forwarded to the OpenAI backend (gpt-4o-mini-tts and OpenAI-compatible "
                    "voice-design servers). Silently ignored by backends that don't support it."
                )
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional TTS provider override. Accepts built-in names "
                    "(edge, openai, elevenlabs, minimax, xai, mistral, gemini, "
                    "neutts, kittentts, piper), user-declared command provider "
                    "names from tts.providers.<name>, or plugin-registered names. "
                    "When omitted, the configured tts.provider from config.yaml is used."
                )
            }
        },
        "required": ["text"]
    }
}

registry.register(
    name="text_to_speech",
    toolset="tts",
    schema=TTS_SCHEMA,
    handler=lambda args, **kw: text_to_speech_tool(
        text=args.get("text", ""),
        **{k: args.get(k) for k in ("output_path", "speed", "instructions", "provider")}),
    check_fn=check_tts_requirements,
    emoji="🔊",
    dynamic_schema_overrides=_tts_schema_overrides)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from concurrent.futures import Future  # noqa: F401,E402
from typing import Iterator  # noqa: F401,E402
from concurrent.futures import ThreadPoolExecutor  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
import base64  # noqa: F401,E402
from dataclasses import dataclass  # noqa: F401,E402
from dataclasses import field  # noqa: F401,E402
import platform  # noqa: F401,E402
import queue  # noqa: F401,E402
import re  # noqa: F401,E402
import shlex  # noqa: F401,E402
import shutil  # noqa: F401,E402
import subprocess  # noqa: F401,E402
import threading  # noqa: F401,E402
import time  # noqa: F401,E402
from urllib.parse import urljoin  # noqa: F401,E402
from urllib.parse import urlparse  # noqa: F401,E402
import uuid  # noqa: F401,E402

GEMINI_TTS_CHANNELS = 1

GEMINI_TTS_SAMPLE_RATE = 24000

GEMINI_TTS_SAMPLE_WIDTH = 2  # 16-bit PCM (L16)

FALLBACK_MAX_TEXT_LENGTH = 4000

MAX_TEXT_LENGTH = FALLBACK_MAX_TEXT_LENGTH


_PLUGIN_COMPAT_LAZY = {
    'AudioDeliveryProfile': ('tools.tts_tool_delivery', 'AudioDeliveryProfile'),
    'COMMAND_TTS_OUTPUT_FORMATS': ('tools.tts_command_provider', 'COMMAND_TTS_OUTPUT_FORMATS'),
    'DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH': ('tools.tts_command_provider', 'DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH'),
    'DEFAULT_COMMAND_TTS_OUTPUT_FORMAT': ('tools.tts_command_provider', 'DEFAULT_COMMAND_TTS_OUTPUT_FORMAT'),
    'DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS': ('tools.tts_command_provider', 'DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS'),
    'DEFAULT_DEEPINFRA_TTS_VOICE': ('tools.tts_tool_openai', 'DEFAULT_DEEPINFRA_TTS_VOICE'),
    'DEFAULT_EDGE_VOICE': ('tools.tts_tool_providers', 'DEFAULT_EDGE_VOICE'),
    'DEFAULT_ELEVENLABS_MODEL_ID': ('tools.tts_tool_providers', 'DEFAULT_ELEVENLABS_MODEL_ID'),
    'DEFAULT_ELEVENLABS_STREAMING_MODEL_ID': ('tools.tts_tool_providers', 'DEFAULT_ELEVENLABS_STREAMING_MODEL_ID'),
    'DEFAULT_ELEVENLABS_VOICE_ID': ('tools.tts_tool_providers', 'DEFAULT_ELEVENLABS_VOICE_ID'),
    'DEFAULT_GEMINI_AUDIO_TAGS': ('tools.tts_tool_providers', 'DEFAULT_GEMINI_AUDIO_TAGS'),
    'DEFAULT_GEMINI_TTS_BASE_URL': ('tools.tts_tool_providers', 'DEFAULT_GEMINI_TTS_BASE_URL'),
    'DEFAULT_GEMINI_TTS_MODEL': ('tools.tts_tool_providers', 'DEFAULT_GEMINI_TTS_MODEL'),
    'DEFAULT_GEMINI_TTS_VOICE': ('tools.tts_tool_providers', 'DEFAULT_GEMINI_TTS_VOICE'),
    'DEFAULT_KITTENTTS_MODEL': ('tools.tts_tool_local', 'DEFAULT_KITTENTTS_MODEL'),
    'DEFAULT_KITTENTTS_VOICE': ('tools.tts_tool_local', 'DEFAULT_KITTENTTS_VOICE'),
    'DEFAULT_MINIMAX_BASE_URL': ('tools.tts_tool_providers', 'DEFAULT_MINIMAX_BASE_URL'),
    'DEFAULT_MINIMAX_CN_BASE_URL': ('tools.tts_tool_providers', 'DEFAULT_MINIMAX_CN_BASE_URL'),
    'DEFAULT_MINIMAX_MODEL': ('tools.tts_tool_providers', 'DEFAULT_MINIMAX_MODEL'),
    'DEFAULT_MINIMAX_VOICE_ID': ('tools.tts_tool_providers', 'DEFAULT_MINIMAX_VOICE_ID'),
    'DEFAULT_MISTRAL_TTS_MODEL': ('tools.tts_tool_providers', 'DEFAULT_MISTRAL_TTS_MODEL'),
    'DEFAULT_MISTRAL_TTS_VOICE_ID': ('tools.tts_tool_providers', 'DEFAULT_MISTRAL_TTS_VOICE_ID'),
    'DEFAULT_OPENAI_BASE_URL': ('tools.tts_tool_openai', 'DEFAULT_OPENAI_BASE_URL'),
    'DEFAULT_OPENAI_MODEL': ('tools.tts_tool_openai', 'DEFAULT_OPENAI_MODEL'),
    'DEFAULT_OPENAI_VOICE': ('tools.tts_tool_openai', 'DEFAULT_OPENAI_VOICE'),
    'DEFAULT_PIPER_VOICE': ('tools.tts_tool_local', 'DEFAULT_PIPER_VOICE'),
    'DEFAULT_XAI_AUTO_SPEECH_TAGS': ('tools.tts_tool_providers', 'DEFAULT_XAI_AUTO_SPEECH_TAGS'),
    'DEFAULT_XAI_BASE_URL': ('tools.tts_tool_providers', 'DEFAULT_XAI_BASE_URL'),
    'DEFAULT_XAI_BIT_RATE': ('tools.tts_tool_providers', 'DEFAULT_XAI_BIT_RATE'),
    'DEFAULT_XAI_LANGUAGE': ('tools.tts_tool_providers', 'DEFAULT_XAI_LANGUAGE'),
    'DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT': ('tools.tts_tool_providers', 'DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT'),
    'DEFAULT_XAI_SAMPLE_RATE': ('tools.tts_tool_providers', 'DEFAULT_XAI_SAMPLE_RATE'),
    'DEFAULT_XAI_SPEED_DEFAULT': ('tools.tts_tool_providers', 'DEFAULT_XAI_SPEED_DEFAULT'),
    'DEFAULT_XAI_SPEED_MAX': ('tools.tts_tool_providers', 'DEFAULT_XAI_SPEED_MAX'),
    'DEFAULT_XAI_SPEED_MIN': ('tools.tts_tool_providers', 'DEFAULT_XAI_SPEED_MIN'),
    'DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT': ('tools.tts_tool_providers', 'DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT'),
    'DEFAULT_XAI_VOICE_ID': ('tools.tts_tool_providers', 'DEFAULT_XAI_VOICE_ID'),
    'ELEVENLABS_MODEL_MAX_TEXT_LENGTH': ('tools.tts_tool_delivery', 'ELEVENLABS_MODEL_MAX_TEXT_LENGTH'),
    'FALLBACK_MAX_TEXT_LENGTH': ('tools.tts_tool_delivery', 'FALLBACK_MAX_TEXT_LENGTH'),
    'GEMINI_AUDIO_TAG_REWRITE_TASK': ('tools.tts_tool_providers', 'GEMINI_AUDIO_TAG_REWRITE_TASK'),
    'MANAGED_OPENAI_TTS_MODELS': ('tools.tts_tool_openai', 'MANAGED_OPENAI_TTS_MODELS'),
    'PROVIDER_MAX_TEXT_LENGTH': ('tools.tts_tool_delivery', 'PROVIDER_MAX_TEXT_LENGTH'),
    'TTS_RESPONSE_BODY_CHUNK_BYTES': ('tools.tts_tool_providers', 'TTS_RESPONSE_BODY_CHUNK_BYTES'),
    'TTS_RESPONSE_BODY_LIMIT_BYTES': ('tools.tts_tool_providers', 'TTS_RESPONSE_BODY_LIMIT_BYTES'),
    'acquire_tts_lease': ('tools.tts_tool_lifecycle', 'acquire_tts_lease'),
    'hermes_xai_user_agent': ('tools.xai_http', 'hermes_xai_user_agent'),
    'managed_nous_tools_enabled': ('tools.tool_backend_helpers', 'managed_nous_tools_enabled'),
    'nous_tool_gateway_unavailable_message': ('tools.tool_backend_helpers', 'nous_tool_gateway_unavailable_message'),
    'read_selection': ('tools.tool_backend_helpers', 'read_selection'),
    'release_tts_lease': ('tools.tts_tool_lifecycle', 'release_tts_lease'),
    'release_tts_provider': ('tools.tts_tool_lifecycle', 'release_tts_provider'),
    'resolve_managed_tool_gateway': ('tools.managed_tool_gateway', 'resolve_managed_tool_gateway'),
    'resolve_openai_audio_api_key': ('tools.tool_backend_helpers', 'resolve_openai_audio_api_key'),
    'selection_error': ('tools.tool_backend_helpers', 'selection_error'),
    'stream_tts_to_speaker': ('tools.tts_tool_speaker', 'stream_tts_to_speaker'),
    'tts_lease_holders': ('tools.tts_tool_lifecycle', 'tts_lease_holders'),
    'warm_tts_provider': ('tools.tts_tool_lifecycle', 'warm_tts_provider'),
    'windows_hide_flags': ('hermes_cli._subprocess_compat', 'windows_hide_flags'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
