#!/usr/bin/env python3
"""
Text-to-Speech Tool Module

Built-in TTS providers:
- Edge TTS (default, free, no API key): Microsoft Edge neural voices
- ElevenLabs (premium): High-quality voices, needs ELEVENLABS_API_KEY
- OpenAI TTS: Good quality, needs OPENAI_API_KEY
- MiniMax TTS: High-quality with voice cloning, needs the selected region's key
- Mistral (Voxtral TTS): Multilingual, native Opus, needs MISTRAL_API_KEY
- Google Gemini TTS: Controllable, 30 prebuilt voices, needs GEMINI_API_KEY
- xAI TTS: Grok voices, uses xAI Grok OAuth credentials or XAI_API_KEY
- NeuTTS (local, free, no API key): On-device TTS via neutts
- KittenTTS (local, free, no API key): On-device 25MB model
- Piper (local, free, no API key): OHF-Voice/piper1-gpl neural VITS, 44 languages

Custom command providers:
- Users can declare any number of named providers with ``type: command``
  under ``tts.providers.<name>`` in ``~/.hermes/config.yaml``. Hermes
  writes the input text to a temp file and runs the configured shell
  command, which must produce the audio file at the expected path.
  See the Local Command section of ``website/docs/user-guide/features/tts.md``.

Output formats:
- Opus (.ogg) for Telegram voice bubbles (requires ffmpeg for Edge TTS)
- MP3 (.mp3) for everything else (CLI, Discord, WhatsApp)

Configuration is loaded from ~/.hermes/config.yaml under the 'tts:' key.
The user chooses the provider and voice; the model just sends text.

Usage:
    from tools.tts_tool import text_to_speech_tool, check_tts_requirements

    result = text_to_speech_tool(text="Hello world")
"""

import asyncio
import base64
import datetime
import importlib.util
import json
import logging
import os
import queue
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Iterator, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)
def get_env_value(name, default=None):
    """Read env values through the live config module.

    Tests may monkeypatch and later restore ``hermes_cli.config.get_env_value``
    before this module is imported. Resolve the helper at call time so TTS does
    not keep a stale imported function for the rest of the test process.
    """
    try:
        from hermes_cli.config import get_env_value as _get_env_value
    except ImportError:
        return os.getenv(name, default)
    value = _get_env_value(name)
    return default if value is None else value


def _resolve_provider_key(env_var: str, provider_id: str) -> str:
    """Resolve a TTS provider API key via the shared voice-key resolver.

    Delegates to ``tools.tool_backend_helpers.resolve_provider_secret`` —
    the single owner of STT/TTS key resolution (config > env/.env > the
    credential pool populated by ``hermes auth add <provider_id>``).
    Resolved at call time so tests that reload the helpers module see the
    live function.
    """
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
        pass
    except Exception:
        pass
    import edge_tts
    return edge_tts

def _import_elevenlabs():
    """Lazy import ElevenLabs client. Returns the class or raises ImportError.

    Calls :func:`tools.lazy_deps.ensure` first so the SDK gets installed on
    demand if the user picked ElevenLabs as their TTS provider but never ran
    the post-setup hook (e.g. enabled it by editing config.yaml directly).
    Raises ``ImportError`` on lazy-install failure so existing callers'
    error-handling paths keep working.
    """
    try:
        from tools.lazy_deps import FeatureUnavailable, ensure
        ensure("tts.elevenlabs", prompt=False)
    except ImportError:
        # lazy_deps module itself missing — fall through to the raw import
        # so older code paths still get a clean ImportError.
        pass
    except Exception:
        pass
    from elevenlabs.client import ElevenLabs
    return ElevenLabs


def _elevenlabs_environment_kwargs(el_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build ElevenLabs client kwargs honoring config base_url/wss_url.

    ``tts.elevenlabs.base_url`` (and optionally ``wss_url``) redirect the SDK
    to a self-hosted / proxy endpoint via an ``ElevenLabsEnvironment``. When
    neither is set the SDK default environment is used. ``wss_url`` defaults
    to the ``base_url`` host with a ``wss://`` scheme when omitted.
    """
    base_url = (el_config.get("base_url") or "").rstrip("/")
    if not base_url:
        return {}
    wss_url = (el_config.get("wss_url") or "").rstrip("/")
    if not wss_url:
        wss_url = re.sub(r"^http", "ws", base_url)
    from elevenlabs.environment import ElevenLabsEnvironment
    return {"environment": ElevenLabsEnvironment(base=base_url, wss=wss_url)}

def _import_openai_client():
    """Lazy import OpenAI client. Returns the class or raises ImportError."""
    from openai import OpenAI as OpenAIClient
    return OpenAIClient

def _import_mistral_client():
    """Lazy import Mistral client. Returns the class or raises ImportError.

    Calls :func:`tools.lazy_deps.ensure` first so the ``mistralai`` SDK gets
    installed on demand if the user picked Mistral as their STT/TTS provider
    but never ran the post-setup hook (e.g. enabled it by editing config.yaml
    directly). Mirrors the ElevenLabs lazy-import path.
    """
    try:
        from tools.lazy_deps import ensure
        ensure("tts.mistral", prompt=False)
    except ImportError:
        pass
    except Exception:
        pass
    from mistralai.client import Mistral
    return Mistral

def _import_sounddevice():
    """Lazy import sounddevice. Returns the module or raises ImportError/OSError."""
    import sounddevice as sd
    return sd


def _import_kittentts():
    """Lazy import KittenTTS. Returns the class or raises ImportError."""
    from kittentts import KittenTTS
    return KittenTTS


def _import_piper():
    """Lazy import Piper. Returns the PiperVoice class or raises ImportError.

    Piper is an optional, fully-local neural TTS engine (Home Assistant /
    Open Home Foundation). ``pip install piper-tts`` provides cross-platform
    wheels (Linux / macOS / Windows, x86_64 + ARM64) with embedded espeak-ng.
    Voice models (.onnx + .onnx.json) are downloaded on first use.
    """
    from piper import PiperVoice
    return PiperVoice


# ===========================================================================
# Defaults
# ===========================================================================
DEFAULT_PROVIDER = "edge"
DEFAULT_EDGE_VOICE = "en-US-AriaNeural"
DEFAULT_ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_STREAMING_MODEL_ID = "eleven_flash_v2_5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-tts"
# The managed OpenAI audio gateway (Nous portal proxy) only proxies these speech
# models. A user's tts.openai.model set for *direct* OpenAI (e.g. "tts-1-hd")
# is rejected with a 400 "Unsupported managed OpenAI speech model", so it must be
# coerced to a supported model when routing through the gateway.
MANAGED_OPENAI_TTS_MODELS = frozenset({"gpt-4o-mini-tts"})
DEFAULT_KITTENTTS_MODEL = "KittenML/kitten-tts-nano-0.8-int8"  # 25MB
DEFAULT_KITTENTTS_VOICE = "Jasper"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"  # balanced size/quality
DEFAULT_OPENAI_VOICE = "alloy"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MINIMAX_MODEL = "speech-02-hd"
DEFAULT_MINIMAX_VOICE_ID = "English_expressive_narrator"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.io/v1/t2a_v2"
DEFAULT_MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/v1/t2a_v2"
# Emotions MiniMax documents for the t2a_v2 voice_setting. "neutral" is not in
# the published list but the API accepts it, and Hermes sent it on every request
# until the hardcoded default was removed, so it stays recognized here: it is
# the documented way back to the old flat delivery.
MINIMAX_EMOTIONS = frozenset({
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
    "calm",
    "neutral",
})
DEFAULT_MISTRAL_TTS_MODEL = "voxtral-mini-tts-2603"
DEFAULT_MISTRAL_TTS_VOICE_ID = "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral
DEFAULT_XAI_VOICE_ID = "eve"
DEFAULT_XAI_LANGUAGE = "en"
DEFAULT_XAI_SAMPLE_RATE = 24000
DEFAULT_XAI_BIT_RATE = 128000
DEFAULT_XAI_AUTO_SPEECH_TAGS = False
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
# xAI TTS `speed` accepts 0.7..1.5; 1.0 is the API default (omitted => default).
DEFAULT_XAI_SPEED_MIN = 0.7
DEFAULT_XAI_SPEED_MAX = 1.5
DEFAULT_XAI_SPEED_DEFAULT = 1.0
# xAI TTS `optimize_streaming_latency` accepts 0, 1, or 2; 0 (best quality) is
# the API default (omitted => default). Values >0 trade quality for time-to-first-audio.
DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT = 0
# xAI TTS `text_normalization` is a boolean (default False). When enabled,
# the model normalizes written-form text (numbers, abbreviations, symbols)
# into spoken-form before generating audio.
DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT = False
DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_GEMINI_TTS_VOICE = "Kore"
DEFAULT_GEMINI_TTS_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_AUDIO_TAGS = False
GEMINI_AUDIO_TAG_REWRITE_TASK = "tts_audio_tags"
# Base URL now resolved via hermes_cli.models.deepinfra_base_url (shared).
DEFAULT_DEEPINFRA_TTS_VOICE = "default"
# PCM output specs for Gemini TTS (fixed by the API)
GEMINI_TTS_SAMPLE_RATE = 24000
GEMINI_TTS_CHANNELS = 1
GEMINI_TTS_SAMPLE_WIDTH = 2  # 16-bit PCM (L16)
TTS_RESPONSE_BODY_LIMIT_BYTES = 16 * 1024 * 1024
TTS_RESPONSE_BODY_CHUNK_BYTES = 64 * 1024

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

# ---------------------------------------------------------------------------
# Per-provider input-character limits (from official provider docs).
# A single global cap was wrong: OpenAI is 4096, xAI is 15k, MiniMax is 10k,
# ElevenLabs is model-dependent (5k / 10k / 30k / 40k), Gemini has a 32k-token
# context window.  Users can override any of these via
# ``tts.<provider>.max_text_length`` in config.yaml.
# ---------------------------------------------------------------------------
PROVIDER_MAX_TEXT_LENGTH: Dict[str, int] = {
    "edge": 5000,         # edge-tts practical sync limit
    "openai": 4096,       # https://platform.openai.com/docs/guides/text-to-speech
    "xai": 15000,         # https://docs.x.ai/developers/model-capabilities/audio/text-to-speech
    "minimax": 10000,     # https://platform.minimax.io/docs/api-reference/speech-t2a-http (sync)
    "mistral": 4000,      # conservative; no published per-request cap
    "gemini": 32000,      # Gemini TTS has a 32k-token context window; char cap is conservative
    "elevenlabs": 10000,  # fallback when model-aware lookup can't resolve (multilingual_v2)
    "neutts": 2000,       # local model, quality falls off on long text
    "kittentts": 2000,    # local 25MB model
    "piper": 5000,        # local VITS model, phoneme-based; practical cap
}

# ElevenLabs caps vary by model_id. https://elevenlabs.io/docs/overview/models
ELEVENLABS_MODEL_MAX_TEXT_LENGTH: Dict[str, int] = {
    "eleven_v3": 5000,
    "eleven_ttv_v3": 5000,
    "eleven_multilingual_v2": 10000,
    "eleven_multilingual_v1": 10000,
    "eleven_english_sts_v2": 10000,
    "eleven_english_sts_v1": 10000,
    "eleven_flash_v2": 30000,
    "eleven_flash_v2_5": 40000,
}


def _config_bool(value: Any, default: bool = False) -> bool:
    """Coerce common YAML/env bool spellings without treating random strings as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _response_has_explicit_stream(response: Any) -> bool:
    iter_content = getattr(response, "iter_content", None)
    if not callable(iter_content):
        return False
    response_type = type(response)
    if response_type.__module__.startswith("requests."):
        return True
    return "iter_content" in vars(response_type)


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _read_tts_response_bytes(
    response: Any,
    *,
    label: str,
    limit: Optional[int] = None,
) -> bytes:
    """Read an upstream TTS response with a hard byte cap."""
    limit = TTS_RESPONSE_BODY_LIMIT_BYTES if limit is None else limit
    chunks: list[bytes] = []
    total = 0
    try:
        if _response_has_explicit_stream(response):
            iterator = response.iter_content(chunk_size=TTS_RESPONSE_BODY_CHUNK_BYTES)
        else:
            content = vars(response).get("content", getattr(type(response), "content", b""))
            if isinstance(content, str):
                content = content.encode("utf-8", errors="replace")
            iterator = (content,) if isinstance(content, (bytes, bytearray)) else ()

        for chunk in iterator:
            if not chunk:
                continue
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", errors="replace")
            chunk = bytes(chunk)
            total += len(chunk)
            if total > limit:
                _close_response(response)
                raise RuntimeError(f"{label} response exceeds {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        _close_response(response)


def _read_tts_response_json(
    response: Any,
    *,
    label: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    raw = _read_tts_response_bytes(response, label=label, limit=limit)
    if raw:
        return json.loads(raw.decode("utf-8"))

    # Unit-test doubles often only provide `.json()`. Real requests.Response
    # objects use the streaming path above, so this fallback does not re-open
    # the production eager-buffering behavior.
    if not _response_has_explicit_stream(response):
        json_reader = getattr(response, "json", None)
        if callable(json_reader):
            parsed = json_reader()
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _write_tts_response_to_file(
    response: Any,
    output_path: str,
    *,
    label: str,
    limit: Optional[int] = None,
) -> None:
    audio_bytes = _read_tts_response_bytes(response, label=label, limit=limit)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

# Final fallback when provider isn't recognized at all.
FALLBACK_MAX_TEXT_LENGTH = 4000

# Back-compat alias. Prefer ``_resolve_max_text_length()`` for new code.
MAX_TEXT_LENGTH = FALLBACK_MAX_TEXT_LENGTH


def _resolve_max_text_length(
    provider: Optional[str],
    tts_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Return the input-character cap for *provider*.

    Resolution order:
      1. ``tts.<provider>.max_text_length`` (user override in config.yaml)
      2. ``tts.providers.<provider>.max_text_length`` for user-declared
         command providers
      3. ElevenLabs model-aware table (keyed on configured ``model_id``)
      4. ``PROVIDER_MAX_TEXT_LENGTH`` default
      5. ``DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH`` when the provider is a
         command-type user provider without an explicit cap
      6. ``FALLBACK_MAX_TEXT_LENGTH`` (4000)

    Non-positive or non-integer overrides fall through to the default so a
    broken config can't accidentally disable truncation entirely.
    """
    if not provider:
        return FALLBACK_MAX_TEXT_LENGTH
    key = provider.lower().strip()
    cfg = tts_config or {}

    # Built-in-style override at tts.<provider>.max_text_length wins first,
    # matching historical behavior.
    prov_cfg = cfg.get(key) if isinstance(cfg.get(key), dict) else {}
    override = prov_cfg.get("max_text_length") if prov_cfg else None
    if isinstance(override, bool):
        override = None
    if isinstance(override, int) and override > 0:
        return override

    if key == "elevenlabs":
        model_id = (prov_cfg or {}).get("model_id") or DEFAULT_ELEVENLABS_MODEL_ID
        mapped = ELEVENLABS_MODEL_MAX_TEXT_LENGTH.get(str(model_id).strip())
        if mapped:
            return mapped

    if key in PROVIDER_MAX_TEXT_LENGTH:
        return PROVIDER_MAX_TEXT_LENGTH[key]

    # User-declared command provider (under tts.providers.<name>)
    if key not in BUILTIN_TTS_PROVIDERS:
        named = _get_named_provider_config(cfg, key)
        if _is_command_provider_config(named):
            named_override = named.get("max_text_length")
            if isinstance(named_override, bool):
                named_override = None
            if isinstance(named_override, int) and named_override > 0:
                return named_override
            return DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH

    return FALLBACK_MAX_TEXT_LENGTH


# ===========================================================================
# Long-form chunking and delivery packing
# ===========================================================================

@dataclass(frozen=True)
class AudioDeliveryProfile:
    """Destination-platform constraints for generated TTS audio."""

    platform: str
    max_file_bytes: int
    safety_ratio: float = 0.85

    @property
    def target_file_bytes(self) -> int:
        """Conservative packing target below the platform hard limit."""
        return max(1, int(self.max_file_bytes * self.safety_ratio))


_PLATFORM_AUDIO_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "discord": {
        "max_file_bytes": 10 * 1024 * 1024,
        "safety_ratio": 0.85,
    },
    "telegram": {
        "max_file_bytes": 50 * 1024 * 1024,
        "safety_ratio": 0.85,
    },
    "default": {
        "max_file_bytes": 10 * 1024 * 1024,
        "safety_ratio": 0.85,
    },
}


def _resolve_audio_delivery_profile(
    platform: Optional[str],
    tts_config: Optional[Dict[str, Any]] = None,
) -> AudioDeliveryProfile:
    """Resolve upload constraints, including optional per-platform overrides."""
    key = (platform or "default").lower().strip() or "default"
    defaults = dict(
        _PLATFORM_AUDIO_DEFAULTS.get(key) or _PLATFORM_AUDIO_DEFAULTS["default"]
    )
    cfg = tts_config or {}
    profiles = cfg.get("delivery_profiles")
    overrides = profiles.get(key, {}) if isinstance(profiles, dict) else {}
    if isinstance(overrides, dict):
        defaults.update({k: v for k, v in overrides.items() if v is not None})

    max_file_bytes = defaults.get("max_file_bytes")
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes <= 0
    ):
        max_file_bytes = _PLATFORM_AUDIO_DEFAULTS["default"]["max_file_bytes"]

    safety_ratio = defaults.get("safety_ratio", 0.85)
    if (
        isinstance(safety_ratio, bool)
        or not isinstance(safety_ratio, (int, float))
        or not 0 < safety_ratio <= 1
    ):
        safety_ratio = 0.85

    return AudioDeliveryProfile(
        platform=key,
        max_file_bytes=max_file_bytes,
        safety_ratio=float(safety_ratio),
    )


def _split_oversized_sentence(sentence: str, max_chars: int) -> List[str]:
    """Split one over-limit sentence on word boundaries, then hard boundaries."""
    words = sentence.split()
    chunks: List[str] = []
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_text_for_tts(text: str, max_chars: int) -> List[str]:
    """Split text under a provider cap without dropping normalized content."""
    if max_chars <= 0:
        max_chars = FALLBACK_MAX_TEXT_LENGTH
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;:,])\s+", normalized)
        if sentence.strip()
    ]
    expanded: List[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            expanded.append(sentence)
        else:
            expanded.extend(_split_oversized_sentence(sentence, max_chars))

    chunks: List[str] = []
    current = ""
    for sentence in expanded:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _pack_audio_files_for_delivery(
    audio_paths: List[str],
    profile: AudioDeliveryProfile,
) -> List[List[str]]:
    """Group already-final-encoded chunks under the conservative size target."""
    groups: List[List[str]] = []
    current: List[str] = []
    current_size = 0
    current_suffix = ""
    for path in audio_paths:
        size = Path(path).stat().st_size
        suffix = Path(path).suffix.lower()
        if current and (
            current_size + size > profile.target_file_bytes
            or suffix != current_suffix
        ):
            groups.append(current)
            current = []
            current_size = 0
            current_suffix = ""
        current.append(path)
        current_size += size
        current_suffix = suffix
    if current:
        groups.append(current)
    return groups


# ===========================================================================
# Config loader -- reads tts: section from ~/.hermes/config.yaml
# ===========================================================================
def _load_tts_config() -> Dict[str, Any]:
    """
    Load TTS configuration from ~/.hermes/config.yaml.

    Returns a dict with provider settings. Falls back to defaults
    for any missing fields.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config.get("tts") or {}
    except ImportError:
        logger.debug("hermes_cli.config not available, using default TTS config")
        return {}
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


@dataclass(frozen=True)
class _MiniMaxTTSRuntime:
    """A region-bound MiniMax endpoint and credential.

    The credential is excluded from ``repr`` so diagnostics cannot expose it
    accidentally.
    """

    region: str
    endpoint: str
    credential_source: str
    api_key: str = field(repr=False)


def _resolve_minimax_tts_runtime(
    tts_config: Dict[str, Any],
) -> _MiniMaxTTSRuntime:
    """Select MiniMax TTS region, endpoint, and credential atomically.

    An explicit ``tts.minimax.region`` wins. Without one, the legacy global
    credential wins when present; a China credential is selected only when it
    is the sole configured MiniMax credential.
    """
    mm_config = tts_config.get("minimax", {})
    if not isinstance(mm_config, dict):
        mm_config = {}

    credentials = {
        "global": (
            "MINIMAX_API_KEY",
            str(_resolve_provider_key("MINIMAX_API_KEY", "minimax") or "").strip(),
        ),
        "cn": (
            "MINIMAX_CN_API_KEY",
            str(_resolve_provider_key("MINIMAX_CN_API_KEY", "minimax") or "").strip(),
        ),
    }
    endpoints = {
        "global": DEFAULT_MINIMAX_BASE_URL,
        "cn": DEFAULT_MINIMAX_CN_BASE_URL,
    }

    configured_region = str(mm_config.get("region") or "").strip().lower()
    if configured_region and configured_region not in endpoints:
        raise ValueError("tts.minimax.region must be 'global' or 'cn'")

    if configured_region:
        region = configured_region
    elif credentials["global"][1]:
        region = "global"
    elif credentials["cn"][1]:
        region = "cn"
    else:
        region = "global"

    credential_source, api_key = credentials[region]
    if not api_key:
        raise ValueError(
            f"{credential_source} not set for MiniMax TTS region {region!r}"
        )

    endpoint = str(mm_config.get("base_url") or endpoints[region]).strip()
    endpoint_host = (urlparse(endpoint).hostname or "").lower()
    official_region_hosts = {
        "global": frozenset({"api.minimax.io", "api.minimax.chat"}),
        "cn": frozenset({"api.minimaxi.com"}),
    }
    other_region = "cn" if region == "global" else "global"
    if endpoint_host in official_region_hosts[other_region]:
        raise ValueError(
            f"tts.minimax.base_url points to the {other_region!r} MiniMax endpoint "
            f"but region is {region!r}"
        )

    return _MiniMaxTTSRuntime(
        region=region,
        endpoint=endpoint,
        credential_source=credential_source,
        api_key=api_key,
    )


# ===========================================================================
# Custom command providers (type: command under tts.providers.<name>)
# ===========================================================================
#
# Users can declare any number of command-type providers alongside the
# built-ins so they can plug any local CLI (Piper, VoxCPM, Kokoro CLIs,
# custom voice-cloning scripts, etc.) into Hermes without any Python code
# changes. The config shape is::
#
#     tts:
#       provider: piper-en
#       providers:
#         piper-en:
#           type: command
#           command: "piper -m ~/model.onnx -f {output_path} < {input_path}"
#           output_format: wav
#
# Hermes writes the input text to a temp UTF-8 file, runs the command with
# placeholder substitution, and reads the audio file the command wrote to
# ``{output_path}``. Supported placeholders: ``{input_path}``,
# ``{text_path}`` (alias for input_path), ``{output_path}``, ``{format}``,
# ``{voice}``, ``{model}``, ``{speed}``. Use ``{{`` / ``}}`` for literal braces.
#
# Built-in provider names always win over an entry with the same name under
# ``tts.providers``, so user config can't silently shadow ``edge`` etc.
#
# Placeholder values are shell-quoted for their surrounding context
# (bare / single / double quote), so paths with spaces work transparently.

# Built-in provider names. Any ``tts.provider`` value NOT in this set is
# interpreted as a reference to ``tts.providers.<name>``.
BUILTIN_TTS_PROVIDERS = frozenset({
    "edge",
    "elevenlabs",
    "openai",
    "minimax",
    "xai",
    "mistral",
    "gemini",
    "neutts",
    "kittentts",
    "piper",
    "deepinfra",
})

DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS = 120
DEFAULT_COMMAND_TTS_OUTPUT_FORMAT = "mp3"
COMMAND_TTS_OUTPUT_FORMATS = frozenset(
    {"mp3", "wav", "ogg", "flac", "m4a", "aac", "amr", "opus"}
)
DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH = 5000

# Platforms whose native voice-bubble delivery requires Ogg/Opus audio.
# Previously only Telegram was recognized, so Matrix/Feishu/WhatsApp/Signal
# voice replies were synthesized as MP3 and rendered as broken attachments
# (#14841, #45557 and siblings).
OPUS_VOICE_PLATFORMS = frozenset({
    "telegram",
    "matrix",
    "feishu",
    "whatsapp",
    "signal",
})


def _get_provider_section(tts_config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return a provider config block if it's a dict, else an empty dict."""
    if not isinstance(tts_config, dict):
        return {}
    section = tts_config.get(name)
    return section if isinstance(section, dict) else {}


def _get_named_provider_config(
    tts_config: Dict[str, Any],
    name: str,
) -> Dict[str, Any]:
    """Return the config dict for a user-declared provider.

    Looks up ``tts.providers.<name>`` first (the canonical location), and
    falls back to ``tts.<name>`` so users who followed the built-in layout
    still work. Returns an empty dict when the provider is not declared.
    """
    providers = _get_provider_section(tts_config, "providers")
    section = providers.get(name) if isinstance(providers, dict) else None
    if isinstance(section, dict):
        return section
    # Back-compat: allow ``tts.<name>`` for user-declared providers too,
    # but only when the name is not a built-in (so a user's ``tts.openai``
    # block still means the OpenAI provider, not a custom command).
    if name.lower() not in BUILTIN_TTS_PROVIDERS:
        legacy = _get_provider_section(tts_config, name)
        if legacy:
            return legacy
    return {}


def _is_command_provider_config(config: Dict[str, Any]) -> bool:
    """Return True when *config* declares a command-type provider."""
    if not isinstance(config, dict):
        return False
    ptype = str(config.get("type") or "").strip().lower()
    if ptype and ptype != "command":
        return False
    command = config.get("command")
    return isinstance(command, str) and bool(command.strip())


def _resolve_command_provider_config(
    provider: str,
    tts_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the provider config if *provider* resolves to a command type.

    Built-in provider names are rejected (they have native handlers).
    Returns None when the name is a built-in, unknown, or not a command
    type.
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_TTS_PROVIDERS:
        return None
    config = _get_named_provider_config(tts_config, key)
    if _is_command_provider_config(config):
        return config
    return None


def _dispatch_to_plugin_provider(
    text: str,
    output_path: str,
    provider: str,
    tts_config: Dict[str, Any],
) -> Optional[str]:
    """Route the call to a plugin-registered TTS provider, or return None.

    Returns the path to the written audio file on dispatch, or ``None``
    to fall through to the next resolution layer (built-in dispatch or
    Edge TTS default).

    Resolution invariants enforced here (matches issue #30398):

    1. Built-in provider names short-circuit — never reach the plugin
       registry. The caller is responsible for the elif chain that
       handles ``edge``/``openai``/etc.; this function explicitly
       rejects those names defensively.
    2. Command-type providers declared under
       ``tts.providers.<name>: type: command`` (PR #17843) win over a
       plugin with the same name. The caller passes us only when its
       own command-provider check returned None — we re-verify here so
       a refactor of the caller can't silently break the invariant.
    3. Plugin dispatch fires only when ``provider`` matches a registered
       :class:`TTSProvider` whose ``name`` equals the configured value.
       Unknown names return None (caller falls through to Edge default).

    Plugin exceptions are caught and re-raised — the outer
    ``text_to_speech_tool`` try/except converts them to the standard
    error envelope, matching how command-provider failures surface.
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_TTS_PROVIDERS:
        return None
    # Defense in depth: command-provider check should already have
    # short-circuited the caller. If a same-name command config exists,
    # bail so the command path wins.
    if _is_command_provider_config(_get_named_provider_config(tts_config, key)):
        return None
    try:
        from agent.tts_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        plugin_provider = get_provider(key)
        if plugin_provider is None:
            # Long-lived sessions may have discovered plugins before the
            # bundled backend was patched in or before config changed.
            # Retry once with a forced refresh before surfacing fall-
            # through. Mirrors the image_gen / browser dispatcher
            # recovery pattern.
            _ensure_plugins_discovered(force=True)
            plugin_provider = get_provider(key)
    except Exception as exc:  # noqa: BLE001 — discovery failure is non-fatal
        logger.debug("tts plugin dispatch skipped (discovery failed): %s", exc)
        return None
    if plugin_provider is None:
        return None

    # Resolve voice / model / format from tts_config — providers should
    # treat all of these as optional and fall back to their own defaults
    # when None is passed (matches the ABC contract documented on
    # ``TTSProvider.synthesize``).
    voice = tts_config.get("voice") if isinstance(tts_config, dict) else None
    model = tts_config.get("model") if isinstance(tts_config, dict) else None
    speed = tts_config.get("speed") if isinstance(tts_config, dict) else None
    fmt = (
        tts_config.get("output_format", DEFAULT_COMMAND_TTS_OUTPUT_FORMAT)
        if isinstance(tts_config, dict)
        else DEFAULT_COMMAND_TTS_OUTPUT_FORMAT
    )

    logger.info(
        "Generating speech with plugin TTS provider '%s'...", key,
    )
    written = plugin_provider.synthesize(
        text,
        output_path,
        voice=voice if isinstance(voice, str) and voice else None,
        model=model if isinstance(model, str) and model else None,
        speed=float(speed) if isinstance(speed, (int, float)) else None,
        format=str(fmt).lower() if fmt else "mp3",
    )
    # Provider contract: returns the (possibly rewritten) output path.
    # Defensive against a provider returning None or a non-string —
    # fall back to the caller's expected output_path.
    return written if isinstance(written, str) and written else output_path


def _plugin_provider_is_voice_compatible(provider: str) -> bool:
    """Return True when the registered plugin provider opts into voice
    bubble delivery via its ``voice_compatible`` property.

    Defensive: any registry or property access failure means False
    (matches the safe default for the command-provider path).
    """
    if not provider:
        return False
    key = provider.lower().strip()
    if key in BUILTIN_TTS_PROVIDERS:
        return False
    try:
        from agent.tts_registry import get_provider

        plugin_provider = get_provider(key)
        if plugin_provider is None:
            return False
        return bool(plugin_provider.voice_compatible)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "tts plugin voice_compatible check failed for '%s': %s", key, exc,
        )
        return False


def _iter_command_providers(tts_config: Dict[str, Any]):
    """Yield (name, config) pairs for every declared command-type provider."""
    if not isinstance(tts_config, dict):
        return
    providers = _get_provider_section(tts_config, "providers")
    for name, cfg in (providers or {}).items():
        if isinstance(name, str) and name.lower() not in BUILTIN_TTS_PROVIDERS:
            if _is_command_provider_config(cfg):
                yield name, cfg


def _get_command_tts_timeout(config: Dict[str, Any]) -> float:
    """Return timeout in seconds, falling back when invalid."""
    raw = config.get("timeout", config.get("timeout_seconds", DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)
    if value <= 0:
        return float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)
    return value


def _get_command_tts_output_format(
    config: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Return the validated output format (mp3/wav/ogg/flac)."""
    if output_path:
        suffix = Path(output_path).suffix.lower().strip().lstrip(".")
        if suffix in COMMAND_TTS_OUTPUT_FORMATS:
            return suffix
    raw = (
        config.get("format")
        or config.get("output_format")
        or DEFAULT_COMMAND_TTS_OUTPUT_FORMAT
    )
    fmt = str(raw).lower().strip().lstrip(".")
    return fmt if fmt in COMMAND_TTS_OUTPUT_FORMATS else DEFAULT_COMMAND_TTS_OUTPUT_FORMAT


def _is_command_tts_voice_compatible(config: Dict[str, Any]) -> bool:
    """Return True only when the user explicitly opted in to voice delivery."""
    value = config.get("voice_compatible", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _shell_quote_context(command_template: str, position: int) -> Optional[str]:
    """Return the shell quote character active right before *position*.

    Returns ``"'"`` / ``'"'`` when inside a single- / double-quoted region
    of the template, ``None`` for bare context.
    """
    quote: Optional[str] = None
    escaped = False
    i = 0
    while i < position:
        char = command_template[i]
        if quote == "'":
            if char == "'":
                quote = None
        elif quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif char == "'":
            quote = "'"
        elif char == '"':
            quote = '"'
        elif char == "\\":
            i += 1
        i += 1
    return quote


def _quote_command_tts_placeholder(value: str, quote_context: Optional[str]) -> str:
    """Quote a placeholder value for its position in a shell command template."""
    if quote_context == "'":
        return value.replace("'", r"'\''")
    if quote_context == '"':
        return (
            value
            .replace("\\", "\\\\")
            .replace('"', r'\"')
            .replace("$", r"\$")
            .replace("`", r"\`")
        )
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def _render_command_tts_template(
    command_template: str,
    placeholders: Dict[str, str],
) -> str:
    """Replace supported placeholders while preserving ``{{`` / ``}}``."""
    names = "|".join(re.escape(name) for name in placeholders)
    pattern = re.compile(
        rf"(?<!\$)(?:\{{\{{(?P<double>{names})\}}\}}|\{{(?P<single>{names})\}})"
    )
    replacements: list[tuple[str, str]] = []

    def replace_match(match: re.Match[str]) -> str:
        name = match.group("double") or match.group("single")
        token = f"__HERMES_TTS_PLACEHOLDER_{len(replacements)}__"
        replacements.append((
            token,
            _quote_command_tts_placeholder(
                placeholders[name],
                _shell_quote_context(command_template, match.start()),
            ),
        ))
        return token

    rendered = pattern.sub(replace_match, command_template)
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    for token, value in replacements:
        rendered = rendered.replace(token, value)
    return rendered


def _terminate_command_tts_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of a shell process and all of its children."""
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            proc.kill()
        return

    import psutil
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
    except psutil.NoSuchProcess:
        return
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        return
    except Exception:
        proc.kill()


def _command_provider_env_passthrough(config: Dict[str, Any]) -> list:
    """Return the provider's ``env_passthrough`` allowlist (opt-out of scrub).

    Command providers legitimately reference their own API keys in the shell
    template (curl one-liners). The child env is scrubbed of Hermes secrets by
    default; ``env_passthrough: [MY_API_KEY, ...]`` copies the named variables
    back from the parent environment so a trusted template keeps working.
    """
    raw = config.get("env_passthrough")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _run_command_tts(
    command: str,
    timeout: float,
    env_passthrough: Optional[list] = None,
) -> subprocess.CompletedProcess:
    """Run a command-provider shell command with process-tree idle cleanup.

    Child env is scrubbed of Hermes secrets (salvage of #56332) while still
    propagating delegated-child lineage markers when applicable.
    """
    from agent.delegation_context import delegated_child_subprocess_env
    from tools.environments.local import hermes_subprocess_env

    scrubbed = hermes_subprocess_env(inherit_credentials=False)
    for key in env_passthrough or []:
        value = os.environ.get(key)
        if value is not None:
            scrubbed[key] = value
    popen_kwargs: Dict[str, Any] = {
        "shell": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        # Lossy UTF-8 decode — locale-mismatched bytes from the TTS command
        # must not raise in the reader threads on non-UTF-8 Windows (#45099).
        "encoding": "utf-8",
        "errors": "replace",
        "env": delegated_child_subprocess_env(scrubbed),
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(command, **popen_kwargs, stdin=subprocess.DEVNULL)
    output_queue: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()
    chunks: Dict[str, list[str]] = {"stdout": [], "stderr": []}
    open_streams = {"stdout", "stderr"}

    def read_stream(name: str, stream: Any) -> None:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        read1 = getattr(getattr(stream, "buffer", None), "read1", None)
        try:
            while True:
                if read1 is None:
                    chunk = stream.read(65536)
                else:
                    data = read1(65536)
                    chunk = data.decode(encoding, errors="replace")
                if not chunk:
                    break
                output_queue.put((name, chunk))
        finally:
            output_queue.put((name, None))

    readers = [
        threading.Thread(
            target=read_stream,
            args=("stdout", proc.stdout),
            daemon=True,
        ),
        threading.Thread(
            target=read_stream,
            args=("stderr", proc.stderr),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout
    timed_out = False
    while open_streams:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            name, chunk = output_queue.get(timeout=min(0.05, remaining))
        except queue.Empty:
            continue
        if chunk is None:
            open_streams.discard(name)
            continue
        chunks[name].append(chunk)
        deadline = time.monotonic() + timeout

    if not timed_out:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True

    if timed_out:
        _terminate_command_tts_process_tree(proc)
        for reader in readers:
            reader.join(timeout=0.5)
        while True:
            try:
                name, chunk = output_queue.get_nowait()
            except queue.Empty:
                break
            if chunk:
                chunks[name].append(chunk)
        stdout = "".join(chunks["stdout"])
        stderr = "".join(chunks["stderr"])
        try:
            raise subprocess.TimeoutExpired(command, timeout)
        except subprocess.TimeoutExpired as exc:
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=stdout,
                stderr=stderr,
            ) from exc

    stdout = "".join(chunks["stdout"])
    stderr = "".join(chunks["stderr"])

    if proc.returncode:
        raise subprocess.CalledProcessError(
            proc.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _configured_command_tts_output_path(path: Path, config: Dict[str, Any]) -> Path:
    """Return an output path whose extension matches the provider's output_format."""
    fmt = _get_command_tts_output_format(config)
    return path.with_suffix(f".{fmt}")


def _generate_command_tts(
    text: str,
    output_path: str,
    provider_name: str,
    config: Dict[str, Any],
    tts_config: Dict[str, Any],
) -> str:
    """Generate speech by running a user-configured shell command.

    Returns the absolute path of the audio file the command wrote.
    Raises ``ValueError`` when the provider config is invalid, and
    ``RuntimeError`` for timeouts / non-zero exits / empty output.
    """
    command_template = str(config.get("command") or "").strip()
    if not command_template:
        raise ValueError(
            f"tts.providers.{provider_name}.command is not configured"
        )

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    timeout = _get_command_tts_timeout(config)
    output_format = _get_command_tts_output_format(config, str(output))
    speed = config.get("speed", tts_config.get("speed", ""))

    with tempfile.TemporaryDirectory() as tmpdir:
        text_path = Path(tmpdir) / "input.txt"
        text_path.write_text(text, encoding="utf-8")

        placeholders = {
            "input_path": str(text_path),
            "text_path": str(text_path),
            "output_path": str(output),
            "format": output_format,
            "voice": str(config.get("voice", "")),
            "model": str(config.get("model", "")),
            "speed": str(speed),
        }
        command = _render_command_tts_template(command_template, placeholders)

        try:
            _run_command_tts(
                command,
                timeout,
                env_passthrough=_command_provider_env_passthrough(config),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"TTS provider '{provider_name}' timed out after {timeout:g}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail_parts = []
            if exc.stderr:
                detail_parts.append(f"stderr: {exc.stderr.strip()}")
            if exc.stdout:
                detail_parts.append(f"stdout: {exc.stdout.strip()}")
            detail = "; ".join(detail_parts) or "no command output"
            raise RuntimeError(
                f"TTS provider '{provider_name}' exited with code "
                f"{exc.returncode}: {detail}"
            ) from exc

    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError(
            f"TTS provider '{provider_name}' produced no output at {output}"
        )
    return str(output)


def _has_any_command_tts_provider(tts_config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when any command-type TTS provider is configured."""
    if tts_config is None:
        tts_config = _load_tts_config()
    for _name, _cfg in _iter_command_providers(tts_config):
        return True
    return False


# ===========================================================================
# ffmpeg Opus conversion (Edge TTS MP3 -> OGG Opus for Telegram)
# ===========================================================================
def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _convert_to_opus(mp3_path: str) -> Optional[str]:
    """
    Convert an audio file (MP3/WAV/anything ffmpeg reads) to OGG Opus
    format for Telegram voice bubbles.

    Args:
        mp3_path: Path to the input audio file.

    Returns:
        Path to the .ogg file, or None if conversion fails.
    """
    if not _has_ffmpeg():
        return None

    ogg_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
    return _ffmpeg_transcode_to_opus(mp3_path, ogg_path)


def _ffmpeg_transcode_to_opus(input_path: str, ogg_path: str) -> Optional[str]:
    """Transcode *input_path* to real Ogg/Opus at *ogg_path* via ffmpeg.

    Safe when ``input_path == ogg_path`` (writes to a temp file, then
    replaces). Returns the output path on success, None on failure.
    """
    if not _has_ffmpeg():
        return None

    in_place = os.path.abspath(input_path) == os.path.abspath(ogg_path)
    work_path = ogg_path + ".tmp.ogg" if in_place else ogg_path
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-acodec", "libopus",
             "-ac", "1", "-b:a", "48k", "-vbr", "on",
             "-application", "voip", "-compression_level", "10", "-f", "ogg",
             work_path, "-y"],
            capture_output=True, timeout=30,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            logger.warning("ffmpeg conversion failed with return code %d: %s",
                          result.returncode, result.stderr.decode('utf-8', errors='ignore')[:200])
            return None
        if os.path.exists(work_path) and os.path.getsize(work_path) > 0:
            if in_place:
                os.replace(work_path, ogg_path)
            return ogg_path
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg OGG conversion timed out after 30s")
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH")
    except Exception as e:
        logger.warning("ffmpeg OGG conversion failed: %s", e, exc_info=True)
    finally:
        if in_place and os.path.exists(work_path):
            try:
                os.remove(work_path)
            except OSError:
                pass
    return None


# ---------------------------------------------------------------------------
# Container sniffing — class-level guard against "MP3/WAV bytes in a .ogg
# file". Several TTS backends silently ignore the requested opus format
# (Edge only emits MP3, Piper writes WAV, xAI writes MP3, some
# OpenAI-compatible servers reject/ignore response_format="opus"), which
# breaks native voice bubbles on Telegram/Matrix/Feishu/WhatsApp. Rather
# than special-casing every provider, sniff the magic bytes once after
# synthesis and repair the container when it doesn't match the extension.
# ---------------------------------------------------------------------------

def _sniff_audio_container(path: str) -> str:
    """Return a container id ('ogg', 'wav', 'mp3', 'flac', ...) or 'unknown'.

    Delegates to the shared magic-byte sniffer in ``tools.audio_container``
    (one module owns container detection for both this outbound repair and
    the inbound gateway audio cache).
    """
    from tools.audio_container import sniff_container

    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
    except OSError:
        return "unknown"
    return sniff_container(head) or "unknown"


def _repair_ogg_container(file_str: str) -> str:
    """Ensure a path claiming ``.ogg`` actually contains an Ogg container.

    When the bytes are MP3/WAV/FLAC (a backend ignored the opus request),
    transcode in place to real Ogg/Opus. On any failure, rename to the
    sniffed real extension so downstream players/platforms at least get an
    honest file instead of a 0-second voice bubble. Returns the (possibly
    updated) path.
    """
    if not file_str.endswith(".ogg"):
        return file_str
    container = _sniff_audio_container(file_str)
    if container in ("ogg", "unknown"):
        return file_str

    logger.info(
        "TTS wrote %s bytes into a .ogg path (%s) — transcoding to real Ogg/Opus",
        container, file_str,
    )
    repaired = _ffmpeg_transcode_to_opus(file_str, file_str)
    if repaired:
        return repaired

    # ffmpeg unavailable/failed: rename to the honest extension.
    honest = file_str[:-4] + "." + container
    try:
        os.replace(file_str, honest)
        logger.warning(
            "Could not transcode %s to Ogg/Opus — renamed to %s so the "
            "file is delivered with its real format", file_str, honest,
        )
        return honest
    except OSError:
        return file_str


# ===========================================================================
# Long-form audio combination and delivery packing
# ===========================================================================

def _concat_audio_files(
    audio_paths: List[str],
    output_path: str,
    *,
    voice_compatible: bool = False,
) -> Optional[str]:
    """Combine independently encoded chunks with ffmpeg.

    OGG/Opus is always decoded and re-encoded, even when a custom provider did
    not opt in to voice-message presentation. Matching MP3 chunks preserve their
    encoded frames. A failed or unavailable combine returns ``None`` so callers
    can preserve the original, individually valid files. Structured audio
    containers are never byte-joined.
    """
    if not audio_paths:
        raise ValueError("No audio chunks to combine")
    if len(audio_paths) == 1:
        source = audio_paths[0]
        if os.path.abspath(source) != os.path.abspath(output_path):
            shutil.copyfile(source, output_path)
        return output_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    concat_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.concat.txt")
    temp_output = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.combining{destination.suffix}"
    )
    try:
        with concat_path.open("w", encoding="utf-8") as concat_file:
            for path in audio_paths:
                concat_file.write(f"file {shlex.quote(os.path.abspath(path))}\n")

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vn",
        ]
        suffix = destination.suffix.lower()
        if voice_compatible or suffix in {".ogg", ".opus"}:
            command.extend([
                "-c:a", "libopus", "-ac", "1", "-b:a", "64k", "-vbr", "off",
            ])
        elif suffix == ".mp3" and all(
            Path(path).suffix.lower() == ".mp3" for path in audio_paths
        ):
            # Matching MP3 provider chunks already share one output codec/config.
            # Preserve those encoded frames instead of imposing a second lossy pass.
            command.extend(["-c:a", "copy"])
        command.append(str(temp_output))

        result = subprocess.run(
            command,
            capture_output=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        if (
            result.returncode == 0
            and temp_output.exists()
            and temp_output.stat().st_size > 0
        ):
            os.replace(temp_output, destination)
            return str(destination)
        logger.warning(
            "ffmpeg audio combine failed: %s",
            result.stderr.decode("utf-8", errors="ignore")[:500],
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg audio combine failed: %s", exc)
    finally:
        for path in (concat_path, temp_output):
            try:
                path.unlink()
            except OSError:
                pass
    return None


def _build_audio_delivery_files(
    audio_paths: List[str],
    output_path: str,
    profile: AudioDeliveryProfile,
    *,
    voice_compatible: bool = False,
) -> Tuple[List[str], bool]:
    """Pack final-encoded chunks and enforce the hard upload limit.

    Packing uses the conservative target. Every combined artifact is then
    checked at its actual post-encoding size; an over-limit group is split and
    retried. If combining fails, the valid constituent files are returned
    separately. A single final-encoded chunk above the hard limit fails closed
    rather than returning an upload that the destination will reject.
    """
    if not audio_paths:
        raise ValueError("No final-encoded TTS audio chunks")
    for path in audio_paths:
        size = Path(path).stat().st_size
        if size > profile.max_file_bytes:
            raise ValueError(
                f"Final-encoded TTS chunk exceeds {profile.platform} delivery "
                f"limit ({size} > {profile.max_file_bytes} bytes): {path}"
            )

    base = Path(output_path)
    scratch_outputs: List[str] = []
    combined_any = False
    combine_index = 0

    def emit(group: List[str]) -> List[str]:
        nonlocal combined_any, combine_index
        if len(group) == 1:
            return list(group)

        combine_index += 1
        scratch = base.with_name(
            f".{base.stem}.delivery{combine_index:03d}.{uuid.uuid4().hex}{base.suffix}"
        )
        combined = _concat_audio_files(
            group, str(scratch), voice_compatible=voice_compatible,
        )
        if not combined:
            return list(group)
        scratch_outputs.append(combined)
        combined_size = Path(combined).stat().st_size
        if combined_size <= profile.max_file_bytes:
            combined_any = True
            return [combined]

        try:
            Path(combined).unlink()
        except OSError:
            pass
        midpoint = max(1, len(group) // 2)
        return emit(group[:midpoint]) + emit(group[midpoint:])

    packed: List[str] = []
    for group in _pack_audio_files_for_delivery(audio_paths, profile):
        packed.extend(emit(group))

    final_paths: List[str] = []
    for index, source in enumerate(packed, start=1):
        if len(packed) == 1:
            destination = base
        else:
            source_suffix = Path(source).suffix or base.suffix
            destination = base.with_name(
                f"{base.stem}.part{index:02d}{source_suffix}"
            )
        if os.path.abspath(source) != os.path.abspath(destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        if destination.stat().st_size > profile.max_file_bytes:
            raise ValueError(
                f"Final TTS deliverable exceeds {profile.platform} delivery limit: "
                f"{destination}"
            )
        final_paths.append(str(destination))

    try:
        return final_paths, combined_any
    finally:
        for scratch in scratch_outputs:
            if scratch not in final_paths:
                try:
                    Path(scratch).unlink()
                except OSError:
                    pass


# ===========================================================================
# Provider: Edge TTS (free)
# ===========================================================================
async def _generate_edge_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using Edge TTS.

    Args:
        text: Text to convert.
        output_path: Where to save the MP3 file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    _edge_tts = _import_edge_tts()
    edge_config = tts_config.get("edge") or {}
    voice = edge_config.get("voice", DEFAULT_EDGE_VOICE)
    speed = float(edge_config.get("speed", tts_config.get("speed", 1.0)))

    kwargs = {"voice": voice}
    if speed != 1.0:
        pct = round((speed - 1.0) * 100)
        kwargs["rate"] = f"{pct:+d}%"

    communicate = _edge_tts.Communicate(text, **kwargs)
    await communicate.save(output_path)
    return output_path


# ===========================================================================
# Provider: ElevenLabs (premium)
# ===========================================================================
def _generate_elevenlabs(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using ElevenLabs.

    Args:
        text: Text to convert.
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    api_key = (_resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs") or "")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set. Get one at https://elevenlabs.io/")

    el_config = tts_config.get("elevenlabs") or {}
    voice_id = el_config.get("voice_id", DEFAULT_ELEVENLABS_VOICE_ID)
    model_id = el_config.get("model_id", DEFAULT_ELEVENLABS_MODEL_ID)

    # Determine output format based on file extension
    if output_path.endswith(".ogg"):
        output_format = "opus_48000_64"
    else:
        output_format = "mp3_44100_128"

    ElevenLabs = _import_elevenlabs()
    client = ElevenLabs(api_key=api_key, **_elevenlabs_environment_kwargs(el_config))
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    )

    # audio_generator yields chunks -- write them all
    with open(output_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    return output_path


def _tts_response_format_from_path(output_path: str) -> str:
    """Pick an OpenAI-compatible TTS response format from the output extension."""
    if output_path.endswith(".ogg"):
        return "opus"
    if output_path.endswith(".wav"):
        return "wav"
    if output_path.endswith(".flac"):
        return "flac"
    return "mp3"


# ===========================================================================
# Provider: OpenAI TTS (also used by every OpenAI-compatible TTS endpoint —
# DeepInfra delegates here via _generate_deepinfra_tts).
# ===========================================================================
def _generate_openai_tts(
    text: str,
    output_path: str,
    tts_config: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
) -> str:
    """Generate audio via the OpenAI ``audio.speech.create`` SDK shape.

    Optional kwargs let OpenAI-compatible backends (DeepInfra etc.) reuse
    this function — they resolve credentials/model themselves and pass
    them through, skipping the OpenAI-only ``_resolve_openai_audio_client_config``.

    Args:
        text: Text to convert.
        output_path: Where to save the audio file.
        tts_config: TTS config dict (used for ``tts.openai`` sub-block
            and the global ``speed`` default).
        api_key: Bearer token. When None, resolved from the OpenAI auth
            chain (config → env → managed gateway).
        base_url: API base URL. When None, falls back to
            ``tts.openai.base_url`` then the OpenAI default.
        model: Model id. When None, reads ``tts.openai.model``.
        voice: Voice id. When None, reads ``tts.openai.voice``.
        speed: Playback speed. When None, reads ``tts.openai.speed`` /
            ``tts.speed``.
        instructions: Optional voice-design guidance (tone, emotion, pacing,
            accent, whispering). Forwarded to `audio.speech.create` when
            truthy; omitted otherwise so ``tts-1``/``tts-1-hd`` and strict
            OpenAI-compatible servers that reject unknown kwargs are
            unaffected.

    Returns:
        Path to the saved audio file.
    """
    # Only resolve the OpenAI auth chain when the caller didn't pass explicit
    # credentials. OpenAI-compatible backends (DeepInfra) pass api_key /
    # base_url / model / voice through and never hit the managed-gateway path.
    fallback_base: Optional[str] = None
    is_managed = False
    explicit_base_url = base_url is not None
    if api_key is None:
        api_key, fallback_base, is_managed = _resolve_openai_audio_client_config()

    # ``tts.openai: null`` in YAML yields None — coalesce so .get() is safe.
    oai_config = (tts_config.get("openai") if isinstance(tts_config, dict) else None) or {}
    if model is None:
        model = oai_config.get("model", DEFAULT_OPENAI_MODEL)
    if voice is None:
        voice = oai_config.get("voice", DEFAULT_OPENAI_VOICE)
    config_base_url = oai_config.get("base_url")
    if base_url is None:
        # Config override wins over the auth-chain fallback (restores the
        # pre-refactor precedence, where tts.openai.base_url beat the resolved
        # default); the auth-chain value is the last-resort default. An
        # explicit base_url arg from an OpenAI-compatible caller (DeepInfra)
        # skips this block entirely and always wins.
        base_url = config_base_url or fallback_base or DEFAULT_OPENAI_BASE_URL
    if speed is None:
        speed_default = tts_config.get("speed", 1.0) if isinstance(tts_config, dict) else 1.0
        speed = float(oai_config.get("speed", speed_default))
    language = oai_config.get("language")

    # The managed OpenAI audio gateway only proxies MANAGED_OPENAI_TTS_MODELS.
    # A model set for direct OpenAI (e.g. "tts-1-hd") 400s there with
    # "Unsupported managed OpenAI speech model", so coerce it — unless the user
    # redirected base_url to their own endpoint, in which case respect it.
    if (
        is_managed
        and not explicit_base_url
        and not config_base_url
        and model not in MANAGED_OPENAI_TTS_MODELS
    ):
        logger.warning(
            "TTS: managed OpenAI audio gateway does not support model %r; "
            "falling back to %s. Set VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY "
            "to use %r directly.",
            model, DEFAULT_OPENAI_MODEL, model,
        )
        model = DEFAULT_OPENAI_MODEL

    response_format = _tts_response_format_from_path(output_path)

    OpenAIClient = _import_openai_client()
    client = OpenAIClient(api_key=api_key, base_url=base_url)
    try:
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
            "extra_headers": {"x-idempotency-key": str(uuid.uuid4())},
        }
        if speed != 1.0:
            create_kwargs["speed"] = max(0.25, min(4.0, speed))
        if instructions:
            create_kwargs["instructions"] = instructions
        if language:
            create_kwargs["extra_body"] = {"lang_code": language}
        response = client.audio.speech.create(**create_kwargs)

        response.stream_to_file(output_path)
        return output_path
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


# ===========================================================================
# Provider: DeepInfra TTS
# ===========================================================================
#
# DeepInfra serves TTS over an OpenAI-compatible /v1/openai/audio/speech
# endpoint. Models are discovered live via the shared catalog helper
# (filtered by the ``tts`` surface tag) — no hardcoded model ids in this
# file, so retired models disappear from hermes the next time the
# catalog is fetched without a patch.


def _generate_deepinfra_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Resolve DeepInfra credentials/model, then delegate to the OpenAI handler.

    DeepInfra's audio endpoint is OpenAI-compatible, so there's no need
    to duplicate the SDK call — we just pass an explicit api_key /
    base_url / model / voice through. Model ids and the base URL come from
    the shared ``hermes_cli.models`` helpers so every DeepInfra surface
    resolves them identically.
    """
    api_key = _resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")
    if not api_key:
        raise ValueError(
            "DEEPINFRA_API_KEY not set. Run `hermes setup` to configure, "
            "or set the env var directly."
        )

    # ``tts.deepinfra: null`` in YAML yields None, not {} — coalesce so the
    # ``.get`` calls below don't raise AttributeError (there is no
    # tts.deepinfra block in DEFAULT_CONFIG to deep-merge over the null).
    di_config = tts_config.get("deepinfra") if isinstance(tts_config, dict) else None
    if not isinstance(di_config, dict):
        di_config = {}

    from hermes_cli.models import deepinfra_base_url, deepinfra_model_ids

    model = di_config.get("model")
    if not isinstance(model, str) or not model.strip():
        candidates = deepinfra_model_ids("tts")
        if not candidates:
            raise ValueError(
                "No DeepInfra TTS model available. Pin one in config.yaml "
                "under tts.deepinfra.model, or check connectivity to "
                "api.deepinfra.com so the live catalog can be fetched."
            )
        model = candidates[0]
    return _generate_openai_tts(
        text,
        output_path,
        tts_config,
        api_key=api_key,
        base_url=deepinfra_base_url(di_config),
        model=model,
        voice=di_config.get("voice", DEFAULT_DEEPINFRA_TTS_VOICE),
        speed=float(di_config.get("speed", tts_config.get("speed", 1.0))),
    )


# ===========================================================================
# Provider: xAI TTS
# ===========================================================================
_XAI_INLINE_SPEECH_TAGS = (
    "pause",
    "long-pause",
    "hum-tune",
    "laugh",
    "chuckle",
    "giggle",
    "cry",
    "tsk",
    "tongue-click",
    "lip-smack",
    "breath",
    "inhale",
    "exhale",
    "sigh",
)
_XAI_WRAPPING_SPEECH_TAGS = (
    "soft",
    "whisper",
    "loud",
    "build-intensity",
    "decrease-intensity",
    "higher-pitch",
    "lower-pitch",
    "slow",
    "fast",
    "sing-song",
    "singing",
    "laugh-speak",
    "emphasis",
)
_XAI_SPEECH_TAG_RE = re.compile(
    r"(\[(?:" + "|".join(_XAI_INLINE_SPEECH_TAGS) + r")\]|</?(?:" + "|".join(_XAI_WRAPPING_SPEECH_TAGS) + r")>)",
    flags=re.IGNORECASE,
)
_XAI_FIRST_SENTENCE_RE = re.compile(r"^(.{12,120}?[.!?…])\s+(?=\S)", flags=re.DOTALL)


def _xai_bool_config(value: Any, default: bool = False) -> bool:
    return _config_bool(value, default=default)


def _apply_xai_auto_speech_tags(text: str) -> str:
    """Add xAI speech tags for more natural voice-mode replies.

    First applies a conservative local transform (inserts [pause] between
    paragraphs and after the first sentence). Then, if the result contains
    no explicit user/model speech tags, asks the configured auxiliary model
    to rewrite the transcript with a richer set of xAI-supported tags
    (laughs, sighs, whispers, soft/loud, slow/fast, etc.) so the voice
    output sounds more expressive. Falls back to the local result on any
    auxiliary-model failure.
    """
    clean = text.strip()
    if not clean:
        return text

    # Local conservative pass: pauses only.
    local = clean
    local = re.sub(r"\n\s*\n+", " [pause] ", local)
    local = re.sub(r"\s*\n\s*", " ", local)
    if not _XAI_SPEECH_TAG_RE.search(local):
        local = _XAI_FIRST_SENTENCE_RE.sub(r"\1 [pause] ", local, count=1)
    local = re.sub(r"\s{2,}", " ", local).strip()

    # If the user/model already supplied explicit speech tags, trust them
    # and don't re-rewrite.
    if _XAI_SPEECH_TAG_RE.search(clean):
        return local

    # Auxiliary rewrite for richer emotion tags (mirrors the Gemini path).
    inline = ", ".join(_XAI_INLINE_SPEECH_TAGS)
    wrapping = ", ".join(_XAI_WRAPPING_SPEECH_TAGS)
    system_prompt = (
        "You rewrite transcripts for the xAI /v1/tts endpoint by inserting "
        "expressive speech tags.\n\n"
        "Valid inline tags (use as `[tag]`): " + inline + ".\n"
        "Valid wrapping tags (use as `[tag]...[/tag]`): " + wrapping + ".\n\n"
        "Rules:\n"
        "- Preserve the spoken words, order, and meaning.\n"
        "- Do not add new spoken sentences or remove existing spoken words.\n"
        "- Use inline `[tag]` for short modifiers (laughs, sighs, pause, etc.).\n"
        "- Use wrapping `[tag]...[/tag]` for sustained effects (whisper, soft, slow, fast, loud, etc.).\n"
        "- Do not use angle-bracket tags like `<tag>...</tag>` — xAI uses BBCode-style closing tags with `[/tag]`.\n"
        "- Do not use SSML.\n"
        "- Do not explain or comment.\n"
        "- Return only the tagged TTS script."
    )
    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            task="tts_audio_tags",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"TRANSCRIPT TO TAG:\n{local}"},
            ],
            temperature=0.7,
        )
        tagged = _extract_auxiliary_message_content(response).strip()
        # Strip markdown fences if the LLM wrapped the response.
        fence = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", tagged, flags=re.DOTALL)
        if fence:
            tagged = fence.group(1).strip()
        return tagged or local
    except Exception as exc:
        logger.debug("xAI TTS audio tag rewrite failed; using locally-tagged text: %s", exc)
        return local


def _generate_xai_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using xAI TTS.

    xAI exposes a dedicated /v1/tts endpoint instead of the OpenAI audio.speech
    API shape, so this is implemented as a separate backend.
    """
    import requests

    from tools.xai_http import resolve_xai_http_credentials

    # TTS is API-billed: a subscription OAuth bearer can authorize chat while
    # returning 403 for /v1/tts (#87045, same root cause as x_search #88040),
    # so prefer an explicit XAI_API_KEY with OAuth as the fallback.
    creds = resolve_xai_http_credentials(prefer_api_key=True)
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise ValueError("No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY.")

    xai_config = tts_config.get("xai") or {}
    voice_id = str(xai_config.get("voice_id", DEFAULT_XAI_VOICE_ID)).strip() or DEFAULT_XAI_VOICE_ID
    language = str(xai_config.get("language", DEFAULT_XAI_LANGUAGE)).strip() or DEFAULT_XAI_LANGUAGE
    sample_rate = int(xai_config.get("sample_rate", DEFAULT_XAI_SAMPLE_RATE))
    bit_rate = int(xai_config.get("bit_rate", DEFAULT_XAI_BIT_RATE))
    auto_speech_tags = _xai_bool_config(
        xai_config.get("auto_speech_tags", xai_config.get("speech_tags")),
        DEFAULT_XAI_AUTO_SPEECH_TAGS,
    )
    # ``tts.xai.speed`` overrides global ``tts.speed``; the xAI TTS API
    # accepts 0.7..1.5 (1.0 = normal). Out-of-range values are clamped so a
    # misconfigured agent can't 400 the request — the API would reject
    # anything outside the band.
    speed = xai_config.get("speed", tts_config.get("speed"))
    if speed is not None and speed != "":
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = None
    if speed is not None:
        speed = max(DEFAULT_XAI_SPEED_MIN, min(DEFAULT_XAI_SPEED_MAX, speed))
    # ``tts.xai.optimize_streaming_latency`` is 0, 1, or 2 (xAI-specific;
    # trades chunk-boundary quality for time-to-first-audio).
    optimize_streaming_latency = xai_config.get(
        "optimize_streaming_latency",
        tts_config.get("optimize_streaming_latency"),
    )
    if optimize_streaming_latency is not None and optimize_streaming_latency != "":
        try:
            optimize_streaming_latency = int(optimize_streaming_latency)
        except (TypeError, ValueError):
            optimize_streaming_latency = None
    if optimize_streaming_latency is not None:
        optimize_streaming_latency = max(0, min(2, optimize_streaming_latency))
    # ``tts.xai.text_normalization`` enables spoken-form normalization
    # (numbers, abbreviations, symbols → words). Defaults to False.
    text_normalization = _xai_bool_config(
        xai_config.get("text_normalization"),
        DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT,
    )
    if auto_speech_tags:
        text = _apply_xai_auto_speech_tags(text)
    if creds.get("provider") == "xai-oauth":
        base_url = str(creds.get("base_url") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")
    else:
        base_url = str(
            xai_config.get("base_url")
            or creds.get("base_url")
            or get_env_value("XAI_BASE_URL")
            or DEFAULT_XAI_BASE_URL
        ).strip().rstrip("/")

    # Match the documented minimal POST /v1/tts shape by default. Only send
    # output_format when Hermes actually needs a non-default format/override.
    codec = "wav" if output_path.endswith(".wav") else "mp3"
    payload: Dict[str, Any] = {
        "text": text,
        "voice_id": voice_id,
        "language": language,
    }
    if (
        codec != "mp3"
        or sample_rate != DEFAULT_XAI_SAMPLE_RATE
        or (codec == "mp3" and bit_rate != DEFAULT_XAI_BIT_RATE)
    ):
        output_format: Dict[str, Any] = {"codec": codec}
        if sample_rate:
            output_format["sample_rate"] = sample_rate
        if codec == "mp3" and bit_rate:
            output_format["bit_rate"] = bit_rate
        payload["output_format"] = output_format
    # Only attach `speed` when the caller asked for something other than the
    # API default (1.0). Keeps the existing minimal-payload contract for
    # users who never touch the knob.
    if speed is not None and speed != DEFAULT_XAI_SPEED_DEFAULT:
        payload["speed"] = speed
    # Only attach `optimize_streaming_latency` when the caller explicitly
    # opts in to a non-default value (anything other than 0).
    if (
        optimize_streaming_latency is not None
        and optimize_streaming_latency != DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT
    ):
        payload["optimize_streaming_latency"] = optimize_streaming_latency
    # Only attach `text_normalization` when explicitly enabled (default is False).
    if text_normalization:
        payload["text_normalization"] = True

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
def _minimax_voice_id(mm_config: Dict[str, Any]) -> str:
    """Resolve the configured MiniMax voice id.

    Both payload shapes need it (t2a_v2 nests it under voice_setting, the older
    text_to_speech endpoint carries it flat), so the default lives in one place
    and the two cannot drift apart.
    """
    return mm_config.get("voice_id", DEFAULT_MINIMAX_VOICE_ID)


def _minimax_voice_setting(mm_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the t2a_v2 voice_setting payload. (minimax-auto-emotion)

    The emotion field is only sent when a config explicitly sets one. With
    the field omitted, MiniMax "automatically selects the most natural
    emotion based on text", which is the API's documented default; the old
    hardcoded "neutral" fallback silently disabled that inference on every
    request, flattening delivery. An empty value or "auto" also omits the
    field, so a config can state the intent explicitly.
    """
    setting = {
        "voice_id": _minimax_voice_id(mm_config),
        "speed": mm_config.get("speed", 1.0),
        "vol": mm_config.get("vol", 1.0),
        "pitch": mm_config.get("pitch", 0),
    }
    emotion = str(mm_config.get("emotion") or "").strip().lower()
    if emotion and emotion != "auto":
        if emotion not in MINIMAX_EMOTIONS:
            logger.warning(
                "tts.minimax.emotion is %r, which is not an emotion MiniMax "
                "documents (%s). Sending it as configured; if the API rejects "
                "it, correct the value or unset it to let MiniMax infer the "
                "emotion from the text.",
                emotion,
                ", ".join(sorted(MINIMAX_EMOTIONS)),
            )
        setting["emotion"] = emotion
    return setting


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
    voice_id = _minimax_voice_id(mm_config)
    base_url = runtime.endpoint
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
            "voice_setting": _minimax_voice_setting(mm_config),
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
    want_opus = platform in OPUS_VOICE_PLATFORMS

    # Determine output path
    if output_path:
        # Reject '..' traversal components in the user-supplied path. An
        # explicit absolute path is fine (the agent legitimately writes
        # audio to user-specified locations), but a path that uses ``..``
        # to escape its declared base is almost always either a bug or
        # prompt-injection-controlled — e.g.
        # ``output_path="audio/../../etc/cron.d/x"``. The terminal tool
        # can still write anywhere with approval; this just keeps the
        # unattended TTS surface from materializing files via traversal.
        from tools.path_security import has_traversal_component
        if has_traversal_component(output_path):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path contains '..' traversal component: "
                    f"{output_path}. Use an absolute path or one relative "
                    "to the current directory without '..'."
                ),
            }, ensure_ascii=False)
        file_path = Path(output_path).expanduser()
        if command_provider_config is not None:
            # Respect caller-supplied path but align the extension with the
            # provider's configured output_format so the command writes to a
            # path the caller actually expects.
            file_path = _configured_command_tts_output_path(
                file_path, command_provider_config
            )
        from agent.file_safety import is_write_approval_required, is_write_denied

        if is_write_denied(str(file_path)) or is_write_approval_required(str(file_path)):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path targets a protected credential or system path: "
                    f"{file_path}. Choose a normal audio output location."
                ),
            }, ensure_ascii=False)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path(_default_output_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        if command_provider_config is not None:
            fmt = _get_command_tts_output_format(command_provider_config)
            file_path = out_dir / f"tts_{timestamp}.{fmt}"
        # Use .ogg for Telegram with providers that support native Opus output,
        # otherwise fall back to .mp3 (Edge TTS will attempt ffmpeg conversion later).
        elif want_opus and provider in {"openai", "elevenlabs", "mistral", "gemini"}:
            file_path = out_dir / f"tts_{timestamp}.ogg"
        else:
            file_path = out_dir / f"tts_{timestamp}.mp3"

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_str = str(file_path)

    try:
        # Generate audio with the configured provider
        if command_provider_config is not None:
            logger.info(
                "Generating speech with command TTS provider '%s'...", provider,
            )
            file_str = _generate_command_tts(
                text, file_str, provider, command_provider_config, tts_config,
            )

        # Plugin-registered TTS backend (issue #30398). Fires when the
        # configured provider is neither a built-in nor a command-type
        # entry, AND a plugin is registered under that name. The walrus
        # binds `_plugin_path` only when the dispatcher returns a path
        # (i.e. a plugin was actually found); a None return falls
        # through to the built-in elif chain so unknown names hit the
        # Edge TTS default at the bottom. The dispatcher itself enforces
        # built-ins-always-win + command-wins-over-plugin defensively.
        elif provider not in BUILTIN_TTS_PROVIDERS and (
            _plugin_path := _dispatch_to_plugin_provider(
                text, file_str, provider, tts_config,
            )
        ) is not None:
            file_str = _plugin_path

        elif provider == "elevenlabs":
            try:
                _import_elevenlabs()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "ElevenLabs provider selected but 'elevenlabs' package not installed. Run: pip install elevenlabs"
                }, ensure_ascii=False)
            logger.info("Generating speech with ElevenLabs...")
            _generate_elevenlabs(text, file_str, tts_config)

        elif provider == "openai":
            try:
                _import_openai_client()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "OpenAI provider selected but 'openai' package not installed."
                }, ensure_ascii=False)
            logger.info("Generating speech with OpenAI TTS...")
            _generate_openai_tts(text, file_str, tts_config, instructions=instructions)

        elif provider == "deepinfra":
            try:
                _import_openai_client()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "DeepInfra TTS uses the 'openai' SDK but it isn't installed."
                }, ensure_ascii=False)
            logger.info("Generating speech with DeepInfra TTS...")
            _generate_deepinfra_tts(text, file_str, tts_config)

        elif provider == "minimax":
            logger.info("Generating speech with MiniMax TTS...")
            _generate_minimax_tts(text, file_str, tts_config)

        elif provider == "xai":
            logger.info("Generating speech with xAI TTS...")
            _generate_xai_tts(text, file_str, tts_config)

        elif provider == "mistral":
            try:
                _import_mistral_client()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "Mistral provider selected but 'mistralai' package not installed. "
                             "Run `hermes setup` to install Mistral support."
                }, ensure_ascii=False)
            logger.info("Generating speech with Mistral Voxtral TTS...")
            _generate_mistral_tts(text, file_str, tts_config)

        elif provider == "gemini":
            logger.info("Generating speech with Google Gemini TTS...")
            _generate_gemini_tts(text, file_str, tts_config)

        elif provider == "neutts":
            if not _check_neutts_available():
                return json.dumps({
                    "success": False,
                    "error": "NeuTTS provider selected but neutts is not installed. "
                             "Run hermes setup and choose NeuTTS, or install espeak-ng and run python -m pip install -U neutts[all]."
                }, ensure_ascii=False)
            logger.info("Generating speech with NeuTTS (local)...")
            _generate_neutts(text, file_str, tts_config)

        elif provider == "kittentts":
            try:
                _import_kittentts()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "KittenTTS provider selected but 'kittentts' package not installed. "
                             "Run 'hermes setup tts' and choose KittenTTS, or install manually: "
                             "pip install https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl"
                }, ensure_ascii=False)
            logger.info("Generating speech with KittenTTS (local, ~25MB)...")
            _generate_kittentts(text, file_str, tts_config)

        elif provider == "piper":
            try:
                _import_piper()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "Piper provider selected but 'piper-tts' package not installed. "
                             "Run 'hermes tools' and select Piper under TTS, or install manually: "
                             "pip install piper-tts",
                }, ensure_ascii=False)
            logger.info("Generating speech with Piper (local)...")
            _generate_piper_tts(text, file_str, tts_config)

        else:
            # Default: Edge TTS (free), with NeuTTS as local fallback
            edge_available = True
            try:
                _import_edge_tts()
            except ImportError:
                edge_available = False

            if edge_available:
                logger.info("Generating speech with Edge TTS...")
                try:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(
                            lambda: asyncio.run(_generate_edge_tts(text, file_str, tts_config))
                        ).result(timeout=60)
                except RuntimeError:
                    asyncio.run(_generate_edge_tts(text, file_str, tts_config))
            elif _check_neutts_available():
                logger.info("Edge TTS not available, falling back to NeuTTS (local)...")
                provider = "neutts"
                _generate_neutts(text, file_str, tts_config)
            else:
                return json.dumps({
                    "success": False,
                    "error": "No TTS provider available. Install edge-tts (pip install edge-tts) "
                             "or set up NeuTTS for local synthesis."
                }, ensure_ascii=False)

        # Check the file was actually created
        if not os.path.exists(file_str) or os.path.getsize(file_str) == 0:
            return json.dumps({
                "success": False,
                "error": f"TTS generation produced no output (provider: {provider})"
            }, ensure_ascii=False)

        # Class-level container repair: several backends silently write
        # MP3/WAV bytes into a .ogg output path (Edge, Piper, xAI,
        # OpenAI-compatible servers without opus support), which platforms
        # like Telegram render as broken 0-second voice bubbles. Sniff the
        # magic bytes once here — covering every current and future
        # provider — and transcode in place when they don't match.
        file_str = _repair_ogg_container(file_str)

        # Try Opus conversion for Telegram compatibility.
        # Edge TTS outputs MP3, NeuTTS/KittenTTS output WAV. Keep those native
        # formats for local/CLI playback and only convert when the current
        # platform actually needs Opus voice delivery.
        voice_compatible = False
        if command_provider_config is not None:
            # Command providers are documents by default. Voice-bubble
            # delivery only kicks in when the user explicitly opts in
            # via ``voice_compatible: true`` in their provider config.
            if _is_command_tts_voice_compatible(command_provider_config):
                if not file_str.endswith(".ogg"):
                    opus_path = _convert_to_opus(file_str)
                    if opus_path:
                        file_str = opus_path
                voice_compatible = file_str.endswith(".ogg")
        elif provider not in BUILTIN_TTS_PROVIDERS:
            # Plugin-registered provider (issue #30398). Voice-bubble
            # delivery opts in via ``TTSProvider.voice_compatible``
            # (mirrors the command-provider opt-in). Plugins that
            # already write Opus skip the ffmpeg conversion.
            plugin_voice_compatible = _plugin_provider_is_voice_compatible(provider)
            if plugin_voice_compatible:
                if not file_str.endswith(".ogg"):
                    opus_path = _convert_to_opus(file_str)
                    if opus_path:
                        file_str = opus_path
                voice_compatible = file_str.endswith(".ogg")
        elif (
            want_opus
            and provider in {"edge", "neutts", "minimax", "xai", "kittentts", "piper"}
            and not file_str.endswith(".ogg")
        ):
            opus_path = _convert_to_opus(file_str)
            if opus_path:
                file_str = opus_path
                voice_compatible = True
        elif provider in {"elevenlabs", "openai", "mistral", "gemini"}:
            voice_compatible = want_opus and file_str.endswith(".ogg")

        file_size = os.path.getsize(file_str)
        logger.info("TTS audio saved: %s (%s bytes, provider: %s)", file_str, f"{file_size:,}", provider)

        # Build response with MEDIA tag for platform delivery
        media_tag = f"MEDIA:{file_str}"
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"

        return json.dumps({
            "success": True,
            "file_path": file_str,
            "media_tag": media_tag,
            "provider": provider,
            "voice_compatible": voice_compatible,
        }, ensure_ascii=False)

    except ValueError as e:
        # Configuration errors (missing API keys, etc.)
        error_msg = f"TTS configuration error ({provider}): {e}"
        logger.error("%s", error_msg)
        return tool_error(error_msg, success=False)
    except FileNotFoundError as e:
        # Missing dependencies or files
        error_msg = f"TTS dependency missing ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)
    except Exception as e:
        # Unexpected errors
        error_msg = f"TTS generation failed ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)


def text_to_speech_tool(
    text: str,
    output_path: Optional[str] = None,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Convert text to speech audio with long-form chunking.

    Long text is normalized, split into provider-safe chunks, synthesized
    sequentially, and packed against destination platform upload limits.
    Each provider request is encoded to its final format before files are
    packed. Multi-chunk voice output is re-encoded when combined; failed
    combines preserve separate valid files, and no over-limit final artifact
    is returned.

    On messaging platforms, the returned MEDIA:<path> tag is intercepted
    by the send pipeline and delivered as a native voice message.
    In CLI mode, the file is saved to ~/voice-memos/.

    Args:
        text: The text to convert to speech. Provider-specific per-request
            character caps apply automatically (OpenAI 4096, xAI 15000,
            MiniMax 10000, ElevenLabs 5k-40k depending on model); longer
            input is split into ordered chunks without silent truncation.
        output_path: Optional custom save path.
        speed: Optional playback speed multiplier (0.25-4.0).
        instructions: Optional voice-design guidance (tone, emotion, pacing).
        provider: Optional TTS provider override.

    Returns:
        str: JSON result with success, file_path, file_paths, and MEDIA tag.
    """
    if not text or not text.strip():
        return tool_error("Text is required", success=False)

    # Normalize text via the shared cleaner: markdown, emoji, think blocks,
    # verifier footer, units, newline flattening.
    try:
        from tools.tts_text_normalize import prepare_spoken_text
        text = prepare_spoken_text(text, max_chars=None)
    except Exception:
        text = text.strip()
    if not text:
        return tool_error("Text is empty after TTS cleanup", success=False)

    tts_config = _load_tts_config()

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

    command_provider_config = _resolve_command_provider_config(provider, tts_config)
    max_len = _resolve_max_text_length(provider, tts_config)
    chunks = _split_text_for_tts(text, max_len)
    if not chunks:
        return tool_error("Text is required", success=False)
    if len(chunks) > 1:
        logger.info(
            "TTS text for provider %s split into %d chunks (input=%d chars, cap=%d)",
            provider,
            len(chunks),
            len(text),
            max_len,
        )

    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "").lower()
    want_opus = platform in OPUS_VOICE_PLATFORMS
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
    chunk_results: List[Dict[str, Any]] = []
    try:
        encoded_paths: List[str] = []
        for index, chunk in enumerate(chunks, start=1):
            if len(chunks) == 1:
                chunk_path = base_path
            else:
                chunk_path = base_path.with_name(
                    f"{base_path.stem}.chunk{index:03d}{base_path.suffix}"
                )
            generated_artifacts.add(str(chunk_path))
            raw_result = _text_to_speech_single(
                text=chunk,
                output_path=str(chunk_path),
                speed=speed,
                instructions=instructions,
                provider=provider,
                tts_config_override=tts_config,
            )
            try:
                chunk_result = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError):
                raise RuntimeError(
                    f"TTS chunk {index} returned invalid JSON: {str(raw_result)[:200]}"
                )
            if not chunk_result.get("success"):
                error_msg = chunk_result.get("error", "unknown error")
                return tool_error(
                    f"TTS chunk {index} failed ({provider}): {error_msg}",
                    success=False,
                )
            actual_path = str(chunk_result.get("file_path") or chunk_path)
            if not os.path.isfile(actual_path) or os.path.getsize(actual_path) <= 0:
                raise RuntimeError(
                    f"TTS chunk {index} produced no final audio: {actual_path}"
                )
            generated_artifacts.add(actual_path)
            encoded_paths.append(actual_path)
            chunk_results.append(chunk_result)

        voice_compatible = bool(chunk_results) and all(
            bool(result.get("voice_compatible")) for result in chunk_results
        )
        delivery_base = base_path.with_suffix(Path(encoded_paths[0]).suffix)
        final_paths, combined_chunks = _build_audio_delivery_files(
            encoded_paths,
            str(delivery_base),
            delivery_profile,
            voice_compatible=voice_compatible,
        )

        for path in final_paths:
            logger.info(
                "TTS audio saved: %s (%s bytes, provider: %s)",
                path,
                f"{os.path.getsize(path):,}",
                provider,
            )
        media_tag = "\n".join(f"MEDIA:{path}" for path in final_paths)
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"

        return json.dumps({
            "success": True,
            "file_path": final_paths[0],
            "file_paths": final_paths,
            "media_tag": media_tag,
            "provider": chunk_results[0].get("provider", provider),
            "voice_compatible": voice_compatible,
            "chunk_count": len(chunks),
            "delivery_file_count": len(final_paths),
            "combined_chunks": bool(combined_chunks),
            "delivery_profile": {
                "platform": delivery_profile.platform,
                "max_file_bytes": delivery_profile.max_file_bytes,
                "target_file_bytes": delivery_profile.target_file_bytes,
            },
        }, ensure_ascii=False)
    except ValueError as exc:
        error_msg = f"TTS delivery error ({provider}): {exc}"
        logger.error("%s", error_msg)
        return tool_error(error_msg, success=False)
    except Exception as exc:
        error_msg = f"TTS long-form generation failed ({provider}): {exc}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)
    finally:
        final_absolute = {os.path.abspath(path) for path in final_paths}
        for artifact in generated_artifacts:
            if os.path.abspath(artifact) in final_absolute:
                continue
            try:
                os.unlink(artifact)
            except OSError:
                pass


# ===========================================================================
# Requirements check
# ===========================================================================
def check_tts_requirements() -> bool:
    """Return whether the explicitly resolved TTS provider can run.

    Availability must mirror :func:`text_to_speech_tool` dispatch. Unrelated
    cloud credentials do not make the default Edge backend usable, and an
    explicitly selected backend is checked on its own requirements.
    """
    tts_config = _load_tts_config()
    provider = _get_provider(tts_config)
    command_config = _resolve_command_provider_config(provider, tts_config)
    if command_config is not None:
        return True

    if provider == "edge":
        try:
            _import_edge_tts()
            return True
        except ImportError:
            return _check_neutts_available()
    if provider == "elevenlabs":
        try:
            _import_elevenlabs()
        except ImportError:
            return False
        return bool(_resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs"))
    if provider == "openai":
        if importlib.util.find_spec("openai") is None:
            return False
        return _has_openai_audio_backend()
    if provider == "deepinfra":
        if importlib.util.find_spec("openai") is None:
            return False
        return bool(_resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra"))
    if provider == "minimax":
        try:
            _resolve_minimax_tts_runtime(tts_config)
        except ValueError:
            return False
        return True
    if provider == "xai":
        try:
            from tools.xai_http import resolve_xai_http_credentials

            return bool(resolve_xai_http_credentials().get("api_key"))
        except Exception:
            return False
    if provider == "gemini":
        return bool(
            _resolve_provider_key("GEMINI_API_KEY", "gemini")
            or _resolve_provider_key("GOOGLE_API_KEY", "gemini")
        )
    if provider == "mistral":
        try:
            _import_mistral_client()
        except ImportError:
            return False
        return bool(_resolve_provider_key("MISTRAL_API_KEY", "mistral"))
    if provider == "neutts":
        return _check_neutts_available()
    if provider == "kittentts":
        return _check_kittentts_available()
    if provider == "piper":
        return _check_piper_available()

    try:
        from agent.tts_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        plugin = get_provider(provider)
        return bool(plugin and plugin.is_available())
    except Exception:
        return False


def _resolve_openai_audio_client_config() -> tuple[str, str, bool]:
    """Return ``(api_key, base_url, is_managed)`` for the OpenAI audio client.

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
                "description": f"Optional custom file path to save the audio. Defaults to {display_hermes_home()}/audio_cache/<timestamp>.mp3"
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
        output_path=args.get("output_path"),
        speed=args.get("speed"),
        instructions=args.get("instructions"),
        provider=args.get("provider")),
    check_fn=check_tts_requirements,
    emoji="🔊",
)
