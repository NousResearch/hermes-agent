"""Constants, result envelopes and tiny config readers shared by every STT module."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, Optional

from tools.tts_command_provider import _get_provider_section as _get_stt_section

# Log-record parity with the origin module.
logger = logging.getLogger("tools.transcription_tools")

DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_MODEL = "base"
DEFAULT_LOCAL_STT_LANGUAGE = "en"
DEFAULT_STT_MODEL = os.getenv("STT_OPENAI_MODEL", "whisper-1")
DEFAULT_GROQ_STT_MODEL = os.getenv("STT_GROQ_MODEL", "whisper-large-v3-turbo")
DEFAULT_MISTRAL_STT_MODEL = os.getenv("STT_MISTRAL_MODEL", "voxtral-mini-latest")
DEFAULT_ELEVENLABS_STT_MODEL = os.getenv("STT_ELEVENLABS_MODEL", "scribe_v2")
DEFAULT_OPENROUTER_STT_MODEL = os.getenv("STT_OPENROUTER_MODEL", "openai/whisper-large-v3")
LOCAL_STT_COMMAND_ENV = "HERMES_LOCAL_STT_COMMAND"
LOCAL_STT_LANGUAGE_ENV = "HERMES_LOCAL_STT_LANGUAGE"
COMMON_LOCAL_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_BASE_URL = os.getenv("STT_OPENAI_BASE_URL", "https://api.openai.com/v1")
XAI_STT_BASE_URL = os.getenv("XAI_STT_BASE_URL", "https://api.x.ai/v1")
ELEVENLABS_STT_BASE_URL = os.getenv("ELEVENLABS_STT_BASE_URL", "https://api.elevenlabs.io/v1")
OPENROUTER_STT_BASE_URL = os.getenv("STT_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# DeepInfra STT base URL is resolved via hermes_cli.models.deepinfra_base_url (shared).

SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".opus", ".aac", ".flac", ".caf"}
LOCAL_NATIVE_AUDIO_FORMATS = {".wav", ".aiff", ".aif"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Known model sets for auto-correction
OPENAI_MODELS = {"whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"}
GROQ_MODELS = {"whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"}
# OpenRouter's transcription catalog is vendor-prefixed (`vendor/model`); this is the suggestion set
# for the pickers, mainstream entries first. Any live slug may be pinned — the catalog moves, so
# membership is a hint, never a gate.
OPENROUTER_STT_MODELS = (
    "openai/whisper-large-v3", "openai/whisper-large-v3-turbo", "openai/whisper-1",
    "openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe", "openai/gpt-transcribe",
    "mistralai/voxtral-mini-transcribe", "google/chirp-3", "deepgram/nova-3",
    "x-ai/grok-stt-1.0", "qwen/qwen3-asr-flash-2026-02-10", "microsoft/mai-transcribe-2",
    "nvidia/parakeet-tdt-0.6b-v3", "fish-audio/transcribe-1", "meta/muse-voice-transcribe-1.0",
    "mistralai/voxtral-small-24b-2507-stt", "qwen/qwen3-asr-1.7b", "qwen/qwen3-asr-0.6b",
    "microsoft/mai-transcribe-1.5", "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
)

# Providers with native handlers. Kept in sync with ``agent.transcription_registry._BUILTIN_NAMES``
# (a regression test fails on drift); plugins may not register under these names and the
# dispatcher short-circuits them before command/plugin lookup.
# The plugin hook from issue #30398-style follow-up rejects plugins registering under any of these names;
# the dispatcher in ``transcribe_audio`` short-circuits them defensively as well.
BUILTIN_STT_PROVIDERS = frozenset({
    "local", "local_command", "groq", "openai", "openrouter", "mistral", "xai", "elevenlabs",
    "deepinfra"})
# Built-in providers that upload audio to a remote API.
CLOUD_STT_PROVIDERS = frozenset(BUILTIN_STT_PROVIDERS - {"local", "local_command"})


# Models whose endpoint refuses the containers our clients record (WebM/Opus from the desktop
# recorder, Ogg/Opus voice notes from messaging). Probed against the live OpenRouter catalog
# 2026-09-14 with an MP3: of 20 transcription models, exactly one rejects non-WAV input —
# meta/muse-voice-transcribe-1.0 — and it additionally requires a 16 kHz or 24 kHz WAV sample rate
# ("Meta transcription requires a 16000 Hz or 24000 Hz WAV sample rate (received 22050 Hz)").
# The DESKTOP converts before upload (WebAudio decode + WAV encode): the browser can always decode
# what it recorded, whereas the server path needs ffmpeg, which is not guaranteed to exist.
STT_WAV_ONLY_MODELS = frozenset({"meta/muse-voice-transcribe-1.0"})
#: Sample rates a WAV-only endpoint accepts; anything else is rejected outright.
STT_WAV_SAMPLE_RATES = (16000, 24000)
STT_WAV_TARGET_SAMPLE_RATE = 16000


def stt_requires_wav(model_name: Optional[str]) -> bool:
    """True when *model_name* accepts only mono WAV at :data:`STT_WAV_SAMPLE_RATES`."""
    return (model_name or "").strip().lower() in STT_WAV_ONLY_MODELS


def openrouter_stt_base_url(section: Any = None) -> str:
    """``stt.openrouter.base_url`` when set, else the env-resolved ``OPENROUTER_STT_BASE_URL``.

    Shared by the relay handler (tools/transcription_cloud.py) and the desktop's client-direct
    resolver (tools/voice_client_config.py) so a configured endpoint never applies to only one path.
    """
    configured = ""
    if isinstance(section, dict):
        configured = str(section.get("base_url") or "").strip()
    return configured.rstrip("/") or OPENROUTER_STT_BASE_URL


def _error_result(error: str, **extra: Any) -> Dict[str, Any]:
    """Standard failure envelope shared by every provider and validator."""
    return {"success": False, "transcript": "", "error": error, **extra}


def _ok_result(transcript: str, provider: str) -> Dict[str, Any]:
    return {"success": True, "transcript": transcript, "provider": provider}


def _lazy_ensure_quietly(dep: str) -> None:
    """Best-effort ``tools.lazy_deps.ensure(dep, prompt=False)``; failures are swallowed.
    prompt=False: a bare input() deadlocks under the interactive CLI where prompt_toolkit owns
    stdin; installs are gated by ``security.allow_lazy_installs``."""
    try:
        from tools.lazy_deps import ensure
        ensure(dep, prompt=False)
    except Exception:
        pass


def _process_error_detail(exc: "subprocess.CalledProcessError") -> str:
    """stderr > stdout > str(exc) for a failed helper binary."""
    return exc.stderr.strip() or exc.stdout.strip() or str(exc)


def _log_prompt_unsupported(label: str) -> None:
    logger.debug("%s does not support transcription prompts — proceeding without the prompt.", label)


def _config_number(cfg: Dict[str, Any], key: str, default, cast=float):
    """Read ``cfg[key]`` through *cast*, falling back to *default* on bad values."""
    try:
        return cast(cfg.get(key, default))
    except (TypeError, ValueError):
        return default
