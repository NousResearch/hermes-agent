"""Constants, result envelopes and tiny config readers shared by every STT module."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict

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
# Seconds for one STT HTTP request; shared by the OpenAI-SDK path and the QQ adapter so a
# self-hosted model's cold start is not cut off at the old fixed 30s (#112939).
DEFAULT_STT_TIMEOUT = 60.0
# /v1/stt's server default moved from grok-voice-transcribe-1.0 to 2.0 between Sep 17 and
# Sep 21 2026 and 1.0 is slated for retirement; naming the model keeps the wire deterministic
# and lets STT_XAI_MODEL / stt.xai.model pin 1.0 for a rollback.
XAI_STT_MODEL_ENV = "STT_XAI_MODEL"
LEGACY_XAI_STT_MODEL = "grok-stt"
LOCAL_STT_COMMAND_ENV = "HERMES_LOCAL_STT_COMMAND"
LOCAL_STT_LANGUAGE_ENV = "HERMES_LOCAL_STT_LANGUAGE"
COMMON_LOCAL_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_BASE_URL = os.getenv("STT_OPENAI_BASE_URL", "https://api.openai.com/v1")
XAI_STT_BASE_URL = os.getenv("XAI_STT_BASE_URL", "https://api.x.ai/v1")
ELEVENLABS_STT_BASE_URL = os.getenv("ELEVENLABS_STT_BASE_URL", "https://api.elevenlabs.io/v1")
# DeepInfra STT base URL is resolved via hermes_cli.models.deepinfra_base_url (shared).

SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".opus", ".aac", ".flac", ".caf"}
LOCAL_NATIVE_AUDIO_FORMATS = {".wav", ".aiff", ".aif"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Per-provider model catalogs keyed by ``stt.<provider>`` section, default first. This one table
# feeds the `hermes tools` picker, the dashboard selects and the auto-correction sets below.
# DeepInfra has no static list: its picker reads the live catalog.
STT_MODEL_CATALOG = {
    "local": ["base", "tiny", "small", "medium", "large-v3", "turbo"],
    "groq": ["whisper-large-v3-turbo", "whisper-large-v3"],
    "openai": ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"],
    "mistral": ["voxtral-mini-latest", "voxtral-mini-2602"],
    "xai": ["grok-voice-transcribe-2.0", "grok-voice-transcribe-1.0"],
    "elevenlabs": ["scribe_v2", "scribe_v1"]}
# ElevenLabs historically uses ``model_id`` instead of ``model``.
STT_MODEL_CONFIG_KEY = {"elevenlabs": "model_id"}

# Known model sets for auto-correction. Groq shut distil-whisper-large-v3-en down on
# 2025-08-23 (console.groq.com/docs/deprecations); a config still naming it is remapped.
OPENAI_MODELS = frozenset(STT_MODEL_CATALOG["openai"])
GROQ_MODELS = frozenset(STT_MODEL_CATALOG["groq"])
RETIRED_GROQ_MODELS = frozenset({"distil-whisper-large-v3-en"})

# Providers with native handlers. Kept in sync with ``agent.transcription_registry._BUILTIN_NAMES``
# (a regression test fails on drift); plugins may not register under these names and the
# dispatcher short-circuits them before command/plugin lookup.
# The plugin hook from issue #30398-style follow-up rejects plugins registering under any of these names;
# the dispatcher in ``transcribe_audio`` short-circuits them defensively as well.
BUILTIN_STT_PROVIDERS = frozenset({
    "local", "local_command", "groq", "openai", "mistral", "xai", "elevenlabs", "deepinfra"})
# Built-in providers that upload audio to a remote API.
CLOUD_STT_PROVIDERS = frozenset(BUILTIN_STT_PROVIDERS - {"local", "local_command"})


def _error_result(error: str, **extra: Any) -> dict[str, Any]:
    """Standard failure envelope shared by every provider and validator."""
    return {"success": False, "transcript": "", "error": error, **extra}


class STTResponseError(ValueError):
    """A provider answered with a structured object carrying no transcript text.

    Raised by ``_extract_transcript_text`` when an SDK object or JSON body reports an
    ``error`` (or neither ``text`` nor ``error``) instead of a usable transcript. The
    message is the provider's own, so the STT failure paths surface it verbatim rather
    than stringifying the response object into its repr (#78098)."""


def normalize_xai_stt_model(model: Any) -> str:
    """Return a valid xAI STT model: blank or Hermes' old ``grok-stt`` alias -> ``STT_XAI_MODEL`` (read
    per call, so a profile's env applies), else the catalog default."""
    value = str(model or "").strip()
    if not value or value == LEGACY_XAI_STT_MODEL:
        return os.getenv(XAI_STT_MODEL_ENV, "").strip() or STT_MODEL_CATALOG["xai"][0]
    return value


def _ok_result(transcript: str, provider: str) -> dict[str, Any]:
    return {"success": True, "transcript": transcript, "provider": provider}


def _lazy_ensure_quietly(extra: str) -> None:
    """Best-effort ``pm.ensure_import(extra)``; failures are swallowed.
    Installs are gated by ``security.allow_lazy_installs`` inside pm."""
    try:
        import pm
        pm.ensure_import(extra)
    except Exception:
        pass


def _process_error_detail(exc: "subprocess.CalledProcessError") -> str:
    """stderr > stdout > str(exc) for a failed helper binary."""
    for output in (exc.stderr, exc.stdout):
        if isinstance(output, bytes):
            detail = output.decode("utf-8", errors="replace").strip()
        else:
            detail = str(output or "").strip()
        if detail:
            return detail
    return str(exc)


def _log_prompt_unsupported(label: str) -> None:
    logger.debug("%s does not support transcription prompts — proceeding without the prompt.", label)


def _config_number(cfg: dict[str, Any], key: str, default, cast=float):
    """Read ``cfg[key]`` through *cast*, falling back to *default* on bad values."""
    try:
        return cast(cfg.get(key, default))
    except (TypeError, ValueError):
        return default
