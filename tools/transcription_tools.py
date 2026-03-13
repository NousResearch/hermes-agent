#!/usr/bin/env python3
"""
Transcription Tools Module

Provides speech-to-text transcription with three providers:

  - **local** (default, free) — faster-whisper running locally, no API key needed.
    Auto-downloads the model (~150 MB for ``base``) on first use.
  - **groq** (free tier) — Groq Whisper API, requires ``GROQ_API_KEY``.
  - **openai** (paid) — OpenAI Whisper API, requires ``VOICE_TOOLS_OPENAI_KEY``.

Used by the messaging gateway to automatically transcribe voice messages
sent by users on Telegram, Discord, WhatsApp, Slack, and Signal.

Supported input formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg

Usage::

    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio("/path/to/audio.ogg")
    if result["success"]:
        print(result["transcript"])
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    from faster_whisper import WhisperModel
    _HAS_FASTER_WHISPER = True
except ImportError:
    _HAS_FASTER_WHISPER = False
    WhisperModel = None  # type: ignore[assignment,misc]

try:
    from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_MODEL = "base"
DEFAULT_STT_MODEL = os.getenv("STT_OPENAI_MODEL", "whisper-1")
DEFAULT_GROQ_STT_MODEL = os.getenv("STT_GROQ_MODEL", "whisper-large-v3-turbo")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_BASE_URL = os.getenv("STT_OPENAI_BASE_URL", "https://api.openai.com/v1")

SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Known model sets for auto-correction
OPENAI_MODELS = {"whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"}
GROQ_MODELS = {"whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"}

# Singleton for the local model — loaded once, reused across calls
_local_model: Optional["WhisperModel"] = None
_local_model_name: Optional[str] = None

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def get_stt_model_from_config() -> Optional[str]:
    """Read the STT model name from ~/.hermes/config.yaml.

    Returns the value of ``stt.model`` if present, otherwise ``None``.
    Silently returns ``None`` on any error (missing file, bad YAML, etc.).
    """
    try:
        import yaml
        cfg_path = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            return data.get("stt", {}).get("model")
    except Exception:
        pass
    return None


def _load_stt_config() -> dict:
    """Load the ``stt`` section from user config, falling back to defaults."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("stt", {})
    except Exception:
        return {}


def _get_provider(stt_config: dict) -> str:
    """Determine which STT provider to use.

    Priority:
      1. Explicit config value  (``stt.provider``)
      2. Auto-detect: local > groq (free) > openai (paid)
      3. Disabled (returns "none")
    """
    provider = stt_config.get("provider", DEFAULT_PROVIDER)

    if provider == "local":
        if _HAS_FASTER_WHISPER:
            return "local"
        # Local requested but not available — fall back to groq, then openai
        if _HAS_OPENAI and os.getenv("GROQ_API_KEY"):
            logger.info("faster-whisper not installed, falling back to Groq Whisper API")
            return "groq"
        if _HAS_OPENAI and os.getenv("VOICE_TOOLS_OPENAI_KEY"):
            logger.info("faster-whisper not installed, falling back to OpenAI Whisper API")
            return "openai"
        return "none"

    if provider == "groq":
        if _HAS_OPENAI and os.getenv("GROQ_API_KEY"):
            return "groq"
        # Groq requested but no key — fall back
        if _HAS_FASTER_WHISPER:
            logger.info("GROQ_API_KEY not set, falling back to local faster-whisper")
            return "local"
        if _HAS_OPENAI and os.getenv("VOICE_TOOLS_OPENAI_KEY"):
            logger.info("GROQ_API_KEY not set, falling back to OpenAI Whisper API")
            return "openai"
        return "none"

    if provider == "openai":
        if _HAS_OPENAI and os.getenv("VOICE_TOOLS_OPENAI_KEY"):
            return "openai"
        # OpenAI requested but no key — fall back
        if _HAS_FASTER_WHISPER:
            logger.info("VOICE_TOOLS_OPENAI_KEY not set, falling back to local faster-whisper")
            return "local"
        if _HAS_OPENAI and os.getenv("GROQ_API_KEY"):
            logger.info("VOICE_TOOLS_OPENAI_KEY not set, falling back to Groq Whisper API")
            return "groq"
        return "none"

    return provider  # Unknown — let it fail downstream

# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_audio_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Validate the audio file.  Returns an error dict or None if OK."""
    audio_path = Path(file_path)

    if not audio_path.exists():
        return {"success": False, "transcript": "", "error": f"Audio file not found: {file_path}"}
    if not audio_path.is_file():
        return {"success": False, "transcript": "", "error": f"Path is not a file: {file_path}"}
    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        return {
            "success": False,
            "transcript": "",
            "error": f"Unsupported format: {audio_path.suffix}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        }
    try:
        file_size = audio_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "success": False,
                "transcript": "",
                "error": f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE / (1024*1024):.0f}MB)",
            }
    except OSError as e:
        return {"success": False, "transcript": "", "error": f"Failed to access file: {e}"}

    return None

# ---------------------------------------------------------------------------
# Provider: local (faster-whisper)
# ---------------------------------------------------------------------------


def _transcribe_local(file_path: str, model_name: str) -> Dict[str, Any]:
    """Transcribe using faster-whisper (local, free)."""
    global _local_model, _local_model_name

    if not _HAS_FASTER_WHISPER:
        return {"success": False, "transcript": "", "error": "faster-whisper not installed"}

    try:
        # Lazy-load the model (downloads on first use, ~150 MB for 'base')
        if _local_model is None or _local_model_name != model_name:
            logger.info("Loading faster-whisper model '%s' (first load downloads the model)...", model_name)
            _local_model = WhisperModel(model_name, device="auto", compute_type="auto")
            _local_model_name = model_name

        segments, info = _local_model.transcribe(file_path, beam_size=5)
        transcript = " ".join(segment.text.strip() for segment in segments)

        logger.info(
            "Transcribed %s via local whisper (%s, lang=%s, %.1fs audio)",
            Path(file_path).name, model_name, info.language, info.duration,
        )

        return {"success": True, "transcript": transcript}

    except Exception as e:
        logger.error("Local transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Local transcription failed: {e}"}

# ---------------------------------------------------------------------------
# Provider: groq (Whisper API — free tier)
# ---------------------------------------------------------------------------


def _transcribe_groq(file_path: str, model_name: str) -> Dict[str, Any]:
    """Transcribe using Groq Whisper API (free tier available)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "GROQ_API_KEY not set"}

    if not _HAS_OPENAI:
        return {"success": False, "transcript": "", "error": "openai package not installed"}

    # Auto-correct model if caller passed an OpenAI-only model
    if model_name in OPENAI_MODELS:
        logger.info("Model %s not available on Groq, using %s", model_name, DEFAULT_GROQ_STT_MODEL)
        model_name = DEFAULT_GROQ_STT_MODEL

    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=30, max_retries=0)

        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
                response_format="text",
            )

        transcript_text = str(transcription).strip()
        logger.info("Transcribed %s via Groq API (%s, %d chars)",
                     Path(file_path).name, model_name, len(transcript_text))

        return {"success": True, "transcript": transcript_text, "provider": "groq"}

    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except APIConnectionError as e:
        return {"success": False, "transcript": "", "error": f"Connection error: {e}"}
    except APITimeoutError as e:
        return {"success": False, "transcript": "", "error": f"Request timeout: {e}"}
    except APIError as e:
        return {"success": False, "transcript": "", "error": f"API error: {e}"}
    except Exception as e:
        logger.error("Groq transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Transcription failed: {e}"}

# ---------------------------------------------------------------------------
# Provider: openai (Whisper API)
# ---------------------------------------------------------------------------


def _transcribe_openai(file_path: str, model_name: str) -> Dict[str, Any]:
    """Transcribe using OpenAI Whisper API (paid)."""
    api_key = os.getenv("VOICE_TOOLS_OPENAI_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "VOICE_TOOLS_OPENAI_KEY not set"}

    if not _HAS_OPENAI:
        return {"success": False, "transcript": "", "error": "openai package not installed"}

    # Auto-correct model if caller passed a Groq-only model
    if model_name in GROQ_MODELS:
        logger.info("Model %s not available on OpenAI, using %s", model_name, DEFAULT_STT_MODEL)
        model_name = DEFAULT_STT_MODEL

    try:
        client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL, timeout=30, max_retries=0)

        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
                response_format="text",
            )

        transcript_text = str(transcription).strip()
        logger.info("Transcribed %s via OpenAI API (%s, %d chars)",
                     Path(file_path).name, model_name, len(transcript_text))

        return {"success": True, "transcript": transcript_text, "provider": "openai"}

    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except APIConnectionError as e:
        return {"success": False, "transcript": "", "error": f"Connection error: {e}"}
    except APITimeoutError as e:
        return {"success": False, "transcript": "", "error": f"Request timeout: {e}"}
    except APIError as e:
        return {"success": False, "transcript": "", "error": f"API error: {e}"}
    except Exception as e:
        logger.error("OpenAI transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Transcription failed: {e}"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_audio(file_path: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe an audio file using the configured STT provider.

    Provider priority:
      1. User config (``stt.provider`` in config.yaml)
      2. Auto-detect: local faster-whisper (free) > Groq (free tier) > OpenAI (paid)

    Args:
        file_path: Absolute path to the audio file to transcribe.
        model:     Override the model. If None, uses config or provider default.

    Returns:
        dict with keys:
          - "success" (bool): Whether transcription succeeded
          - "transcript" (str): The transcribed text (empty on failure)
          - "error" (str, optional): Error message if success is False
          - "provider" (str, optional): Which provider was used
    """
    # Validate input
    error = _validate_audio_file(file_path)
    if error:
        return error

    # Load config and determine provider
    stt_config = _load_stt_config()
    provider = _get_provider(stt_config)

    if provider == "local":
        local_cfg = stt_config.get("local", {})
        model_name = model or local_cfg.get("model", DEFAULT_LOCAL_MODEL)
        return _transcribe_local(file_path, model_name)

    if provider == "groq":
        model_name = model or DEFAULT_GROQ_STT_MODEL
        return _transcribe_groq(file_path, model_name)

    if provider == "openai":
        openai_cfg = stt_config.get("openai", {})
        model_name = model or openai_cfg.get("model", DEFAULT_STT_MODEL)
        return _transcribe_openai(file_path, model_name)

    # No provider available
    return {
        "success": False,
        "transcript": "",
        "error": (
            "No STT provider available. Install faster-whisper for free local "
            "transcription, set GROQ_API_KEY for free Groq Whisper, "
            "or set VOICE_TOOLS_OPENAI_KEY for the OpenAI Whisper API."
        ),
    }
