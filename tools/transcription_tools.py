#!/usr/bin/env python3
"""
Transcription Tools Module

Provides speech-to-text transcription using OpenAI-compatible Whisper APIs.
Supports multiple providers with automatic fallback:
  1. OpenAI (VOICE_TOOLS_OPENAI_KEY) -- paid
  2. Groq  (GROQ_API_KEY)           -- free tier available

Used by the messaging gateway to automatically transcribe voice messages
sent by users on Telegram, Discord, WhatsApp, and Slack.

Supported models:
  OpenAI: whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe
  Groq:   whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en

Supported input formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg

Usage:
    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio("/path/to/audio.ogg")
    if result["success"]:
        print(result["transcript"])
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# Default STT models per provider
DEFAULT_STT_MODEL = "whisper-1"
DEFAULT_GROQ_STT_MODEL = "whisper-large-v3-turbo"

# Provider endpoints
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _resolve_stt_provider() -> Tuple[Optional[str], Optional[str], str]:
    """Resolve which STT provider to use based on available API keys.

    Returns:
        Tuple of (api_key, base_url, provider_name).
        api_key is None if no provider is available.
    """
    openai_key = os.getenv("VOICE_TOOLS_OPENAI_KEY")
    if openai_key:
        return openai_key, OPENAI_BASE_URL, "openai"

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return groq_key, GROQ_BASE_URL, "groq"

    return None, None, "none"

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}

# Maximum file size (25MB - OpenAI limit)
MAX_FILE_SIZE = 25 * 1024 * 1024


def transcribe_audio(file_path: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe an audio file using an OpenAI-compatible Whisper API.

    Automatically selects the provider based on available API keys:
    VOICE_TOOLS_OPENAI_KEY (OpenAI) > GROQ_API_KEY (Groq).

    Args:
        file_path: Absolute path to the audio file to transcribe.
        model:     Whisper model to use. Defaults per provider if not specified.

    Returns:
        dict with keys:
          - "success" (bool): Whether transcription succeeded
          - "transcript" (str): The transcribed text (empty on failure)
          - "error" (str, optional): Error message if success is False
          - "provider" (str, optional): Which provider was used
    """
    api_key, base_url, provider = _resolve_stt_provider()
    if not api_key:
        return {
            "success": False,
            "transcript": "",
            "error": "No STT API key set. Set VOICE_TOOLS_OPENAI_KEY or GROQ_API_KEY.",
        }

    audio_path = Path(file_path)
    
    # Validate file exists
    if not audio_path.exists():
        return {
            "success": False,
            "transcript": "",
            "error": f"Audio file not found: {file_path}",
        }
    
    if not audio_path.is_file():
        return {
            "success": False,
            "transcript": "",
            "error": f"Path is not a file: {file_path}",
        }
    
    # Validate file extension
    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        return {
            "success": False,
            "transcript": "",
            "error": f"Unsupported file format: {audio_path.suffix}. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}",
        }
    
    # Validate file size
    try:
        file_size = audio_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "success": False,
                "transcript": "",
                "error": f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE / (1024*1024)}MB)",
            }
    except OSError as e:
        logger.error("Failed to get file size for %s: %s", file_path, e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Failed to access file: {e}",
        }

    # Use provided model, or fall back to provider default.
    # If the caller passed an OpenAI-only model but we resolved to Groq, override it.
    OPENAI_MODELS = {"whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"}
    GROQ_MODELS = {"whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"}

    if model is None:
        model = DEFAULT_GROQ_STT_MODEL if provider == "groq" else DEFAULT_STT_MODEL
    elif provider == "groq" and model in OPENAI_MODELS:
        logger.info("Model %s not available on Groq, using %s", model, DEFAULT_GROQ_STT_MODEL)
        model = DEFAULT_GROQ_STT_MODEL
    elif provider == "openai" and model in GROQ_MODELS:
        logger.info("Model %s not available on OpenAI, using %s", model, DEFAULT_STT_MODEL)
        model = DEFAULT_STT_MODEL

    try:
        from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

        client = OpenAI(api_key=api_key, base_url=base_url)

        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text",
            )

        # The response is a plain string when response_format="text"
        transcript_text = str(transcription).strip()

        logger.info("Transcribed %s (%d chars, provider=%s)", audio_path.name, len(transcript_text), provider)

        return {
            "success": True,
            "transcript": transcript_text,
            "provider": provider,
        }

    except PermissionError:
        logger.error("Permission denied accessing file: %s", file_path, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Permission denied: {file_path}",
        }
    except APIConnectionError as e:
        logger.error("API connection error during transcription: %s", e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Connection error: {e}",
        }
    except APITimeoutError as e:
        logger.error("API timeout during transcription: %s", e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Request timeout: {e}",
        }
    except APIError as e:
        logger.error("OpenAI API error during transcription: %s", e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"API error: {e}",
        }
    except Exception as e:
        logger.error("Unexpected error during transcription: %s", e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Transcription failed: {e}",
        }
