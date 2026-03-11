#!/usr/bin/env python3
"""
Transcription Tools Module

Provides speech-to-text transcription using OpenAI's Whisper API, with an
optional local command fallback for offline/local transcription workflows.
Used by the messaging gateway to automatically transcribe voice messages
sent by users on Telegram, Discord, WhatsApp, and Slack.

Supported OpenAI input formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg

Environment variables:
  - VOICE_TOOLS_OPENAI_KEY: Preferred OpenAI key for STT/TTS
  - OPENAI_API_KEY: Fallback OpenAI key for STT when voice-specific key is unset
  - HERMES_LOCAL_STT_COMMAND: Optional shell command template for local fallback.
    Supported placeholders:
      * {input_path}
      * {output_dir}
      * {language}
  - HERMES_LOCAL_STT_LANGUAGE: Optional default language for local fallback

Usage:
    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio("/path/to/audio.ogg")
    if result["success"]:
        print(result["transcript"])
"""

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# Default STT model -- cheapest and widely available
DEFAULT_STT_MODEL = "whisper-1"
DEFAULT_LOCAL_STT_LANGUAGE = "en"
LOCAL_STT_COMMAND_ENV = "HERMES_LOCAL_STT_COMMAND"
LOCAL_STT_LANGUAGE_ENV = "HERMES_LOCAL_STT_LANGUAGE"

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}

# Formats typically safest to feed directly to local CLI pipelines
LOCAL_NATIVE_AUDIO_FORMATS = {".wav", ".aiff", ".aif"}

# Maximum file size (25MB - OpenAI limit)
MAX_FILE_SIZE = 25 * 1024 * 1024


def _resolve_openai_api_key() -> str:
    """Prefer the voice-tools key, but fall back to the normal OpenAI key."""
    return os.getenv("VOICE_TOOLS_OPENAI_KEY", "") or os.getenv("OPENAI_API_KEY", "")


def _has_local_stt_command() -> bool:
    return bool(os.getenv(LOCAL_STT_COMMAND_ENV, "").strip())


def _find_ffmpeg_binary() -> Optional[str]:
    """Find ffmpeg, preferring the common Homebrew location on macOS."""
    homebrew_ffmpeg = "/opt/homebrew/bin/ffmpeg"
    if os.path.exists(homebrew_ffmpeg):
        return homebrew_ffmpeg
    return shutil.which("ffmpeg")


def _prepare_local_audio(file_path: str, work_dir: str) -> tuple[Optional[str], Optional[str]]:
    """
    Normalize audio for local STT when needed.

    Returns:
        (prepared_path, error)
    """
    audio_path = Path(file_path)
    if audio_path.suffix.lower() in LOCAL_NATIVE_AUDIO_FORMATS:
        return file_path, None

    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return None, "Local STT fallback requires ffmpeg for non-WAV inputs, but ffmpeg was not found"

    converted_path = os.path.join(work_dir, f"{audio_path.stem}.wav")
    cmd = [ffmpeg, "-y", "-i", file_path, converted_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return converted_path, None
    except subprocess.CalledProcessError as e:
        details = e.stderr.strip() or e.stdout.strip() or str(e)
        logger.error("ffmpeg conversion failed for %s: %s", file_path, details)
        return None, f"Failed to convert audio for local STT: {details}"


def _transcribe_audio_openai(file_path: str, model: str, api_key: str) -> Dict[str, Any]:
    """Transcribe an audio file using OpenAI's Whisper API."""
    try:
        from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text",
            )

        transcript_text = str(transcription).strip()
        logger.info("Transcribed %s via OpenAI (%d chars)", Path(file_path).name, len(transcript_text))
        return {"success": True, "transcript": transcript_text}

    except PermissionError:
        logger.error("Permission denied accessing file: %s", file_path, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except APIConnectionError as e:
        logger.error("API connection error during transcription: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Connection error: {e}"}
    except APITimeoutError as e:
        logger.error("API timeout during transcription: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Request timeout: {e}"}
    except APIError as e:
        logger.error("OpenAI API error during transcription: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"API error: {e}"}
    except Exception as e:
        logger.error("Unexpected error during OpenAI transcription: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Transcription failed: {e}"}


def _transcribe_audio_local(file_path: str) -> Dict[str, Any]:
    """Run the configured local STT command template and read back a .txt transcript."""
    command_template = os.getenv(LOCAL_STT_COMMAND_ENV, "").strip()
    if not command_template:
        return {
            "success": False,
            "transcript": "",
            "error": f"{LOCAL_STT_COMMAND_ENV} not configured",
        }

    language = os.getenv(LOCAL_STT_LANGUAGE_ENV, DEFAULT_LOCAL_STT_LANGUAGE)

    try:
        with tempfile.TemporaryDirectory(prefix="hermes-local-stt-") as output_dir:
            prepared_input, prep_error = _prepare_local_audio(file_path, output_dir)
            if prep_error:
                return {"success": False, "transcript": "", "error": prep_error}

            command = command_template.format(
                input_path=shlex.quote(prepared_input),
                output_dir=shlex.quote(output_dir),
                language=shlex.quote(language),
            )
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

            txt_files = sorted(Path(output_dir).glob("*.txt"))
            if not txt_files:
                return {
                    "success": False,
                    "transcript": "",
                    "error": "Local STT command completed but did not produce a .txt transcript",
                }

            transcript_text = txt_files[0].read_text(encoding="utf-8").strip()
            logger.info("Transcribed %s via local STT (%d chars)", Path(file_path).name, len(transcript_text))
            return {"success": True, "transcript": transcript_text}

    except KeyError as e:
        return {
            "success": False,
            "transcript": "",
            "error": f"Invalid {LOCAL_STT_COMMAND_ENV} template, missing placeholder: {e}",
        }
    except subprocess.CalledProcessError as e:
        details = e.stderr.strip() or e.stdout.strip() or str(e)
        logger.error("Local STT command failed for %s: %s", file_path, details)
        return {"success": False, "transcript": "", "error": f"Local STT failed: {details}"}
    except Exception as e:
        logger.error("Unexpected error during local transcription: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Local transcription failed: {e}"}


def transcribe_audio(file_path: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe an audio file using OpenAI's Whisper API, with optional local fallback.

    Args:
        file_path: Absolute path to the audio file to transcribe.
        model:     Whisper model to use. Defaults to config or "whisper-1".

    Returns:
        dict with keys:
          - "success" (bool): Whether transcription succeeded
          - "transcript" (str): The transcribed text (empty on failure)
          - "error" (str, optional): Error message if success is False
    """
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

    if model is None:
        model = DEFAULT_STT_MODEL

    api_key = _resolve_openai_api_key()
    local_stt_configured = _has_local_stt_command()

    if api_key:
        openai_result = _transcribe_audio_openai(file_path, model, api_key)
        if openai_result["success"]:
            return openai_result
        if local_stt_configured:
            local_result = _transcribe_audio_local(file_path)
            if local_result["success"]:
                return local_result
            return {
                "success": False,
                "transcript": "",
                "error": f"{openai_result['error']}; local fallback failed: {local_result['error']}",
            }
        return openai_result

    if local_stt_configured:
        return _transcribe_audio_local(file_path)

    return {
        "success": False,
        "transcript": "",
        "error": (
            "Neither VOICE_TOOLS_OPENAI_KEY nor OPENAI_API_KEY is set, and "
            f"{LOCAL_STT_COMMAND_ENV} is not configured"
        ),
    }
