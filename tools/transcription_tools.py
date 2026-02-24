#!/usr/bin/env python3
"""
Transcription Tools Module

Provides speech-to-text transcription using OpenAI's Whisper API.
Used by the messaging gateway to automatically transcribe voice messages
sent by users on Telegram, Discord, WhatsApp, and Slack.

Supported models:
  - whisper-1        (cheapest, good quality)
  - gpt-4o-mini-transcribe  (better quality, higher cost)
  - gpt-4o-transcribe       (best quality, highest cost)

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
from typing import Optional

logger = logging.getLogger(__name__)


# Default STT model -- cheapest and widely available
DEFAULT_STT_MODEL = "whisper-1"


def transcribe_audio(file_path: str, model: Optional[str] = None) -> dict:
    """
    Transcribe an audio file using OpenAI's Whisper API.

    This function calls the OpenAI Audio Transcriptions endpoint directly
    (not via OpenRouter, since Whisper isn't available there).

    Args:
        file_path: Absolute path to the audio file to transcribe.
        model:     Whisper model to use. Defaults to config or "whisper-1".

    Returns:
        dict with keys:
          - "success" (bool): Whether transcription succeeded
          - "transcript" (str): The transcribed text (empty on failure)
          - "error" (str, optional): Error message if success is False
    """
    # Use VOICE_TOOLS_OPENAI_KEY to avoid interference with the OpenAI SDK's
    # auto-detection of OPENAI_API_KEY (which would break OpenRouter calls).
    # Falls back to OPENAI_API_KEY for backward compatibility.
    api_key = os.getenv("VOICE_TOOLS_OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "success": False,
            "transcript": "",
            "error": "VOICE_TOOLS_OPENAI_KEY not set",
        }

    audio_path = Path(file_path)
    if not audio_path.is_file():
        return {
            "success": False,
            "transcript": "",
            "error": f"Audio file not found: {file_path}",
        }

    # Use provided model, or fall back to default
    if model is None:
        model = DEFAULT_STT_MODEL

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text",
            )

        # The response is a plain string when response_format="text"
        transcript_text = str(transcription).strip()

        logger.info("Transcribed %s (%d chars)", audio_path.name, len(transcript_text))

        return {
            "success": True,
            "transcript": transcript_text,
        }

    except Exception as e:
        logger.error("Transcription error: %s", e)
        return {
            "success": False,
            "transcript": "",
            "error": str(e),
        }
