#!/usr/bin/env python3
"""
Text-to-Speech Tool Module

Supports three TTS providers:
- Edge TTS (default, free, no API key): Microsoft Edge neural voices
- ElevenLabs (premium): High-quality voices, needs ELEVENLABS_API_KEY
- OpenAI TTS: Good quality, needs OPENAI_API_KEY

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
import datetime
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports -- providers degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import edge_tts
    _HAS_EDGE_TTS = True
except ImportError:
    _HAS_EDGE_TTS = False

try:
    from elevenlabs.client import ElevenLabs
    _HAS_ELEVENLABS = True
except ImportError:
    _HAS_ELEVENLABS = False

# openai is a core dependency, but guard anyway
try:
    from openai import OpenAI as OpenAIClient
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ===========================================================================
# Defaults
# ===========================================================================
DEFAULT_PROVIDER = "edge"
DEFAULT_EDGE_VOICE = "en-US-AriaNeural"
DEFAULT_ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_VOICE = "alloy"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/voice-memos")
MAX_TEXT_LENGTH = 4000


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
        return config.get("tts", {})
    except Exception:
        return {}


def _get_provider(tts_config: Dict[str, Any]) -> str:
    """Get the configured TTS provider name."""
    return tts_config.get("provider", DEFAULT_PROVIDER).lower().strip()


# ===========================================================================
# ffmpeg Opus conversion (Edge TTS MP3 -> OGG Opus for Telegram)
# ===========================================================================
def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _convert_to_opus(mp3_path: str) -> Optional[str]:
    """
    Convert an MP3 file to OGG Opus format for Telegram voice bubbles.

    Args:
        mp3_path: Path to the input MP3 file.

    Returns:
        Path to the .ogg file, or None if conversion fails.
    """
    if not _has_ffmpeg():
        return None

    ogg_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
    try:
        subprocess.run(
            ["ffmpeg", "-i", mp3_path, "-acodec", "libopus",
             "-ac", "1", "-b:a", "64k", "-vbr", "off", ogg_path, "-y"],
            capture_output=True, timeout=30,
        )
        if os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 0:
            return ogg_path
    except Exception as e:
        logger.warning("ffmpeg OGG conversion failed: %s", e)
    return None


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
    edge_config = tts_config.get("edge", {})
    voice = edge_config.get("voice", DEFAULT_EDGE_VOICE)

    communicate = edge_tts.Communicate(text, voice)
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
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set. Get one at https://elevenlabs.io/")

    el_config = tts_config.get("elevenlabs", {})
    voice_id = el_config.get("voice_id", DEFAULT_ELEVENLABS_VOICE_ID)
    model_id = el_config.get("model_id", DEFAULT_ELEVENLABS_MODEL_ID)

    # Determine output format based on file extension
    if output_path.endswith(".ogg"):
        output_format = "opus_48000_64"
    else:
        output_format = "mp3_44100_128"

    client = ElevenLabs(api_key=api_key)
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


# ===========================================================================
# Provider: OpenAI TTS
# ===========================================================================
def _generate_openai_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """
    Generate audio using OpenAI TTS.

    Args:
        text: Text to convert.
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    api_key = os.getenv("VOICE_TOOLS_OPENAI_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("VOICE_TOOLS_OPENAI_KEY not set. Get one at https://platform.openai.com/api-keys")

    oai_config = tts_config.get("openai", {})
    model = oai_config.get("model", DEFAULT_OPENAI_MODEL)
    voice = oai_config.get("voice", DEFAULT_OPENAI_VOICE)

    # Determine response format from extension
    if output_path.endswith(".ogg"):
        response_format = "opus"
    else:
        response_format = "mp3"

    client = OpenAIClient(api_key=api_key, base_url="https://api.openai.com/v1")
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format,
    )

    response.stream_to_file(output_path)
    return output_path


# ===========================================================================
# Main tool function
# ===========================================================================
def text_to_speech_tool(
    text: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Convert text to speech audio.

    Reads provider/voice config from ~/.hermes/config.yaml (tts: section).
    The model sends text; the user configures voice and provider.

    On messaging platforms, the returned MEDIA:<path> tag is intercepted
    by the send pipeline and delivered as a native voice message.
    In CLI mode, the file is saved to ~/voice-memos/.

    Args:
        text: The text to convert to speech.
        output_path: Optional custom save path. Defaults to ~/voice-memos/<timestamp>.mp3

    Returns:
        str: JSON result with success, file_path, and optionally MEDIA tag.
    """
    if not text or not text.strip():
        return json.dumps({"success": False, "error": "Text is required"}, ensure_ascii=False)

    # Truncate very long text with a warning
    if len(text) > MAX_TEXT_LENGTH:
        logger.warning("TTS text too long (%d chars), truncating to %d", len(text), MAX_TEXT_LENGTH)
        text = text[:MAX_TEXT_LENGTH]

    tts_config = _load_tts_config()
    provider = _get_provider(tts_config)

    # Detect platform from gateway env var to choose the best output format.
    # Telegram voice bubbles require Opus (.ogg); OpenAI and ElevenLabs can
    # produce Opus natively (no ffmpeg needed).  Edge TTS always outputs MP3
    # and needs ffmpeg for conversion.
    platform = os.getenv("HERMES_SESSION_PLATFORM", "").lower()
    want_opus = (platform == "telegram")

    # Determine output path
    if output_path:
        file_path = Path(output_path).expanduser()
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(DEFAULT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Use .ogg for Telegram with providers that support native Opus output,
        # otherwise fall back to .mp3 (Edge TTS will attempt ffmpeg conversion later).
        if want_opus and provider in ("openai", "elevenlabs"):
            file_path = out_dir / f"tts_{timestamp}.ogg"
        else:
            file_path = out_dir / f"tts_{timestamp}.mp3"

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_str = str(file_path)

    try:
        # Generate audio with the configured provider
        if provider == "elevenlabs":
            if not _HAS_ELEVENLABS:
                return json.dumps({
                    "success": False,
                    "error": "ElevenLabs provider selected but 'elevenlabs' package not installed. Run: pip install elevenlabs"
                }, ensure_ascii=False)
            logger.info("Generating speech with ElevenLabs...")
            _generate_elevenlabs(text, file_str, tts_config)

        elif provider == "openai":
            if not _HAS_OPENAI:
                return json.dumps({
                    "success": False,
                    "error": "OpenAI provider selected but 'openai' package not installed."
                }, ensure_ascii=False)
            logger.info("Generating speech with OpenAI TTS...")
            _generate_openai_tts(text, file_str, tts_config)

        else:
            # Default: Edge TTS (free)
            if not _HAS_EDGE_TTS:
                return json.dumps({
                    "success": False,
                    "error": "Edge TTS not available. Run: pip install edge-tts"
                }, ensure_ascii=False)
            logger.info("Generating speech with Edge TTS...")
            # Edge TTS is async, run it
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(
                        lambda: asyncio.run(_generate_edge_tts(text, file_str, tts_config))
                    ).result(timeout=60)
            except RuntimeError:
                asyncio.run(_generate_edge_tts(text, file_str, tts_config))

        # Check the file was actually created
        if not os.path.exists(file_str) or os.path.getsize(file_str) == 0:
            return json.dumps({
                "success": False,
                "error": f"TTS generation produced no output (provider: {provider})"
            }, ensure_ascii=False)

        # Try Opus conversion for Telegram compatibility (Edge TTS only outputs MP3)
        voice_compatible = False
        if provider == "edge" and file_str.endswith(".mp3"):
            opus_path = _convert_to_opus(file_str)
            if opus_path:
                file_str = opus_path
                voice_compatible = True
        elif provider in ("elevenlabs", "openai"):
            # These providers can output Opus natively if the path ends in .ogg
            voice_compatible = file_str.endswith(".ogg")

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

    except Exception as e:
        error_msg = f"TTS generation failed ({provider}): {e}"
        logger.error("%s", error_msg)
        return json.dumps({"success": False, "error": error_msg}, ensure_ascii=False)


# ===========================================================================
# Requirements check
# ===========================================================================
def check_tts_requirements() -> bool:
    """
    Check if at least one TTS provider is available.

    Edge TTS needs no API key and is the default, so if the package
    is installed, TTS is available.

    Returns:
        bool: True if at least one provider can work.
    """
    if _HAS_EDGE_TTS:
        return True
    if _HAS_ELEVENLABS and os.getenv("ELEVENLABS_API_KEY"):
        return True
    if _HAS_OPENAI and (os.getenv("VOICE_TOOLS_OPENAI_KEY") or os.getenv("OPENAI_API_KEY")):
        return True
    return False


# ===========================================================================
# Main -- quick diagnostics
# ===========================================================================
if __name__ == "__main__":
    print("🔊 Text-to-Speech Tool Module")
    print("=" * 50)

    print(f"\nProvider availability:")
    print(f"  Edge TTS:   {'✅ installed' if _HAS_EDGE_TTS else '❌ not installed (pip install edge-tts)'}")
    print(f"  ElevenLabs: {'✅ installed' if _HAS_ELEVENLABS else '❌ not installed (pip install elevenlabs)'}")
    print(f"    API Key:  {'✅ set' if os.getenv('ELEVENLABS_API_KEY') else '❌ not set'}")
    print(f"  OpenAI:     {'✅ installed' if _HAS_OPENAI else '❌ not installed'}")
    print(f"    API Key:  {'✅ set' if (os.getenv('VOICE_TOOLS_OPENAI_KEY') or os.getenv('OPENAI_API_KEY')) else '❌ not set'}")
    print(f"  ffmpeg:     {'✅ found' if _has_ffmpeg() else '❌ not found (needed for Telegram Opus)'}")
    print(f"\n  Output dir: {DEFAULT_OUTPUT_DIR}")

    config = _load_tts_config()
    provider = _get_provider(config)
    print(f"  Configured provider: {provider}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

TTS_SCHEMA = {
    "name": "text_to_speech",
    "description": "Convert text to speech audio. Returns a MEDIA: path that the platform delivers as a voice message. On Telegram it plays as a voice bubble, on Discord/WhatsApp as an audio attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider are user-configured, not model-selected.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to convert to speech. Keep under 4000 characters."
            },
            "output_path": {
                "type": "string",
                "description": "Optional custom file path to save the audio. Defaults to ~/voice-memos/<timestamp>.mp3"
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
        output_path=args.get("output_path")),
    check_fn=check_tts_requirements,
)
