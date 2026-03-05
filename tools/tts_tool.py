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
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, Any, Optional

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

try:
    import sounddevice as sd
    _HAS_AUDIO = True
except (ImportError, OSError):
    sd = None  # type: ignore[assignment]
    _HAS_AUDIO = False


# ===========================================================================
# Defaults
# ===========================================================================
DEFAULT_PROVIDER = "edge"
DEFAULT_EDGE_VOICE = "en-US-AriaNeural"
DEFAULT_ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_STREAMING_MODEL_ID = "eleven_flash_v2_5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_VOICE = "alloy"
DEFAULT_OUTPUT_DIR = str(Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "audio_cache")
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
    except ImportError:
        logger.debug("hermes_cli.config not available, using default TTS config")
        return {}
    except Exception as e:
        logger.warning("Failed to load TTS config: %s", e, exc_info=True)
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
        result = subprocess.run(
            ["ffmpeg", "-i", mp3_path, "-acodec", "libopus",
             "-ac", "1", "-b:a", "64k", "-vbr", "off", ogg_path, "-y"],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg conversion failed with return code %d: %s", 
                          result.returncode, result.stderr.decode('utf-8', errors='ignore')[:200])
            return None
        if os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 0:
            return ogg_path
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg OGG conversion timed out after 30s")
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH")
    except Exception as e:
        logger.warning("ffmpeg OGG conversion failed: %s", e, exc_info=True)
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
    api_key = os.getenv("VOICE_TOOLS_OPENAI_KEY", "")
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

    except ValueError as e:
        # Configuration errors (missing API keys, etc.)
        error_msg = f"TTS configuration error ({provider}): {e}"
        logger.error("%s", error_msg)
        return json.dumps({"success": False, "error": error_msg}, ensure_ascii=False)
    except FileNotFoundError as e:
        # Missing dependencies or files
        error_msg = f"TTS dependency missing ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
        return json.dumps({"success": False, "error": error_msg}, ensure_ascii=False)
    except Exception as e:
        # Unexpected errors
        error_msg = f"TTS generation failed ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
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
    if _HAS_OPENAI and os.getenv("VOICE_TOOLS_OPENAI_KEY"):
        return True
    return False


# ===========================================================================
# Streaming TTS: sentence-by-sentence pipeline for ElevenLabs
# ===========================================================================
# Sentence boundary pattern: punctuation followed by space or newline
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])(?:\s|\n)|(?:\n\n)')

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


def _strip_markdown_for_tts(text: str) -> str:
    """Remove markdown formatting that shouldn't be spoken aloud."""
    text = _MD_CODE_BLOCK.sub(' ', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_URL.sub('', text)
    text = _MD_BOLD.sub(r'\1', text)
    text = _MD_ITALIC.sub(r'\1', text)
    text = _MD_INLINE_CODE.sub(r'\1', text)
    text = _MD_HEADER.sub('', text)
    text = _MD_LIST_ITEM.sub('', text)
    text = _MD_HR.sub('', text)
    text = _MD_EXCESS_NL.sub('\n\n', text)
    return text.strip()


def stream_tts_to_speaker(
    text_queue: queue.Queue,
    stop_event: threading.Event,
    tts_done_event: threading.Event,
    display_callback: Optional[Callable[[str], None]] = None,
):
    """Consume text deltas from *text_queue*, buffer them into sentences,
    and stream each sentence through ElevenLabs TTS to the speaker in
    real-time.

    Protocol:
        * The producer puts ``str`` deltas onto *text_queue*.
        * A ``None`` sentinel signals end-of-text (flush remaining buffer).
        * *stop_event* can be set to abort early (e.g. user interrupt).
        * *tts_done_event* is **set** in the ``finally`` block so callers
          waiting on it (continuous voice mode) know playback is finished.
    """
    tts_done_event.clear()

    try:
        # --- TTS client setup (optional -- display_callback works without it) ---
        client = None
        output_stream = None
        voice_id = DEFAULT_ELEVENLABS_VOICE_ID
        model_id = DEFAULT_ELEVENLABS_STREAMING_MODEL_ID

        tts_config = _load_tts_config()
        el_config = tts_config.get("elevenlabs", {})
        voice_id = el_config.get("voice_id", voice_id)
        model_id = el_config.get("streaming_model_id",
                                 el_config.get("model_id", model_id))

        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY not set; streaming TTS audio disabled")
        elif _HAS_ELEVENLABS:
            client = ElevenLabs(api_key=api_key)

            # Open a single sounddevice output stream for the lifetime of
            # this function.  ElevenLabs pcm_24000 produces signed 16-bit
            # little-endian mono PCM at 24 kHz.
            use_sd = _HAS_AUDIO and sd is not None
            if use_sd:
                try:
                    import numpy as _np
                    output_stream = sd.OutputStream(
                        samplerate=24000, channels=1, dtype="int16",
                    )
                    output_stream.start()
                except Exception as exc:
                    logger.warning("sounddevice OutputStream failed: %s", exc)
                    output_stream = None

        sentence_buf = ""
        in_think = False  # track <think>...</think> blocks
        min_sentence_len = 20
        long_flush_len = 100
        queue_timeout = 0.5

        def _speak_sentence(sentence: str):
            """Display sentence and optionally generate + play audio."""
            if stop_event.is_set():
                return
            cleaned = _strip_markdown_for_tts(sentence).strip()
            if not cleaned:
                return
            # Display raw sentence on screen before TTS processing
            if display_callback is not None:
                display_callback(sentence)
            # Skip audio generation if no TTS client available
            if client is None:
                return
            # Truncate very long sentences
            if len(cleaned) > MAX_TEXT_LENGTH:
                cleaned = cleaned[:MAX_TEXT_LENGTH]
            try:
                audio_iter = client.text_to_speech.convert(
                    text=cleaned,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format="pcm_24000",
                )
                if output_stream is not None:
                    for chunk in audio_iter:
                        if stop_event.is_set():
                            break
                        import numpy as _np
                        audio_array = _np.frombuffer(chunk, dtype=_np.int16)
                        output_stream.write(audio_array.reshape(-1, 1))
                else:
                    # Fallback: write chunks to temp file and play via system player
                    _play_via_tempfile(audio_iter, stop_event)
            except Exception as exc:
                logger.warning("Streaming TTS sentence failed: %s", exc)

        def _play_via_tempfile(audio_iter, stop_evt):
            """Write PCM chunks to a temp WAV file and play it."""
            try:
                import wave
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(24000)
                    for chunk in audio_iter:
                        if stop_evt.is_set():
                            break
                        wf.writeframes(chunk)
                from tools.voice_mode import play_audio_file
                play_audio_file(tmp_path)
                os.unlink(tmp_path)
            except Exception as exc:
                logger.warning("Temp-file TTS fallback failed: %s", exc)

        while not stop_event.is_set():
            # Read next delta from queue
            try:
                delta = text_queue.get(timeout=queue_timeout)
            except queue.Empty:
                # Timeout: if we have accumulated a long buffer, flush it
                if len(sentence_buf) > long_flush_len:
                    _speak_sentence(sentence_buf)
                    sentence_buf = ""
                continue

            if delta is None:
                # End-of-text sentinel: flush remaining buffer
                if sentence_buf.strip():
                    _speak_sentence(sentence_buf)
                break

            # --- Think block filtering ---
            # Process delta character by character for think tags
            i = 0
            filtered_delta = []
            while i < len(delta):
                # Check for opening <think tag
                if delta[i:].startswith("<think"):
                    in_think = True
                    # Skip past the tag
                    end = delta.find(">", i)
                    if end != -1:
                        i = end + 1
                    else:
                        i = len(delta)
                    continue
                # Check for closing </think> tag
                if delta[i:].startswith("</think>"):
                    in_think = False
                    i += len("</think>")
                    continue
                if not in_think:
                    filtered_delta.append(delta[i])
                i += 1

            text = "".join(filtered_delta)
            if not text:
                continue

            sentence_buf += text

            # Check for sentence boundaries
            while True:
                m = _SENTENCE_BOUNDARY_RE.search(sentence_buf)
                if m is None:
                    break
                end_pos = m.end()
                sentence = sentence_buf[:end_pos]
                sentence_buf = sentence_buf[end_pos:]
                # Merge short fragments into the next sentence
                if len(sentence.strip()) < min_sentence_len:
                    sentence_buf = sentence + sentence_buf
                    break
                _speak_sentence(sentence)

        # Drain any remaining items from the queue
        while True:
            try:
                text_queue.get_nowait()
            except queue.Empty:
                break

        # Close the audio output stream
        if output_stream is not None:
            try:
                output_stream.stop()
                output_stream.close()
            except Exception:
                pass

    except Exception as exc:
        logger.warning("Streaming TTS pipeline error: %s", exc)
    finally:
        tts_done_event.set()


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
    print(f"    API Key:  {'✅ set' if os.getenv('VOICE_TOOLS_OPENAI_KEY') else '❌ not set (VOICE_TOOLS_OPENAI_KEY)'}")
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
                "description": "Optional custom file path to save the audio. Defaults to ~/.hermes/audio_cache/<timestamp>.mp3"
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
