#!/usr/bin/env python3
"""
Podcast Generation Tool

Generates NotebookLM-style two-host conversational podcasts from source content.
Pipeline: Ingest → Script (LLM) → Synthesize (TTS) → Assemble (pydub) → Save.

Supports multiple TTS backends with fallback chain:
  VoxCPM (local, studio-grade) → NeuTTS (local) → Edge TTS (free, cloud)

Configuration in ~/.hermes/config.yaml under the 'podcast:' key.

Usage:
    from tools.podcast_tool import podcast_generate_tool
    result = podcast_generate_tool(source="~/notes/doc.md")
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================
MAX_SOURCE_CHARS = 100_000
DEFAULT_STYLE = "deep_dive"
DEFAULT_FORMAT = "mp3"
SILENCE_BETWEEN_SPEAKERS_MS = 350
SILENCE_WITHIN_SPEAKER_MS = 150
CROSSFADE_MS = 50


def _get_podcasts_dir() -> Path:
    """Return the base directory for podcast output."""
    from hermes_constants import get_hermes_dir
    return get_hermes_dir("podcasts", "podcasts")


def _get_voice_cache_dir() -> Path:
    """Return the directory for cached voice embeddings."""
    from hermes_constants import get_hermes_dir
    return get_hermes_dir("cache/podcast_voices", "podcast_voice_cache")


# ===========================================================================
# Config
# ===========================================================================
_DEFAULT_VOICES = {
    "Alex": {
        "description": "A warm, authoritative male voice, mid-30s, American English, like a seasoned podcast host",
        "edge_voice": "en-US-GuyNeural",
        "speed": 1.0,
    },
    "Sam": {
        "description": "An energetic, curious female voice, late-20s, American English, like a tech journalist",
        "edge_voice": "en-US-JennyNeural",
        "speed": 1.05,
    },
}


def _load_podcast_config() -> Dict[str, Any]:
    """Load podcast configuration from ~/.hermes/config.yaml."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config.get("podcast", {})
    except ImportError:
        logger.debug("hermes_cli.config not available, using default podcast config")
        return {}
    except Exception as e:
        logger.warning("Failed to load podcast config: %s", e, exc_info=True)
        return {}


def _get_voice_config(podcast_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Get voice configuration, merging user overrides with defaults."""
    import copy
    voices = copy.deepcopy(_DEFAULT_VOICES)
    user_voices = podcast_config.get("voices", {})
    for name, overrides in user_voices.items():
        canonical = name.capitalize()
        if canonical in voices:
            voices[canonical].update(overrides)
        else:
            voices[canonical] = overrides
    return voices


def _get_tts_provider(podcast_config: Dict[str, Any]) -> str:
    """Get the configured TTS provider for podcasts."""
    return (podcast_config.get("tts_provider") or "edge").lower().strip()


# ===========================================================================
# Step 1: Source Ingestion
# ===========================================================================
def _ingest_source(source: str) -> str:
    """Read source content from file, directory, or inline text.

    Returns the raw text content, truncated to MAX_SOURCE_CHARS.
    """
    source_path = Path(source).expanduser()

    # Single file
    if source_path.is_file():
        content = source_path.read_text(encoding="utf-8", errors="replace")
        logger.info("Ingested file: %s (%d chars)", source_path, len(content))
        return content[:MAX_SOURCE_CHARS]

    # Directory — glob for text files and concatenate
    if source_path.is_dir():
        parts = []
        extensions = {".md", ".txt", ".rst", ".html"}
        files = sorted(source_path.rglob("*"))
        for f in files:
            if f.suffix.lower() in extensions and f.is_file():
                header = f"## {f.relative_to(source_path)}\n\n"
                body = f.read_text(encoding="utf-8", errors="replace")
                parts.append(header + body)
        content = "\n\n---\n\n".join(parts)
        logger.info("Ingested directory: %s (%d files, %d chars)",
                     source_path, len(parts), len(content))
        return content[:MAX_SOURCE_CHARS]

    # Inline text (not a valid file or directory path)
    logger.info("Treating source as inline text (%d chars)", len(source))
    return source[:MAX_SOURCE_CHARS]


# ===========================================================================
# Step 2: Script Generation (LLM)
# ===========================================================================

# Style-specific prompt fragments
_STYLE_PROMPTS = {
    "deep_dive": (
        "Create a thorough, exploratory conversation (20-30 turns, ~8-12 minutes). "
        "Alex explains concepts in depth. Sam asks probing follow-up questions, "
        "offers analogies, and pushes for practical implications."
    ),
    "quick_brief": (
        "Create a concise summary conversation (8-12 turns, ~2-4 minutes). "
        "Alex highlights the top 3-5 key points. Sam reacts and asks one "
        "clarifying question per point. Keep it brisk and punchy."
    ),
    "debate": (
        "Create a structured debate (15-25 turns, ~6-10 minutes). "
        "Alex takes the position that the source material's conclusions are sound. "
        "Sam plays devil's advocate, challenging assumptions and raising counterpoints. "
        "Both hosts are respectful but intellectually rigorous."
    ),
}

_SCRIPT_SYSTEM_PROMPT = """You are a podcast scriptwriter. Generate a natural, engaging two-host podcast script from the provided source material.

The two hosts are:
- **Alex**: The explainer and narrator. Warm, authoritative, knowledgeable. Leads the conversation.
- **Sam**: The curious co-host. Energetic, asks great follow-up questions, provides reactions and analogies.

{style_instruction}

## Output Format

Return ONLY a valid JSON array with NO markdown formatting around it. Each element:
```
{{"speaker": "Alex"|"Sam", "text": "...", "emotion": "...", "pace": "..."}}
```

Allowed emotions: neutral, excited, curious, thoughtful, surprised, emphatic, amused, serious
Allowed paces: slow, normal, fast

## Conversation Guidelines

- Open with a hook that grabs attention — NOT "Welcome to the show"
- Include natural conversation patterns: interruptions ("Wait, hold on—"), backchannels ("Right, exactly"), reactions ("That's fascinating")
- Don't read lists or data verbatim — narrativize them ("So if you look at the numbers, what jumps out is...")
- Each turn should be 1-4 sentences. Avoid monologues.
- Close with a concrete takeaway or call-to-action
- For data-heavy content, Alex summarizes trends while Sam asks "what does that mean for..."
- If the source is long, focus on the 3-5 most interesting or important themes"""


def _generate_script(content: str, style: str) -> List[Dict[str, Any]]:
    """Generate a podcast script by calling the LLM.

    Uses OpenRouter/OpenAI API directly (same pattern as podcast_gen.py)
    to avoid recursive agent invocation.
    """
    import urllib.request
    import urllib.error
    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        load_hermes_dotenv()
    except ImportError:
        pass  # Env already loaded when running within agent context

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No LLM API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in ~/.hermes/.env"
        )

    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    url = "https://openrouter.ai/api/v1/chat/completions" if has_openrouter else "https://api.openai.com/v1/chat/completions"
    model = "anthropic/claude-sonnet-4" if has_openrouter else "gpt-4o"

    style_instruction = _STYLE_PROMPTS.get(style, _STYLE_PROMPTS["deep_dive"])
    system_prompt = _SCRIPT_SYSTEM_PROMPT.format(style_instruction=style_instruction)

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a podcast script from this source material:\n\n{content}"},
        ],
        "temperature": 0.8,
        "max_tokens": 8192,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    logger.info("Generating podcast script via LLM (style=%s)...", style)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM API error {e.code}: {body}") from e

    raw_content = result["choices"][0]["message"]["content"]
    script = _parse_script_json(raw_content)

    logger.info("Generated script with %d turns", len(script))
    return script


def _parse_script_json(raw: str) -> List[Dict[str, Any]]:
    """Parse script JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return _validate_script(parsed)
        if isinstance(parsed, dict) and "script" in parsed:
            return _validate_script(parsed["script"])
        if isinstance(parsed, dict) and "dialogue" in parsed:
            return _validate_script(parsed["dialogue"])
    except json.JSONDecodeError:
        pass

    # Strip markdown code block wrapper
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if code_block:
        try:
            parsed = json.loads(code_block.group(1))
            if isinstance(parsed, list):
                return _validate_script(parsed)
        except json.JSONDecodeError:
            pass

    # Last resort: find the first [ ... ] in the text
    bracket_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(0))
            if isinstance(parsed, list):
                return _validate_script(parsed)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse script JSON from LLM response (first 200 chars): {raw[:200]}")


def _validate_script(script: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize script entries."""
    valid_emotions = {"neutral", "excited", "curious", "thoughtful", "surprised",
                      "emphatic", "amused", "serious"}
    valid_paces = {"slow", "normal", "fast"}
    validated = []

    for entry in script:
        speaker = entry.get("speaker", "Alex")
        text = entry.get("text", "").strip()
        if not text:
            continue

        emotion = entry.get("emotion", "neutral").lower()
        if emotion not in valid_emotions:
            emotion = "neutral"

        pace = entry.get("pace", "normal").lower()
        if pace not in valid_paces:
            pace = "normal"

        validated.append({
            "speaker": speaker if speaker in ("Alex", "Sam") else "Alex",
            "text": text,
            "emotion": emotion,
            "pace": pace,
        })

    if not validated:
        raise ValueError("Script validation produced zero valid turns")

    return validated


# ===========================================================================
# Step 3: Voice Synthesis
# ===========================================================================

# VoxCPM emotion → inline style descriptor mapping
_EMOTION_TO_VOXCPM_STYLE = {
    "neutral": "",
    "excited": "excited and energetic tone",
    "curious": "curious and inquisitive tone",
    "thoughtful": "thoughtful and measured tone",
    "surprised": "surprised tone",
    "emphatic": "emphatic and passionate tone",
    "amused": "amused and light tone",
    "serious": "serious and focused tone",
}

_PACE_TO_VOXCPM_STYLE = {
    "slow": "speaking slowly",
    "normal": "",
    "fast": "speaking quickly",
}


def _check_voxcpm_available() -> bool:
    """Check if VoxCPM is importable."""
    try:
        import importlib.util
        return importlib.util.find_spec("voxcpm") is not None
    except Exception:
        return False


def _build_voxcpm_style_prefix(voice_description: str, emotion: str, pace: str) -> str:
    """Build a VoxCPM inline style prefix from voice description + emotion + pace.

    VoxCPM uses parenthesized text at the start of the synthesis input:
        "(warm male voice, excited tone, speaking quickly)Hello world"
    """
    parts = []
    if voice_description:
        parts.append(voice_description)

    emotion_style = _EMOTION_TO_VOXCPM_STYLE.get(emotion, "")
    if emotion_style:
        parts.append(emotion_style)

    pace_style = _PACE_TO_VOXCPM_STYLE.get(pace, "")
    if pace_style:
        parts.append(pace_style)

    if not parts:
        return ""
    return f"({', '.join(parts)})"


def _synthesize_voxcpm(
    text: str,
    output_path: str,
    voice_description: str = "",
    reference_wav: str = "",
    reference_text: str = "",
    emotion: str = "neutral",
    pace: str = "normal",
    podcast_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Synthesize a single line using VoxCPM via subprocess.

    Runs voxcpm_synth.py in a separate process to isolate the ~8GB model.
    """
    voxcpm_config = (podcast_config or {}).get("voxcpm", {})
    cfg_value = voxcpm_config.get("cfg_value", 2.0)
    inference_steps = voxcpm_config.get("inference_steps", 10)
    device = voxcpm_config.get("device", "auto")

    # Build style prefix and prepend to synthesis text
    style_prefix = _build_voxcpm_style_prefix(voice_description, emotion, pace)
    synth_text = f"{style_prefix}{text}" if style_prefix else text

    synth_script = str(Path(__file__).parent / "voxcpm_synth.py")

    # VoxCPM outputs WAV natively — use .wav for synthesis
    wav_path = output_path
    needs_conversion = False
    if not output_path.endswith(".wav"):
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"
        needs_conversion = True

    cmd = [
        sys.executable, synth_script,
        "--text", synth_text,
        "--out", wav_path,
        "--cfg-value", str(cfg_value),
        "--inference-steps", str(inference_steps),
        "--device", device,
    ]

    if voice_description and not reference_wav:
        cmd.extend(["--voice-description", voice_description])

    if reference_wav:
        cmd.extend(["--reference-wav", reference_wav])
        if reference_text:
            cmd.extend(["--reference-text", reference_text])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        error_lines = [l for l in stderr.splitlines()
                       if not l.startswith("INFO:") and not l.startswith("OK:")]
        raise RuntimeError(
            f"VoxCPM synthesis failed: {chr(10).join(error_lines) or 'unknown error'}"
        )

    # Convert WAV to target format if needed
    if needs_conversion and os.path.exists(wav_path):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            conv_cmd = [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path]
            subprocess.run(conv_cmd, check=True, timeout=30)
            os.remove(wav_path)
        else:
            os.rename(wav_path, output_path)

    return output_path


def _synthesize_edge_tts(
    text: str,
    voice_name: str,
    output_path: str,
    speed: float = 1.0,
) -> str:
    """Synthesize a single line using Edge TTS."""
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

    rate_str = f"{int((speed - 1) * 100):+d}%"
    communicate = edge_tts.Communicate(text, voice_name, rate=rate_str)

    # Run the async synthesis
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(
                lambda: asyncio.run(communicate.save(output_path))
            ).result(timeout=60)
    except RuntimeError:
        asyncio.run(communicate.save(output_path))

    return output_path


def _synthesize_voices(
    script: List[Dict[str, Any]],
    segments_dir: Path,
    podcast_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Synthesize all script lines, returning segment metadata.

    Each segment is saved as a numbered file in segments_dir.
    Returns a list of dicts with path, speaker, duration info.
    """
    provider = _get_tts_provider(podcast_config)
    voices = _get_voice_config(podcast_config)
    segments = []

    for i, line in enumerate(script):
        speaker = line["speaker"]
        text = line["text"]
        voice_cfg = voices.get(speaker, voices.get("Alex", {}))

        segment_name = f"{i:03d}_{speaker.lower()}.mp3"
        segment_path = segments_dir / segment_name

        speed = voice_cfg.get("speed", 1.0)

        # Map pace to speed multiplier
        pace = line.get("pace", "normal")
        pace_multiplier = {"slow": 0.9, "normal": 1.0, "fast": 1.1}.get(pace, 1.0)
        effective_speed = speed * pace_multiplier

        emotion = line.get("emotion", "neutral")
        actual_provider = provider

        if provider in ("voxcpm", "auto"):
            if _check_voxcpm_available():
                try:
                    voice_desc = voice_cfg.get("description", "")
                    ref_wav = voice_cfg.get("reference_wav", "")
                    ref_text = voice_cfg.get("reference_text", "")
                    # VoxCPM outputs WAV; segment name reflects that
                    segment_name = f"{i:03d}_{speaker.lower()}.wav"
                    segment_path = segments_dir / segment_name
                    _synthesize_voxcpm(
                        text=text,
                        output_path=str(segment_path),
                        voice_description=voice_desc,
                        reference_wav=ref_wav,
                        reference_text=ref_text,
                        emotion=emotion,
                        pace=pace,
                        podcast_config=podcast_config,
                    )
                    actual_provider = "voxcpm"
                except Exception as e:
                    logger.warning("VoxCPM synthesis failed for segment %d, "
                                   "falling back to Edge TTS: %s", i, e)
                    segment_name = f"{i:03d}_{speaker.lower()}.mp3"
                    segment_path = segments_dir / segment_name
                    edge_voice = voice_cfg.get("edge_voice", "en-US-GuyNeural")
                    _synthesize_edge_tts(text, edge_voice, str(segment_path), effective_speed)
                    actual_provider = "edge"
            else:
                if provider == "voxcpm":
                    logger.warning("VoxCPM not installed, falling back to Edge TTS. "
                                   "Install with: pip install voxcpm")
                edge_voice = voice_cfg.get("edge_voice", "en-US-GuyNeural")
                _synthesize_edge_tts(text, edge_voice, str(segment_path), effective_speed)
                actual_provider = "edge"
        elif provider == "edge":
            edge_voice = voice_cfg.get("edge_voice", "en-US-GuyNeural")
            _synthesize_edge_tts(text, edge_voice, str(segment_path), effective_speed)
        else:
            edge_voice = voice_cfg.get("edge_voice", "en-US-GuyNeural")
            logger.warning("Unknown provider '%s', falling back to Edge TTS", provider)
            _synthesize_edge_tts(text, edge_voice, str(segment_path), effective_speed)
            actual_provider = "edge"

        if not segment_path.exists() or segment_path.stat().st_size == 0:
            logger.warning("Segment %d failed to synthesize, skipping", i)
            continue

        segments.append({
            "index": i,
            "speaker": speaker,
            "text": text,
            "emotion": emotion,
            "pace": pace,
            "path": str(segment_path),
            "provider": actual_provider,
        })

        logger.info("Synthesized segment %d/%d (%s: %d chars)",
                     i + 1, len(script), speaker, len(text))

    return segments


# ===========================================================================
# Step 4: Audio Assembly
# ===========================================================================
def _assemble_audio(
    segments: List[Dict[str, Any]],
    output_path: Path,
    output_format: str = "mp3",
) -> float:
    """Concatenate audio segments with silences and crossfades.

    Returns the total duration in seconds.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError("pydub not installed. Run: pip install pydub")

    combined = AudioSegment.empty()
    prev_speaker = None

    for seg in segments:
        try:
            audio = AudioSegment.from_file(seg["path"])
        except Exception as e:
            logger.warning("Failed to load segment %s: %s", seg["path"], e)
            continue

        # Add silence between turns
        if prev_speaker is not None:
            same_speaker = (seg["speaker"] == prev_speaker)
            silence_ms = SILENCE_WITHIN_SPEAKER_MS if same_speaker else SILENCE_BETWEEN_SPEAKERS_MS
            combined += AudioSegment.silent(duration=silence_ms)

        # Apply crossfade if we have enough audio
        if len(combined) > CROSSFADE_MS and len(audio) > CROSSFADE_MS:
            combined = combined.append(audio, crossfade=CROSSFADE_MS)
        else:
            combined += audio

        prev_speaker = seg["speaker"]

    # Export
    format_map = {"mp3": "mp3", "m4a": "mp4", "wav": "wav"}
    export_format = format_map.get(output_format, "mp3")

    export_params = {}
    if export_format == "mp3":
        export_params["bitrate"] = "128k"
    elif export_format == "mp4":
        export_params["codec"] = "aac"
        export_params["bitrate"] = "96k"

    combined.export(str(output_path), format=export_format, **export_params)

    duration_seconds = len(combined) / 1000.0
    logger.info("Assembled podcast: %.1f seconds, %s format", duration_seconds, output_format)
    return duration_seconds


# ===========================================================================
# Step 5: Metadata & Transcript
# ===========================================================================
def _save_metadata(
    podcast_dir: Path,
    title: str,
    source: str,
    style: str,
    output_format: str,
    duration: float,
    script: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
) -> None:
    """Save transcript and metadata JSON files."""
    # Transcript with timing estimates
    transcript_entries = []
    estimated_time = 0.0
    for seg in segments:
        words = len(seg["text"].split())
        est_duration = words / 2.5  # ~150 wpm
        transcript_entries.append({
            "speaker": seg["speaker"],
            "text": seg["text"],
            "emotion": seg.get("emotion", "neutral"),
            "start_time": round(estimated_time, 2),
            "end_time": round(estimated_time + est_duration, 2),
        })
        estimated_time += est_duration

    transcript_path = podcast_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(transcript_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Metadata
    import datetime
    metadata = {
        "title": title,
        "source": source,
        "style": style,
        "format": output_format,
        "duration_seconds": round(duration, 1),
        "turn_count": len(script),
        "created_at": datetime.datetime.now().isoformat(),
        "tts_provider": segments[0].get("provider", "edge") if segments else "unknown",
    }
    metadata_path = podcast_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ===========================================================================
# Main Tool Function
# ===========================================================================
def podcast_generate_tool(
    source: str,
    title: Optional[str] = None,
    style: Optional[str] = None,
    output_format: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Generate a two-host conversational podcast from source content.

    Pipeline: Ingest → Script → Synthesize → Assemble → Save.

    Args:
        source: Path to file/directory, or inline text content.
        title: Episode title. Auto-generated if omitted.
        style: Conversation style (deep_dive, quick_brief, debate).
        output_format: Audio format (mp3, m4a, wav).
        provider: TTS provider (auto, voxcpm, edge). Overrides config.

    Returns:
        JSON string with success status, file paths, and MEDIA tag.
    """
    style = style or DEFAULT_STYLE
    output_format = output_format or DEFAULT_FORMAT

    if style not in _STYLE_PROMPTS:
        return json.dumps({
            "success": False,
            "error": f"Invalid style '{style}'. Choose from: {', '.join(_STYLE_PROMPTS.keys())}",
        })

    podcast_config = _load_podcast_config()

    # Override provider if explicitly specified
    if provider:
        podcast_config["tts_provider"] = provider

    # Create output directory
    episode_id = str(uuid.uuid4())[:8]
    podcast_dir = _get_podcasts_dir() / episode_id
    podcast_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = podcast_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Ingest
        logger.info("Step 1/4: Ingesting source...")
        content = _ingest_source(source)

        # Step 2: Generate script
        logger.info("Step 2/4: Generating podcast script...")
        script = _generate_script(content, style)

        # Auto-generate title from first line if not provided
        if not title:
            first_line = script[0]["text"] if script else "Untitled Episode"
            title = first_line[:60].rstrip(".!?,") if len(first_line) > 60 else first_line

        # Step 3: Synthesize voices
        logger.info("Step 3/4: Synthesizing voices (%d segments)...", len(script))
        segments = _synthesize_voices(script, segments_dir, podcast_config)

        if not segments:
            return json.dumps({
                "success": False,
                "error": "Voice synthesis produced no segments. Check TTS provider availability.",
            })

        # Step 4: Assemble audio
        logger.info("Step 4/4: Assembling final audio...")
        audio_filename = f"podcast.{output_format}"
        audio_path = podcast_dir / audio_filename
        duration = _assemble_audio(segments, audio_path, output_format)

        # Save metadata and transcript
        _save_metadata(podcast_dir, title, source, style, output_format,
                        duration, script, segments)

        # Clean up individual segment files
        for seg in segments:
            try:
                os.remove(seg["path"])
            except OSError:
                pass
        try:
            segments_dir.rmdir()
        except OSError:
            pass  # Not empty — keep it

        # Format duration
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_str = f"{minutes}:{seconds:02d}"

        media_tag = f"MEDIA:{audio_path}"

        return json.dumps({
            "success": True,
            "title": title,
            "file_path": str(audio_path),
            "transcript_path": str(podcast_dir / "transcript.json"),
            "metadata_path": str(podcast_dir / "metadata.json"),
            "duration": duration_str,
            "duration_seconds": round(duration, 1),
            "turn_count": len(script),
            "style": style,
            "media_tag": media_tag,
            "episode_id": episode_id,
        }, ensure_ascii=False)

    except FileNotFoundError as e:
        return json.dumps({"success": False, "error": f"Source not found: {e}"})
    except ValueError as e:
        return json.dumps({"success": False, "error": f"Script generation error: {e}"})
    except RuntimeError as e:
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.error("Podcast generation failed: %s", e, exc_info=True)
        return json.dumps({"success": False, "error": f"Unexpected error: {e}"})


# ===========================================================================
# Requirements Check
# ===========================================================================
def _check_podcast_requirements() -> bool:
    """Check if podcast generation dependencies are available.

    Requires pydub + at least one TTS provider (Edge TTS or VoxCPM).
    """
    try:
        from pydub import AudioSegment  # noqa: F401
    except ImportError:
        return False
    # At least one TTS provider must be available
    has_edge = False
    try:
        import edge_tts  # noqa: F401
        has_edge = True
    except ImportError:
        pass
    has_voxcpm = _check_voxcpm_available()
    return has_edge or has_voxcpm


# ===========================================================================
# Registry
# ===========================================================================
from tools.registry import registry

PODCAST_SCHEMA = {
    "name": "podcast_generate",
    "description": (
        "Generate a two-host conversational audio podcast from source content "
        "(file, directory, or text). Two AI hosts — Alex (the explainer) and "
        "Sam (the curious co-host) — discuss the material in an engaging, "
        "natural conversation. Returns MEDIA: path for platform delivery. "
        "Styles: deep_dive (thorough, ~10 min), quick_brief (~3 min), debate (opposing views)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": (
                    "Source content: path to a file (.md, .txt), a directory of files, "
                    "or inline text. For cron output, use ~/.hermes/cron/output/{job_id}/{timestamp}.md"
                ),
            },
            "title": {
                "type": "string",
                "description": "Episode title. Auto-generated from content if omitted.",
            },
            "style": {
                "type": "string",
                "enum": ["deep_dive", "quick_brief", "debate"],
                "description": "Conversation style. deep_dive (default): thorough exploration. quick_brief: concise summary. debate: hosts take opposing views.",
            },
            "output_format": {
                "type": "string",
                "enum": ["mp3", "m4a", "wav"],
                "description": "Output audio format. Default: mp3.",
            },
            "provider": {
                "type": "string",
                "enum": ["auto", "voxcpm", "edge"],
                "description": (
                    "TTS provider. auto (default): try VoxCPM, fall back to Edge TTS. "
                    "voxcpm: studio-grade local synthesis (~8GB VRAM). "
                    "edge: free cloud-based Microsoft voices."
                ),
            },
        },
        "required": ["source"],
    },
}

registry.register(
    name="podcast_generate",
    toolset="podcast",
    schema=PODCAST_SCHEMA,
    handler=lambda args, **kw: podcast_generate_tool(
        source=args.get("source", ""),
        title=args.get("title"),
        style=args.get("style"),
        output_format=args.get("output_format"),
        provider=args.get("provider"),
    ),
    check_fn=_check_podcast_requirements,
    emoji="🎙️",
    mutates=True,
)
