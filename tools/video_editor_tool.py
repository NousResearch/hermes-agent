#!/usr/bin/env python3
"""
Video Editor Tool (Native Hermes Builtin)
===========================================

8 tools for conversational video editing via FFmpeg + faster-whisper.
Wraps the hermes-video-mcp package as native Hermes tools, eliminating
the MCP subprocess overhead.

Tools registered under toolset "video_editor":
  - video_probe:       Get video metadata via ffprobe
  - video_transcribe:  Speech-to-text with word-level timestamps
  - video_cut:         Extract video segments
  - video_overlay:     Overlay images with fade transitions
  - video_captions:    Burn animated subtitles (SRT/ASS karaoke)
  - video_zoom:        Dynamic zoom/pan (Ken Burns effect)
  - video_split_screen: Multi-video layouts
  - video_render:      Final composition with codec/quality settings

Requirements:
  - ffmpeg and ffprobe on PATH (>= 6.0)
  - faster-whisper installed (pip install faster-whisper)
  - Pillow installed (pip install pillow)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions (compact — follows Hermes tool schema convention)
# ---------------------------------------------------------------------------

_PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to video file"},
    },
    "required": ["input_path"],
}

_TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to video/audio file"},
        "model": {"type": "string", "description": "Whisper model: tiny, base, small, medium, large-v3", "default": "base"},
        "language": {"type": "string", "description": "Language code or 'auto'", "default": "auto"},
        "word_timestamps": {"type": "boolean", "description": "Include word-level timestamps", "default": True},
    },
    "required": ["input_path"],
}

_CUT_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to input video file"},
        "start": {"type": "string", "description": "Start time (HH:MM:SS.mmm or seconds)"},
        "end": {"type": "string", "description": "End time (HH:MM:SS.mmm or seconds)"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "reencode": {"type": "boolean", "description": "Re-encode instead of stream copy", "default": False},
    },
    "required": ["input_path", "start", "end"],
}

_OVERLAY_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string", "description": "Path to base video file"},
        "image_path": {"type": "string", "description": "Path to overlay image (PNG with alpha recommended)"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "position": {"type": "string", "description": "Position: top-left, top-right, bottom-left, bottom-right, center, or x:y", "default": "center"},
        "start": {"type": "string", "description": "Start time for overlay", "default": "0"},
        "end": {"type": "string", "description": "End time (overlay end)"},
        "fade_in": {"type": "number", "description": "Fade in duration in seconds", "default": 0},
        "fade_out": {"type": "number", "description": "Fade out duration in seconds", "default": 0},
        "scale": {"type": "string", "description": "Scale overlay: 'width:height' or 'iw*0.5:ih*0.5'", "default": "original"},
        "opacity": {"type": "number", "description": "Overlay opacity 0.0-1.0", "default": 1.0},
    },
    "required": ["video_path", "image_path"],
}

_CAPTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string", "description": "Path to input video file"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "segments": {
            "type": "array",
            "description": "Transcript segments: [{start, end, text}, ...]",
            "items": {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}, "text": {"type": "string"}}, "required": ["start", "end", "text"]},
        },
        "srt_content": {"type": "string", "description": "Raw SRT content string"},
        "font_size": {"type": "number", "description": "Font size relative to video height (0.0-1.0)", "default": 0.06},
        "font_color": {"type": "string", "description": "Font color (hex or name)", "default": "#FFFFFF"},
        "style": {"type": "string", "description": "Animation style: none, fade, karaoke", "default": "karaoke"},
        "position": {"type": "string", "description": "Vertical position: top, center, bottom", "default": "bottom"},
    },
    "required": ["video_path"],
}

_ZOOM_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to input video file"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "effect": {"type": "string", "description": "Effect: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down, ken_burns", "default": "zoom_in"},
        "zoom_start": {"type": "number", "description": "Starting zoom level (1.0 = no zoom)", "default": 1.0},
        "zoom_end": {"type": "number", "description": "Ending zoom level", "default": 1.5},
        "duration": {"type": "string", "description": "Duration of effect or 'full'", "default": "full"},
    },
    "required": ["input_path"],
}

_SPLIT_SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "input_paths": {
            "type": "array",
            "description": "List of input video file paths (2-4 videos)",
            "items": {"type": "string"},
        },
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "layout": {"type": "string", "description": "Layout: side_by_side, top_bottom, grid_2x2, picture_in_picture", "default": "side_by_side"},
        "duration": {"type": "string", "description": "Output duration: shortest, longest", "default": "shortest"},
    },
    "required": ["input_paths"],
}

_RENDER_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to input video file (edited intermediate)"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "resolution": {"type": "string", "description": "Output resolution (e.g., 1920x1080, original)", "default": "original"},
        "video_codec": {"type": "string", "description": "Video codec (libx264, libx265, copy)", "default": "libx264"},
        "audio_codec": {"type": "string", "description": "Audio codec (aac, copy, mp3)", "default": "aac"},
        "crf": {"type": "number", "description": "Quality (0-51, lower=better)", "default": 23},
        "preset": {"type": "string", "description": "Encoding preset (ultrafast, fast, medium, slow, veryslow)", "default": "medium"},
        "normalize_audio": {"type": "boolean", "description": "Normalize audio levels", "default": True},
    },
    "required": ["input_path"],
}

# ---------------------------------------------------------------------------
# FFmpeg helpers (self-contained, no dependency on hermes-video-mcp package)
# ---------------------------------------------------------------------------

def _run_ffmpeg(args: List[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run ffmpeg with args (no shell). Raises RuntimeError on non-zero exit."""
    cmd = ["ffmpeg"] + args
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed (exit {result.returncode}): {result.stderr[-500:]}")
    return result


def _run_ffprobe(args: List[str]) -> dict:
    """Run ffprobe with JSON output. Raises RuntimeError on failure."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {result.stderr[-500:]}")
    return json.loads(result.stdout)


def _ensure_ffmpeg() -> None:
    """Raise RuntimeError if ffmpeg/ffprobe not found."""
    for cmd in ("ffmpeg", "ffprobe"):
        if not shutil.which(cmd):
            raise RuntimeError(f"{cmd} not found on PATH. Install FFmpeg >= 6.0.")


def _probe_video(input_path: str) -> dict:
    """Get structured video metadata."""
    _ensure_ffmpeg()
    data = _run_ffprobe(["-show_format", "-show_streams", input_path])
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fps_str = vs.get("r_frame_rate", "0/1")
    try:
        fps = float(fps_str) if "/" not in fps_str else eval(fps_str)
    except Exception:
        fps = 0
    return {
        "duration": float(fmt.get("duration", 0)),
        "size": int(fmt.get("size", 0)),
        "bit_rate": int(fmt.get("bit_rate", 0)) if fmt.get("bit_rate") else 0,
        "video": {
            "codec": vs.get("codec_name", ""),
            "width": vs.get("width", 0),
            "height": vs.get("height", 0),
            "fps": fps,
        } if vs else None,
        "audio": {
            "codec": audio.get("codec_name", ""),
            "sample_rate": int(audio.get("sample_rate", 0)),
            "channels": audio.get("channels", 0),
        } if audio else None,
    }


def _parse_timecode(tc: str | float | int) -> float:
    """Parse timecode to seconds."""
    if isinstance(tc, (int, float)):
        return float(tc)
    import re
    tc = tc.strip()
    m = re.match(r"^(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", tc)
    if m:
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    m = re.match(r"^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", tc)
    if m:
        return int(m[1]) * 60 + float(m[2])
    m = re.match(r"^(\d+(?:\.\d+)?)$", tc)
    if m:
        return float(m[1])
    raise ValueError(f"Invalid timecode: {tc}")


def _output_path(input_path: str, suffix: str, extension: str = ".mp4") -> str:
    """Generate auto output path: input_name_suffix.ext."""
    base, _ = os.path.splitext(input_path)
    return f"{base}{suffix}{extension}"


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _handle_probe(kwargs: dict) -> dict:
    """Get video metadata."""
    return _probe_video(kwargs["input_path"])


def _handle_transcribe(kwargs: dict) -> dict:
    """Transcribe audio from video using faster-whisper."""
    _ensure_ffmpeg()
    input_path = kwargs["input_path"]
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    from faster_whisper import WhisperModel

    model_name = kwargs.get("model", "base")
    lang = None if kwargs.get("language", "auto") == "auto" else kwargs["language"]
    word_ts = kwargs.get("word_timestamps", True)

    model_instance = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model_instance.transcribe(input_path, language=lang, word_timestamps=word_ts, vad_filter=True)

    seg_list = []
    for seg in segments:
        d = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        if word_ts and seg.words:
            d["words"] = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in seg.words]
        seg_list.append(d)

    return {"language": info.language, "language_probability": info.language_probability, "duration": info.duration, "segments": seg_list}


def _handle_cut(kwargs: dict) -> dict:
    """Extract video segment."""
    _ensure_ffmpeg()
    input_path = kwargs["input_path"]
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    start = _parse_timecode(kwargs["start"])
    end = _parse_timecode(kwargs["end"])
    if end <= start:
        raise ValueError(f"End ({end}) must be after start ({start})")

    output = kwargs.get("output_path") or _output_path(input_path, "_cut")
    reencode = kwargs.get("reencode", False)

    if reencode:
        cmd = ["-ss", str(start), "-i", input_path, "-t", str(end - start), "-c:v", "libx264", "-c:a", "aac", "-y", output]
    else:
        cmd = ["-ss", str(start), "-i", input_path, "-t", str(end - start), "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", output]

    _run_ffmpeg(cmd)
    return {"output_path": output, "start": start, "end": end, "duration": end - start, "reencoded": reencode}


def _handle_overlay(kwargs: dict) -> dict:
    """Overlay image onto video."""
    _ensure_ffmpeg()
    video_path = kwargs["video_path"]
    image_path = kwargs["image_path"]
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    output = kwargs.get("output_path") or _output_path(video_path, "_overlay")
    position = kwargs.get("position", "center")
    start = _parse_timecode(kwargs.get("start", "0"))
    vi = _probe_video(video_path)
    end = _parse_timecode(kwargs.get("end", str(vi["duration"]))) if kwargs.get("end") else vi["duration"]
    fade_in = kwargs.get("fade_in", 0)
    fade_out = kwargs.get("fade_out", 0)
    scale = kwargs.get("scale", "original")
    opacity = kwargs.get("opacity", 1.0)

    pos_map = {
        "top-left": "10:10", "top-right": "main_w-overlay_w-10:10",
        "bottom-left": "10:main_h-overlay_h-10", "bottom-right": "main_w-overlay_w-10:main_h-overlay_h-10",
        "center": "(main_w-overlay_w)/2:(main_h-overlay_h)/2",
    }
    pos = pos_map.get(position, position)

    filters = []
    overlay_input = "[1:v]"

    if scale != "original":
        filters.append(f"[1:v]scale={scale}[overlay_scaled]")
        overlay_input = "[overlay_scaled]"
    if opacity < 1.0:
        filters.append(f"{overlay_input}format=rgba,colorchannelmixer=aa={opacity}[overlay_opacity]")
        overlay_input = "[overlay_opacity]"
    if fade_in > 0:
        filters.append(f"{overlay_input}fade=type=in:start_time={start}:duration={fade_in}[fade_in]")
        overlay_input = "[fade_in]"
    if fade_out > 0:
        filters.append(f"{overlay_input}fade=type=out:start_time={end - fade_out}:duration={fade_out}[fade_out]")
        overlay_input = "[fade_out]"

    overlay_filter = f"[0:v]{overlay_input}overlay={pos}:enable='between(t,{start},{end})'"
    filters.append(overlay_filter)

    filtergraph = ";".join(filters)
    _run_ffmpeg(["-i", video_path, "-i", image_path, "-filter_complex", filtergraph, "-c:a", "copy", "-y", output])

    return {"output_path": output, "position": position, "start": start, "end": end, "fade_in": fade_in, "fade_out": fade_out}


def _handle_zoom(kwargs: dict) -> dict:
    """Apply dynamic zoom/pan (Ken Burns effect)."""
    _ensure_ffmpeg()
    input_path = kwargs["input_path"]
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    output = kwargs.get("output_path") or _output_path(input_path, "_zoom")
    effect = kwargs.get("effect", "zoom_in")
    zoom_start = kwargs.get("zoom_start", 1.0)
    zoom_end = kwargs.get("zoom_end", 1.5)
    vi = _probe_video(input_path)
    vw, vh = vi["video"]["width"], vi["video"]["height"]
    fps = vi["video"]["fps"]
    total_frames = int(vi["duration"] * fps)
    duration_frames = total_frames

    z = f"min({zoom_start}+(({zoom_end}-{zoom_start})*in/{duration_frames}),10)"
    x = "(iw-iw/zoom)/2"
    y = "(ih-ih/zoom)/2"

    if effect == "zoom_out":
        z = f"min({zoom_end}+(({zoom_start}-{zoom_end})*in/{duration_frames}),10)"
    elif effect in ("pan_left", "ken_burns"):
        x = f"max(0,(iw-iw/zoom)*(1-in/{duration_frames}))"
    elif effect in ("pan_right",):
        x = f"max(0,(iw-iw/zoom)*(in/{duration_frames}))"
    elif effect == "pan_up":
        y = f"max(0,(ih-ih/zoom)*(1-in/{duration_frames}))"
    elif effect == "pan_down":
        y = f"max(0,(ih-ih/zoom)*(in/{duration_frames}))"

    _run_ffmpeg(["-i", input_path, "-vf", f"zoompan=zoom='{z}':x='{x}':y='{y}':d={total_frames}:s={vw}x{vh}:fps={fps},scale={vw}:{vh}", "-c:a", "copy", "-y", output])
    return {"output_path": output, "effect": effect, "zoom_start": zoom_start, "zoom_end": zoom_end}


def _handle_split_screen(kwargs: dict) -> dict:
    """Compose split-screen from multiple videos."""
    _ensure_ffmpeg()
    paths = kwargs["input_paths"]
    if len(paths) < 2:
        raise ValueError("Need at least 2 input videos")

    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Input not found: {p}")

    output = kwargs.get("output_path") or _output_path(paths[0], "_split")
    layout = kwargs.get("layout", "side_by_side")

    cmd = []
    for p in paths:
        cmd.extend(["-i", p])

    filters = []
    for i in range(len(paths)):
        filters.append(f"[{i}:v]scale=640:480,setsar=1[v{i}]")

    if layout == "side_by_side":
        filters.append("[v0][v1]hstack=inputs=2[outv]")
        filters.append("[0:a][1:a]amix=inputs=2:duration=shortest[outa]")
    elif layout == "top_bottom":
        filters.append("[v0][v1]vstack=inputs=2[outv]")
        filters.append("[0:a][1:a]amix=inputs=2:duration=shortest[outa]")
    elif layout == "picture_in_picture":
        filters.append("[v1]scale=213:160[pip]")
        filters.append("[v0][pip]overlay=427:310:shortest=1[outv]")
        filters.append("[0:a][1:a]amix=inputs=2:duration=shortest[outa]")
    elif layout == "grid_2x2" and len(paths) == 4:
        filters.append("[v0][v1]hstack=inputs=2[top]")
        filters.append("[v2][v3]hstack=inputs=2[bottom]")
        filters.append("[top][bottom]vstack=inputs=2[outv]")
        filters.append("[0:a][1:a][2:a][3:a]amix=inputs=4:duration=shortest[outa]")
    else:
        raise ValueError(f"Unsupported layout '{layout}' for {len(paths)} inputs")

    filter_complex = ";".join(filters)
    cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]", "-map", "[outa]", "-y", output])
    _run_ffmpeg(cmd)
    return {"output_path": output, "layout": layout, "inputs": len(paths)}


def _handle_captions(kwargs: dict) -> dict:
    """Burn subtitles into video."""
    _ensure_ffmpeg()
    video_path = kwargs["video_path"]
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    output = kwargs.get("output_path") or _output_path(video_path, "_captioned")
    segments = kwargs.get("segments")
    srt_content = kwargs.get("srt_content")
    if not segments and not srt_content:
        raise ValueError("Must provide segments or srt_content")

    # Build SRT from segments
    if not srt_content:
        lines = []
        for i, seg in enumerate(segments, 1):
            h = int(seg["start"] // 3600)
            m = int((seg["start"] % 3600) // 60)
            s = int(seg["start"] % 60)
            ms = int((seg["start"] % 1) * 1000)
            start_srt = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            h2 = int(seg["end"] // 3600)
            m2 = int((seg["end"] % 3600) // 60)
            s2 = int(seg["end"] % 60)
            ms2 = int((seg["end"] % 1) * 1000)
            end_srt = f"{h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}"
            lines.append(f"{i}\n{start_srt} --> {end_srt}\n{seg['text'].strip()}\n")
        srt_content = "\n".join(lines)

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
        f.write(srt_content)
        srt_path = f.name

    try:
        style = kwargs.get("style", "karaoke")
        if style == "karaoke":
            # Use subtitles filter with karaoke-style force_style
            font_size = int(480 * kwargs.get("font_size", 0.06))
            force_style = f"FontName=DejaVu Sans,FontSize={max(12,font_size)},PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,MarginV=20"
            _run_ffmpeg(["-i", video_path, "-vf", f"subtitles={srt_path}:force_style='{force_style}'", "-c:a", "copy", "-y", output])
        else:
            _run_ffmpeg(["-i", video_path, "-vf", f"subtitles={srt_path}", "-c:a", "copy", "-y", output])
    finally:
        os.unlink(srt_path)

    return {"output_path": output, "style": style}


def _handle_render(kwargs: dict) -> dict:
    """Render final video composition."""
    _ensure_ffmpeg()
    input_path = kwargs["input_path"]
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    output = kwargs.get("output_path") or _output_path(input_path, "_final")
    resolution = kwargs.get("resolution", "original")
    video_codec = kwargs.get("video_codec", "libx264")
    audio_codec = kwargs.get("audio_codec", "aac")
    crf = kwargs.get("crf", 23)
    preset = kwargs.get("preset", "medium")
    normalize = kwargs.get("normalize_audio", True)

    if not 0 <= crf <= 51:
        raise ValueError(f"CRF must be between 0 and 51, got {crf}")

    cmd = ["-i", input_path]
    vfilters = []
    if resolution != "original":
        parts = resolution.split("x")
        if len(parts) == 2:
            vfilters.append(f"scale={parts[0]}:{parts[1]}")
    if vfilters:
        cmd.extend(["-vf", ",".join(vfilters)])

    if normalize and audio_codec != "copy":
        cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])

    if video_codec == "copy":
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend(["-c:v", video_codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"])

    if audio_codec == "copy":
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.extend(["-c:a", audio_codec, "-b:a", "192k"])

    cmd.extend(["-movflags", "+faststart", "-y", output])
    _run_ffmpeg(cmd)

    out_info = _probe_video(output)
    return {"output_path": output, "resolution": f"{out_info['video']['width']}x{out_info['video']['height']}", "video_codec": video_codec, "audio_codec": audio_codec, "crf": crf, "preset": preset}


# ---------------------------------------------------------------------------
# Check functions (tool availability)
# ---------------------------------------------------------------------------

def _check_ffmpeg(kwargs: dict = None) -> str | None:
    """Check if ffmpeg/ffprobe are available. Returns None if OK, error string if not."""
    import shutil
    for cmd in ("ffmpeg", "ffprobe"):
        if not shutil.which(cmd):
            return f"{cmd} not found on PATH. Install FFmpeg >= 6.0."
    return None


def _check_whisper(kwargs: dict = None) -> str | None:
    """Check if faster-whisper is available. Returns None if OK, error string if not."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return "faster-whisper not installed. Run: pip install faster-whisper"
    return _check_ffmpeg()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="video_probe",
    toolset="video_editor",
    schema=_PROBE_SCHEMA,
    handler=_handle_probe,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)

registry.register(
    name="video_transcribe",
    toolset="video_editor",
    schema=_TRANSCRIBE_SCHEMA,
    handler=_handle_transcribe,
    check_fn=_check_whisper,
    requires_env=[],
    is_async=False,
    emoji="",
)

registry.register(
    name="video_cut",
    toolset="video_editor",
    schema=_CUT_SCHEMA,
    handler=_handle_cut,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)

registry.register(
    name="video_overlay",
    toolset="video_editor",
    schema=_OVERLAY_SCHEMA,
    handler=_handle_overlay,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)

registry.register(
    name="video_captions",
    toolset="video_editor",
    schema=_CAPTIONS_SCHEMA,
    handler=_handle_captions,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)

registry.register(
    name="video_zoom",
    toolset="video_editor",
    schema=_ZOOM_SCHEMA,
    handler=_handle_zoom,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)

registry.register(
    name="video_split_screen",
    toolset="video_editor",
    schema=_SPLIT_SCREEN_SCHEMA,
    handler=_handle_split_screen,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)

registry.register(
    name="video_render",
    toolset="video_editor",
    schema=_RENDER_SCHEMA,
    handler=_handle_render,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)

# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

_TRANSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "input_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1 or 2 video paths. 1=fade in/out, 2=crossfade.",
        },
        "transition_type": {
            "type": "string",
            "enum": ["fade_to_black", "fade_from_black", "fade_in", "fade_out", "crossfade", "wipe_left", "wipe_right"],
            "default": "fade_in",
        },
        "duration": {"type": "number", "default": 1.0},
    },
    "required": ["input_paths"],
}

def _call_mcp_tool(tool_name: str, kwargs: dict) -> dict:
    """Import the MCP tool handler and call it synchronously."""
    import importlib
    handlers = {
        "transition": ("hermes_video_mcp.tools.transition", "transition_handler"),
        "audio_mix": ("hermes_video_mcp.tools.audio_mix", "audio_mix_handler"),
        "watermark": ("hermes_video_mcp.tools.watermark", "watermark_handler"),
    }
    if tool_name not in handlers:
        raise ValueError(f"Unknown MCP tool: {tool_name}")
    mod_name, func_name = handlers[tool_name]
    mod = importlib.import_module(mod_name)
    handler = getattr(mod, func_name)
    # Run the async handler in a new event loop
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(handler(**kwargs))
    finally:
        loop.close()
    # Convert Path objects to strings
    if isinstance(result, dict):
        return {k: str(v) if isinstance(v, __import__("pathlib").Path) else v for k, v in result.items()}
    return result


def _handle_transition(kwargs: dict) -> dict:
    """Import and call the MCP transition handler."""
    return _call_mcp_tool("transition", kwargs)

# ---------------------------------------------------------------------------
# Audio Mix
# ---------------------------------------------------------------------------

_AUDIO_MIX_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string"},
        "mode": {
            "type": "string",
            "enum": ["replace", "adjust_volume", "normalize", "add_background", "extract", "mute"],
            "default": "normalize",
        },
        "audio_path": {"type": "string"},
        "volume": {"type": "number", "default": 1.0},
        "background_volume": {"type": "number", "default": 0.3},
        "duck": {"type": "boolean", "default": False},
    },
    "required": ["video_path", "mode"],
}

def _handle_audio_mix(kwargs: dict) -> dict:
    return _call_mcp_tool("audio_mix", kwargs)

# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------

_WATERMARK_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string"},
        "mode": {"type": "string", "enum": ["text", "image"], "default": "text"},
        "text": {"type": "string"},
        "image_path": {"type": "string"},
        "position": {"type": "string", "default": "bottom-right"},
        "opacity": {"type": "number", "default": 0.5},
        "font_size": {"type": "number", "default": 24},
        "font_color": {"type": "string", "default": "#FFFFFF"},
        "start": {"type": "string", "default": "0"},
        "end": {"type": "string"},
        "scale": {"type": "string"},
    },
    "required": ["video_path", "mode"],
}

def _handle_watermark(kwargs: dict) -> dict:
    return _call_mcp_tool("watermark", kwargs)

# ---------------------------------------------------------------------------
# Registry entries for new tools
# ---------------------------------------------------------------------------

registry.register(
    name="video_transition",
    toolset="video_editor",
    schema=_TRANSITION_SCHEMA,
    handler=_handle_transition,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)

registry.register(
    name="video_audio_mix",
    toolset="video_editor",
    schema=_AUDIO_MIX_SCHEMA,
    handler=_handle_audio_mix,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)

registry.register(
    name="video_watermark",
    toolset="video_editor",
    schema=_WATERMARK_SCHEMA,
    handler=_handle_watermark,
    check_fn=_check_ffmpeg,
    requires_env=[],
    is_async=False,
    emoji="",
)
