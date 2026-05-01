"""Vietnamese language teaching video caption tool for Hermes Agent.

For short teaching videos (YouTube Shorts style) that mix English narration
with Vietnamese words/phrases being taught:

  - Vietnamese segments → main text (Vietnamese with diacritics) on top,
    English phonetic guide in brackets below  e.g.  không biết
                                                      [humm biet]
  - English segments    → English text only, no second line

Pipeline:
  1. faster-whisper transcribes audio (auto language detect, fully local)
  2. Kimi K2.5 via NVIDIA NIM: corrects Vietnamese diacritics, classifies
     each segment as "en" or "vi", and generates English phonetic guides
  3. ASS subtitle file built with MAIN + PHONETIC styles
  4. FFmpeg burns captions into the video

Required dependencies:
    pip install faster-whisper openai

Required env var for phonetic generation (add to ~/.hermes/.env):
    NVIDIA_API_KEY=nvapi-...

FFmpeg must be installed system-wide:
    macOS: brew install ffmpeg
    Linux: sudo apt install ffmpeg
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from tools.registry import registry
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".ts", ".mts"}
_ASS_COLOR_RE = re.compile(r"^&H[0-9A-Fa-f]{8}$")

_DEFAULT_STYLE = {
    "font": "Arial",
    "font_size": 48,
    "primary_color": "&H00FFFFFF",   # white   (ASS: &HAABBGGRR)
    "outline_color": "&H00000000",   # black
    "outline_width": 3,
    "alignment": 2,                  # 2 = bottom-center (ASS numpad alignment)
    "margin_bottom": 80,
    "max_line_length": 42,
}


def _load_style() -> dict:
    """Load caption style from config, falling back to built-in defaults."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        user_style = cfg.get("caption", {}).get("style", {})
        return {**_DEFAULT_STYLE, **{k: v for k, v in user_style.items() if v is not None}}
    except Exception:
        return dict(_DEFAULT_STYLE)


def _ass_color(hex_color: str) -> str:
    """Ensure hex_color is in ASS &HAABBGGRR format; pass through if already valid."""
    if _ASS_COLOR_RE.match(hex_color):
        return hex_color
    m = re.match(r"^#([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})$", hex_color)
    if m:
        r, g, b = m.group(1), m.group(2), m.group(3)
        return f"&H00{b}{g}{r}".upper()
    return "&H00FFFFFF"


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert float seconds to ASS timestamp H:MM:SS.cs"""
    cs = int(round(seconds * 100)) % 100
    total_s = int(seconds)
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _wrap_text(text: str, max_len: int) -> str:
    """Wrap text to max_len characters with \\N (ASS hard break)."""
    if len(text) <= max_len:
        return text
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return r"\N".join(lines)


def _build_ass_content(segments: list[dict], style: dict) -> str:
    """Build ASS subtitle file for teaching layout.

    Vietnamese segment: MAIN line (Vietnamese text, bold, larger)
                        PHONETIC line ([english guide], italic, smaller, below)
    English segment:    MAIN line only (English text)
    """
    font = style["font"]
    size_main = int(style["font_size"])
    size_phonetic = max(int(size_main * 0.75), 24)
    primary = _ass_color(style.get("primary_color", "&H00FFFFFF"))
    outline = _ass_color(style.get("outline_color", "&H00000000"))
    outline_w = int(style.get("outline_width", 3))
    alignment = int(style.get("alignment", 2))
    margin_v = int(style.get("margin_bottom", 80))
    max_len = int(style.get("max_line_length", 42))

    # PHONETIC sits at the base margin; MAIN sits directly above it
    phonetic_margin = margin_v
    main_margin = margin_v + size_phonetic + 8

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    )

    # MAIN: bold, full-size — used for both EN and VI main text
    style_main = (
        f"Style: MAIN,{font},{size_main},{primary},&H000000FF,{outline},&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline_w},1,{alignment},10,10,{main_margin},1\n"
    )
    # PHONETIC: smaller, italic, slightly transparent — only for VI segments
    phonetic_color = "&H99FFFFFF"  # 60% alpha white for visual hierarchy
    style_phonetic = (
        f"Style: PHONETIC,{font},{size_phonetic},{phonetic_color},&H000000FF,{outline},&H80000000,"
        f"0,1,0,0,100,100,0,0,1,{outline_w},1,{alignment},10,10,{phonetic_margin},1\n"
    )

    events_header = (
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    dialogue_lines: list[str] = []
    for seg in segments:
        start = _seconds_to_ass_time(float(seg["start"]))
        end = _seconds_to_ass_time(float(seg["end"]))
        text = _wrap_text((seg.get("text") or "").strip(), max_len)
        phonetic = _wrap_text((seg.get("phonetic") or "").strip(), max_len)
        lang = seg.get("lang", "en")

        if not text:
            continue

        if lang == "vi" and phonetic:
            # Vietnamese: main text on top + phonetic guide below
            dialogue_lines.append(f"Dialogue: 0,{start},{end},MAIN,,0,0,0,,{text}")
            dialogue_lines.append(f"Dialogue: 0,{start},{end},PHONETIC,,0,0,0,,{phonetic}")
        else:
            # English (or Vietnamese with no phonetics yet): main text only
            dialogue_lines.append(f"Dialogue: 0,{start},{end},MAIN,,0,0,0,,{text}")

    return header + style_main + style_phonetic + events_header + "\n".join(dialogue_lines) + "\n"


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def transcribe(video_path: str) -> list[dict]:
    """Transcribe audio with auto language detection using faster-whisper.

    Fully local — no internet required after model download. No rate limits.
    Returns segments: {id, start, end, text, lang, phonetic}
    lang and phonetic are blank — filled in by generate_phonetics().
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )

    model_name = os.getenv("HERMES_WHISPER_MODEL", "medium")
    logger.info("Loading faster-whisper model: %s", model_name)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    logger.info("Transcribing: %s", video_path)
    segments_raw, info = model.transcribe(
        video_path,
        language=None,        # auto-detect — works for mixed EN/VI
        vad_filter=True,
        word_timestamps=False,
    )
    logger.info(
        "Detected language: %s (prob %.2f)", info.language, info.language_probability
    )

    segments: list[dict] = []
    for seg in segments_raw:
        text = seg.text.strip()
        if text:
            segments.append({
                "id": len(segments),
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
                "lang": "",       # filled by generate_phonetics()
                "phonetic": "",
            })
    return segments


def generate_phonetics(segments: list[dict], api_key: str | None = None) -> list[dict]:
    """Use Kimi K2.5 to classify, correct, and add phonetics to segments.

    For each segment Kimi will:
      - Decide if it is English narration ("en") or a Vietnamese teaching moment ("vi")
      - Correct Vietnamese diacritics if garbled by Whisper
      - Generate an English phonetic guide for Vietnamese segments  e.g. [humm biet]

    Falls back to all-English (no phonetics) if NVIDIA_API_KEY is not set.
    """
    _api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
    if not _api_key:
        logger.warning(
            "NVIDIA_API_KEY not set — skipping phonetic generation. "
            "All segments treated as English. Add key to ~/.hermes/.env to enable."
        )
        return [{**s, "lang": "en", "phonetic": ""} for s in segments]

    try:
        import openai  # type: ignore
    except ImportError:
        logger.warning("openai not installed — skipping phonetics. Run: pip install openai")
        return [{**s, "lang": "en", "phonetic": ""} for s in segments]

    input_lines = "\n".join(
        f'{{"id": {s["id"]}, "text": {json.dumps(s["text"])}}}'
        for s in segments
    )

    prompt = (
        "You are captioning a Vietnamese language teaching short video. "
        "The audio is mostly English narration with Vietnamese words and phrases being taught. "
        "Whisper (the speech transcriber) may have garbled Vietnamese words into approximate English spellings.\n\n"
        "For each segment below, return a JSON array where every item has exactly these fields:\n"
        '- "id": same integer as input\n'
        '- "text": if English narration → corrected English text. '
        'If the segment contains Vietnamese being taught (even if garbled) → rewrite with correct Vietnamese spelling and diacritics.\n'
        '- "lang": "en" for English narration, "vi" for Vietnamese words/phrases being taught.\n'
        '- "phonetic": if lang is "vi" → a simple English pronunciation guide in square brackets '
        '(e.g. "[humm biet]", "[zuh yee]", "[toh-ee ten la]"). '
        'If lang is "en" → empty string "".\n\n'
        "Return ONLY a valid JSON array, no markdown, no explanation.\n\n"
        f"Segments:\n{input_lines}"
    )

    try:
        client = openai.OpenAI(
            api_key=_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        choice = response.choices[0]
        raw = getattr(choice.message, "content", None) or ""
        # Kimi NVIDIA NIM quirk: output sometimes in reasoning_content
        if not raw.strip():
            raw = getattr(choice.message, "reasoning_content", None) or ""

    except Exception as e:
        logger.warning("Kimi API call failed: %s — treating all segments as English", e)
        return [{**s, "lang": "en", "phonetic": ""} for s in segments]

    # Strip markdown code fences if present
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        kimi_results = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning(
            "Kimi returned non-JSON — treating all segments as English. Raw: %s", raw[:300]
        )
        return [{**s, "lang": "en", "phonetic": ""} for s in segments]

    result_map = {item["id"]: item for item in kimi_results if isinstance(item, dict)}
    result_segments = []
    for seg in segments:
        kimi = result_map.get(seg["id"], {})
        result_segments.append({
            **seg,
            "text": kimi.get("text", seg["text"]),
            "lang": kimi.get("lang", "en"),
            "phonetic": kimi.get("phonetic", ""),
        })
    return result_segments


def build_ass(segments: list[dict], output_path: str | None = None) -> str:
    """Build an ASS subtitle file from *segments* and write it to *output_path*."""
    style = _load_style()
    content = _build_ass_content(segments, style)

    if output_path is None:
        cache_dir = get_hermes_home() / "cache" / "captions"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(cache_dir / f"captions_{uuid.uuid4().hex[:8]}.ass")

    Path(output_path).write_text(content, encoding="utf-8")
    logger.info("ASS subtitle file written: %s", output_path)
    return output_path


def burn(video_path: str, ass_path: str, output_path: str | None = None) -> str:
    """Burn ASS subtitles into *video_path* using FFmpeg."""
    _check_ffmpeg()

    if output_path is None:
        base = Path(video_path).stem
        cache_dir = get_hermes_home() / "cache" / "captions"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(cache_dir / f"{base}_captioned_{uuid.uuid4().hex[:8]}.mp4")

    escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"ass={escaped_ass}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
    ]

    logger.info("Burning captions: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg burn failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ffmpeg() -> None:
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "FFmpeg not found.\n  macOS: brew install ffmpeg\n  Linux: sudo apt install ffmpeg"
        )


def save_caption_job(
    job_id: str,
    video_path: str,
    output_path: str,
    segments: list[dict],
    style: dict,
) -> Path:
    """Persist caption job state to ~/.hermes/caption-jobs/{job_id}.json.

    This enables the dashboard visual editor to load, edit, and re-burn
    the job without any LLM involvement.
    """
    jobs_dir = get_hermes_home() / "caption-jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_path = jobs_dir / f"{job_id}.json"
    job = {
        "id": job_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_path": str(video_path),
        "output_path": str(output_path),
        "style": style,
        "segments": segments,
    }
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Caption job saved: %s", job_path)
    return job_path


def check_requirements() -> bool:
    try:
        import faster_whisper  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def _handle_caption(args: dict, **kw: Any) -> str:
    operation = args.get("operation", "caption")
    video_path = args.get("video_path", "")
    segments_raw = args.get("segments")
    output_path = args.get("output_path")

    if not video_path and operation not in ("build_ass",):
        return json.dumps({"error": "video_path is required"})
    if video_path and not os.path.exists(video_path):
        return json.dumps({"error": f"Video file not found: {video_path}"})

    try:
        if operation == "caption":
            _check_ffmpeg()
            segments = transcribe(video_path)
            if not segments:
                return json.dumps({"error": "No speech detected in video"})
            segments = generate_phonetics(segments)
            style = _load_style()
            ass_path = build_ass(segments)
            out = burn(video_path, ass_path, output_path)

            # Persist job for dashboard visual editor
            job_id = uuid.uuid4().hex[:12]
            save_caption_job(job_id, video_path, out, segments, style)
            dashboard_url = f"http://localhost:9119/captions/{job_id}"

            vi_count = sum(1 for s in segments if s.get("lang") == "vi")
            en_count = sum(1 for s in segments if s.get("lang") == "en")
            caption_display = "\n".join(
                (
                    f"{i+1}. [{'VI' if s.get('lang') == 'vi' else 'EN'}] {s['text']}"
                    + (f"\n    {s['phonetic']}" if s.get("phonetic") else "")
                )
                for i, s in enumerate(segments)
            )
            return json.dumps({
                "success": True,
                "output_video": out,
                "ass_file": ass_path,
                "job_id": job_id,
                "dashboard_url": dashboard_url,
                "segments": segments,
                "segment_count": len(segments),
                "vi_segments": vi_count,
                "en_segments": en_count,
                "message": (
                    f"Done! {len(segments)} segments ({vi_count} Vietnamese, {en_count} English).\n"
                    f"Output: MEDIA:{out}\n\n"
                    f"Open the caption editor to visually edit text, phonetics, and style:\n"
                    f"{dashboard_url}\n\n"
                    f"Or review below and tell me what to fix:\n\n{caption_display}"
                ),
            })

        elif operation == "transcribe":
            segments = transcribe(video_path)
            return json.dumps({"success": True, "segments": segments, "count": len(segments)})

        elif operation == "generate_phonetics":
            if not segments_raw:
                return json.dumps({"error": "segments is required"})
            segments = generate_phonetics(segments_raw)
            return json.dumps({"success": True, "segments": segments})

        elif operation == "build_ass":
            if not segments_raw:
                return json.dumps({"error": "segments is required"})
            ass_path = build_ass(segments_raw, output_path)
            return json.dumps({"success": True, "ass_file": ass_path})

        elif operation == "burn":
            ass_path = args.get("ass_path")
            if not ass_path:
                return json.dumps({"error": "ass_path is required"})
            if not os.path.exists(ass_path):
                return json.dumps({"error": f"ASS file not found: {ass_path}"})
            out = burn(video_path, ass_path, output_path)
            return json.dumps({"success": True, "output_video": out})

        elif operation == "reburn":
            if not segments_raw:
                return json.dumps({"error": "segments is required"})
            _check_ffmpeg()
            original = args.get("original_video_path", video_path)
            ass_path = build_ass(segments_raw)
            out = burn(original, ass_path, output_path)
            return json.dumps({
                "success": True,
                "output_video": out,
                "ass_file": ass_path,
                "message": f"Re-burned with corrections. MEDIA:{out}",
            })

        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})

    except RuntimeError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.exception("video_caption tool error")
        return json.dumps({"error": f"Unexpected error: {e}"})


# ---------------------------------------------------------------------------
# Schema & registration
# ---------------------------------------------------------------------------

_SCHEMA = {
    "name": "video-caption",
    "description": (
        "Caption a Vietnamese language teaching video. Transcribes mixed EN/VI audio (auto language detect), "
        "classifies each segment as English or Vietnamese, corrects Vietnamese diacritics, and generates "
        "English phonetic guides (e.g. [humm biet]) for Vietnamese segments via Kimi K2.5. "
        "Burns EN/VI-aware captions into the video.\n\n"
        "Caption layout:\n"
        "  Vietnamese segment: Vietnamese text on top + [phonetic guide] below\n"
        "  English segment:    English text only\n\n"
        "Operations:\n"
        "- caption (default): full pipeline — transcribe + classify + phonetics + burn.\n"
        "- transcribe: transcribe audio only, returns raw segments.\n"
        "- generate_phonetics: classify segments and generate phonetics via Kimi.\n"
        "- build_ass: build ASS subtitle file from segments.\n"
        "- burn: burn an existing ASS file into video.\n"
        "- reburn: apply corrected segments and re-burn (use after user edits).\n\n"
        "Segment format: {id, start, end, text, lang ('en'|'vi'), phonetic}\n"
        "Requirements: faster-whisper, ffmpeg. NVIDIA_API_KEY for phonetics."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["caption", "transcribe", "generate_phonetics", "build_ass", "burn", "reburn"],
                "description": "Operation to perform. Default: caption (full pipeline).",
            },
            "video_path": {
                "type": "string",
                "description": "Absolute path to the input video file.",
            },
            "segments": {
                "type": "array",
                "description": "Caption segments. Each item: {id, start, end, text, lang, phonetic}.",
                "items": {"type": "object"},
            },
            "ass_path": {
                "type": "string",
                "description": "Path to the ASS subtitle file (for burn operation).",
            },
            "original_video_path": {
                "type": "string",
                "description": "Original video path for reburn when video_path is the captioned output.",
            },
            "output_path": {
                "type": "string",
                "description": "Optional explicit output path for the result file.",
            },
        },
        "required": [],
    },
}

registry.register(
    name="video-caption",
    toolset="video-caption",
    schema=_SCHEMA,
    handler=lambda args, **kw: _handle_caption(args, **kw),
    check_fn=check_requirements,
    requires_env=[],
    emoji="🎬",
)
