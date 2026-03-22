"""
Video Analysis Tool

Extracts frames from a video using ffmpeg/ffprobe and analyses them via the
centralized vision LLM router.  Follows the same registration pattern as
vision_tools.py.
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Dict, Optional

from agent.auxiliary_client import async_call_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ---------------------------------------------------------------------------

def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def _get_video_duration(path: str) -> float:
    """Return the duration of a video file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(result.stdout.strip())


def _calculate_frame_interval(duration: float) -> float:
    """Return the interval (seconds) between extracted frames.

    Tiered strategy:
      <1 min  -> every 1 s
      1-5 min -> every 5 s
      5 min+  -> every 10 s

    Hard cap: max 30 frames total — interval is increased proportionally if
    the naive tier would exceed this.
    """
    MAX_FRAMES = 30

    if duration < 60:
        interval = 1.0
    elif duration < 300:
        interval = 5.0
    else:
        interval = 10.0

    # Enforce max-frames cap
    estimated_frames = duration / interval
    if estimated_frames > MAX_FRAMES:
        interval = duration / MAX_FRAMES

    return interval


def _extract_frames(video_path: str, output_dir: str, interval: float) -> list[str]:
    """Extract JPEG frames from *video_path* at the given interval.

    Returns a sorted list of absolute paths to the extracted frame files.
    """
    pattern = os.path.join(output_dir, "frame_%04d.jpg")
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=1/{interval},scale=768:-1",
            "-q:v", "2",
            pattern,
        ],
        capture_output=True,
        timeout=120,
        check=True,
    )
    frames = sorted(
        str(p) for p in Path(output_dir).glob("frame_*.jpg")
    )
    return frames


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

async def video_analyze_tool(
    video_path: str,
    user_prompt: str,
    model: Optional[str] = None,
) -> str:
    """Analyse a video by extracting frames and sending them to a vision LLM.

    Args:
        video_path: Absolute path to the video file.
        user_prompt: The user's question or instruction about the video.
        model: Optional override for the vision model.

    Returns:
        JSON string with keys: success, analysis, duration, frames_analyzed.
    """
    tmp_dir: Optional[str] = None
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return json.dumps({"success": False, "error": "Interrupted"})

        path = Path(video_path)
        if not path.is_file():
            return json.dumps({
                "success": False,
                "error": f"Video file not found: {video_path}",
                "analysis": "The video file could not be found at the specified path.",
            })

        if not _has_ffmpeg() or not _has_ffprobe():
            return json.dumps({
                "success": False,
                "error": "ffmpeg/ffprobe not found on PATH",
                "analysis": (
                    "Video analysis requires ffmpeg and ffprobe to be installed. "
                    "Please install ffmpeg (https://ffmpeg.org) and try again."
                ),
            })

        # Get duration
        logger.info("Getting video duration for %s", video_path)
        duration = await asyncio.to_thread(_get_video_duration, video_path)
        logger.info("Video duration: %.1f seconds", duration)

        # Calculate frame extraction parameters
        interval = _calculate_frame_interval(duration)
        expected_frames = int(duration / interval) + 1
        logger.info(
            "Extracting frames: interval=%.1fs, expected=%d",
            interval, expected_frames,
        )

        # Extract frames to a temp directory
        tmp_dir = tempfile.mkdtemp(prefix="hermes_video_frames_")
        frames = await asyncio.to_thread(
            _extract_frames, video_path, tmp_dir, interval,
        )

        if not frames:
            return json.dumps({
                "success": False,
                "error": "No frames extracted from video",
                "analysis": "ffmpeg did not produce any frames from the video.",
            })

        logger.info("Extracted %d frames", len(frames))

        # Build multi-image vision message
        content: list[dict] = []

        # Text block with prompt + metadata
        meta_text = (
            f"{user_prompt}\n\n"
            f"--- Video metadata ---\n"
            f"Duration: {duration:.1f}s | "
            f"Frames shown: {len(frames)} | "
            f"Interval: {interval:.1f}s between frames"
        )
        content.append({"type": "text", "text": meta_text})

        # One image_url block per frame (base64 data URL)
        for frame_path in frames:
            data = Path(frame_path).read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        messages = [{"role": "user", "content": content}]

        logger.info("Calling vision LLM with %d frames ...", len(frames))
        call_kwargs: dict[str, Any] = {
            "task": "vision",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        if model:
            call_kwargs["model"] = model

        response = await async_call_llm(**call_kwargs)
        analysis = response.choices[0].message.content.strip()
        logger.info("Video analysis complete (%d chars)", len(analysis))

        return json.dumps({
            "success": True,
            "analysis": analysis,
            "duration": round(duration, 1),
            "frames_analyzed": len(frames),
        }, ensure_ascii=False)

    except Exception as e:
        logger.error("Video analysis failed: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "error": str(e),
            "analysis": f"Video analysis failed: {e}",
        }, ensure_ascii=False)

    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as cleanup_err:
                logger.warning("Could not clean up temp frames: %s", cleanup_err)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_video_requirements() -> bool:
    """Return True if ffmpeg, ffprobe, and a vision provider are all available."""
    if not (_has_ffmpeg() and _has_ffprobe()):
        return False
    try:
        from agent.auxiliary_client import resolve_vision_provider_client
        _provider, client, _model = resolve_vision_provider_client()
        return client is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

VIDEO_ANALYZE_SCHEMA = {
    "name": "video_analyze",
    "description": (
        "Analyze a video by extracting frames and describing its content "
        "using AI vision. Requires ffmpeg."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video_path": {
                "type": "string",
                "description": "Absolute local file path to the video.",
            },
            "question": {
                "type": "string",
                "description": (
                    "Your question or instruction about the video. "
                    "The AI will describe the video content and answer this."
                ),
            },
        },
        "required": ["video_path", "question"],
    },
}


def _handle_video_analyze(args: Dict[str, Any], **kw: Any) -> Awaitable[str]:
    video_path = args.get("video_path", "")
    question = args.get("question", "")
    full_prompt = (
        "You are analyzing frames extracted from a video. "
        "Describe what happens in the video, then answer the following "
        f"question:\n\n{question}"
    )
    model = os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None
    return video_analyze_tool(video_path, full_prompt, model)


registry.register(
    name="video_analyze",
    toolset="vision",
    schema=VIDEO_ANALYZE_SCHEMA,
    handler=_handle_video_analyze,
    check_fn=check_video_requirements,
    is_async=True,
    emoji="\U0001f3ac",
)
