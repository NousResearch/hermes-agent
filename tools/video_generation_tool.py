#!/usr/bin/env python3
"""
Video Generation Tools Module

Generates short MP4 videos from text prompts using FAL.ai video models.
Follows the same pattern as image_generation_tool.py.

Available tools:
- video_generate_tool: Generate video from a text prompt

Supported models:
- kling  → fal-ai/kling-video/v2/master/text-to-video  (default)
- luma   → fal-ai/luma-dream-machine
- minimax → fal-ai/minimax/video-01-live

Output: local MP4 file path (downloaded from FAL.ai CDN).

Usage:
    from tools.video_generation_tool import video_generate_tool

    result = video_generate_tool(
        prompt="A cat walking through a neon-lit Tokyo street at night",
        model="kling",
        duration=5,
        aspect_ratio="landscape",
    )
"""

import json
import logging
import os
import tempfile
import datetime
import urllib.request
from typing import Optional

import fal_client
from tools.debug_helpers import DebugSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "kling":   "fal-ai/kling-video/v2/master/text-to-video",
    "luma":    "fal-ai/luma-dream-machine",
    "minimax": "fal-ai/minimax/video-01-live",
}

DEFAULT_MODEL_KEY = "kling"

# ---------------------------------------------------------------------------
# Parameter constants
# ---------------------------------------------------------------------------

# Aspect ratio display → API value mappings per model family
_KLING_ASPECT = {
    "landscape": "16:9",
    "portrait":  "9:16",
    "square":    "1:1",
}

_LUMA_ASPECT = {
    "landscape": "16:9",
    "portrait":  "9:16",
    "square":    "1:1",
}

# MiniMax video-01-live accepts prompt only (no aspect_ratio param)

VALID_DURATIONS = [5, 10]          # seconds; Kling supports both, others default to 5
VALID_ASPECT_RATIOS = ["landscape", "portrait", "square"]
DEFAULT_ASPECT_RATIO = "landscape"
DEFAULT_DURATION = 5

# ---------------------------------------------------------------------------
# Debug session
# ---------------------------------------------------------------------------

_debug = DebugSession("video_tools", env_var="VIDEO_TOOLS_DEBUG")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_arguments(model_key: str, prompt: str, duration: int, aspect_ratio: str,
                     negative_prompt: Optional[str]) -> dict:
    """Build the FAL.ai arguments dict for the given model."""
    args: dict = {"prompt": prompt.strip()}

    if model_key == "kling":
        args["duration"] = str(duration)          # Kling expects string "5" / "10"
        args["aspect_ratio"] = _KLING_ASPECT[aspect_ratio]
        if negative_prompt:
            args["negative_prompt"] = negative_prompt

    elif model_key == "luma":
        args["aspect_ratio"] = _LUMA_ASPECT[aspect_ratio]
        args["loop"] = False
        # Luma doesn't support duration parameter

    elif model_key == "minimax":
        pass  # minimax/video-01-live only needs prompt

    return args


def _extract_video_url(result: dict) -> Optional[str]:
    """
    Extract video URL from FAL.ai response.
    Models return either {"video": {"url": ...}} or {"videos": [{"url": ...}]}.
    """
    if not result:
        return None

    # Single video object
    if "video" in result:
        v = result["video"]
        return v.get("url") if isinstance(v, dict) else None

    # List of videos
    if "videos" in result:
        videos = result["videos"]
        if videos and isinstance(videos[0], dict):
            return videos[0].get("url")

    return None


def _download_video(url: str) -> str:
    """Download video from URL to a temp file. Returns local file path."""
    suffix = ".mp4"
    # Preserve extension if present in URL
    url_path = url.split("?")[0]
    if "." in url_path.split("/")[-1]:
        ext = "." + url_path.split("/")[-1].rsplit(".", 1)[-1]
        if ext.lower() in (".mp4", ".mov", ".webm", ".avi"):
            suffix = ext

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="hermes_video_")
    tmp_path = tmp.name
    tmp.close()

    logger.info("Downloading video from FAL.ai CDN → %s", tmp_path)
    urllib.request.urlretrieve(url, tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    logger.info("Downloaded %.1f MB", size_mb)
    return tmp_path


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def video_generate_tool(
    prompt: str,
    model: str = DEFAULT_MODEL_KEY,
    duration: int = DEFAULT_DURATION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    negative_prompt: Optional[str] = None,
) -> str:
    """
    Generate a short video from a text prompt using FAL.ai.

    Uses fal_client.submit() (sync) to avoid event-loop lifecycle issues
    in the gateway thread-pool pattern — same reason as image_generation_tool.

    Args:
        prompt:          Text description of the desired video.
        model:           "kling" (default), "luma", or "minimax".
        duration:        Video length in seconds — 5 (default) or 10.
                         Only Kling supports 10s; others use 5s.
        aspect_ratio:    "landscape" (16:9, default), "portrait" (9:16), "square" (1:1).
        negative_prompt: Things to avoid in the video (Kling only).

    Returns:
        JSON string:
        {
            "success": bool,
            "video_path": str | null,   # local temp file path, ready for send_message
            "video_url":  str | null,   # original CDN URL
            "model": str,
            "duration_seconds": int
        }
    """
    start = datetime.datetime.now()
    debug_data: dict = {
        "prompt": prompt,
        "model": model,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "error": None,
        "success": False,
    }

    def _fail(msg: str) -> str:
        logger.error(msg)
        debug_data["error"] = msg
        _debug.log_call("video_generate_tool", debug_data)
        _debug.save()
        return json.dumps({"success": False, "video_path": None, "video_url": None,
                           "model": model, "duration_seconds": duration})

    # --- Validation ---
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return _fail("prompt is required and must be a non-empty string")

    if not os.getenv("FAL_KEY"):
        return _fail("FAL_KEY environment variable not set")

    model_key = model.lower().strip()
    if model_key not in MODEL_MAP:
        return _fail(f"Unknown model '{model}'. Valid choices: {list(MODEL_MAP.keys())}")

    if duration not in VALID_DURATIONS:
        logger.warning("Invalid duration %s, defaulting to %s", duration, DEFAULT_DURATION)
        duration = DEFAULT_DURATION

    aspect_key = aspect_ratio.lower().strip()
    if aspect_key not in VALID_ASPECT_RATIOS:
        logger.warning("Invalid aspect_ratio '%s', defaulting to '%s'", aspect_ratio, DEFAULT_ASPECT_RATIO)
        aspect_key = DEFAULT_ASPECT_RATIO

    fal_model_id = MODEL_MAP[model_key]
    arguments = _build_arguments(model_key, prompt, duration, aspect_key, negative_prompt)

    logger.info("Generating video | model=%s | duration=%ss | aspect=%s", model_key, duration, aspect_key)
    logger.info("Prompt: %s", prompt[:120])
    logger.info("FAL model ID: %s", fal_model_id)

    try:
        # Sync fal_client — same pattern as image_generation_tool.py.
        # submit_async() caches a global httpx.AsyncClient via @cached_property
        # which breaks when asyncio.run() destroys the event loop between calls
        # (gateway thread-pool pattern). submit() uses httpx.Client (no loop).
        handler = fal_client.submit(fal_model_id, arguments=arguments)
        result = handler.get()

        elapsed = (datetime.datetime.now() - start).total_seconds()
        logger.info("FAL.ai responded in %.1fs", elapsed)

        video_url = _extract_video_url(result)
        if not video_url:
            return _fail(f"FAL.ai returned no video URL. Raw response keys: {list(result.keys()) if result else 'empty'}")

        # Download to local temp file so send_message_tool can attach it directly
        video_path = _download_video(video_url)

        total_elapsed = (datetime.datetime.now() - start).total_seconds()
        logger.info("Video ready in %.1fs total | path=%s", total_elapsed, video_path)

        debug_data.update({"success": True, "video_url": video_url, "video_path": video_path})
        _debug.log_call("video_generate_tool", debug_data)
        _debug.save()

        return json.dumps({
            "success": True,
            "video_path": video_path,
            "video_url": video_url,
            "model": model_key,
            "duration_seconds": duration,
        }, indent=2)

    except Exception as exc:
        return _fail(f"Error generating video: {exc}")


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def check_video_generation_requirements() -> bool:
    """Return True if FAL_KEY is set and fal_client is importable."""
    try:
        if not os.getenv("FAL_KEY"):
            return False
        import fal_client  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402

VIDEO_GENERATE_SCHEMA = {
    "name": "video_generate",
    "description": (
        "Generate a short MP4 video from a text prompt using FAL.ai AI video models. "
        "Returns a local file path — pass it to send_message to deliver the video. "
        "Generation takes 30–120 seconds. "
        "Models: 'kling' (default, best quality, supports 10s), 'luma' (cinematic), 'minimax' (fast)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed text description of the video to generate.",
            },
            "model": {
                "type": "string",
                "enum": ["kling", "luma", "minimax"],
                "description": "Video generation model. Default: 'kling'.",
                "default": "kling",
            },
            "duration": {
                "type": "integer",
                "enum": [5, 10],
                "description": "Video length in seconds. 10s only supported by kling. Default: 5.",
                "default": 5,
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["landscape", "portrait", "square"],
                "description": "Video aspect ratio — landscape (16:9), portrait (9:16), square (1:1). Default: landscape.",
                "default": "landscape",
            },
            "negative_prompt": {
                "type": "string",
                "description": "Things to avoid in the video (supported by kling only). Optional.",
            },
        },
        "required": ["prompt"],
    },
}


def _handle_video_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "prompt is required for video generation"})
    return video_generate_tool(
        prompt=prompt,
        model=args.get("model", DEFAULT_MODEL_KEY),
        duration=int(args.get("duration", DEFAULT_DURATION)),
        aspect_ratio=args.get("aspect_ratio", DEFAULT_ASPECT_RATIO),
        negative_prompt=args.get("negative_prompt"),
    )


registry.register(
    name="video_generate",
    toolset="video_gen",
    schema=VIDEO_GENERATE_SCHEMA,
    handler=_handle_video_generate,
    check_fn=check_video_generation_requirements,
    requires_env=["FAL_KEY"],
    is_async=False,
    emoji="🎬",
)
