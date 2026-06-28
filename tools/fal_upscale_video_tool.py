#!/usr/bin/env python3
"""
FAL Video Upscale Tool

Upscales videos using SeedVR2 Video via FAL.ai.

Note: the public documentation snippet for the video endpoint was incomplete,
so this implementation uses the standard SeedVR2 video-upscale shape
(video_url + upscale_factor / target_resolution) and the canonical
``fal-ai/seedvr/upscale/video`` endpoint. Verify against the live FAL schema
if you hit parameter errors.
"""

import json
import logging
from typing import Any, Dict, Optional

from tools.image_generation_tool import _submit_fal_request, check_fal_api_key
from tools.fal_video_input import resolve_fal_video_input
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

SEEDVR_UPSCALE_VIDEO_MODEL = "fal-ai/seedvr/upscale/video"

VALID_MODES = ("target", "factor")
VALID_TARGET_RESOLUTIONS = ("720p", "1080p", "1440p", "2160p")
VALID_OUTPUT_FORMATS = ("X264 (.mp4)", "VP9 (.webm)", "PRORES4444 (.mov)", "GIF (.gif)")
VALID_OUTPUT_QUALITIES = ("low", "medium", "high", "maximum")
VALID_OUTPUT_WRITE_MODES = ("fast", "balanced", "small")


def _copy_present_video_metadata(response: Dict[str, Any], video: Dict[str, Any]) -> None:
    for key in ("width", "height", "fps", "duration", "num_frames"):
        value = video.get(key)
        if value not in (None, "", 0):
            response[key] = value
    if video.get("content_type"):
        response["content_type"] = video.get("content_type")


def upscale_video_tool(
    video_url: str,
    upscale_mode: str = "factor",
    upscale_factor: float = 2.0,
    target_resolution: str = "1080p",
    noise_scale: float = 0.1,
    output_format: str = "X264 (.mp4)",
    output_quality: str = "high",
    output_write_mode: str = "balanced",
    seed: Optional[int] = None,
    sync_mode: bool = False,
) -> str:
    """Upscale a video using SeedVR2 Video."""
    response_data: Dict[str, Any] = {
        "success": False,
        "video": None,
        "error": None,
        "error_type": None,
    }

    try:
        resolved_video_url = resolve_fal_video_input(
            video_url,
            image_error=(
                "video_url must reference a video file, not an image. "
                "Use image-to-video generation first, then pass the resulting video URL."
            ),
        )

        if not check_fal_api_key():
            raise ValueError(
                "FAL_KEY is not set and no managed FAL gateway is available."
            )

        if upscale_mode not in VALID_MODES:
            raise ValueError(
                f"Invalid upscale_mode '{upscale_mode}'. Choose one of: {', '.join(VALID_MODES)}"
            )
        if target_resolution not in VALID_TARGET_RESOLUTIONS:
            raise ValueError(
                f"Invalid target_resolution '{target_resolution}'. "
                f"Choose one of: {', '.join(VALID_TARGET_RESOLUTIONS)}"
            )
        if not (1.0 <= upscale_factor <= 10.0):
            raise ValueError("upscale_factor must be between 1.0 and 10.0")
        if not (0.0 <= noise_scale <= 1.0):
            raise ValueError("noise_scale must be between 0.0 and 1.0")
        if output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format '{output_format}'. Choose one of: {', '.join(VALID_OUTPUT_FORMATS)}"
            )
        if output_quality not in VALID_OUTPUT_QUALITIES:
            raise ValueError(
                f"Invalid output_quality '{output_quality}'. Choose one of: {', '.join(VALID_OUTPUT_QUALITIES)}"
            )
        if output_write_mode not in VALID_OUTPUT_WRITE_MODES:
            raise ValueError(
                f"Invalid output_write_mode '{output_write_mode}'. "
                f"Choose one of: {', '.join(VALID_OUTPUT_WRITE_MODES)}"
            )

        arguments: Dict[str, Any] = {
            "video_url": resolved_video_url,
            "upscale_mode": upscale_mode,
            "upscale_factor": upscale_factor,
            "target_resolution": target_resolution,
            "noise_scale": noise_scale,
            "output_format": output_format,
            "output_quality": output_quality,
            "output_write_mode": output_write_mode,
            "sync_mode": sync_mode,
        }
        if seed is not None:
            arguments["seed"] = seed

        logger.info("Upscaling video %s with SeedVR2", resolved_video_url)
        handler = _submit_fal_request(SEEDVR_UPSCALE_VIDEO_MODEL, arguments=arguments)
        result = handler.get()

        if not result or "video" not in result:
            raise ValueError("Invalid response from FAL.ai API — no video returned")

        video = result["video"]
        response_data = {
            "success": True,
            "video": video.get("url"),
            "output_url": video.get("url"),
            "media_type": "video",
            "seed": result.get("seed"),
        }
        _copy_present_video_metadata(response_data, video)

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except Exception as exc:
        logger.error("Error upscaling video: %s", exc, exc_info=True)
        response_data["error"] = str(exc)
        response_data["error_type"] = type(exc).__name__
        return json.dumps(response_data, indent=2, ensure_ascii=False)


def _handle_upscale_video(args: Dict[str, Any], **kw) -> str:
    video_url = args.get("video_url")
    if not video_url:
        return tool_error("video_url is required")

    return upscale_video_tool(
        video_url=video_url,
        upscale_mode=args.get("upscale_mode", "factor"),
        upscale_factor=args.get("upscale_factor", 2.0),
        target_resolution=args.get("target_resolution", "1080p"),
        noise_scale=args.get("noise_scale", 0.1),
        output_format=args.get("output_format", "X264 (.mp4)"),
        output_quality=args.get("output_quality", "high"),
        output_write_mode=args.get("output_write_mode", "balanced"),
        seed=args.get("seed"),
        sync_mode=args.get("sync_mode", False),
    )


UPSCALE_VIDEO_SCHEMA = {
    "name": "upscale_video",
    "description": (
        "Upscale a video using FAL.ai SeedVR2 Video. Supports factor-based "
        "or target-resolution upscaling."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video_url": {
                "type": "string",
                "description": "URL or absolute path of the video to upscale.",
            },
            "upscale_mode": {
                "type": "string",
                "enum": list(VALID_MODES),
                "description": "'factor' multiplies dimensions; 'target' uses target_resolution.",
                "default": "factor",
            },
            "upscale_factor": {
                "type": "number",
                "description": "Multiplier when upscale_mode='factor' (1.0-10.0).",
                "default": 2.0,
            },
            "target_resolution": {
                "type": "string",
                "enum": list(VALID_TARGET_RESOLUTIONS),
                "description": "Target resolution when upscale_mode='target'.",
                "default": "1080p",
            },
            "noise_scale": {
                "type": "number",
                "description": "Noise scale for the generation process (0.0-1.0).",
                "default": 0.1,
            },
            "output_format": {
                "type": "string",
                "enum": list(VALID_OUTPUT_FORMATS),
                "description": "Container/codec of the output video.",
                "default": "X264 (.mp4)",
            },
            "output_quality": {
                "type": "string",
                "enum": list(VALID_OUTPUT_QUALITIES),
                "description": "Output video quality.",
                "default": "high",
            },
            "output_write_mode": {
                "type": "string",
                "enum": list(VALID_OUTPUT_WRITE_MODES),
                "description": "Trade-off between speed, size, and quality.",
                "default": "balanced",
            },
            "seed": {
                "type": "integer",
                "description": "Optional seed for reproducible output.",
            },
            "sync_mode": {
                "type": "boolean",
                "description": "Return media as a data URI and skip request history.",
                "default": False,
            },
        },
        "required": ["video_url"],
    },
}


registry.register(
    name="upscale_video",
    toolset="video_upscale",
    schema=UPSCALE_VIDEO_SCHEMA,
    handler=_handle_upscale_video,
    check_fn=check_fal_api_key,
    requires_env=[],
    is_async=False,
    emoji="📼",
)
