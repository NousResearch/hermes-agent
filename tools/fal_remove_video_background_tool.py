#!/usr/bin/env python3
"""
FAL Video Background Removal Tool

Removes backgrounds from videos using BiRefNet V2 Video via FAL.ai.
"""

import json
import logging
from typing import Any, Dict

from tools.image_generation_tool import _submit_fal_request, check_fal_api_key
from tools.fal_video_input import resolve_fal_video_input
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

BIREFNET_VIDEO_MODEL = "fal-ai/birefnet/v2/video"

VALID_MODELS = (
    "General Use (Light)",
    "General Use (Light 2K)",
    "General Use (Heavy)",
    "Matting",
    "Portrait",
    "General Use (Dynamic)",
)

VALID_RESOLUTIONS = ("1024x1024", "2048x2048", "2304x2304")
VALID_OUTPUT_TYPES = ("X264 (.mp4)", "VP9 (.webm)", "PRORES4444 (.mov)", "GIF (.gif)")
VALID_QUALITIES = ("low", "medium", "high", "maximum")
VALID_WRITE_MODES = ("fast", "balanced", "small")


def _copy_present_video_metadata(response: Dict[str, Any], video: Dict[str, Any]) -> None:
    for key in ("width", "height", "fps", "duration", "num_frames"):
        value = video.get(key)
        if value not in (None, "", 0):
            response[key] = value
    if video.get("content_type"):
        response["content_type"] = video.get("content_type")


def remove_video_background_tool(
    video_url: str,
    model: str = "General Use (Light)",
    operating_resolution: str = "1024x1024",
    video_output_type: str = "X264 (.mp4)",
    video_quality: str = "high",
    video_write_mode: str = "balanced",
    output_mask: bool = False,
    refine_foreground: bool = True,
    sync_mode: bool = False,
) -> str:
    """Remove the background from a video using BiRefNet V2 Video."""
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
                "Use remove_background for still images."
            ),
        )

        if not check_fal_api_key():
            raise ValueError(
                "FAL_KEY is not set and no managed FAL gateway is available."
            )

        if model not in VALID_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. Choose one of: {', '.join(VALID_MODELS)}"
            )
        if operating_resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid operating_resolution '{operating_resolution}'. "
                f"Choose one of: {', '.join(VALID_RESOLUTIONS)}"
            )
        if video_output_type not in VALID_OUTPUT_TYPES:
            raise ValueError(
                f"Invalid video_output_type '{video_output_type}'. "
                f"Choose one of: {', '.join(VALID_OUTPUT_TYPES)}"
            )
        if video_quality not in VALID_QUALITIES:
            raise ValueError(
                f"Invalid video_quality '{video_quality}'. "
                f"Choose one of: {', '.join(VALID_QUALITIES)}"
            )
        if video_write_mode not in VALID_WRITE_MODES:
            raise ValueError(
                f"Invalid video_write_mode '{video_write_mode}'. "
                f"Choose one of: {', '.join(VALID_WRITE_MODES)}"
            )
        if operating_resolution == "2304x2304" and model != "General Use (Dynamic)":
            raise ValueError(
                "Resolution 2304x2304 is only available for the "
                "'General Use (Dynamic)' model."
            )

        arguments: Dict[str, Any] = {
            "video_url": resolved_video_url,
            "model": model,
            "operating_resolution": operating_resolution,
            "video_output_type": video_output_type,
            "video_quality": video_quality,
            "video_write_mode": video_write_mode,
            "output_mask": output_mask,
            "refine_foreground": refine_foreground,
            "sync_mode": sync_mode,
        }

        logger.info("Removing background from video %s with BiRefNet V2", resolved_video_url)
        handler = _submit_fal_request(BIREFNET_VIDEO_MODEL, arguments=arguments)
        result = handler.get()

        if not result or "video" not in result:
            raise ValueError("Invalid response from FAL.ai API — no video returned")

        video = result["video"]
        response_data = {
            "success": True,
            "video": video.get("url"),
            "output_url": video.get("url"),
            "media_type": "video",
        }
        _copy_present_video_metadata(response_data, video)

        if result.get("mask_video"):
            mask = result["mask_video"]
            response_data["mask_video"] = {
                "url": mask.get("url"),
            }
            _copy_present_video_metadata(response_data["mask_video"], mask)

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except Exception as exc:
        logger.error("Error removing video background: %s", exc, exc_info=True)
        response_data["error"] = str(exc)
        response_data["error_type"] = type(exc).__name__
        return json.dumps(response_data, indent=2, ensure_ascii=False)


def _handle_remove_video_background(args: Dict[str, Any], **kw) -> str:
    video_url = args.get("video_url")
    if not video_url:
        return tool_error("video_url is required")

    return remove_video_background_tool(
        video_url=video_url,
        model=args.get("model", "General Use (Light)"),
        operating_resolution=args.get("operating_resolution", "1024x1024"),
        video_output_type=args.get("video_output_type", "X264 (.mp4)"),
        video_quality=args.get("video_quality", "high"),
        video_write_mode=args.get("video_write_mode", "balanced"),
        output_mask=args.get("output_mask", False),
        refine_foreground=args.get("refine_foreground", True),
        sync_mode=args.get("sync_mode", False),
    )


REMOVE_VIDEO_BACKGROUND_SCHEMA = {
    "name": "remove_video_background",
    "description": (
        "Remove the background from a video using FAL.ai BiRefNet V2 Video. "
        "Returns the foreground video (and optionally a mask video)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "video_url": {
                "type": "string",
                "description": "URL or absolute path of the video to process.",
            },
            "model": {
                "type": "string",
                "enum": list(VALID_MODELS),
                "description": "BiRefNet model variant.",
                "default": "General Use (Light)",
            },
            "operating_resolution": {
                "type": "string",
                "enum": list(VALID_RESOLUTIONS),
                "description": "Processing resolution. 2304x2304 requires 'General Use (Dynamic)'.",
                "default": "1024x1024",
            },
            "video_output_type": {
                "type": "string",
                "enum": list(VALID_OUTPUT_TYPES),
                "description": "Container/codec of the output video.",
                "default": "X264 (.mp4)",
            },
            "video_quality": {
                "type": "string",
                "enum": list(VALID_QUALITIES),
                "description": "Output video quality.",
                "default": "high",
            },
            "video_write_mode": {
                "type": "string",
                "enum": list(VALID_WRITE_MODES),
                "description": "Trade-off between speed, size, and quality.",
                "default": "balanced",
            },
            "output_mask": {
                "type": "boolean",
                "description": "Also return the segmentation mask video.",
                "default": False,
            },
            "refine_foreground": {
                "type": "boolean",
                "description": "Refine the foreground using the estimated mask.",
                "default": True,
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
    name="remove_video_background",
    toolset="video_background_removal",
    schema=REMOVE_VIDEO_BACKGROUND_SCHEMA,
    handler=_handle_remove_video_background,
    check_fn=check_fal_api_key,
    requires_env=[],
    is_async=False,
    emoji="🎥",
)
