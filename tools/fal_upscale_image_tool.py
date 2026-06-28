#!/usr/bin/env python3
"""
FAL Image Upscale Tool

Upscales images using SeedVR2 via FAL.ai.
"""

import json
import logging
from typing import Any, Dict, Optional

from tools.image_generation_tool import _submit_fal_request, check_fal_api_key
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

SEEDVR_UPSCALE_IMAGE_MODEL = "fal-ai/seedvr/upscale/image"

VALID_MODES = ("target", "factor")
VALID_TARGET_RESOLUTIONS = ("720p", "1080p", "1440p", "2160p")
VALID_OUTPUT_FORMATS = ("png", "jpg", "webp")


def upscale_image_tool(
    image_url: str,
    upscale_mode: str = "factor",
    upscale_factor: float = 2.0,
    target_resolution: str = "1080p",
    noise_scale: float = 0.1,
    output_format: str = "jpg",
    seed: Optional[int] = None,
    sync_mode: bool = False,
) -> str:
    """Upscale an image using SeedVR2.

    Either ``upscale_factor`` (when mode='factor') or ``target_resolution``
    (when mode='target') is used by the API.
    """
    response_data: Dict[str, Any] = {
        "success": False,
        "image": None,
        "error": None,
        "error_type": None,
    }

    try:
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            raise ValueError("image_url is required")

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
        if output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format '{output_format}'. "
                f"Choose one of: {', '.join(VALID_OUTPUT_FORMATS)}"
            )
        if not (1.0 <= upscale_factor <= 10.0):
            raise ValueError("upscale_factor must be between 1.0 and 10.0")
        if not (0.0 <= noise_scale <= 1.0):
            raise ValueError("noise_scale must be between 0.0 and 1.0")

        arguments: Dict[str, Any] = {
            "image_url": image_url.strip(),
            "upscale_mode": upscale_mode,
            "upscale_factor": upscale_factor,
            "target_resolution": target_resolution,
            "noise_scale": noise_scale,
            "output_format": output_format,
            "sync_mode": sync_mode,
        }
        if seed is not None:
            arguments["seed"] = seed

        logger.info("Upscaling image %s with SeedVR2", image_url)
        handler = _submit_fal_request(SEEDVR_UPSCALE_IMAGE_MODEL, arguments=arguments)
        result = handler.get()

        if not result or "image" not in result:
            raise ValueError("Invalid response from FAL.ai API — no image returned")

        img = result["image"]
        response_data = {
            "success": True,
            "image": img.get("url"),
            "output_url": img.get("url"),
            "media_type": "image",
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "content_type": img.get("content_type"),
            "seed": result.get("seed"),
        }

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except Exception as exc:
        logger.error("Error upscaling image: %s", exc, exc_info=True)
        response_data["error"] = str(exc)
        response_data["error_type"] = type(exc).__name__
        return json.dumps(response_data, indent=2, ensure_ascii=False)


def _handle_upscale_image(args: Dict[str, Any], **kw) -> str:
    image_url = args.get("image_url")
    if not image_url:
        return tool_error("image_url is required")

    return upscale_image_tool(
        image_url=image_url,
        upscale_mode=args.get("upscale_mode", "factor"),
        upscale_factor=args.get("upscale_factor", 2.0),
        target_resolution=args.get("target_resolution", "1080p"),
        noise_scale=args.get("noise_scale", 0.1),
        output_format=args.get("output_format", "jpg"),
        seed=args.get("seed"),
        sync_mode=args.get("sync_mode", False),
    )


UPSCALE_IMAGE_SCHEMA = {
    "name": "upscale_image",
    "description": (
        "Upscale an image using FAL.ai SeedVR2. Supports factor-based or "
        "target-resolution upscaling."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "URL or absolute path of the image to upscale.",
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
                "description": "Output image format.",
                "default": "jpg",
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
        "required": ["image_url"],
    },
}


registry.register(
    name="upscale_image",
    toolset="image_upscale",
    schema=UPSCALE_IMAGE_SCHEMA,
    handler=_handle_upscale_image,
    check_fn=check_fal_api_key,
    requires_env=[],
    is_async=False,
    emoji="💎",
)
