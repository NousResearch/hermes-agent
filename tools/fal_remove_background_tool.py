#!/usr/bin/env python3
"""
FAL Background Removal Tool

Removes backgrounds from images using Ideogram Remove Background or
BiRefNet V2 via FAL.ai.
"""

import json
import logging
from typing import Any, Dict

from tools.image_generation_tool import _submit_fal_request, check_fal_api_key
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

IDEOGRAM_REMOVE_BACKGROUND_MODEL = "fal-ai/ideogram/remove-background"
BIREFNET_MODEL = "fal-ai/birefnet/v2"
IDEOGRAM_MODEL = "Ideogram Remove Background"

BIREFNET_MODELS = (
    "General Use (Light)",
    "General Use (Light 2K)",
    "General Use (Heavy)",
    "Matting",
    "Portrait",
    "General Use (Dynamic)",
)

VALID_MODELS = (IDEOGRAM_MODEL, *BIREFNET_MODELS)
VALID_RESOLUTIONS = ("1024x1024", "2048x2048", "2304x2304")

VALID_OUTPUT_FORMATS = ("webp", "png", "gif")


def remove_background_tool(
    image_url: str,
    model: str = IDEOGRAM_MODEL,
    operating_resolution: str = "1024x1024",
    output_format: str = "png",
    output_mask: bool = False,
    mask_only: bool = False,
    refine_foreground: bool = True,
    sync_mode: bool = False,
) -> str:
    """Remove the background from an image using Ideogram or BiRefNet V2.

    Returns a JSON string with ``success``, ``image`` (URL), and optionally
    ``mask_image``.
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

        if model not in VALID_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. Choose one of: {', '.join(VALID_MODELS)}"
            )

        if model == IDEOGRAM_MODEL:
            arguments = {
                "image_url": image_url.strip(),
                "sync_mode": sync_mode,
            }
            endpoint = IDEOGRAM_REMOVE_BACKGROUND_MODEL
            logger.info("Removing background from %s with Ideogram", image_url)
        else:
            if operating_resolution not in VALID_RESOLUTIONS:
                raise ValueError(
                    f"Invalid operating_resolution '{operating_resolution}'. "
                    f"Choose one of: {', '.join(VALID_RESOLUTIONS)}"
                )

            if output_format not in VALID_OUTPUT_FORMATS:
                raise ValueError(
                    f"Invalid output_format '{output_format}'. "
                    f"Choose one of: {', '.join(VALID_OUTPUT_FORMATS)}"
                )

            # 2304x2304 is only valid for the Dynamic model.
            if operating_resolution == "2304x2304" and model != "General Use (Dynamic)":
                raise ValueError(
                    "Resolution 2304x2304 is only available for the "
                    "'General Use (Dynamic)' model."
                )

            arguments = {
                "image_url": image_url.strip(),
                "model": model,
                "operating_resolution": operating_resolution,
                "output_format": output_format,
                "output_mask": output_mask,
                "mask_only": mask_only,
                "refine_foreground": refine_foreground,
                "sync_mode": sync_mode,
            }
            endpoint = BIREFNET_MODEL
            logger.info("Removing background from %s with BiRefNet V2", image_url)

        handler = _submit_fal_request(endpoint, arguments=arguments)
        result = handler.get()

        if not result or "image" not in result:
            raise ValueError("Invalid response from FAL.ai API — no image returned")

        img = result["image"]
        response_data = {
            "success": True,
            "image": img.get("url"),
            "output_url": img.get("url"),
            "media_type": "image",
            "model": model,
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "content_type": img.get("content_type"),
        }

        if result.get("mask_image"):
            mask = result["mask_image"]
            response_data["mask_image"] = {
                "url": mask.get("url"),
                "width": mask.get("width", 0),
                "height": mask.get("height", 0),
                "content_type": mask.get("content_type"),
            }

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except Exception as exc:
        logger.error("Error removing background: %s", exc, exc_info=True)
        response_data["error"] = str(exc)
        response_data["error_type"] = type(exc).__name__
        return json.dumps(response_data, indent=2, ensure_ascii=False)


def _handle_remove_background(args: Dict[str, Any], **kw) -> str:
    image_url = args.get("image_url")
    if not image_url:
        return tool_error("image_url is required")

    return remove_background_tool(
        image_url=image_url,
        model=args.get("model", IDEOGRAM_MODEL),
        operating_resolution=args.get("operating_resolution", "1024x1024"),
        output_format=args.get("output_format", "png"),
        output_mask=args.get("output_mask", False),
        mask_only=args.get("mask_only", False),
        refine_foreground=args.get("refine_foreground", True),
        sync_mode=args.get("sync_mode", False),
    )


REMOVE_BACKGROUND_SCHEMA = {
    "name": "remove_background",
    "description": (
        "Remove the background from an image using FAL.ai Ideogram Remove "
        "Background by default, or BiRefNet V2 variants when selected. "
        "Returns the foreground image (and optionally a mask for BiRefNet)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "URL or absolute path of the image to process.",
            },
            "model": {
                "type": "string",
                "enum": list(VALID_MODELS),
                "description": (
                    "Background-removal model. Ideogram is the default; "
                    "BiRefNet variants remain available for mask output and "
                    "resolution/model controls."
                ),
                "default": IDEOGRAM_MODEL,
            },
            "operating_resolution": {
                "type": "string",
                "enum": list(VALID_RESOLUTIONS),
                "description": "Processing resolution. 2304x2304 requires 'General Use (Dynamic)'.",
                "default": "1024x1024",
            },
            "output_format": {
                "type": "string",
                "enum": list(VALID_OUTPUT_FORMATS),
                "description": "Output image format.",
                "default": "png",
            },
            "output_mask": {
                "type": "boolean",
                "description": "Also return the segmentation mask used to remove the background.",
                "default": False,
            },
            "mask_only": {
                "type": "boolean",
                "description": "Return only the segmentation mask without applying it.",
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
        "required": ["image_url"],
    },
}


registry.register(
    name="remove_background",
    toolset="image_background_removal",
    schema=REMOVE_BACKGROUND_SCHEMA,
    handler=_handle_remove_background,
    check_fn=check_fal_api_key,
    requires_env=[],
    is_async=False,
    emoji="🧵",
)
