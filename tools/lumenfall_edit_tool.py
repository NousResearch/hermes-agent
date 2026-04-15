"""
Image Edit Tool — Lumenfall Provider

Edits images using Lumenfall's unified API, which routes to the best
available provider with automatic fallback.  Supports inpainting,
background removal, upscaling, and general edits.

Available tool:
  lumenfall_image_edit — Edit images via Lumenfall

Requires:
  LUMENFALL_API_KEY environment variable (get one at https://lumenfall.ai)

Optional:
  LUMENFALL_BASE_URL — Override the API base URL (default: https://api.lumenfall.ai/openai/v1)
"""

import json
import logging
import datetime
from typing import Optional

from tools.lumenfall_client import (
    check_lumenfall_available,
    edit_image,
    LumenfallAuthError,
    LumenfallBalanceError,
    LumenfallError,
)

logger = logging.getLogger(__name__)


def lumenfall_image_edit_tool(
    image_url: str,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    mask_url: Optional[str] = None,
    output_format: str = "png",
) -> str:
    """Edit an image using Lumenfall.

    Uses synchronous HTTP to avoid event-loop lifecycle issues in the
    gateway's thread-pool pattern (same rationale as the image gen tool).

    Args:
        image_url: URL of the source image to edit.
        prompt: Text description of the desired edit.
        model: Model to use (e.g. "gpt-image-1.5").  If not specified,
               the server picks the best available default.
        mask_url: Optional mask URL for inpainting (white = edit region).
        output_format: "png", "jpeg", or "webp".

    Returns:
        JSON string with {"success": bool, "image": str|null, ...}
    """
    start_time = datetime.datetime.now()

    try:
        # Validate image_url
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            raise ValueError("image_url is required and must be a non-empty string")

        # Validate output_format
        if output_format not in ("png", "jpeg", "webp"):
            output_format = "png"

        logger.info(
            "Lumenfall image edit: model=%s, prompt=%.80s",
            model or "(default)",
            prompt or "(none)",
        )

        # Call Lumenfall API
        result = edit_image(
            image_url=image_url,
            prompt=prompt,
            model=model,
            mask_url=mask_url,
            output_format=output_format,
            response_format="url",
        )

        generation_time = (datetime.datetime.now() - start_time).total_seconds()

        # Extract image URLs from response
        data = result.get("data", [])
        if not data:
            raise ValueError("No images returned from Lumenfall API")

        images = []
        for item in data:
            url = item.get("url")
            if url:
                images.append(url)

        if not images:
            raise ValueError("No valid image URLs in response")

        # Extract metadata for logging
        metadata = result.get("metadata", {})
        provider = metadata.get("provider_name", "unknown")
        executed_model = metadata.get("executed_model", model or "default")
        cost = metadata.get("cost")

        logger.info(
            "Edited image in %.1fs via %s (model=%s, cost=%s)",
            generation_time,
            provider,
            executed_model,
            f"${cost:.4f}" if cost else "n/a",
        )

        # Return in the same minimal format as the existing image tools
        response_data = {
            "success": True,
            "image": images[0],
        }
        # Include extra images if multiple were requested
        if len(images) > 1:
            response_data["images"] = images

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except LumenfallAuthError as e:
        logger.error("Lumenfall auth error: %s", e)
        return json.dumps({
            "success": False,
            "image": None,
            "error": str(e),
            "error_type": "auth_error",
        }, indent=2)

    except LumenfallBalanceError as e:
        logger.error("Lumenfall balance error: %s", e)
        return json.dumps({
            "success": False,
            "image": None,
            "error": str(e),
            "error_type": "balance_error",
        }, indent=2)

    except (LumenfallError, ValueError) as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.error("Image edit error: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "image": None,
            "error": str(e),
            "error_type": type(e).__name__,
        }, indent=2)

    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.error("Unexpected error in image edit: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "image": None,
            "error": str(e),
            "error_type": type(e).__name__,
        }, indent=2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error  # noqa: E402

LUMENFALL_IMAGE_EDIT_SCHEMA = {
    "name": "lumenfall_image_edit",
    "description": (
        "Edit images using Lumenfall. Supports inpainting, background removal, "
        "upscaling, and general edits. Provide a source image URL and an optional "
        "prompt describing the desired edit. "
        "Returns an image URL. Display it using markdown: ![description](URL)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": (
                    "URL of the source image to edit. Must be a publicly "
                    "accessible URL (PNG, JPEG, or WebP)."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Text description of the desired edit. "
                    "Describe what to change, add, or remove."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Image edit model to use. "
                    "Leave empty for the best available default."
                ),
            },
            "mask_url": {
                "type": "string",
                "description": (
                    "URL of a mask image for inpainting. White regions "
                    "indicate areas to edit; black regions are preserved. "
                    "Must be the same dimensions as the source image."
                ),
            },
        },
        "required": ["image_url"],
    },
}


def _handle_lumenfall_image_edit(args, **kw):
    image_url = args.get("image_url", "")
    if not image_url:
        return tool_error("image_url is required for image editing")
    return lumenfall_image_edit_tool(
        image_url=image_url,
        prompt=args.get("prompt"),
        model=args.get("model"),
        mask_url=args.get("mask_url"),
        output_format=args.get("output_format", "png"),
    )


registry.register(
    name="lumenfall_image_edit",
    toolset="lumenfall",
    schema=LUMENFALL_IMAGE_EDIT_SCHEMA,
    handler=_handle_lumenfall_image_edit,
    check_fn=check_lumenfall_available,
    requires_env=[],
    is_async=False,
    emoji="✏️",
)
