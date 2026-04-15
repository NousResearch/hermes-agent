"""
Image Generation Tool — Lumenfall Provider

Generates images using Lumenfall's unified API, which routes to the best
available provider (Vertex/Imagen, OpenAI/DALL-E, Black Forest Labs/FLUX,
Stability AI, and more) with automatic fallback.

This tool registers alongside the existing FAL-based image_generate tool.
When LUMENFALL_API_KEY is set, it becomes available as an alternative image
generation backend with multi-model support.

Available tool:
  lumenfall_image_generate — Generate images from text prompts via Lumenfall

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
    generate_image,
    LumenfallAuthError,
    LumenfallBalanceError,
    LumenfallError,
)

logger = logging.getLogger(__name__)

# Aspect ratio mapping — simplified choices for the model to select,
# mapped to ratio strings the Lumenfall API accepts.
ASPECT_RATIO_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


def lumenfall_image_generate_tool(
    prompt: str,
    model: Optional[str] = None,
    aspect_ratio: str = "landscape",
    num_images: int = 1,
    output_format: str = "png",
) -> str:
    """Generate images from text prompts using Lumenfall.

    Uses synchronous HTTP to avoid event-loop lifecycle issues in the
    gateway's thread-pool pattern (same rationale as the FAL tool).

    Args:
        prompt: Text prompt describing the desired image.
        model: Model to use (e.g. "gemini-3.1-flash-image-preview",
               "gpt-image-1.5", "flux.2-max"). If not specified, the server
               picks the best available default.
        aspect_ratio: "landscape" (16:9), "square" (1:1), or "portrait" (9:16).
        num_images: Number of images to generate (1-4).
        output_format: "png", "jpeg", or "webp".

    Returns:
        JSON string with {"success": bool, "image": str|null, ...}
    """
    start_time = datetime.datetime.now()

    try:
        # Validate prompt
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt is required and must be a non-empty string")

        # Validate aspect_ratio
        aspect_ratio_lower = aspect_ratio.lower().strip() if aspect_ratio else "landscape"
        if aspect_ratio_lower not in ASPECT_RATIO_MAP:
            logger.warning(
                "Invalid aspect_ratio '%s', defaulting to 'landscape'", aspect_ratio
            )
            aspect_ratio_lower = "landscape"

        ratio = ASPECT_RATIO_MAP[aspect_ratio_lower]

        # Clamp num_images
        num_images = max(1, min(4, int(num_images)))

        # Validate output_format
        if output_format not in ("png", "jpeg", "webp"):
            output_format = "png"

        logger.info(
            "Lumenfall image generation: model=%s, ratio=%s, n=%d, prompt=%.80s",
            model or "(default)",
            ratio,
            num_images,
            prompt,
        )

        # Call Lumenfall API
        result = generate_image(
            prompt=prompt,
            model=model,
            n=num_images,
            aspect_ratio=ratio,
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
            "Generated %d image(s) in %.1fs via %s (model=%s, cost=%s)",
            len(images),
            generation_time,
            provider,
            executed_model,
            f"${cost:.4f}" if cost else "n/a",
        )

        # Return in the same minimal format as the existing image_generate tool
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
        logger.error("Image generation error: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "image": None,
            "error": str(e),
            "error_type": type(e).__name__,
            "hint": "Use lumenfall_list_models to discover available models and their capabilities.",
        }, indent=2)

    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.error("Unexpected error in image generation: %s", e, exc_info=True)
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

LUMENFALL_IMAGE_GENERATE_SCHEMA = {
    "name": "lumenfall_image_generate",
    "description": (
        "Generate high-quality images from text prompts using Lumenfall. "
        "Supports many models across providers: FLUX, DALL-E, Imagen, "
        "Stable Diffusion, and more. Optionally specify a model or let the "
        "server pick the best default. Generation can take 5-30 seconds "
        "depending on the model — let the user know before calling. "
        "Returns an image URL. "
        "Display it using markdown: ![description](URL)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Text prompt describing the desired image. "
                    "Be detailed and descriptive for best results."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Image model to use. Examples: "
                    "gemini-3.1-flash-image-preview, gpt-image-1.5, "
                    "flux.2-max, gemini-3-pro-image-preview, qwen-image-2512. "
                    "Leave empty for the best available default."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["landscape", "square", "portrait"],
                "description": (
                    "Image aspect ratio. "
                    "'landscape' is 16:9 wide, "
                    "'portrait' is 9:16 tall, "
                    "'square' is 1:1."
                ),
                "default": "landscape",
            },
        },
        "required": ["prompt"],
    },
}


def _handle_lumenfall_image_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for image generation")
    return lumenfall_image_generate_tool(
        prompt=prompt,
        model=args.get("model"),
        aspect_ratio=args.get("aspect_ratio", "landscape"),
        num_images=1,
        output_format="png",
    )


registry.register(
    name="lumenfall_image_generate",
    toolset="lumenfall",
    schema=LUMENFALL_IMAGE_GENERATE_SCHEMA,
    handler=_handle_lumenfall_image_generate,
    check_fn=check_lumenfall_available,
    requires_env=[],
    is_async=False,
    emoji="🎨",
)
