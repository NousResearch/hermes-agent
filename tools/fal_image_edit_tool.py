#!/usr/bin/env python3
"""
FAL Image Edit Tool

Dedicated image-to-image / editing tool for FAL.ai backends.
Uses the active ``image_gen.model`` (default: Gemini 3.1 Flash Image Preview)
and routes to its declared ``edit_endpoint``.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from tools.image_generation_tool import (
    DEFAULT_ASPECT_RATIO,
    FAL_MODELS,
    VALID_ASPECT_RATIOS,
    _build_fal_edit_payload,
    _resolve_fal_model,
    _submit_fal_request,
    check_fal_api_key,
)
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


def _resolve_edit_model(model_id: Optional[str] = None) -> tuple:
    """Return the model id + metadata to use for editing.

    If ``model_id`` is provided and known, use it; otherwise fall back to the
    configured image_gen model. Raises ValueError if the chosen model has no
    edit endpoint.
    """
    if model_id and model_id.strip() in FAL_MODELS:
        chosen = model_id.strip()
        meta = FAL_MODELS[chosen]
    else:
        chosen, meta = _resolve_fal_model()

    if not meta.get("edit_endpoint"):
        raise ValueError(
            f"Model '{meta.get('display', chosen)}' ({chosen}) does not support "
            f"image editing. Choose an edit-capable model via `hermes tools` → "
            f"Image Generation."
        )

    return chosen, meta


def image_edit_tool(
    prompt: str,
    image_url: str,
    reference_image_urls: Optional[List[str]] = None,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    seed: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    """Edit or transform an existing image using the active FAL edit model.

    Required: ``prompt`` (edit instruction) and ``image_url`` (source image).
    Optional ``reference_image_urls`` for style / composition guidance.
    """
    response_data: Dict[str, Any] = {
        "success": False,
        "image": None,
        "error": None,
        "error_type": None,
    }

    try:
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required and must be a non-empty string")
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            raise ValueError("image_url is required for image editing")

        if not check_fal_api_key():
            raise ValueError(
                "FAL_KEY is not set and no managed FAL gateway is available. "
                "Set FAL_KEY or enable the Nous Subscription image_gen provider."
            )

        model_id, meta = _resolve_edit_model(model)
        edit_endpoint = meta["edit_endpoint"]

        aspect_lc = (aspect_ratio or DEFAULT_ASPECT_RATIO).lower().strip()
        if aspect_lc not in VALID_ASPECT_RATIOS:
            logger.warning(
                "Invalid aspect_ratio '%s', defaulting to '%s'",
                aspect_ratio, DEFAULT_ASPECT_RATIO,
            )
            aspect_lc = DEFAULT_ASPECT_RATIO

        sources = [image_url.strip()]
        if reference_image_urls:
            for ref in reference_image_urls:
                if isinstance(ref, str) and ref.strip():
                    sources.append(ref.strip())

        max_refs = int(meta.get("max_reference_images") or 1)
        clamped_sources = sources[:max_refs] if max_refs > 0 else sources

        arguments = _build_fal_edit_payload(
            model_id, prompt, clamped_sources, aspect_lc, seed=seed
        )

        logger.info(
            "Editing image with %s (%s) — %d source image(s), prompt: %s",
            meta.get("display", model_id), edit_endpoint, len(clamped_sources),
            prompt[:80],
        )

        handler = _submit_fal_request(edit_endpoint, arguments=arguments)
        result = handler.get()

        if not result or "images" not in result:
            raise ValueError("Invalid response from FAL.ai API — no images returned")

        images = result.get("images", [])
        if not images:
            raise ValueError("No edited images were returned")

        img = images[0]
        response_data = {
            "success": True,
            "image": img.get("url"),
            "output_url": img.get("url"),
            "media_type": "image",
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "modality": "image",
        }

        if len(images) > 1:
            response_data["images"] = [
                {"url": i.get("url"), "width": i.get("width", 0), "height": i.get("height", 0)}
                for i in images
            ]

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except Exception as exc:
        logger.error("Error editing image: %s", exc, exc_info=True)
        response_data["error"] = str(exc)
        response_data["error_type"] = type(exc).__name__
        return json.dumps(response_data, indent=2, ensure_ascii=False)


def _handle_image_edit(args: Dict[str, Any], **kw) -> str:
    prompt = args.get("prompt", "")
    image_url = args.get("image_url")
    reference_image_urls = args.get("reference_image_urls")
    aspect_ratio = args.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    seed = args.get("seed")
    model = args.get("model")

    if not prompt:
        return tool_error("prompt is required for image editing")
    if not image_url:
        return tool_error("image_url is required for image editing")

    return image_edit_tool(
        prompt=prompt,
        image_url=image_url,
        reference_image_urls=reference_image_urls,
        aspect_ratio=aspect_ratio,
        seed=seed,
        model=model,
    )


IMAGE_EDIT_SCHEMA = {
    "name": "image_edit",
    "description": (
        "Edit or transform an existing image using a FAL.ai image-to-image "
        "model (default: Gemini 3.1 Flash Image Preview). Provide a natural-"
        "language edit instruction in ``prompt`` and the source image in "
        "``image_url``. Add ``reference_image_urls`` for style or composition "
        "references when supported."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The edit instruction, e.g. 'make the background a sunset beach'.",
            },
            "image_url": {
                "type": "string",
                "description": "Public URL or absolute local path of the image to edit.",
            },
            "reference_image_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional reference images for style / composition. "
                    "Capped per-model."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(VALID_ASPECT_RATIOS),
                "description": "Output aspect ratio. 'landscape' is 16:9, 'portrait' is 9:16, 'square' is 1:1.",
                "default": DEFAULT_ASPECT_RATIO,
            },
            "seed": {
                "type": "integer",
                "description": "Optional seed for reproducible output.",
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional FAL model override. If omitted, the configured "
                    "``image_gen.model`` is used."
                ),
            },
        },
        "required": ["prompt", "image_url"],
    },
}


registry.register(
    name="image_edit",
    toolset="image_edit",
    schema=IMAGE_EDIT_SCHEMA,
    handler=_handle_image_edit,
    check_fn=check_fal_api_key,
    requires_env=[],
    is_async=False,
    emoji="🖌️",
)
