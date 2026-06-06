#!/usr/bin/env python3
"""Prompt-guided image editing tool.

Routes ``image_edit`` calls to the active image generation provider when that
provider advertises edit support.  Providers keep their own API-specific image
normalisation and upload rules; the tool is responsible for schema validation,
configuration lookup, and JSON-serialising the provider response.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO
from tools.image_generation_tool import (
    _read_configured_image_model,
    _read_configured_image_provider,
)
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_EDIT_ASPECT_RATIOS = (
    "landscape", "square", "portrait",
    "16:9", "5:4", "4:3", "3:2", "1:1", "2:3", "3:4", "4:5", "9:16",
)


IMAGE_EDIT_SCHEMA = {
    "name": "image_edit",
    "description": (
        "Edit an existing image using a text prompt and one or more reference "
        "images. The active image generation provider must support editing. "
        "Provide an image as an absolute Hermes image-cache path, HTTP(S) URL, "
        "or data:image URL. Returns a URL or absolute file path in `image`."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Instruction describing how to change the image.",
            },
            "image": {
                "type": "string",
                "description": (
                    "Primary input/reference image as an absolute local path, "
                    "HTTP(S) URL, or data:image URL."
                ),
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of input/reference images. If `image` is "
                    "omitted, the first item is used as the primary image."
                ),
            },
            "reference_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Alias for `images` for compatibility with agents that use reference-image wording.",
            },
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Alias for `images`/`reference_images`; ordered reference images for compatibility.",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(_EDIT_ASPECT_RATIOS),
                "description": "Target aspect ratio/shape for the edited image.",
                "default": DEFAULT_ASPECT_RATIO,
            },
            "size": {
                "type": "string",
                "description": "Optional explicit output size such as 1024x1024 for providers that support it.",
            },
            "quality_tier": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Optional provider quality tier when supported.",
            },
        },
        "required": ["prompt"],
    },
}


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _primary_image(args: Dict[str, Any]) -> Optional[str]:
    image = args.get("image") or args.get("image_path") or args.get("image_url")
    if isinstance(image, str) and image.strip():
        return image.strip()
    for key in ("images", "reference_images", "references"):
        values = _string_list(args.get(key))
        if values:
            return values[0]
    return None


def _get_active_edit_provider():
    configured = _read_configured_image_provider()
    if not configured:
        return None, configured

    try:
        from agent.image_gen_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        provider = get_provider(configured)
        if provider is None:
            _ensure_plugins_discovered(force=True)
            provider = get_provider(configured)
        return provider, configured
    except Exception as exc:
        logger.debug("image_edit provider discovery failed: %s", exc)
        return None, configured


def _dispatch_to_edit_provider(prompt: str, image: str, aspect_ratio: str, args: Dict[str, Any]) -> str:
    provider, configured = _get_active_edit_provider()
    if not configured:
        return json.dumps({
            "success": False,
            "image": None,
            "error": (
                "image_edit requires an image_gen.provider configured to a backend "
                "that supports editing (for example `openai-codex`)."
            ),
            "error_type": "provider_not_configured",
        })

    configured_model = _read_configured_image_model()

    if provider is None:
        return json.dumps({
            "success": False,
            "image": None,
            "error": (
                f"image_gen.provider='{configured}' is set but no plugin registered "
                "that name. Run `hermes plugins list` to see available image gen backends."
            ),
            "error_type": "provider_not_registered",
        })

    supports_edit = getattr(provider, "supports_edit", None)
    if not callable(supports_edit) or not supports_edit():
        return json.dumps({
            "success": False,
            "image": None,
            "error": f"Image provider '{getattr(provider, 'name', configured)}' does not support image editing",
            "error_type": "unsupported",
            "provider": getattr(provider, "name", configured),
        })
    if not callable(getattr(provider, "edit", None)):
        return json.dumps({
            "success": False,
            "image": None,
            "error": f"Image provider '{getattr(provider, 'name', configured)}' does not expose an edit method",
            "error_type": "unsupported",
            "provider": getattr(provider, "name", configured),
        })

    try:
        kwargs: Dict[str, Any] = {"prompt": prompt, "image": image, "aspect_ratio": aspect_ratio}
        if configured_model:
            kwargs["model"] = configured_model
        for key in ("size", "quality_tier"):
            if args.get(key) is not None:
                kwargs[key] = args[key]
        images = (
            _string_list(args.get("images"))
            or _string_list(args.get("reference_images"))
            or _string_list(args.get("references"))
        )
        if images:
            kwargs["images"] = images
        result = provider.edit(**kwargs)
    except Exception as exc:
        logger.warning("Image edit provider '%s' raised: %s", getattr(provider, "name", "?"), exc)
        return json.dumps({
            "success": False,
            "image": None,
            "error": f"Provider '{getattr(provider, 'name', '?')}' error: {exc}",
            "error_type": "provider_exception",
        })

    if not isinstance(result, dict):
        return json.dumps({
            "success": False,
            "image": None,
            "error": "Provider returned a non-dict result",
            "error_type": "provider_contract",
        })
    return json.dumps(result)


def check_image_edit_requirements() -> bool:
    provider, configured = _get_active_edit_provider()
    if not configured or provider is None:
        return False
    supports_edit = getattr(provider, "supports_edit", None)
    edit = getattr(provider, "edit", None)
    try:
        available = provider.is_available() if callable(getattr(provider, "is_available", None)) else True
        return bool(available and callable(supports_edit) and supports_edit() and callable(edit))
    except Exception:
        return False


def _handle_image_edit(args: Dict[str, Any], **kw: Any) -> str:
    prompt = (args.get("prompt") or "").strip() if isinstance(args.get("prompt"), str) else ""
    if not prompt:
        return tool_error("prompt is required for image editing")
    image = _primary_image(args)
    if not image:
        return tool_error("image is required for image editing")
    aspect_ratio = args.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    return _dispatch_to_edit_provider(prompt, image, aspect_ratio, args)


registry.register(
    name="image_edit",
    toolset="image_gen",
    schema=IMAGE_EDIT_SCHEMA,
    handler=_handle_image_edit,
    check_fn=check_image_edit_requirements,
    requires_env=[],
    is_async=False,
    emoji="🖼️",
)
