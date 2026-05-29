#!/usr/bin/env python3
"""Opt-in image editing tool.

Dispatches prompt-guided image edits to the configured ``image_gen`` provider
when that provider explicitly advertises edit support.  The tool is registered
under the existing ``image_gen`` toolset so it is not exposed unless users opt
into image generation tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, VALID_ASPECT_RATIOS
from agent.image_reference import ImageReferenceError, validate_image_reference
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

IMAGE_EDIT_SCHEMA = {
    "name": "image_edit",
    "description": (
        "Edit an existing image using a text instruction and a reference image. "
        "Use only when image generation tools are enabled and the configured "
        "image backend supports editing. Returns either a URL or an absolute "
        "file path in the `image` field; display it with markdown "
        "![description](url-or-path)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Instruction describing the edit to apply while preserving "
                    "unchanged parts of the source image."
                ),
            },
            "image": {
                "type": "string",
                "description": (
                    "Reference image as an HTTP(S) URL, data:image URL, or local "
                    "file path under Hermes' image cache."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(VALID_ASPECT_RATIOS),
                "description": "Desired output aspect ratio.",
                "default": DEFAULT_ASPECT_RATIO,
            },
            "size": {
                "type": "string",
                "description": "Optional provider-specific exact output size, for example WIDTHxHEIGHT.",
            },
            "model": {
                "type": "string",
                "description": "Optional provider-specific model override.",
            },
            "quality_tier": {
                "type": "string",
                "description": "Optional provider-specific quality tier override.",
            },
        },
        "required": ["prompt", "image"],
    },
}


def _read_configured_image_provider():
    """Reuse image_generate's provider selection logic."""
    try:
        from tools.image_generation_tool import _read_configured_image_provider as _reader

        return _reader()
    except Exception as exc:
        logger.debug("Could not read configured image provider for image_edit: %s", exc)
        return None


def _read_configured_image_model():
    """Reuse image_generate's selected model as an optional provider hint."""
    try:
        from tools.image_generation_tool import _read_configured_image_model as _reader

        return _reader()
    except Exception as exc:
        logger.debug("Could not read configured image model for image_edit: %s", exc)
        return None


def _get_active_edit_provider():
    configured = _read_configured_image_provider()
    if not configured or configured == "fal":
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


def _provider_error(message: str, error_type: str) -> str:
    return json.dumps({"success": False, "image": None, "error": message, "error_type": error_type})


def _dispatch_to_plugin_provider(
    prompt: str,
    image: str,
    aspect_ratio: str,
    *,
    size: str | None = None,
    model: str | None = None,
    quality_tier: str | None = None,
) -> str:
    provider, configured = _get_active_edit_provider()
    if configured is None or configured == "fal":
        return _provider_error(
            "image_edit requires image_gen.provider to be set to an edit-capable backend.",
            "provider_not_configured",
        )

    if provider is None:
        return _provider_error(
            f"image_gen.provider='{configured}' is set but no plugin registered that name.",
            "provider_not_registered",
        )

    if not getattr(provider, "supports_edit", lambda: False)():
        return _provider_error(
            f"Image provider '{getattr(provider, 'name', configured)}' does not support image_edit",
            "unsupported",
        )

    try:
        validated = validate_image_reference(image)
    except ImageReferenceError as exc:
        return _provider_error(str(exc), exc.error_type)

    try:
        edit_kwargs: Dict[str, Any] = {}
        if size:
            edit_kwargs["size"] = size
        if quality_tier:
            edit_kwargs["quality_tier"] = quality_tier
        configured_model = _read_configured_image_model()
        if model:
            edit_kwargs["model"] = model
        elif configured_model:
            edit_kwargs["model"] = configured_model

        result = provider.edit(
            prompt=prompt,
            image=validated.value,
            aspect_ratio=aspect_ratio,
            **edit_kwargs,
        )
    except Exception as exc:
        logger.warning("Image edit provider '%s' raised: %s", getattr(provider, "name", "?"), exc)
        return _provider_error(
            f"Provider '{getattr(provider, 'name', '?')}' error: {exc}",
            "provider_exception",
        )

    if not isinstance(result, dict):
        return _provider_error("Provider returned a non-dict result", "provider_contract")
    return json.dumps(result)


def check_image_edit_requirements() -> bool:
    provider, configured = _get_active_edit_provider()
    if not configured or provider is None:
        return False
    try:
        return bool(provider.is_available() and provider.supports_edit())
    except Exception:
        return False


def _handle_image_edit(args: Dict[str, Any], **kw):
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("prompt is required for image editing")

    image = args.get("image") or args.get("image_path") or args.get("image_url")
    if not isinstance(image, str) or not image.strip():
        return tool_error("image is required for image editing and must be a path or URL")

    return _dispatch_to_plugin_provider(
        prompt,
        image.strip(),
        args.get("aspect_ratio", DEFAULT_ASPECT_RATIO),
        size=args.get("size"),
        model=args.get("model"),
        quality_tier=args.get("quality_tier"),
    )


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
