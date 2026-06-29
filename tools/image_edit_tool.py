#!/usr/bin/env python3
"""
Image Edit Tool
===============

Exposes ``image_edit`` — a separate tool surface from ``image_generate`` for
image-to-image / image editing. It is a thin dispatcher:

1. Validate the agent inputs (``prompt`` + ``image`` required).
2. Resolve the active backend via the existing ``image_gen.provider`` logic
   (:func:`agent.image_gen_registry.get_active_provider`).
3. If the provider advertises edit support, call ``provider.edit(...)``.
4. Otherwise return a clear ``unsupported_capability`` result — generate-only
   backends (FAL, OpenAI, Krea, ...) are never forced to implement editing.

The provider-specific request shape (e.g. xAI's ``/v1/images/edits``) lives in
the provider; this module stays backend-agnostic so the same surface works for
any future edit-capable provider.
"""

import json
import logging
from typing import Any, Dict

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, VALID_ASPECT_RATIOS

# Import the generation tool for its side effect: it registers the ``image_gen``
# toolset (and its availability check) at module-import time. Because the tool
# discovery walk imports modules in alphabetical order, ``image_edit_tool``
# would otherwise register the toolset first and bind the toolset-level
# availability check to *edit* readiness — hiding the broader image_gen toolset
# whenever no edit-capable provider is configured. Importing here guarantees the
# generation tool anchors the shared toolset check regardless of import order.
import tools.image_generation_tool  # noqa: F401
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def check_image_edit_requirements() -> bool:
    """True when at least one registered provider supports editing and is available.

    Keeps ``image_edit`` hidden from the model unless an edit-capable backend
    (xAI today) is actually usable, so the agent isn't offered a tool that can
    only ever return ``unsupported_capability``.
    """
    try:
        from agent.image_gen_registry import list_providers
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        for provider in list_providers():
            try:
                if provider.supports_edit() and provider.is_available():
                    return True
            except Exception:
                continue
    except Exception as exc:
        logger.debug("image_edit availability probe failed: %s", exc)
    return False


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _dispatch_image_edit(prompt: str, image: str, aspect_ratio: str) -> str:
    """Resolve the active provider and route the edit request to it."""
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:
        logger.debug("image_edit plugin discovery skipped: %s", exc)

    try:
        from agent import image_gen_registry

        provider = image_gen_registry.get_active_provider()
    except Exception as exc:
        logger.warning("image_edit could not resolve a provider: %s", exc)
        return json.dumps({
            "success": False,
            "image": None,
            "error": f"Could not resolve an image provider: {exc}",
            "error_type": "no_provider",
        })

    if provider is None:
        return json.dumps({
            "success": False,
            "image": None,
            "error": (
                "No image generation provider is configured. Set one via "
                "`hermes tools` → Image Generation (an edit-capable backend "
                "such as xAI is required for image_edit)."
            ),
            "error_type": "no_provider",
        })

    provider_name = getattr(provider, "name", "?")

    try:
        supports = bool(provider.supports_edit())
    except Exception as exc:
        logger.debug("provider %s.supports_edit() raised: %s", provider_name, exc)
        supports = False

    if not supports:
        return json.dumps({
            "success": False,
            "image": None,
            "error": (
                f"Provider '{provider_name}' does not support image editing. "
                f"Configure an edit-capable image_gen.provider (e.g. xai)."
            ),
            "error_type": "unsupported_capability",
            "provider": provider_name,
        })

    try:
        result = provider.edit(prompt=prompt, image=image, aspect_ratio=aspect_ratio)
    except Exception as exc:
        logger.warning("Image edit provider '%s' raised: %s", provider_name, exc)
        return json.dumps({
            "success": False,
            "image": None,
            "error": f"Provider '{provider_name}' error: {exc}",
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


def _handle_image_edit(args: Dict[str, Any], **kw: Any) -> str:
    prompt = args.get("prompt", "")
    if not prompt or not str(prompt).strip():
        return tool_error("prompt is required for image edit")

    image = args.get("image", "")
    if not image or not str(image).strip():
        return tool_error("image is required for image edit")

    aspect_ratio = args.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    return _dispatch_image_edit(str(prompt), str(image), aspect_ratio)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

IMAGE_EDIT_SCHEMA = {
    "name": "image_edit",
    "description": (
        "Edit or transform an existing image guided by a text prompt "
        "(image-to-image). Provide the source image as a local file path, an "
        "http(s) URL, or a data URI; the configured backend must support "
        "editing (e.g. xAI). Returns either a URL or an absolute file path in "
        "the `image` field; display it with markdown ![description](url-or-path) "
        "and the gateway will deliver it. If the active backend cannot edit, "
        "the result has success=false and error_type='unsupported_capability'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Instruction describing the desired edit / transformation.",
            },
            "image": {
                "type": "string",
                "description": (
                    "The source image to edit: a local file path, an http(s) "
                    "URL, or a data URI."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(VALID_ASPECT_RATIOS),
                "description": "Output aspect ratio. 'landscape' is 16:9 wide, 'portrait' is 16:9 tall, 'square' is 1:1.",
                "default": DEFAULT_ASPECT_RATIO,
            },
        },
        "required": ["prompt", "image"],
    },
}


registry.register(
    name="image_edit",
    toolset="image_gen",
    schema=IMAGE_EDIT_SCHEMA,
    handler=_handle_image_edit,
    check_fn=check_image_edit_requirements,
    requires_env=[],
    is_async=False,
    emoji="🖌️",
)
