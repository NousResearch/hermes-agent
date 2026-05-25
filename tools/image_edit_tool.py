#!/usr/bin/env python3
"""Image editing tool.

Provides a provider-dispatched tool for prompt-guided image-to-image
editing.  The first local implementation targets the openai-codex image_gen
backend, which passes reference images through the Codex Responses API.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, VALID_ASPECT_RATIOS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

IMAGE_EDIT_SCHEMA = {
    "name": "image_edit",
    "description": (
        "Edit an existing image using a text instruction and a reference image. "
        "Mandatory: when the user provides, uploads, links, or names any "
        "reference/source/product/person image, use this image-to-image tool "
        "rather than image_generate/text-to-image. "
        "The active image backend is user-configured. Currently this is intended "
        "for backends that support image-to-image editing, such as OpenAI Codex "
        "auth with GPT Image 2. Returns either a URL or an absolute file path in "
        "the ``image`` field; display it with markdown "
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
                    "Reference image as an HTTP(S) URL, data URL, or absolute "
                    "local file path."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(VALID_ASPECT_RATIOS),
                "description": (
                    "Desired output aspect ratio. 'landscape' is 16:9 wide, "
                    "'portrait' is 16:9 tall, 'square' is 1:1."
                ),
                "default": DEFAULT_ASPECT_RATIO,
            },
        },
        "required": ["prompt", "image"],
    },
}


# ---------------------------------------------------------------------------
# Plugin dispatch
# ---------------------------------------------------------------------------


def _read_configured_image_provider() -> str | None:
    """Return the value of ``image_gen.provider`` from config.yaml, or None."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            value = section.get("provider")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception as exc:
        logger.debug("Could not read image_gen.provider: %s", exc)
    return None


def _dispatch_edit(prompt: str, image: str, aspect_ratio: str) -> str | None:
    """Route the edit call to a plugin-registered provider.

    Returns a JSON string on dispatch, or ``None`` to fall through (which
    yields an unsupported error for editing since FAL can't edit).
    """
    configured = _read_configured_image_provider()
    if not configured or configured == "fal":
        return None

    try:
        from agent.image_gen_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        provider = get_provider(configured)
    except Exception as exc:
        logger.debug("image_edit plugin dispatch skipped: %s", exc)
        return None

    if provider is None:
        try:
            _ensure_plugins_discovered(force=True)
            provider = get_provider(configured)
        except Exception as exc:
            logger.debug("image_edit plugin force-refresh skipped: %s", exc)

    if provider is None:
        return json.dumps({
            "success": False,
            "image": None,
            "error": (
                f"image_gen.provider='{configured}' is set but no plugin "
                f"registered that name. Run `hermes plugins list` to see "
                f"available image gen backends."
            ),
            "error_type": "provider_not_registered",
        })

    if not getattr(provider, "supports_edit", lambda: False)():
        return json.dumps({
            "success": False,
            "image": None,
            "error": (
                f"Image editing is not supported by provider "
                f"'{getattr(provider, 'name', configured)}'."
            ),
            "error_type": "unsupported_operation",
            "provider": getattr(provider, "name", configured),
        })

    try:
        result = provider.edit(prompt=prompt, image=image, aspect_ratio=aspect_ratio)
    except Exception as exc:
        logger.warning(
            "Image edit provider '%s' raised: %s",
            getattr(provider, "name", "?"), exc,
        )
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


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _handle_image_edit(args: Dict[str, Any], **kw: Any) -> str:
    prompt = (args.get("prompt") or "").strip()
    image = (args.get("image") or "").strip()

    if not prompt:
        from tools.registry import tool_error
        return tool_error("prompt is required for image editing", success=False)
    if not image:
        from tools.registry import tool_error
        return tool_error("image is required for image editing", success=False)

    aspect_ratio = args.get("aspect_ratio", DEFAULT_ASPECT_RATIO)

    dispatched = _dispatch_edit(prompt, image, aspect_ratio)
    if dispatched is not None:
        return dispatched

    # No plugin provider available that supports editing.
    return json.dumps({
        "success": False,
        "image": None,
        "error": (
            "No image editing provider is configured. Set "
            "``image_gen.provider`` in config.yaml to a backend that supports "
            "image-to-image editing (e.g. ``openai-codex``), then restart or "
            "/reset."
        ),
        "error_type": "unsupported_operation",
    })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry


def _edit_check_fn() -> bool:
    """Gate registration: require a plugin-capable Hermes version."""
    try:
        from agent.image_gen_registry import get_provider  # noqa: F401
        return True
    except ImportError:
        return False


registry.register(
    name="image_edit",
    toolset="image_gen",
    schema=IMAGE_EDIT_SCHEMA,
    handler=_handle_image_edit,
    check_fn=_edit_check_fn,
    requires_env=[],
    is_async=False,
    emoji="🖌️",
)
