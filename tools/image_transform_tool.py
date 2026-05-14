"""Image transform tool — vision → describe → generate pipeline using MiniMax.

Three MiniMax API calls, all on the same API key:

1. MiniMax LLM (vision, /v1 endpoint) — analyzes the input image and
   produces an ultra-detailed description suitable for image generation.
2. MiniMax LLM — takes the description + user's transformation instruction
   and crafts an optimized generation prompt.
3. MiniMax image-01 — generates the transformed image from the prompt.

Accepts URLs or base64 data URLs (data:image/jpeg;base64,...).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict

import requests

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

from tools.registry import registry

logger = logging.getLogger(__name__)

_CHAT_URL = "https://api.minimax.io/v1/chat/completions"
_VISION_MODEL = "MiniMax-M2.7"

_DESCRIBE_PROMPT = (
    "Describe this image in extreme detail for AI image generation. "
    "Cover: every subject (appearance, pose, expression, clothing, age), "
    "colors, textures, lighting direction and quality, atmosphere, background, "
    "art style, composition, camera angle, depth of field, mood. "
    "Be precise and exhaustive. Output ONLY the description, no preamble."
)

_TRANSFORM_SYSTEM = (
    "You are an expert AI image prompt engineer. "
    "Given a detailed image description and a transformation request, "
    "write a single optimized image generation prompt. "
    "Preserve every detail that should stay the same. "
    "Apply the transformation faithfully. "
    "Output ONLY the final prompt, nothing else."
)


def _llm_call(messages: list, api_key: str, max_tokens: int = 600) -> str:
    """Single MiniMax chat completion call. Raises on failure."""
    resp = requests.post(
        _CHAT_URL,
        json={"model": _VISION_MODEL, "messages": messages, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    # Strip <think>...</think> reasoning blocks (MiniMax-M2.7 chain-of-thought)
    content = _THINK_RE.sub("", content).strip()
    return content


def image_transform(
    input_image: str,
    transformation: str,
    aspect_ratio: str = "1:1",
) -> Dict[str, Any]:
    """Transform an image via the MiniMax vision → describe → generate pipeline."""

    api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "error": "MINIMAX_API_KEY not set"}
    if not input_image:
        return {"success": False, "error": "input_image is required"}
    if not transformation:
        return {"success": False, "error": "transformation is required"}

    # ── Step 1: describe the source image ──────────────────────────────────
    try:
        description = _llm_call(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": input_image}},
                    {"type": "text", "text": _DESCRIBE_PROMPT},
                ],
            }],
            api_key=api_key,
            max_tokens=800,
        )
        logger.debug("image_transform description: %s...", description[:120])
    except Exception as exc:
        logger.exception("image_transform: vision analysis failed")
        return {"success": False, "error": "Vision analysis failed", "error_type": "vision_error"}

    # ── Step 2: craft generation prompt ────────────────────────────────────
    try:
        gen_prompt = _llm_call(
            messages=[
                {"role": "system", "content": _TRANSFORM_SYSTEM},
                {"role": "user", "content": (
                    f"Original image:\n{description}\n\n"
                    f"Transformation: {transformation}\n\n"
                    f"Keep the final prompt under 1400 characters."
                )},
            ],
            api_key=api_key,
            max_tokens=400,
        )
        logger.debug("image_transform gen_prompt: %s...", gen_prompt[:120])
    except Exception as exc:
        # Graceful fallback: combine directly
        gen_prompt = f"{description}. Style/change: {transformation}."
        logger.debug("Prompt crafting failed (%s), using fallback prompt", exc)

    # ── Step 3: generate the transformed image ──────────────────────────────
    # MiniMax image API limits prompt to 1500 chars — truncate at word boundary
    if len(gen_prompt) > 1480:
        gen_prompt = gen_prompt[:1480].rsplit(" ", 1)[0] + "."

    try:
        from tools.image_generation_tool import dispatch_to_provider
        result_json = dispatch_to_provider(gen_prompt, aspect_ratio)
        if result_json:
            result = json.loads(result_json)
            result["source_description"] = description[:300]
            result["generation_prompt"] = gen_prompt
            return result
    except Exception as exc:
        logger.exception("image_transform: image generation failed")
        return {"success": False, "error": "Image generation failed", "error_type": "generation_error"}

    return {"success": False, "error": "No image gen provider available"}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

_SCHEMA = {
    "name": "image_transform",
    "description": (
        "Transform an existing image by describing what to change. "
        "Uses MiniMax vision to analyze the source, then generates a new "
        "image with the requested modification applied. "
        "Accepts a URL or base64 data URL (data:image/jpeg;base64,...). "
        "Good for: style changes ('make it cyberpunk', 'oil painting style'), "
        "scene edits ('change background to snowy mountains', 'add rain'), "
        "mood/lighting ('dramatic lighting', 'golden hour'). "
        "Returns the transformed image path or URL in the `image` field."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "input_image": {
                "type": "string",
                "description": (
                    "Source image to transform. Accepts a public URL "
                    "or a base64 data URL (data:image/jpeg;base64,...)."
                ),
            },
            "transformation": {
                "type": "string",
                "description": (
                    "What to change. Be specific: "
                    "'make it look like a neon cyberpunk night scene', "
                    "'convert to watercolor painting style', "
                    "'change the background to a snowy forest', "
                    "'add dramatic stormy lighting'."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"],
                "description": "Output aspect ratio. Defaults to 1:1.",
                "default": "1:1",
            },
        },
        "required": ["input_image", "transformation"],
    },
}


def _handle(args, **kw):
    result = image_transform(
        input_image=args.get("input_image", ""),
        transformation=args.get("transformation", ""),
        aspect_ratio=args.get("aspect_ratio", "1:1"),
    )
    return json.dumps(result)


registry.register(
    name="image_transform",
    toolset="image_gen",
    schema=_SCHEMA,
    handler=_handle,
)
