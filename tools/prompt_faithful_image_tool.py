#!/usr/bin/env python3
"""Prompt-faithful image generation via an OpenAI-compatible chat endpoint.

This tool is intentionally thin:
- send the user's prompt through unchanged
- call a configured chat/completions-compatible image endpoint
- extract the returned image URL
- download it into a Hermes-managed cache directory
- return a MEDIA tag so the current platform can send it natively
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from hermes_constants import get_hermes_dir
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.nxfl.cc/v1/chat/completions"
DEFAULT_MODEL = "gemini-imagen"
IMAGE_CACHE_DIR = get_hermes_dir("cache/generated-images", "generated")
_IMAGE_CONTENT_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def _ensure_cache_dir() -> Path:
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGE_CACHE_DIR


def _resolve_api_url() -> str:
    return str(os.getenv("NXFL_IMAGE_API_URL", DEFAULT_API_URL)).strip() or DEFAULT_API_URL


def _resolve_model() -> str:
    return str(os.getenv("NXFL_IMAGE_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL


def _resolve_api_key() -> str:
    return str(os.getenv("NXFL_API_KEY", "")).strip()


def check_prompt_faithful_image_requirements() -> bool:
    return bool(_resolve_api_key())


def build_request_payload(prompt: str) -> dict:
    return {
        "model": _resolve_model(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": str(prompt),
                    }
                ],
            }
        ],
        "stream": False,
    }


def request_provider(prompt: str) -> dict:
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError("Missing NXFL_API_KEY for prompt-faithful image generation")

    req = urllib.request.Request(
        _resolve_api_url(),
        data=json.dumps(build_request_payload(prompt), ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "HermesPromptFaithfulImage/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc


def extract_image_url(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Image provider returned no choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    text_parts: list[str] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
                image_url = item.get("image_url")
                if isinstance(image_url, str):
                    text_parts.append(image_url)
                elif isinstance(image_url, dict):
                    url = image_url.get("url")
                    if isinstance(url, str):
                        text_parts.append(url)

    raw = "\n".join(text_parts)
    markdown_match = re.search(r"!\[[^\]]*\]\((https?://[^)]+)\)", raw)
    if markdown_match:
        return markdown_match.group(1)

    direct_match = re.search(r"(https?://[^\s)]+)", raw)
    if direct_match:
        return direct_match.group(1)

    raw_json = json.dumps(response, ensure_ascii=False)
    fallback_match = re.search(r"(https?://[^\s\"\\]+)", raw_json)
    if fallback_match:
        return fallback_match.group(1)

    raise RuntimeError("Failed to extract image URL from provider response")


def _infer_extension(image_url: str) -> str:
    path = urlparse(image_url).path.lower()
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".png"


def _normalize_content_type(value: str) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def _sniff_image_type(data: bytes) -> tuple[str | None, str | None]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    return None, None


def download_image(image_url: str) -> str:
    cache_dir = _ensure_cache_dir()
    req = urllib.request.Request(
        image_url,
        headers={
            "User-Agent": "HermesPromptFaithfulImage/1.0",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        content_type = _normalize_content_type(getattr(resp, "headers", {}).get("Content-Type", ""))
        data = resp.read()

    sniffed_mime, sniffed_ext = _sniff_image_type(data)
    if content_type:
        if not content_type.startswith("image/"):
            raise RuntimeError(f"Downloaded URL did not return an image (Content-Type: {content_type})")
        if sniffed_mime and content_type in _IMAGE_CONTENT_TYPES and sniffed_mime != content_type:
            raise RuntimeError(
                f"Downloaded image content does not match Content-Type ({content_type} vs {sniffed_mime})"
            )

    if not sniffed_mime or not sniffed_ext:
        raise RuntimeError("Downloaded file does not look like a supported image")

    ext = _IMAGE_CONTENT_TYPES.get(content_type) or sniffed_ext or _infer_extension(image_url)
    filename = f"prompt-faithful-{int(time.time() * 1000)}{ext}"
    output_path = cache_dir / filename

    output_path.write_bytes(data)
    return str(output_path.resolve())


def generate_prompt_faithful_image(prompt: str) -> dict:
    text = str(prompt or "")
    if not text.strip():
        raise ValueError("prompt is required")

    response = request_provider(text)
    image_url = extract_image_url(response)
    local_path = download_image(image_url)
    return {
        "success": True,
        "prompt": text,
        "image_url": image_url,
        "local_path": local_path,
        "media_tag": f"MEDIA:{local_path}",
    }


def prompt_faithful_image_generate_tool(prompt: str) -> str:
    try:
        return json.dumps(generate_prompt_faithful_image(prompt), ensure_ascii=False)
    except Exception as exc:
        logger.error("Prompt-faithful image generation failed: %s", exc, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "prompt": str(prompt or ""),
                "error": str(exc),
            },
            ensure_ascii=False,
        )


PROMPT_FAITHFUL_IMAGE_GENERATE_SCHEMA = {
    "name": "prompt_faithful_image_generate",
    "description": (
        "Generate exactly one image from the user's literal prompt using the configured "
        "OpenAI-compatible image endpoint, download it to a local file, and return a "
        "native MEDIA attachment path for the current chat. Preserve the user's prompt "
        "as-is unless they explicitly ask you to rewrite it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The exact image prompt to send to the provider. Copy the user's "
                    "intended prompt literally instead of embellishing it."
                ),
            }
        },
        "required": ["prompt"],
    },
}


def _handle_prompt_faithful_image_generate(args, **kw):
    prompt = str(args.get("prompt", "") or "")
    if not prompt.strip():
        return tool_error("prompt is required for prompt-faithful image generation")
    return prompt_faithful_image_generate_tool(prompt)


registry.register(
    name="prompt_faithful_image_generate",
    toolset="image_gen",
    schema=PROMPT_FAITHFUL_IMAGE_GENERATE_SCHEMA,
    handler=_handle_prompt_faithful_image_generate,
    check_fn=check_prompt_faithful_image_requirements,
    requires_env=["NXFL_API_KEY"],
    is_async=False,
    emoji="🖼️",
)
