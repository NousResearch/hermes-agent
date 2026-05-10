#!/usr/bin/env python3
"""
Video Generation Tool — Generate videos via xAI's grok-imagine-video API.

Supports text-to-video, image-to-video, video editing, and video extension.
Uses the async submit + polling pattern:
POST /v1/videos/generations|edits|extensions, then GET /v1/videos/{request_id}.

Requires ``XAI_API_KEY`` in ``~/.hermes/.env``.

Configuration (optional, in config.yaml)::

    video_generation:
      model: grok-imagine-video     # default
      timeout_seconds: 240          # default
      poll_interval_seconds: 5      # default

Usage::

    from tools.video_generation_tool import video_generate_tool, check_video_generation_requirements

    result = await video_generate_tool(prompt="A cat playing piano")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-imagine-video"
DEFAULT_OPERATION = "generate"
DEFAULT_DURATION = 8
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_RESOLUTION = "720p"
DEFAULT_TIMEOUT_SECONDS = 240
DEFAULT_POLL_INTERVAL_SECONDS = 5

VALID_ASPECT_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"}
VALID_RESOLUTIONS = {"480p", "720p"}
VALID_SIZES = {"848x480", "1696x960", "1280x720", "1920x1080"}
VALID_OPERATIONS = {"generate", "edit", "extend"}
OPERATION_ENDPOINTS = {
    "generate": "generations",
    "edit": "edits",
    "extend": "extensions",
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_base_url() -> str:
    return (os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config().get("video_generation", {})
    except Exception:
        return {}


def _get_model() -> str:
    return (_load_config().get("model") or DEFAULT_MODEL).strip()


def _get_timeout() -> int:
    raw = _load_config().get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    try:
        return max(30, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


def _get_poll_interval() -> int:
    raw = _load_config().get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_POLL_INTERVAL_SECONDS


# ---------------------------------------------------------------------------
# Requirement check
# ---------------------------------------------------------------------------


def check_video_generation_requirements() -> bool:
    """Return True if XAI_API_KEY is set."""
    return bool(os.getenv("XAI_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


def _xai_headers() -> Dict[str, str]:
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("XAI_API_KEY not set")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": hermes_xai_user_agent(),
    }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize_operation(operation: Optional[str]) -> str:
    op = (operation or DEFAULT_OPERATION).strip().lower()
    if op not in VALID_OPERATIONS:
        raise ValueError(f"operation must be one of {sorted(VALID_OPERATIONS)}")
    return op


def _normalize_duration(duration: Optional[int], operation: str) -> int:
    if duration is None:
        return 6 if operation == "extend" else DEFAULT_DURATION
    try:
        d = int(duration)
    except (TypeError, ValueError):
        return DEFAULT_DURATION
    if operation == "extend":
        return max(1, min(d, 10))
    return max(1, min(d, 15))


def _normalize_aspect_ratio(ar: Optional[str]) -> str:
    normalized = (ar or DEFAULT_ASPECT_RATIO).strip()
    if normalized not in VALID_ASPECT_RATIOS:
        raise ValueError(f"aspect_ratio must be one of {sorted(VALID_ASPECT_RATIOS)}")
    return normalized


def _normalize_resolution(res: Optional[str]) -> str:
    normalized = (res or DEFAULT_RESOLUTION).strip().lower()
    if normalized not in VALID_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {sorted(VALID_RESOLUTIONS)}")
    return normalized


def _normalize_size(size: Optional[str]) -> Optional[str]:
    if not size:
        return None
    normalized = size.strip().lower()
    if normalized not in VALID_SIZES:
        raise ValueError(f"size must be one of {sorted(VALID_SIZES)}")
    return normalized


def _normalize_reference_images(
    images: Optional[List[str]],
) -> List[str]:
    if not images:
        return []
    cleaned: List[str] = []
    for img in images:
        if isinstance(img, str) and img.strip():
            cleaned.append(img.strip())
    return cleaned[:7]  # xAI allows up to 7 reference images


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------


async def video_generate_tool(
    prompt: str = "",
    operation: str = DEFAULT_OPERATION,
    duration: Optional[int] = None,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    resolution: str = DEFAULT_RESOLUTION,
    size: Optional[str] = None,
    video_url: Optional[str] = None,
    image_url: Optional[str] = None,
    reference_images: Optional[List[str]] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Generate, edit, or extend a video using xAI's grok-imagine-video.

    Returns JSON with ``video_url``, ``status``, and metadata.
    """
    # Validate inputs
    try:
        normalized_op = _normalize_operation(operation)
    except ValueError as exc:
        return tool_error(str(exc))

    if normalized_op in ("edit", "extend") and not video_url:
        return tool_error(f"operation '{normalized_op}' requires video_url")

    if normalized_op == "generate" and not prompt and not image_url:
        return tool_error("generate requires prompt or image_url")

    try:
        normalized_duration = _normalize_duration(duration, normalized_op)
        normalized_ar = _normalize_aspect_ratio(aspect_ratio)
        normalized_res = _normalize_resolution(resolution)
        normalized_size = _normalize_size(size)
    except ValueError as exc:
        return tool_error(str(exc))

    ref_images = _normalize_reference_images(reference_images)

    if timeout_seconds < 30:
        return tool_error("timeout_seconds must be at least 30")
    if poll_interval_seconds < 1:
        return tool_error("poll_interval_seconds must be at least 1")

    # Build submit body
    submit_body: Dict[str, Any] = {
        "model": _get_model(),
    }

    if prompt:
        submit_body["prompt"] = prompt.strip()
    if image_url and normalized_op == "generate":
        submit_body["image"] = {"url": image_url.strip()}
    if video_url and normalized_op == "edit":
        submit_body["video_url"] = video_url.strip()
    if video_url and normalized_op == "extend":
        submit_body["video"] = {"url": video_url.strip()}
    if ref_images:
        submit_body["reference_images"] = [{"url": img} for img in ref_images]

    # Duration for generate/extend
    if normalized_op in ("generate", "extend"):
        submit_body["duration"] = normalized_duration

    # Aspect ratio + resolution for generate
    if normalized_op == "generate":
        submit_body["aspect_ratio"] = normalized_ar
        submit_body["resolution"] = normalized_res
        if normalized_size:
            submit_body["size"] = normalized_size

    try:
        headers = _xai_headers()
    except ValueError as exc:
        return tool_error(str(exc))
    base_url = _get_base_url()

    async with httpx.AsyncClient() as client:
        # Step 1: Submit generation request
        try:
            submit_response = await client.post(
                f"{base_url}/videos/{OPERATION_ENDPOINTS[normalized_op]}",
                headers=headers,
                json=submit_body,
                timeout=30,
            )
            submit_response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                err_msg = exc.response.json().get("error", {}).get("message", exc.response.text[:300])
            except Exception:
                err_msg = exc.response.text[:300]
            logger.error("video_gen submit failed (%d): %s", status, err_msg)
            if status in (400, 401, 403):
                return tool_error(f"Video generation auth error ({status}): {err_msg}")
            return tool_error(f"Video generation submit failed ({status}): {err_msg}")
        except httpx.TimeoutException:
            return tool_error("Video generation submit timed out")
        except httpx.ConnectError as exc:
            return tool_error(f"Video generation connection error: {exc}")

        submit_payload = submit_response.json()
        request_id = submit_payload.get("request_id")
        if not request_id:
            return tool_error("Video generation: no request_id returned")

        logger.info("video_gen submitted: request_id=%s, op=%s", request_id, normalized_op)

        # Step 2: Poll for completion
        elapsed = 0
        last_status = "queued"

        while elapsed < timeout_seconds:
            await asyncio.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds

            try:
                status_response = await client.get(
                    f"{base_url}/videos/{request_id}",
                    headers=headers,
                    timeout=15,
                )
                status_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.warning("video_gen poll failed: %s", exc)
                continue
            except httpx.TimeoutException:
                logger.warning("video_gen poll timeout at %ds", elapsed)
                continue

            status_payload = status_response.json()
            last_status = (status_payload.get("status") or "").lower()

            if last_status == "done":
                video = status_payload.get("video") or {}
                video_url_result = video.get("url") or status_payload.get("video_url", "")

                result = {
                    "tool": "video_generate",
                    "status": "done",
                    "video_url": video_url_result,
                    "duration": normalized_duration,
                    "operation": normalized_op,
                    "model": status_payload.get("model", _get_model()),
                    "usage": status_payload.get("usage"),
                }

                logger.info(
                    "video_gen completed: request_id=%s, url=%s",
                    request_id, video_url_result[:100] if video_url_result else "none",
                )

                return json.dumps(result, ensure_ascii=False)

            if last_status in {"failed", "error", "expired", "cancelled"}:
                err_msg = (
                    status_payload.get("error", {}).get("message")
                    or status_payload.get("message")
                    or f"Video generation ended with status '{last_status}'"
                )
                logger.error("video_gen failed: request_id=%s, status=%s", request_id, last_status)
                return tool_error(f"Video generation {last_status}: {err_msg}")

            logger.debug("video_gen status: %s (%ds elapsed)", last_status, elapsed)

        # Timeout
        return tool_error(
            f"Video generation timed out after {timeout_seconds}s (status: {last_status})"
        )


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

VIDEO_GENERATE_SCHEMA = {
    "name": "video_generate",
    "description": (
        "Generate, edit, or extend short videos with xAI grok-imagine-video. "
        "Supports text-to-video, image-to-video, reference-image-guided generation, "
        "native video edits, and native video extensions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Describe the video to generate, edit, or extend. "
                    "Optional only for image-to-video where the image alone is the instruction."
                ),
            },
            "operation": {
                "type": "string",
                "enum": sorted(VALID_OPERATIONS),
                "description": (
                    "Video mode: 'generate' for new videos, 'edit' to modify existing, "
                    "'extend' to continue an existing video."
                ),
                "default": DEFAULT_OPERATION,
            },
            "duration": {
                "type": "integer",
                "description": (
                    "Duration in seconds. Generate: 1-15s. Extend: 1-10s. "
                    "Edit retains source duration."
                ),
                "default": DEFAULT_DURATION,
            },
            "aspect_ratio": {
                "type": "string",
                "enum": sorted(VALID_ASPECT_RATIOS),
                "description": "Output aspect ratio for generate mode.",
                "default": DEFAULT_ASPECT_RATIO,
            },
            "resolution": {
                "type": "string",
                "enum": sorted(VALID_RESOLUTIONS),
                "description": "Output resolution for generate mode.",
                "default": DEFAULT_RESOLUTION,
            },
            "size": {
                "type": "string",
                "enum": sorted(VALID_SIZES),
                "description": "Optional explicit output size for generate mode.",
            },
            "video_url": {
                "type": "string",
                "description": "Required for edit and extend modes. Source video URL.",
            },
            "image_url": {
                "type": "string",
                "description": "Optional source image URL for image-to-video generation.",
            },
            "reference_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional reference image URLs to guide generation (max 5).",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Handler + registration
# ---------------------------------------------------------------------------


async def _handle_video_generate(args: Dict[str, Any], **kw: Any) -> str:
    return await video_generate_tool(
        prompt=args.get("prompt", ""),
        operation=args.get("operation", DEFAULT_OPERATION),
        duration=args.get("duration"),
        aspect_ratio=args.get("aspect_ratio", DEFAULT_ASPECT_RATIO),
        resolution=args.get("resolution", DEFAULT_RESOLUTION),
        size=args.get("size"),
        video_url=args.get("video_url"),
        image_url=args.get("image_url"),
        reference_images=args.get("reference_images"),
    )


registry.register(
    name="video_generate",
    toolset="video_gen",
    schema=VIDEO_GENERATE_SCHEMA,
    handler=_handle_video_generate,
    check_fn=check_video_generation_requirements,
    requires_env=["XAI_API_KEY"],
    is_async=True,
    emoji="🎬",
)
