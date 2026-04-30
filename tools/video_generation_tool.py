#!/usr/bin/env python3
"""Video generation tool backed by FAL-compatible video endpoints."""

from __future__ import annotations

import datetime
import json
import re
import uuid
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import fal_client  # noqa: F401 — imported for requirement checks and direct backend availability

from hermes_constants import get_hermes_home
from tools.image_generation_tool import _submit_fal_request, check_fal_api_key
from tools.registry import registry, tool_error

DEFAULT_TEXT_TO_VIDEO_MODEL = "fal-ai/kling-video/v2.1/master/text-to-video"
DEFAULT_IMAGE_TO_VIDEO_MODEL = "fal-ai/kling-video/v2.1/master/image-to-video"
DEFAULT_DURATION = "5"
DEFAULT_ASPECT_RATIO = "landscape"
DEFAULT_NEGATIVE_PROMPT = "blur, distort, and low quality"
DEFAULT_CFG_SCALE = 0.5
DEFAULT_DOWNLOAD_TIMEOUT = 300

MODE_ALIASES = {
    "text": "text_to_video",
    "txt2vid": "text_to_video",
    "t2v": "text_to_video",
    "text_to_video": "text_to_video",
    "image": "image_to_video",
    "img2vid": "image_to_video",
    "i2v": "image_to_video",
    "image_to_video": "image_to_video",
}

ASPECT_RATIO_MAP = {
    "landscape": "16:9",
    "wide": "16:9",
    "16:9": "16:9",
    "portrait": "9:16",
    "vertical": "9:16",
    "9:16": "9:16",
    "square": "1:1",
    "1:1": "1:1",
}

VALID_DURATIONS = {"5", "10"}


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "text_to_video").strip().lower().replace("-", "_")
    if normalized not in MODE_ALIASES:
        raise ValueError("mode must be one of: text_to_video, image_to_video")
    return MODE_ALIASES[normalized]


def _normalize_aspect_ratio(aspect_ratio: str) -> str:
    normalized = str(aspect_ratio or DEFAULT_ASPECT_RATIO).strip().lower()
    if normalized not in ASPECT_RATIO_MAP:
        raise ValueError("aspect_ratio must be one of: landscape, portrait, square, 16:9, 9:16, 1:1")
    return ASPECT_RATIO_MAP[normalized]


def _normalize_duration(duration: Any) -> str:
    normalized = str(duration or DEFAULT_DURATION).strip()
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    if normalized not in VALID_DURATIONS:
        raise ValueError("duration must be 5 or 10 seconds for the default FAL video endpoints")
    return normalized


def _normalize_cfg_scale(cfg_scale: Any) -> float:
    try:
        value = float(cfg_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("cfg_scale must be a number between 0 and 1") from exc
    if value < 0 or value > 1:
        raise ValueError("cfg_scale must be between 0 and 1")
    return value


def _model_for_mode(mode: str) -> str:
    """Resolve the FAL-compatible video model for the requested mode.

    Env vars are intentionally kept as low-level escape hatches so admins can
    point the generic tool at any FAL-compatible endpoint without changing the
    agent-facing schema.
    """
    import os

    if mode == "image_to_video":
        return os.getenv("VIDEO_FAL_IMAGE_TO_VIDEO_MODEL", DEFAULT_IMAGE_TO_VIDEO_MODEL).strip() or DEFAULT_IMAGE_TO_VIDEO_MODEL
    return os.getenv("VIDEO_FAL_TEXT_TO_VIDEO_MODEL", DEFAULT_TEXT_TO_VIDEO_MODEL).strip() or DEFAULT_TEXT_TO_VIDEO_MODEL


def _safe_video_filename(video_url: str, file_name: Optional[str] = None) -> str:
    candidate = file_name or Path(str(video_url).split("?", 1)[0]).name
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate or "")
    if not candidate or "." not in candidate:
        candidate = f"video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
    if not candidate.lower().endswith(".mp4"):
        candidate = f"{candidate}.mp4"
    return candidate


def _download_video(video_url: str, *, file_name: Optional[str] = None) -> str:
    out_dir = get_hermes_home() / "generated-videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = _safe_video_filename(video_url, file_name=file_name)
    out_path = out_dir / base_name
    if out_path.exists():
        out_path = out_dir / f"{out_path.stem}_{uuid.uuid4().hex[:8]}{out_path.suffix}"

    request = urllib.request.Request(video_url, headers={"User-Agent": "Hermes/1.0"})
    with urllib.request.urlopen(request, timeout=DEFAULT_DOWNLOAD_TIMEOUT) as response:
        out_path.write_bytes(response.read())
    return str(out_path)


def _extract_video_info(result: Dict[str, Any]) -> Dict[str, Optional[str]]:
    video = result.get("video") if isinstance(result, dict) else None
    if isinstance(video, dict):
        return {
            "url": video.get("url"),
            "file_name": video.get("file_name"),
            "content_type": video.get("content_type"),
        }
    if isinstance(video, str):
        return {"url": video, "file_name": None, "content_type": None}
    if isinstance(result, dict) and isinstance(result.get("video_url"), str):
        return {"url": result.get("video_url"), "file_name": None, "content_type": None}
    return {"url": None, "file_name": None, "content_type": None}


def video_generate_tool(
    prompt: str,
    mode: str = "text_to_video",
    image_url: Optional[str] = None,
    duration: Any = DEFAULT_DURATION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    cfg_scale: Any = DEFAULT_CFG_SCALE,
) -> str:
    """Generate a short MP4 video through a FAL-compatible video backend."""
    start_time = datetime.datetime.now()
    try:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt is required for video generation")
        if not check_fal_api_key():
            raise ValueError("FAL_KEY environment variable not set and managed FAL gateway is unavailable")

        normalized_mode = _normalize_mode(mode)
        normalized_duration = _normalize_duration(duration)
        normalized_aspect_ratio = _normalize_aspect_ratio(aspect_ratio)
        normalized_cfg_scale = _normalize_cfg_scale(cfg_scale)

        arguments: Dict[str, Any] = {
            "prompt": prompt,
            "duration": normalized_duration,
            "aspect_ratio": normalized_aspect_ratio,
            "negative_prompt": str(negative_prompt or DEFAULT_NEGATIVE_PROMPT),
            "cfg_scale": normalized_cfg_scale,
        }

        if normalized_mode == "image_to_video":
            image_url = str(image_url or "").strip()
            if not image_url:
                raise ValueError("image_url is required for image_to_video mode")
            arguments["image_url"] = image_url

        model = _model_for_mode(normalized_mode)
        handler = _submit_fal_request(model, arguments)
        raw_result = handler.get()
        video_info = _extract_video_info(raw_result if isinstance(raw_result, dict) else {})
        video_url = video_info.get("url")
        if not video_url:
            raise ValueError("Invalid response from FAL video API - no video URL returned")

        response_data: Dict[str, Any] = {
            "success": True,
            "provider": "fal",
            "mode": normalized_mode,
            "model": model,
            "video": video_url,
            "video_url": video_url,
            "media_path": None,
            "generation_time": (datetime.datetime.now() - start_time).total_seconds(),
        }

        try:
            response_data["media_path"] = _download_video(video_url, file_name=video_info.get("file_name"))
        except Exception as download_error:  # Keep remote URL usable if local download fails.
            response_data["download_error"] = str(download_error)

        return json.dumps(response_data, indent=2, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "video": None,
                "video_url": None,
                "media_path": None,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "generation_time": (datetime.datetime.now() - start_time).total_seconds(),
            },
            indent=2,
            ensure_ascii=False,
        )


def check_video_generation_requirements() -> bool:
    try:
        import fal_client as _fal_client  # noqa: F401
    except ImportError:
        return False
    return check_fal_api_key()


VIDEO_GENERATE_SCHEMA = {
    "name": "video_generate",
    "description": "Generate short MP4 videos through FAL-compatible video models. Supports text-to-video and image-to-video. Returns a video URL and a local media_path when the MP4 can be downloaded.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt describing the video, camera motion, subject motion, style, and mood.",
            },
            "mode": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video"],
                "description": "Use text_to_video for prompt-only generation, or image_to_video to animate a source image.",
                "default": "text_to_video",
            },
            "image_url": {
                "type": "string",
                "description": "Source image URL. Required when mode is image_to_video.",
            },
            "duration": {
                "type": "string",
                "enum": ["5", "10"],
                "description": "Video length in seconds for the default FAL video endpoints.",
                "default": "5",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["landscape", "portrait", "square"],
                "description": "Video frame shape. landscape=16:9, portrait=9:16, square=1:1.",
                "default": "landscape",
            },
            "negative_prompt": {
                "type": "string",
                "description": "What to avoid in the generated video.",
                "default": DEFAULT_NEGATIVE_PROMPT,
            },
            "cfg_scale": {
                "type": "number",
                "description": "Prompt adherence for default Kling endpoints. Range 0 to 1.",
                "default": DEFAULT_CFG_SCALE,
            },
        },
        "required": ["prompt"],
    },
}


def _handle_video_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for video generation")
    return video_generate_tool(
        prompt=prompt,
        mode=args.get("mode", "text_to_video"),
        image_url=args.get("image_url"),
        duration=args.get("duration", DEFAULT_DURATION),
        aspect_ratio=args.get("aspect_ratio", DEFAULT_ASPECT_RATIO),
        negative_prompt=args.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        cfg_scale=args.get("cfg_scale", DEFAULT_CFG_SCALE),
    )


registry.register(
    name="video_generate",
    toolset="video_gen",
    schema=VIDEO_GENERATE_SCHEMA,
    handler=_handle_video_generate,
    check_fn=check_video_generation_requirements,
    requires_env=[],
    is_async=False,
    emoji="🎬",
)
