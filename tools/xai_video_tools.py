#!/usr/bin/env python3
"""xAI-specific Imagine video edit and extend tools."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from hermes_cli.config import load_config
from plugins.video_gen.xai import (
    has_xai_video_credentials,
    run_xai_video_edit,
    run_xai_video_extend,
)
from tools.registry import registry, tool_error


def _configured_for_xai_video() -> bool:
    try:
        cfg = load_config()
    except Exception:
        return False
    section = cfg.get("video_gen") if isinstance(cfg, dict) else None
    return isinstance(section, dict) and section.get("provider") == "xai"


def _check_xai_video_requirements() -> bool:
    return _configured_for_xai_video() and has_xai_video_credentials()


def _clean_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _provider_not_configured_error() -> str:
    return json.dumps({
        "success": False,
        "error": (
            "xAI video edit/extend tools require `video_gen.provider` to be "
            "configured as `xai` via `hermes tools` -> Video Generation."
        ),
        "error_type": "provider_not_configured",
        "provider": "xai",
    })


def _normalize_public_video_url(video_url: Any) -> Optional[str]:
    """Require a public HTTPS MP4 URL (``http``/``https`` only)."""
    cleaned = _clean_string(video_url)
    if not cleaned:
        return None
    if cleaned.lower().startswith(("http://", "https://")):
        return cleaned
    return None


XAI_VIDEO_EDIT_SCHEMA: Dict[str, Any] = {
    "name": "xai_video_edit",
    "description": (
        "Edit an existing video with xAI Imagine. This is separate from "
        "`video_generate` because video editing is provider-specific. "
        "`video_url` must be the public HTTPS MP4 URL from a prior Imagine "
        "result (`video` or `public_url` on files-cdn)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Instruction for how xAI should modify the source video.",
            },
            "video_url": {
                "type": "string",
                "description": (
                    "Public HTTPS MP4 URL of the source video — the `video` or "
                    "`public_url` from a prior xAI Imagine result."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional xAI Imagine model override.",
            },
        },
        "required": ["prompt", "video_url"],
    },
}


XAI_VIDEO_EXTEND_SCHEMA: Dict[str, Any] = {
    "name": "xai_video_extend",
    "description": (
        "Extend an existing video with xAI Imagine. This is separate from "
        "`video_generate` because video extension is provider-specific. "
        "`video_url` must be the public HTTPS MP4 URL from a prior Imagine "
        "result (`video` or `public_url` on files-cdn)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Instruction for how xAI should continue the source video.",
            },
            "video_url": {
                "type": "string",
                "description": (
                    "Public HTTPS MP4 URL of the source video — the `video` or "
                    "`public_url` from a prior xAI Imagine result."
                ),
            },
            "duration": {
                "type": "integer",
                "description": (
                    "Desired extension duration in seconds. xAI clamps this "
                    "to its supported range."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional xAI Imagine model override.",
            },
        },
        "required": ["prompt", "video_url"],
    },
}


def _handle_xai_video_edit(args: Dict[str, Any], **_kw: Any) -> str:
    prompt = _clean_string(args.get("prompt"))
    video_url = _normalize_public_video_url(args.get("video_url"))
    model = _clean_string(args.get("model"))

    if not prompt:
        return tool_error("prompt is required for xAI video edit")
    if not video_url:
        return tool_error(
            "video_url must be a public HTTPS MP4 URL (the `video`/`public_url` "
            "from a prior Imagine result)"
        )
    if not _configured_for_xai_video():
        return _provider_not_configured_error()

    result = run_xai_video_edit(
        prompt=prompt,
        video_url=video_url,
        model=model,
    )
    return json.dumps(result)


def _handle_xai_video_extend(args: Dict[str, Any], **_kw: Any) -> str:
    prompt = _clean_string(args.get("prompt"))
    video_url = _normalize_public_video_url(args.get("video_url"))
    model = _clean_string(args.get("model"))
    duration = _coerce_int(args.get("duration"))

    if not prompt:
        return tool_error("prompt is required for xAI video extend")
    if not video_url:
        return tool_error(
            "video_url must be a public HTTPS MP4 URL (the `video`/`public_url` "
            "from a prior Imagine result)"
        )
    if not _configured_for_xai_video():
        return _provider_not_configured_error()

    result = run_xai_video_extend(
        prompt=prompt,
        video_url=video_url,
        duration=duration,
        model=model,
    )
    return json.dumps(result)


registry.register(
    name="xai_video_edit",
    toolset="video_gen",
    schema=XAI_VIDEO_EDIT_SCHEMA,
    handler=_handle_xai_video_edit,
    check_fn=_check_xai_video_requirements,
    requires_env=[],
    is_async=False,
    emoji="video",
)

registry.register(
    name="xai_video_extend",
    toolset="video_gen",
    schema=XAI_VIDEO_EXTEND_SCHEMA,
    handler=_handle_xai_video_extend,
    check_fn=_check_xai_video_requirements,
    requires_env=[],
    is_async=False,
    emoji="video",
)
