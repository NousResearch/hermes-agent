#!/usr/bin/env python3
"""xAI-specific Imagine video edit and extend tools."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from hermes_cli.config import load_config
from plugins.video_gen.xai import has_xai_video_credentials, run_xai_video_edit, run_xai_video_extend
from tools.registry import registry, tool_error


def _configured_for_xai_video() -> bool:
    try:
        section = load_config().get("video_gen")
    except Exception:
        return False
    return isinstance(section, dict) and section.get("provider") == "xai"


def _check_xai_video_requirements() -> bool:
    return _configured_for_xai_video() and has_xai_video_credentials()


def _clean_string(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _coerce_int(value: Any) -> Optional[int]:
    # bool is rejected (unlike video_generation_tool._coerce_int) so duration=true never becomes 1.
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_VIDEO_URL_PARAM = {
    "type": "string",
    "description": (
        "Public HTTPS MP4 URL of the source video — the `video` or "
        "`public_url` from a prior xAI Imagine result."
    ),
}
_MODEL_PARAM = {"type": "string", "description": "Optional xAI Imagine model override."}


def _xai_video_schema(name: str, verb: str, noun: str, prompt_verb: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "description": (
            f"{verb} an existing video with xAI Imagine. This is separate from "
            f"`video_generate` because video {noun} is provider-specific. "
            "`video_url` must be the public HTTPS MP4 URL from a prior Imagine "
            "result (`video` or `public_url` on files-cdn)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": f"Instruction for how xAI should {prompt_verb} the source video.",
                },
                "video_url": _VIDEO_URL_PARAM,
                **extra,
                "model": _MODEL_PARAM,
            },
            "required": ["prompt", "video_url"],
        },
    }


XAI_VIDEO_EDIT_SCHEMA: Dict[str, Any] = _xai_video_schema("xai_video_edit", "Edit", "editing", "modify", {})

XAI_VIDEO_EXTEND_SCHEMA: Dict[str, Any] = _xai_video_schema(
    "xai_video_extend", "Extend", "extension", "continue", {
        "duration": {
            "type": "integer",
            "description": (
                "Desired extension duration in seconds. xAI clamps this "
                "to its supported range."
            ),
        },
    },
)


def _run_xai_video_tool(args: Dict[str, Any], op: str, run, **extra: Any) -> str:
    prompt, video_url = _clean_string(args.get("prompt")), _clean_string(args.get("video_url"))
    if not prompt:
        return tool_error(f"prompt is required for xAI video {op}")
    if not (video_url and video_url.lower().startswith(("http://", "https://"))):  # public URL only
        return tool_error(
            "video_url must be a public HTTPS MP4 URL (the `video`/`public_url` "
            "from a prior Imagine result)"
        )
    if not _configured_for_xai_video():
        return json.dumps({
            "success": False,
            "error": (
                "xAI video edit/extend tools require `video_gen.provider` to be "
                "configured as `xai` via `hermes tools` -> Video Generation."
            ),
            "error_type": "provider_not_configured",
            "provider": "xai",
        })
    model = _clean_string(args.get("model"))
    return json.dumps(run(prompt=prompt, video_url=video_url, model=model, **extra))


def _handle_xai_video_edit(args: Dict[str, Any], **_kw: Any) -> str:
    return _run_xai_video_tool(args, "edit", run_xai_video_edit)


def _handle_xai_video_extend(args: Dict[str, Any], **_kw: Any) -> str:
    return _run_xai_video_tool(args, "extend", run_xai_video_extend, duration=_coerce_int(args.get("duration")))


for _name, _schema, _handler in (
    ("xai_video_edit", XAI_VIDEO_EDIT_SCHEMA, _handle_xai_video_edit),
    ("xai_video_extend", XAI_VIDEO_EXTEND_SCHEMA, _handle_xai_video_extend),
):
    registry.register(
        name=_name, toolset="video_gen", schema=_schema, handler=_handler,
        check_fn=_check_xai_video_requirements, requires_env=[], is_async=False, emoji="video",
    )
