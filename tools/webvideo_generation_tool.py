#!/usr/bin/env python3
"""Web-video generation tool for EasyHermes.

``webvideo_build`` turns a short storyboard into a self-contained animated HTML
page that plays like a video in the browser (auto-advancing scenes, transitions,
progress bar, play/pause), and saves it under ``$HERMES_HOME/cache/webvideos/``.

Deterministic and dependency-free — not an exported mp4, no render service, no
network, no LLM call. The agent (EasyHermes itself) writes the storyboard `spec`;
this tool renders it. Rendering lives in :mod:`tools.webvideo_render`.

For a real mp4 (camera motion, photoreal), use `video_generate` (KIE/Seedance)
instead. This tool is for fast, lightweight promo/intro web videos built from
text + background images.

Always available — pure rendering, no API key required.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from tools.registry import registry, tool_error
from tools.webvideo_render import extract_spec_json, palette_names, render_webvideo_html

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 800


WEBVIDEO_BUILD_SCHEMA: Dict[str, Any] = {
    "name": "webvideo_build",
    "description": (
        "Build a lightweight 'web video' — a self-contained animated HTML page "
        "that auto-plays scene-by-scene in the browser (big headline text, "
        "transitions, progress bar) — from a short storyboard, and save it to "
        "disk. Deterministic: YOU (the assistant) write the `spec` storyboard "
        "from the user's topic; this tool only renders it. Good for fast "
        "promo/intro clips. For a real mp4 with photoreal motion, use "
        "`video_generate` instead. Pass background image URLs in `images` "
        "(generate them with `image_generate` first); they cycle across scenes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "description": (
                    "The storyboard. Fields: title (str), scenes (list of "
                    "{headline, sub}). Use 5-8 scenes; first = title scene, last "
                    "= call-to-action. Headlines should be short and punchy."
                ),
                "properties": {
                    "title": {"type": "string"},
                    "scenes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "headline": {"type": "string"},
                                "sub": {"type": "string"},
                            },
                            "required": ["headline"],
                        },
                    },
                },
                "required": ["title", "scenes"],
            },
            "style": {
                "type": "string",
                "enum": palette_names(),
                "description": "Color palette. Default 科技蓝.",
                "default": "科技蓝",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds each scene stays on screen (min 2). Default 4.",
                "default": 4,
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional public background image URLs; cycled across scenes.",
            },
        },
        "required": ["spec"],
    },
}


def _webvideos_cache_dir() -> Path:
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "webvideos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_spec(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        return extract_spec_json(raw)
    raise ValueError("`spec` must be an object (or a JSON string).")


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in value if item and str(item).strip()]


def _handle_webvideo_build(args, **kw):
    try:
        spec = _coerce_spec(args.get("spec"))
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"Invalid storyboard spec: {exc}")
    if not isinstance(spec, dict):
        return tool_error("`spec` must be an object.")
    if not (spec.get("scenes") or []):
        return tool_error("`spec.scenes` is empty — add at least one scene.")

    style = args.get("style") or "科技蓝"
    try:
        seconds = int(args.get("seconds") or 4)
    except (TypeError, ValueError):
        seconds = 4
    images = _as_list(args.get("images"))

    try:
        html_out = render_webvideo_html(spec, images, style=style, seconds=seconds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("webvideo_build render failed: %s", exc, exc_info=True)
        return tool_error(f"Web-video render failed: {exc}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _webvideos_cache_dir() / f"webvideo_{ts}_{short}.html"
    try:
        path.write_text(html_out, encoding="utf-8")
    except OSError as exc:
        return tool_error(f"Could not save web video: {exc}")

    n_scenes = len(spec.get("scenes", []) or [])
    logger.info("webvideo_build wrote %d chars (%d scenes) -> %s", len(html_out), n_scenes, path)
    return json.dumps({
        "success": True,
        "html_path": str(path),
        "bytes": len(html_out.encode("utf-8")),
        "scenes": n_scenes,
        "style": style,
        "seconds_per_scene": max(2, seconds),
        "preview": html_out[:_PREVIEW_CHARS],
        "message": (
            f"Built a {n_scenes}-scene web video → {path}. Open it in a browser "
            f"to watch (auto-plays)."
        ),
    })


registry.register(
    name="webvideo_build",
    toolset="webvideo_gen",
    schema=WEBVIDEO_BUILD_SCHEMA,
    handler=_handle_webvideo_build,
    check_fn=None,  # pure rendering — always available
    requires_env=[],
    is_async=False,
    emoji="🎞️",
)
