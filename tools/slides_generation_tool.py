#!/usr/bin/env python3
"""Slide-deck (PPT) generation tool for EasyHermes.

``slides_build`` turns a slide outline into a self-contained HTML slide deck
(keyboard nav, scroll-snap, print-to-PDF friendly) and saves it under
``$HERMES_HOME/cache/slides/``.

Deterministic and dependency-free — no LibreOffice / python-pptx, no network,
no LLM call. The agent (EasyHermes itself) writes the outline `spec`; this tool
renders it. Rendering lives in :mod:`tools.slides_render`.

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
from tools.slides_render import extract_spec_json, render_slides_html, theme_names

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 800


SLIDES_BUILD_SCHEMA: Dict[str, Any] = {
    "name": "slides_build",
    "description": (
        "Build a self-contained HTML slide deck (a presentation / PPT) from a "
        "structured outline and save it to disk. Deterministic — YOU (the "
        "assistant) write the `spec` outline from the user's topic; this tool "
        "only renders it (no model call). The deck supports keyboard nav, "
        "on-screen controls, and prints cleanly to PDF (one slide per page). "
        "For per-slide images, generate them with `image_generate` first and "
        "pass their URLs in `images` (assigned to content slides in order)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "description": (
                    "The slide outline. Fields: title (cover title), subtitle "
                    "(cover subtitle), slides (list of {title, bullets:[str], "
                    "note}). Keep 3-5 punchy bullets per slide."
                ),
                "properties": {
                    "title": {"type": "string"},
                    "subtitle": {"type": "string"},
                    "slides": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "bullets": {"type": "array", "items": {"type": "string"}},
                                "note": {"type": "string"},
                            },
                            "required": ["title"],
                        },
                    },
                },
                "required": ["title", "slides"],
            },
            "style": {
                "type": "string",
                "enum": theme_names(),
                "description": "Deck theme. Default 商务.",
                "default": "商务",
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional public image URLs, assigned to content slides in order (one per slide).",
            },
        },
        "required": ["spec"],
    },
}


def _slides_cache_dir() -> Path:
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "slides"
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


def _handle_slides_build(args, **kw):
    try:
        spec = _coerce_spec(args.get("spec"))
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"Invalid slide spec: {exc}")
    if not isinstance(spec, dict):
        return tool_error("`spec` must be an object.")
    if not (spec.get("slides") or []):
        return tool_error("`spec.slides` is empty — add at least one content slide.")

    style = args.get("style") or "商务"
    images = _as_list(args.get("images"))

    try:
        html_out = render_slides_html(spec, images, style=style)
    except Exception as exc:  # noqa: BLE001
        logger.warning("slides_build render failed: %s", exc, exc_info=True)
        return tool_error(f"Slide render failed: {exc}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _slides_cache_dir() / f"deck_{ts}_{short}.html"
    try:
        path.write_text(html_out, encoding="utf-8")
    except OSError as exc:
        return tool_error(f"Could not save slide deck: {exc}")

    n_slides = len(spec.get("slides", []) or []) + 1  # + cover
    logger.info("slides_build wrote %d chars (%d slides) -> %s", len(html_out), n_slides, path)
    return json.dumps({
        "success": True,
        "html_path": str(path),
        "bytes": len(html_out.encode("utf-8")),
        "slides": n_slides,
        "style": style,
        "preview": html_out[:_PREVIEW_CHARS],
        "message": (
            f"Built a {n_slides}-slide deck → {path}. Open it in a browser "
            f"(←/→ to navigate), or print to PDF (one slide per page)."
        ),
    })


registry.register(
    name="slides_build",
    toolset="slides_gen",
    schema=SLIDES_BUILD_SCHEMA,
    handler=_handle_slides_build,
    check_fn=None,  # pure rendering — always available
    requires_env=[],
    is_async=False,
    emoji="🖼️",
)
