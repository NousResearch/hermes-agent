#!/usr/bin/env python3
"""Website generation tool for EasyHermes.

``website_build`` turns a structured site spec into a complete, responsive
single-file HTML website and saves it under ``$HERMES_HOME/cache/sites/``.

Deterministic and dependency-free: the agent (EasyHermes itself) produces the
spec from the user's request, and this tool renders it — no LLM call, no
network, no publish. The rendering logic lives in :mod:`tools.website_render`
(ported from the Kari/Langflow website renderer; Langflow is not involved).

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
from tools.website_render import extract_spec_json, render_website_html, style_names

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 800


WEBSITE_BUILD_SCHEMA: Dict[str, Any] = {
    "name": "website_build",
    "description": (
        "Build a complete, responsive single-file HTML website from a structured "
        "site spec and save it to disk. Deterministic — YOU (the assistant) write "
        "the `spec` from the user's request; this tool only renders it (no model "
        "call, no publish). Returns the saved HTML file path. Use for landing "
        "pages / company sites / product showcases. For images in the hero and "
        "cards, generate them first with `image_generate` and pass their URLs in "
        "`images` (first image = hero background, rest fill section cards)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "description": (
                    "The site spec. Fields: site_title (str), brand "
                    "{name, tagline}, nav (list of str), hero {headline, subhead, "
                    "cta}, sections (list of {title, intro, body, items:[{title, "
                    "desc}], phone, email, address}), seo {description}, footer (str)."
                ),
                "properties": {
                    "site_title": {"type": "string"},
                    "brand": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "tagline": {"type": "string"},
                        },
                    },
                    "nav": {"type": "array", "items": {"type": "string"}},
                    "hero": {
                        "type": "object",
                        "properties": {
                            "headline": {"type": "string"},
                            "subhead": {"type": "string"},
                            "cta": {"type": "string"},
                        },
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "intro": {"type": "string"},
                                "body": {"type": "string"},
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "desc": {"type": "string"},
                                        },
                                    },
                                },
                                "phone": {"type": "string"},
                                "email": {"type": "string"},
                                "address": {"type": "string"},
                            },
                        },
                    },
                    "seo": {
                        "type": "object",
                        "properties": {"description": {"type": "string"}},
                    },
                    "footer": {"type": "string"},
                },
            },
            "style": {
                "type": "string",
                "enum": style_names(),
                "description": "Visual style / color palette. Default 简约商务.",
                "default": "简约商务",
            },
            "primary_color": {
                "type": "string",
                "description": "Optional brand accent color as hex (e.g. #2563eb). Overrides the style default.",
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional public image URLs. First = hero background; the rest fill section cards in order.",
            },
        },
        "required": ["spec"],
    },
}


def _sites_cache_dir() -> Path:
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "sites"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_spec(raw: Any) -> Dict[str, Any]:
    """Accept either a spec object or a JSON string (tolerant of code fences)."""
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


def _handle_website_build(args, **kw):
    try:
        spec = _coerce_spec(args.get("spec"))
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"Invalid site spec: {exc}")
    if not isinstance(spec, dict):
        return tool_error("`spec` must be an object.")

    style = args.get("style") or "简约商务"
    primary_color = args.get("primary_color")
    images = _as_list(args.get("images"))

    try:
        html_out = render_website_html(spec, images, style=style, primary_color=primary_color)
    except Exception as exc:  # noqa: BLE001
        logger.warning("website_build render failed: %s", exc, exc_info=True)
        return tool_error(f"Website render failed: {exc}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _sites_cache_dir() / f"site_{ts}_{short}.html"
    try:
        path.write_text(html_out, encoding="utf-8")
    except OSError as exc:
        return tool_error(f"Could not save site HTML: {exc}")

    n_sections = len(spec.get("sections", []) or [])
    logger.info("website_build wrote %d chars -> %s", len(html_out), path)
    return json.dumps({
        "success": True,
        "html_path": str(path),
        "bytes": len(html_out.encode("utf-8")),
        "sections": n_sections,
        "style": style,
        "preview": html_out[:_PREVIEW_CHARS],
        "message": (
            f"Built a {n_sections}-section site → {path}. Open it in a browser to "
            f"view, or read the file for the full HTML."
        ),
    })


registry.register(
    name="website_build",
    toolset="website_gen",
    schema=WEBSITE_BUILD_SCHEMA,
    handler=_handle_website_build,
    check_fn=None,  # pure rendering — always available
    requires_env=[],
    is_async=False,
    emoji="🌐",
)
