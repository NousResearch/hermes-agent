#!/usr/bin/env python3
"""Draw temporary, non-interactive annotations over an explicitly captured display."""

import json
import math

from tools import desktop_ui
from tools.registry import registry, tool_error

KINDS = {"arrow", "circle", "label", "line", "point", "rect"}
COLORS = {"amber", "cyan", "emerald", "rose", "white"}
TWO_POINT_KINDS = {"arrow", "circle", "line", "rect"}


def _coordinate(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0 <= number <= 1 else None


def _normalize_annotation(value):
    if not isinstance(value, dict) or value.get("kind") not in KINDS:
        return None
    kind = value["kind"]
    x, y = _coordinate(value.get("x")), _coordinate(value.get("y"))
    if x is None or y is None:
        return None

    result = {"kind": kind, "x": x, "y": y, "color": value.get("color") if value.get("color") in COLORS else "cyan"}
    label = str(value.get("label") or "").strip()[:120]
    if label:
        result["label"] = label
    if kind in TWO_POINT_KINDS:
        x2, y2 = _coordinate(value.get("x2")), _coordinate(value.get("y2"))
        if x2 is None or y2 is None:
            return None
        result.update(x2=x2, y2=y2)
    return result


def _normalize_guide(value):
    if not isinstance(value, dict):
        return None
    guide_id = str(value.get("id") or "").strip()[:80]
    title = str(value.get("title") or "").strip()[:100]
    instruction = str(value.get("instruction") or "").strip()[:240]
    try:
        step = max(1, min(99, round(float(value.get("step")))))
        total = max(step, min(99, round(float(value.get("total")))))
    except (TypeError, ValueError):
        return None
    if not guide_id or not title or not instruction:
        return None
    result = {"id": guide_id, "title": title, "instruction": instruction, "step": step, "total": total}
    success_check = str(value.get("success_check") or "").strip()[:240]
    if success_check:
        result["success_check"] = success_check
    return result


def screen_tutor_annotate_tool(display_id, annotations, mode="replace", ttl_seconds=30, frozen=False, guide=None):
    display_id = str(display_id or "").strip()
    if not display_id:
        return tool_error("screen_tutor_annotate needs the display_id supplied in the Screen Tutor prompt.")
    if not isinstance(annotations, list):
        return tool_error("annotations must be a list of annotation objects.")

    normalized = [item for item in (_normalize_annotation(value) for value in annotations[:24]) if item]
    if not normalized:
        return tool_error("No valid annotations were supplied. Coordinates must be between 0 and 1.")

    try:
        ttl_ms = max(3_000, min(300_000, round(float(ttl_seconds) * 1000)))
    except (TypeError, ValueError):
        ttl_ms = 30_000

    payload = {
        "annotations": normalized,
        "display_id": display_id,
        "frozen": bool(frozen),
        "mode": "append" if mode == "append" else "replace",
        "ttl_ms": ttl_ms,
    }
    normalized_guide = _normalize_guide(guide)
    if normalized_guide:
        payload["guide"] = normalized_guide
    try:
        ok = desktop_ui.emit("screen.tutor.annotations", payload)
    except Exception as exc:
        return tool_error(f"Failed to show screen annotations: {exc}")
    if not ok:
        return tool_error("screen_tutor_annotate is only available in the Hermes desktop app.")

    return json.dumps({"count": len(normalized), "display_id": display_id, "frozen": bool(frozen), "success": True})


SCREEN_TUTOR_ANNOTATE_SCHEMA = {
    "name": "screen_tutor_annotate",
    "description": (
        "Draw temporary, click-through visual guidance over the fresh Screen Tutor screenshot. "
        "Use only after an explicit Screen Tutor capture and only for visible features you can locate confidently. "
        "Coordinates are normalized from 0 to 1. This tool cannot click, type, trade, or operate another app."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "display_id": {"type": "string"},
            "annotations": {
                "type": "array",
                "maxItems": 24,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": sorted(KINDS)},
                        "x": {"type": "number", "minimum": 0, "maximum": 1},
                        "y": {"type": "number", "minimum": 0, "maximum": 1},
                        "x2": {"type": "number", "minimum": 0, "maximum": 1},
                        "y2": {"type": "number", "minimum": 0, "maximum": 1},
                        "label": {"type": "string"},
                        "color": {"type": "string", "enum": sorted(COLORS)},
                    },
                    "required": ["kind", "x", "y"],
                },
            },
            "mode": {"type": "string", "enum": ["replace", "append"], "default": "replace"},
            "ttl_seconds": {"type": "number", "minimum": 3, "maximum": 300, "default": 30},
            "frozen": {"type": "boolean", "default": False},
            "guide": {
                "type": "object",
                "description": "Optional closed-loop teaching step shown in the HUD.",
                "properties": {
                    "id": {"type": "string", "description": "Stable id reused for every step in this guide."},
                    "title": {"type": "string"},
                    "instruction": {"type": "string"},
                    "step": {"type": "integer", "minimum": 1, "maximum": 99},
                    "total": {"type": "integer", "minimum": 1, "maximum": 99},
                    "success_check": {"type": "string", "description": "What must be visibly true before advancing."},
                },
                "required": ["id", "title", "instruction", "step", "total"],
            },
        },
        "required": ["display_id", "annotations"],
    },
}


registry.register(
    name="screen_tutor_annotate",
    toolset="desktop_ui",
    schema=SCREEN_TUTOR_ANNOTATE_SCHEMA,
    handler=lambda args, **kw: screen_tutor_annotate_tool(
        display_id=args.get("display_id", ""),
        annotations=args.get("annotations"),
        mode=args.get("mode", "replace"),
        ttl_seconds=args.get("ttl_seconds", 30),
        frozen=args.get("frozen", False),
        guide=args.get("guide"),
    ),
    emoji="🖍️",
)
