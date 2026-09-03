#!/usr/bin/env python3
"""Show one non-interactive pointer on a captured desktop display.

Screen Tutor captures a display only after an explicit composer action. The
model receives that image and may use this desktop-only tool once to point at a
visible control. Coordinates are normalized to the captured display so mixed
DPI and negative multi-monitor origins remain Electron's responsibility.
"""

import json
import math

from tools import desktop_ui
from tools.registry import registry, tool_error


def screen_tutor_point_tool(display_id: str, x: float, y: float, label: str = "") -> str:
    display_id = str(display_id or "").strip()
    label = str(label or "").strip()[:120]

    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return tool_error("screen_tutor_point needs numeric x and y coordinates between 0 and 1.")

    if not display_id:
        return tool_error("screen_tutor_point needs the display_id supplied in the Screen Tutor prompt.")

    if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
        return tool_error("screen_tutor_point x and y must each be between 0 and 1.")

    payload = {"display_id": display_id, "x": x, "y": y}
    if label:
        payload["label"] = label

    try:
        ok = desktop_ui.emit("screen.tutor.point", payload)
    except Exception as exc:
        return tool_error(f"Failed to show the Screen Tutor pointer: {exc}")

    if not ok:
        return tool_error("screen_tutor_point is only available in the Hermes desktop app.")

    return json.dumps({"display_id": display_id, "success": True, "x": x, "y": y}, ensure_ascii=False)


SCREEN_TUTOR_POINT_SCHEMA = {
    "name": "screen_tutor_point",
    "description": (
        "Show one temporary, click-through pointer on the screenshot's display. "
        "Use only in Screen Tutor mode, only after inspecting the attached fresh "
        "screenshot, and only when the requested visible control can be located "
        "with confidence. Coordinates are normalized within the screenshot: "
        "x=0/y=0 is top-left and x=1/y=1 is bottom-right. This tool points only; "
        "it never clicks, types, or controls the other app. Do not guess."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "display_id": {
                "type": "string",
                "description": "Exact display id supplied in the Screen Tutor prompt.",
            },
            "x": {"type": "number", "minimum": 0, "maximum": 1},
            "y": {"type": "number", "minimum": 0, "maximum": 1},
            "label": {"type": "string", "description": "Optional short label for the control."},
        },
        "required": ["display_id", "x", "y"],
    },
}


registry.register(
    name="screen_tutor_point",
    toolset="desktop_ui",
    schema=SCREEN_TUTOR_POINT_SCHEMA,
    handler=lambda args, **kw: screen_tutor_point_tool(
        display_id=args.get("display_id", ""),
        x=args.get("x"),
        y=args.get("y"),
        label=args.get("label", ""),
    ),
    emoji="🎯",
)
