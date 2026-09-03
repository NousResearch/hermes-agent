#!/usr/bin/env python3
"""Draw instruction marks on the user's screen from the Hermes desktop GUI.

``annotate_preview`` marks the page inside the in-app browser; this is its
sibling for the rest of the screen. HUD mode floats Hermes over whatever the
user is actually working in — a chess app, a DAW, a spreadsheet — and "which
move should I make?" wants an answer drawn on the board, not a paragraph of
square names. The desktop's Electron main process owns a transparent,
click-through, always-on-top overlay window and paints the shapes; this module
is just schema + a thin dispatcher over the platform-injected callback.

Coordinates ride the agent's own eyes: the model passes the pixel size of the
screenshot it analyzed (``frame_width``/``frame_height``) and every shape's
coordinates in that same pixel space. Electron resolves the target window's
current bounds and maps frame pixels onto screen points, so Retina/DPI scaling
never leaks into the schema.

Round-trips through the gateway's blocking-prompt bridge like ``tour``:
tui_gateway emits ``screen.annotate.request``, the desktop renderer asks its
main process (which owns the overlay window) and answers
``screen.annotate.respond`` with the outcome.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

ACTIONS = ("draw", "clear")
SHAPE_KINDS = ("circle", "rect", "arrow", "line", "label", "polyline")
COLORS = ("red", "green", "blue", "yellow", "orange", "purple", "white", "black")

# A trend, channel edge, or wedge is one polyline. Cap vertices so a model
# cannot dump a sampled path that paints over the whole chart.
MAX_POLYLINE_POINTS = 24
MAX_STEP = 12

# Required numeric fields per shape kind. `label` additionally needs `text`,
# and `polyline` needs `points` — both checked separately.
_REQUIRED_FIELDS = {
    "circle": ("x", "y"),
    "rect": ("x", "y", "width", "height"),
    "arrow": ("from_x", "from_y", "to_x", "to_y"),
    "line": ("from_x", "from_y", "to_x", "to_y"),
    "label": ("x", "y"),
    "polyline": (),
}


def _is_number(value) -> bool:
    """True for real numbers only — bool is an int in Python and must not pass."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_shape(index: int, shape) -> Optional[str]:
    """Return an error string for one shape, or None when it is drawable."""
    if not isinstance(shape, dict):
        return f"shapes[{index}] must be an object."

    kind = shape.get("kind")
    if kind not in SHAPE_KINDS:
        return f"shapes[{index}].kind must be one of: {', '.join(SHAPE_KINDS)}."

    for field in _REQUIRED_FIELDS[kind]:
        if not _is_number(shape.get(field)):
            return f"shapes[{index}] ({kind}) needs numeric '{field}'."

    if kind == "label" and not str(shape.get("text") or "").strip():
        return f"shapes[{index}] (label) needs non-empty 'text'."

    if kind == "polyline":
        problem = _validate_polyline_points(index, shape.get("points"))
        if problem:
            return problem

    dashed = shape.get("dashed")
    if dashed is not None and not isinstance(dashed, bool):
        return f"shapes[{index}].dashed must be a boolean."

    step_problem = _validate_step(index, shape.get("step"))
    if step_problem:
        return step_problem

    color = shape.get("color")
    if color is not None and color not in COLORS:
        return f"shapes[{index}].color must be one of: {', '.join(COLORS)}."

    return None


def _validate_polyline_points(index: int, points) -> Optional[str]:
    """A polyline is a vertex list — at least a segment, at most a sketch."""
    if not isinstance(points, list) or len(points) < 2:
        return f"shapes[{index}] (polyline) needs at least two points."
    if len(points) > MAX_POLYLINE_POINTS:
        return (
            f"shapes[{index}] (polyline) has at most {MAX_POLYLINE_POINTS} points."
        )
    for i, point in enumerate(points):
        if not isinstance(point, dict):
            return f"shapes[{index}].points[{i}] must be an object."
        if not _is_number(point.get("x")) or not _is_number(point.get("y")):
            return f"shapes[{index}].points[{i}] needs numeric 'x' and 'y'."
    return None


def _validate_step(index: int, value) -> Optional[str]:
    """A walkthrough badge. None is fine; anything else must be 1–MAX_STEP."""
    if value is None:
        return None
    if isinstance(value, bool) or not _is_number(value):
        return f"shapes[{index}].step must be an integer 1–{MAX_STEP}."
    number = int(value)
    if number != value or not (1 <= number <= MAX_STEP):
        return f"shapes[{index}].step must be an integer 1–{MAX_STEP}."
    return None


def annotate_screen_tool(
    action: str = "draw",
    target: Optional[str] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    shapes: Optional[list] = None,
    ttl_seconds: Optional[float] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Dispatch one screen-annotation action to the desktop and return its outcome."""
    if callback is None:
        return tool_error("annotate_screen is only available in the Hermes desktop app.")

    verb = (action or "draw").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")

    if ttl_seconds is not None and (not _is_number(ttl_seconds) or ttl_seconds <= 0):
        return tool_error("ttl_seconds must be a positive number.")

    if verb == "draw":
        if not isinstance(shapes, list) or not shapes:
            return tool_error("draw needs a non-empty shapes array.")
        for i, shape in enumerate(shapes):
            problem = _validate_shape(i, shape)
            if problem:
                return tool_error(problem)
        if not (_is_number(frame_width) and frame_width > 0 and _is_number(frame_height) and frame_height > 0):
            return tool_error(
                "draw needs frame_width and frame_height — the pixel size of the "
                "screenshot your coordinates come from."
            )

    payload = {
        key: val
        for key, val in (
            ("action", verb),
            ("target", (target or "").strip() or None),
            ("frame", {"width": frame_width, "height": frame_height} if verb == "draw" else None),
            ("shapes", shapes if verb == "draw" else None),
            ("ttl_seconds", ttl_seconds),
        )
        if val is not None
    }

    try:
        raw = callback(payload)
    except Exception as exc:
        return tool_error(f"Failed to annotate the screen: {exc}")

    if not raw:
        return tool_error(
            "The annotation timed out, or no GUI window answered. "
            "The Hermes desktop app must be in the foreground of this session."
        )

    # The renderer answers with a JSON object; pass it through, else wrap it.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


_SHAPE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": list(SHAPE_KINDS),
            "description": (
                "What to draw. circle/rect outline a spot, arrow points from A "
                "to B, line is a single segment, polyline is a path (trend, "
                "channel edge, wedge), label is standalone text."
            ),
        },
        "x": {"type": "number", "description": "Center x for circle, top-left x for rect, anchor x for label."},
        "y": {"type": "number", "description": "Center y for circle, top-left y for rect, anchor y for label."},
        "radius": {"type": "number", "description": "Circle radius in frame pixels. Omit for a sensible default."},
        "width": {"type": "number", "description": "Rect width."},
        "height": {"type": "number", "description": "Rect height."},
        "from_x": {"type": "number", "description": "Arrow/line start x."},
        "from_y": {"type": "number", "description": "Arrow/line start y."},
        "to_x": {"type": "number", "description": "Arrow/line end x (the arrowhead)."},
        "to_y": {"type": "number", "description": "Arrow/line end y (the arrowhead)."},
        "text": {"type": "string", "description": "For label: the text to draw. Newlines make stacked lines."},
        "font_size": {
            "type": "number",
            "description": "For label: font size in frame pixels. Omit for the default caption size.",
        },
        "label": {
            "type": "string",
            "description": "Optional short caption drawn beside a circle/rect/arrow/line/polyline.",
        },
        "dashed": {
            "type": "boolean",
            "description": (
                "Draw the stroke dashed. Use on line, arrow, or polyline for a "
                "projected level, a weaker structure, or the second parallel "
                "of a channel."
            ),
        },
        "step": {
            "type": "integer",
            "description": (
                f"Optional walkthrough number 1–{MAX_STEP}, drawn as a badge "
                "on the mark. Use on a how-to: step 1 on the menu, step 2 on "
                "the button, step 3 on the panel. Omit for a single glance."
            ),
        },
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Vertex x in frame pixels."},
                    "y": {"type": "number", "description": "Vertex y in frame pixels."},
                },
                "required": ["x", "y"],
            },
            "description": (
                "For polyline: vertices in order, in frame pixels. At least "
                f"two, at most {MAX_POLYLINE_POINTS}. One path per structure "
                "— do not fake a trend with a pile of lines."
            ),
        },
        "color": {
            "type": "string",
            "enum": list(COLORS),
            "description": "Mark color. Defaults to red.",
        },
    },
    "required": ["kind"],
}

ANNOTATE_SCREEN_SCHEMA = {
    "name": "annotate_screen",
    "description": (
        "Draw instruction marks — circles, rectangles, arrows, lines, "
        "polylines, text labels — directly on the user's SCREEN, over the "
        "app they are working in, via a transparent click-through overlay "
        "the Hermes desktop app owns. Use it to point instead of describing: "
        "the chess piece to move, the button to click, a numbered how-to on "
        "the live UI (Photoshop, a DAW, a settings panel), or a chart/diagram "
        "read (trendline, channel, support/resistance, up vs down) drawn on "
        "the candles rather than narrated. Workflow: look at the target "
        "first (a screenshot of the window or screen), then pass that "
        "image's pixel size as frame_width/frame_height and give every "
        "shape's coordinates in that same image-pixel space — the desktop "
        "maps them onto the live window, handling display scaling for you. "
        "`target` names the app to anchor to (the window's app name, e.g. "
        "'Photoshop' or 'Chrome'); omit it to anchor to the window directly "
        "behind the Hermes window, or pass 'screen' when your coordinates "
        "cover the whole display. Marks never intercept clicks, a new draw "
        "replaces the previous one, and everything fades out after "
        "ttl_seconds (default 30, a glance) — for a walkthrough or a chart "
        "read pass ~180 so the user can follow it; action='clear' removes "
        "them sooner. Keep it to a few decisive shapes; numbered steps on "
        "the real buttons say more than a paragraph of menu paths."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "draw: put shapes up (replaces any previous set). clear: take them all down now.",
            },
            "target": {
                "type": "string",
                "description": (
                    "App name of the window to anchor coordinates to (matched "
                    "case-insensitively, e.g. 'Chess'), or 'screen' for the whole "
                    "display. Omit to use the window directly behind the Hermes window."
                ),
            },
            "frame_width": {
                "type": "integer",
                "description": "Pixel width of the screenshot your coordinates come from. Required for draw.",
            },
            "frame_height": {
                "type": "integer",
                "description": "Pixel height of the screenshot your coordinates come from. Required for draw.",
            },
            "shapes": {
                "type": "array",
                "items": _SHAPE_SCHEMA,
                "description": "For draw: the marks to put up, in frame-pixel coordinates.",
            },
            "ttl_seconds": {
                "type": "number",
                "description": (
                    "Seconds before the marks fade on their own. Default 30 "
                    "(a glance: chess move, which button). For a walkthrough "
                    "or a chart the user will sit with, pass ~180. "
                    "Clamped to 3–900."
                ),
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="annotate_screen",
    toolset="desktop_ui",
    schema=ANNOTATE_SCREEN_SCHEMA,
    handler=lambda args, **kw: annotate_screen_tool(
        action=args.get("action", "draw"),
        target=args.get("target"),
        frame_width=args.get("frame_width"),
        frame_height=args.get("frame_height"),
        shapes=args.get("shapes"),
        ttl_seconds=args.get("ttl_seconds"),
        callback=kw.get("callback"),
    ),
    emoji="🎯",
)
