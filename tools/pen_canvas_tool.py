#!/usr/bin/env python3
"""Drive a pen.dev design canvas from the Hermes desktop GUI.

Hermes desktop hosts the user's installed pen.dev editor in "Canvas" tabs
(apps/desktop/electron/pen-canvas.ts). This tool is the agent's door into that
canvas: it round-trips through the gateway's blocking-prompt bridge — the same
one ``read_preview`` uses — so it works wherever the CLIENT is, remote
backends included. tui_gateway emits ``pen.tool.request``, the renderer runs
the pen operation against the live canvas (or the user's running pen.dev
desktop app when no Canvas tab is open — the HUD-mode path) and answers with
``pen.tool.respond``.

This module is just schema + a thin dispatcher over the platform-injected
callback; the pen operation names and payloads pass through verbatim so the
editor's own tool surface stays the source of truth.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions.
"""

import base64
import binascii
import json
import os
import time
from typing import Any, Callable, Optional

from tools.registry import registry, tool_error

# Pen results are design-document JSON — schemas, node trees, guideline text.
# Cap what crosses into model context; the tail is truncated with a note.
_MAX_RESULT_CHARS = 48_000

# A string field this long that decodes as base64 is image data (screenshots,
# exports) — materialize it to disk instead of flooding the context window.
_BASE64_MATERIALIZE_THRESHOLD = 4_096

_ACTIONS = {
    "open",
    "close",
    "execute",
    "get_app_state",
    "get_guidelines",
    "get_screenshot",
    "get_selection",
    "revert",
    "export_nodes",
    "export_html",
}


def _screenshot_dir() -> str:
    root = os.path.join(
        os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes")), "pen_canvas"
    )
    os.makedirs(root, exist_ok=True)
    return root


def _materialize_images(value: Any) -> Any:
    """Replace embedded base64 image payloads with saved file paths.

    pen's ``get_screenshot`` / ``TakeScreenshot()`` answers carry raw base64
    image data. The agent reads images through vision, not through tool text,
    so write them to ``~/.hermes/pen_canvas/`` and hand back the path.
    """
    if isinstance(value, dict):
        return {key: _materialize_images(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_images(item) for item in value]
    if not isinstance(value, str) or len(value) < _BASE64_MATERIALIZE_THRESHOLD:
        return value

    raw = value
    suffix = "png"
    if raw.startswith("data:image/"):
        header, _, raw = raw.partition(",")
        suffix = header.removeprefix("data:image/").partition(";")[0] or "png"
    try:
        blob = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return value

    path = os.path.join(_screenshot_dir(), f"canvas-{int(time.time() * 1000)}.{suffix}")
    try:
        with open(path, "wb") as handle:
            handle.write(blob)
    except OSError:
        return value
    return {"saved_to": path, "note": "image written to disk — view it with vision_analyze"}


def pen_canvas_tool(
    action: str = "",
    args: Optional[dict] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Run a pen.dev canvas operation and return its result as a JSON string."""
    if callback is None:
        return tool_error("pen_canvas is only available in the Hermes desktop app.")

    action = str(action or "").strip()
    if action not in _ACTIONS:
        return tool_error(
            f"Unknown action {action!r}. One of: {', '.join(sorted(_ACTIONS))}."
        )
    if args is not None and not isinstance(args, dict):
        return tool_error("args must be an object.")

    try:
        raw = callback(action, args or {})
    except Exception as exc:
        return tool_error(f"Failed to reach the pen canvas: {exc}")

    if not raw:
        return tool_error(
            "No answer from the desktop app — is a Canvas tab open (or the "
            "pen.dev app running)? Open one with pen_canvas(action='open'), "
            "or the operation timed out."
        )

    try:
        result = _materialize_images(json.loads(raw))
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)

    text = json.dumps(result, ensure_ascii=False)
    if len(text) > _MAX_RESULT_CHARS:
        text = json.dumps(
            {
                "truncated": True,
                "note": (
                    f"result was {len(text)} chars; showing the first "
                    f"{_MAX_RESULT_CHARS}. Ask for less — a smaller node, "
                    "fewer schema sections, one guideline at a time."
                ),
                "head": text[:_MAX_RESULT_CHARS],
            },
            ensure_ascii=False,
        )
    return text


PEN_CANVAS_SCHEMA = {
    "name": "pen_canvas",
    "description": (
        "Design on a pen.dev canvas in the Hermes desktop app — the Canvas tab "
        "beside this chat. You and the user share one live canvas: your edits "
        "render instantly, and the user can draw alongside you with the full "
        "editor. Actions: 'open' opens a Canvas tab (args: {name?: short "
        "friendly title, path?: absolute .pen file, template?: name} — ALWAYS "
        "pass `name` when creating a new canvas: 2-4 words from the design "
        "brief, e.g. 'Robot factory dashboard' — it names the .pen file the "
        "way chats get auto-titles; omit name only when opening an existing "
        "path); 'close' "
        "puts the canvas away (the file is untouched and stays in the "
        "library); "
        "'get_app_state' reads the document + editor state (args: "
        "{include_schema, include_canvas_design, include_scripts_and_shaders}: "
        "booleans — pass include_schema true on your first call to learn the "
        ".pen node schema); 'get_guidelines' lists/loads design guides and "
        "styles (args: {} to list, then {category: 'guide'|'style', name, "
        "params?}); 'execute' runs pen's design JavaScript to insert/update/"
        "delete/read nodes (args: {input: snippet} — e.g. hero=Insert(document,"
        "{type:'frame',name:'Hero',x:0,y:0,width:1440,height:900,fill:'#0A0A0A'}); "
        "on a failed snippet, patch it with {editId, edits:[{find,replace}]}); "
        "'get_screenshot' renders a node to an image file you can view with "
        "vision_analyze (args: {nodeId} or nodeId 'document' — expensive, use "
        "after a section is done, not after every execute); 'get_selection' "
        "reads what the user has selected RIGHT NOW in the editor (no args; "
        "returns {nodes: [{id, name, type, bounds}]}) — call it whenever the "
        "user says 'this'/'these'/'the selected one' so you act on their "
        "actual selection instead of guessing from names; 'export_nodes' / "
        "'export_html' write PNG/JPEG/WEBP/PDF or HTML files (args: pen's "
        "export payloads: {nodeIds, outputDir|outputPath, format, ...}). "
        "Workflow: open → get_app_state (schema) → get_guidelines → execute in "
        "small steps → screenshot to verify. If no Canvas tab is open but the "
        "user has the pen.dev desktop app running (e.g. Hermes is in HUD mode "
        "over it), every action except 'open' reaches their live pen.dev "
        "document directly — design there without opening a canvas."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(_ACTIONS),
                "description": "The canvas operation to run.",
            },
            "args": {
                "type": "object",
                "description": (
                    "Arguments for the action, passed to the pen editor "
                    "verbatim (see the per-action shapes in the tool "
                    "description). Omit when the action needs none."
                ),
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="pen_canvas",
    toolset="desktop_ui",
    schema=PEN_CANVAS_SCHEMA,
    handler=lambda args, **kw: pen_canvas_tool(
        action=args.get("action", ""),
        args=args.get("args"),
        callback=kw.get("callback"),
    ),
    emoji="✏️",
)
