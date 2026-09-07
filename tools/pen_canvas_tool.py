#!/usr/bin/env python3
"""Drive a pen.dev design canvas from the Hermes desktop GUI.

Hermes desktop embeds the hosted pen.dev editor (app.pen.dev/new?embed) in a
Canvas pane. This tool is the agent's door into that canvas: it round-trips
through the gateway's blocking-prompt bridge — the same one ``read_preview``
uses — so it works wherever the CLIENT is, remote backends included.
tui_gateway emits ``pen.tool.request``, the renderer runs the operation
against the live canvas and answers with ``pen.tool.respond``.

Host actions (``open`` / ``close`` / ``schema``) own the pane and the live
tool list. ``open`` and ``schema`` send ``get-mcp-schema`` so the agent
sees Pencil's current tools. Everything else is ``mcp-tool-call``.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions.
"""

import base64
import binascii
import json
import os
import time
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error

# Pen results are design-document JSON — schemas, node trees, guideline text.
# Cap what crosses into model context; the tail is truncated with a note.
_MAX_RESULT_CHARS = 48_000

# A string field this long that decodes as base64 is image data (screenshots,
# exports) — materialize it to disk instead of flooding the context window.
_BASE64_MATERIALIZE_THRESHOLD = 4_096


def _screenshot_dir() -> str:
    root = os.path.join(str(get_hermes_home()), "pen_canvas")
    os.makedirs(root, exist_ok=True)
    return root


def _materialize_images(value: Any) -> Any:
    """Replace embedded base64 image payloads with saved file paths."""
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
    if not action:
        return tool_error("action is required.")
    if args is not None and not isinstance(args, dict):
        return tool_error("args must be an object.")

    try:
        raw = callback(action, args or {})
    except Exception as exc:
        return tool_error(f"Failed to reach the pen canvas: {exc}")

    if not raw:
        return tool_error(
            "No answer from the desktop app — is a Canvas tab open? "
            "Open one with pen_canvas(action='open')."
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
        "beside this chat. You and the user share one live canvas. "
        "'open' opens a tab (args: {name?: 2-4 word title from the brief, "
        "path?: absolute .pen file} — ALWAYS pass name when creating). "
        "'close' puts the canvas away (file stays in the library). "
        "'schema' re-fetches the editor's live MCP tool list (get-mcp-schema) "
        "— call it when tools look stale; Pencil changes them. "
        "open already returns that list. Any other action is an editor tool "
        "from that list, forwarded verbatim. Workflow: open → use the "
        "returned tools → edit in small steps. If no Canvas tab is open, "
        "call open first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "'open' or 'close' for the pane, 'schema' for the live "
                    "MCP tool list; any other string is an editor tool name "
                    "from that list."
                ),
            },
            "args": {
                "type": "object",
                "description": (
                    "Arguments for the action, passed to the editor verbatim. "
                    "Omit when the action needs none."
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
