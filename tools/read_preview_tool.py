#!/usr/bin/env python3
"""Read the in-app browser / preview pane in the Hermes desktop GUI.

The preview's content lives in the desktop renderer (a sandboxed ``<webview>``
for URL tabs), so this tool round-trips through the gateway's blocking-prompt
bridge — the same one ``read_terminal`` uses: tui_gateway emits
``preview.read.request``, the renderer serializes the active preview tab and
answers with ``preview.read.respond``. This module is just schema + a thin
dispatcher over the platform-injected callback.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error


def read_preview_tool(
    start: Optional[int] = None,
    count: Optional[int] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Return the active preview tab's contents (+ metadata) as a JSON string."""
    if callback is None:
        return tool_error("read_preview is only available in the Hermes desktop app.")

    try:
        window = {
            key: max(floor, int(val))
            for key, val, floor in (("start", start, 0), ("count", count, 1))
            if val is not None
        }
    except (TypeError, ValueError):
        return tool_error("start and count must be integers.")

    try:
        raw = callback(**window)
    except Exception as exc:
        return tool_error(f"Failed to read the preview pane: {exc}")

    if not raw:
        return tool_error("No preview tab is open, or the read timed out.")

    # Desktop answers with a JSON object; pass it through, else wrap the raw text.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


READ_PREVIEW_SCHEMA = {
    "name": "read_preview",
    "description": (
        "Read what's currently shown in the in-app browser / preview pane of the "
        "Hermes desktop GUI (the pane open_preview opens beside this chat). Call "
        "with no arguments for the first window of the active tab's content. "
        "Returns JSON {kind, url, title, text, start, end, total_chars, note?}: "
        "a URL (Browser) tab's text is the rendered page's visible text — page "
        "through longer pages with `start`/`count` (character offsets, capped "
        "per read); a file tab answers identity only (read the file with "
        "read_file); an artifact tab points back at the conversation. Use after "
        "open_preview, or whenever the user refers to what's on screen in the "
        "browser ('what does this page say?'). To close the pane, use close_preview."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start": {
                "type": "integer",
                "description": "0-indexed character offset into the page text. Omit for the start.",
            },
            "count": {
                "type": "integer",
                "description": "Characters to return from start. Defaults to (and is capped at) the per-read maximum.",
            },
        },
    },
}


registry.register(
    name="read_preview",
    toolset="desktop_ui",
    schema=READ_PREVIEW_SCHEMA,
    handler=lambda args, **kw: read_preview_tool(
        start=args.get("start"),
        count=args.get("count"),
        callback=kw.get("callback"),
    ),
    emoji="🔍",
)
