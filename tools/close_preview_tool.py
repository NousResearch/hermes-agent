#!/usr/bin/env python3
"""Close the Hermes desktop GUI's preview pane, or one of its tabs.

Lives in the ``desktop_ui`` toolset (same as ``open_preview``), which the GUI
gateway enables only for a session whose source is the desktop app. Emits
``preview.close`` through the shared ``desktop_ui`` bridge; the renderer drops
the matching tab — or the whole pane when no url is given — for the window
that asked and never steals a background session's view.
"""

import json

from tools import desktop_ui
from tools.open_preview_tool import _normalize_target
from tools.registry import registry, tool_error


def close_preview_tool(url: str = "") -> str:
    """Ask the desktop GUI to close the preview pane, or the tab for ``url``."""
    target = _normalize_target(url or "")

    try:
        ok = desktop_ui.emit("preview.close", {"url": target})
    except Exception as exc:
        return tool_error(f"Failed to close the preview pane: {exc}")
    if not ok:
        return tool_error("The preview pane is only available in the Hermes desktop app.")

    return json.dumps({"success": True, "url": target}, ensure_ascii=False)


CLOSE_PREVIEW_SCHEMA = {
    "name": "close_preview",
    "description": (
        "Close the preview pane beside the chat in the Hermes desktop app, or one "
        "tab inside it. Use this when the user asks to close, hide, or dismiss the "
        "preview — e.g. \"close the preview pane\", \"close cnn.com\", \"hide the "
        "preview\". Omit url to close the whole pane (every tab). Pass a web URL, "
        "localhost address, or file path to close only that tab. Counterpart of "
        "open_preview."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "Optional. The tab to close: a web URL (https://… or a bare "
                    "domain), a localhost URL, or a file path. Omit to close the "
                    "whole preview pane."
                ),
            },
        },
    },
}


registry.register(
    name="close_preview",
    toolset="desktop_ui",
    schema=CLOSE_PREVIEW_SCHEMA,
    handler=lambda args, **kw: close_preview_tool(url=args.get("url") or ""),
    emoji="🖼️",
)
