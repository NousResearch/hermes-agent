#!/usr/bin/env python3
"""Read which OS window sits directly underneath the Hermes desktop window.

The window list lives with the OS, so this tool round-trips through the
gateway's blocking-prompt bridge — the same one `read_terminal` uses:
tui_gateway emits ``window.read.request``, the desktop renderer asks its main
process (which owns native window enumeration) and answers with
``window.read.respond``. This module is just schema + a thin dispatcher over
the platform-injected callback.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error


def read_window_below_tool(callback: Optional[Callable] = None) -> str:
    """Return the window underneath the Hermes window as a JSON string."""
    if callback is None:
        return tool_error(
            "read_window_below is only available in the Hermes desktop app."
        )

    try:
        raw = callback()
    except Exception as exc:
        return tool_error(f"Failed to read the window below: {exc}")

    if not raw:
        return tool_error(
            "Could not determine the window underneath (the desktop app did "
            "not answer, or window enumeration is unavailable on this system)."
        )

    # Desktop answers with a JSON object; pass it through, else wrap the raw text.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


READ_WINDOW_BELOW_SCHEMA = {
    "name": "read_window_below",
    "description": (
        "Identify the application window directly underneath (behind) the "
        "Hermes desktop window — what the user is working in behind this app. "
        "Returns JSON: {window: {app, title, bounds{x,y,width,height}, id}, "
        "frontmost: {app, title}, platform}. `title` may be empty when the OS "
        "withholds window titles (e.g. macOS without the Screen Recording "
        "permission — never prompted for, noted in `note`). Other Hermes "
        "windows are skipped: the nearest non-Hermes window is reported. "
        "Metadata only; this never captures pixels or content of other windows."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


registry.register(
    name="read_window_below",
    toolset="desktop_ui",
    schema=READ_WINDOW_BELOW_SCHEMA,
    handler=lambda args, **kw: read_window_below_tool(callback=kw.get("callback")),
    emoji="🪟",
)
