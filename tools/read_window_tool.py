#!/usr/bin/env python3
"""Read which OS window sits directly underneath the Hermes desktop window.

The window list lives with the OS, so this round-trips through the gateway's
blocking-prompt bridge like `read_terminal`: ``window.read.request`` -> the renderer's
main process (native window enumeration) -> ``window.read.respond``.
"""

from typing import Callable, Optional

from tools.read_terminal_tool import read_pane
from tools.registry import registry


def read_window_below_tool(callback: Optional[Callable] = None) -> str:
    """Return the window underneath the Hermes window as a JSON string."""
    return read_pane(callback, (), (
        "read_window_below is only available in the Hermes desktop app.",
        "",
        "Failed to read the window below: ",
        "Could not determine the window underneath (the desktop app did "
        "not answer, or window enumeration is unavailable on this system).",
    ))


READ_WINDOW_BELOW_SCHEMA = {
    "name": "read_window_below",
    "description": (
        "Identify the app window directly behind the Hermes desktop window "
        "(what the user is working in). JSON: {window: {app, title, bounds, "
        "id}, frontmost, platform}. title may be empty when the OS withholds "
        "it (noted in `note`); where windows cannot be enumerated at all, "
        "{error, platform} says what would fix it — relay that instead of "
        "retrying. Metadata only; never captures pixels."
    ),
    "parameters": {
        "type": "object", "properties": {}
    },
}


registry.register(
    name="read_window_below",
    toolset="desktop_ui",
    schema=READ_WINDOW_BELOW_SCHEMA,
    handler=lambda args, **kw: read_window_below_tool(callback=kw.get("callback")),
    emoji="🪟",
)
