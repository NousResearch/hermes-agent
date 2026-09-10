#!/usr/bin/env python3
"""Read the in-app terminal pane in the Hermes desktop GUI.

The buffer lives in the desktop renderer (xterm.js), so this round-trips through the
gateway's blocking-prompt bridge (as `clarify` does): tui_gateway emits
``terminal.read.request``, the renderer answers ``terminal.read.respond``. Lives in the
``desktop_ui`` toolset, enabled only for desktop-sourced sessions.
"""

from typing import Callable, Optional

from tools.desktop_ui import passthrough_json
from tools.registry import registry, tool_error


def read_pane(callback: Optional[Callable], window, errors: tuple) -> str:
    """Shared body of the read_terminal / read_preview / read_window bridges. ``window`` is
    ``((key, value, floor), ...)`` (None omitted, else int-coerced and floored); ``errors`` =
    (not_desktop, not_integers, fail_prefix, empty)."""
    if callback is None:
        return tool_error(errors[0])
    try:
        kwargs = {key: max(floor, int(val)) for key, val, floor in window if val is not None}
    except (TypeError, ValueError):
        return tool_error(errors[1])
    try:
        raw = callback(**kwargs)
    except Exception as exc:
        return tool_error(f"{errors[2]}{exc}")
    if not raw:
        return tool_error(errors[3])
    return passthrough_json(raw)


def read_terminal_tool(
    start_line: Optional[int] = None,
    count: Optional[int] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Return the in-app terminal's contents (+ line metadata) as a JSON string."""
    return read_pane(callback, (("start", start_line, 0), ("count", count, 1)), (
        "read_terminal is only available in the Hermes desktop app.",
        "start_line and count must be integers.",
        "Failed to read terminal: ",
        "No in-app terminal is open, or the read timed out.",
    ))


READ_TERMINAL_SCHEMA = {
    "name": "read_terminal",
    "description": (
        "Read the in-app terminal pane beside this chat. No args = visible "
        "screen + total_lines; page scrollback with start_line (0 = oldest) "
        "+ count. JSON: {total_lines, start, end, viewport_rows, cursor_row, "
        "text}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start_line": {
                "type": "integer",
                "description": "0-indexed first line (0 = oldest). Omit for the visible screen.",
            },
            "count": {
                "type": "integer",
                "description": "Lines to read from start_line. Defaults to the visible row count.",
            },
        },
    },
}


registry.register(
    name="read_terminal",
    toolset="desktop_ui",
    schema=READ_TERMINAL_SCHEMA,
    handler=lambda args, **kw: read_terminal_tool(
        start_line=args.get("start_line"), count=args.get("count"), callback=kw.get("callback")
    ),
    emoji="🖥️",
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
