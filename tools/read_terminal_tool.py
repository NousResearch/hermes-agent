#!/usr/bin/env python3
"""Read the in-app terminal pane in the Hermes desktop GUI.

The embedded terminal's buffer lives in the desktop renderer (xterm.js), so this
tool round-trips through the gateway's blocking-prompt bridge — the same one
`clarify` uses: tui_gateway emits ``terminal.read.request``, the renderer answers
with ``terminal.read.respond``. This module is just schema + a thin dispatcher
over the platform-injected callback.
"""

import json
import os
from typing import Callable, Optional

from tools.registry import registry, tool_error
from utils import env_var_enabled


def read_terminal_tool(
    start_line: Optional[int] = None,
    count: Optional[int] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Return the in-app terminal's contents (+ line metadata) as a JSON string."""
    if callback is None:
        return tool_error("read_terminal is only available in the Hermes desktop app.")

    try:
        window = {
            key: max(floor, int(val))
            for key, val, floor in (("start", start_line, 0), ("count", count, 1))
            if val is not None
        }
    except (TypeError, ValueError):
        return tool_error("start_line and count must be integers.")

    try:
        raw = callback(**window)
    except Exception as exc:
        return tool_error(f"Failed to read terminal: {exc}")

    if not raw:
        return tool_error("No in-app terminal is open, or the read timed out.")

    # Desktop answers with a JSON object; pass it through, else wrap the raw text.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


def check_read_terminal_requirements() -> bool:
    """Desktop GUI only — HERMES_DESKTOP is set on the gateway the app spawns."""
    return env_var_enabled("HERMES_DESKTOP")


READ_TERMINAL_SCHEMA = {
    "name": "read_terminal",
    "description": (
        "Read what's currently shown in the in-app terminal pane of the Hermes "
        "desktop GUI (the embedded shell beside this chat). Call with no arguments "
        "to get the visible screen plus the total line count (`total_lines`). To "
        "page through scrollback, pass `start_line` (0 = oldest line) and `count`; "
        "valid lines are [0, total_lines). Returns JSON: "
        "{total_lines, start, end, viewport_rows, cursor_row, text}."
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
    toolset="terminal",
    schema=READ_TERMINAL_SCHEMA,
    handler=lambda args, **kw: read_terminal_tool(
        start_line=args.get("start_line"),
        count=args.get("count"),
        callback=kw.get("callback"),
    ),
    check_fn=check_read_terminal_requirements,
    emoji="🖥️",
)
