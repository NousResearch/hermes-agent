#!/usr/bin/env python3
"""Read the in-app terminal pane in the Hermes desktop GUI.

The buffer lives in the desktop renderer (xterm.js), so this round-trips through the
gateway's blocking-prompt bridge (as `clarify` does): tui_gateway emits
``terminal.read.request``, the renderer answers ``terminal.read.respond``. Lives in the
``desktop_ui`` toolset, enabled only for desktop-sourced sessions.
"""

import json
from typing import Callable, Optional

from agent.redact import redact_sensitive_text
from tools.desktop_ui import passthrough_json
from tools.registry import registry, tool_error

REDACT_FAILED = "[redacted: failed]"


def _redact_leaves(node):
    """Redact every string leaf of a decoded JSON value; other types pass through.
    A redaction error yields a placeholder, never the raw text."""
    if isinstance(node, str):
        try:
            return redact_sensitive_text(node, force=True, redact_url_credentials=True)
        except Exception:
            return REDACT_FAILED
    if isinstance(node, list):
        return [_redact_leaves(item) for item in node]
    if isinstance(node, dict):
        return {key: _redact_leaves(val) for key, val in node.items()}
    return node


def read_pane(callback: Optional[Callable], window, errors: tuple) -> str:
    """Shared body of the read_terminal / read_preview / read_window bridges. ``window`` is
    ``((key, value, floor), ...)`` (None omitted, else int-coerced and floored); ``errors`` =
    (not_desktop, not_integers, fail_prefix, empty). String values in the answer are
    secret-redacted before they reach the model, like read_file / terminal output."""
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
    return json.dumps(_redact_leaves(json.loads(passthrough_json(raw))), ensure_ascii=False)


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
        "No in-app terminal answered: nothing is open, or the bridge timed out. "
        "If the terminal pane IS open, the desktop app may be older than this "
        "backend — update it and retry.",
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
