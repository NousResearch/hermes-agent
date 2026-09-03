#!/usr/bin/env python3
"""Start/stop live subtitle translation over the app behind the Hermes desktop.

``annotate_screen`` draws the agent's own marks; this drives the desktop's
LIVE-SUBTITLE session — a background loop, owned entirely by the desktop app
and its backend, that watches the subtitle band of the window the user is
watching, OCRs each new line, translates it, and paints the translation over
the original. The agent's only job is this switch: one ``start`` when the user
asks for translated subtitles, one ``stop`` when they are done. No model turn
runs per subtitle line — a two-hour movie is thousands of lines, and the loop
handles all of them without touching the conversation.

Round-trips through the gateway's blocking-prompt bridge like ``tour`` and
``annotate_screen``: tui_gateway emits ``subtitles.control.request``, the
desktop renderer asks its main process (which owns the snapshot loop and the
overlay) and answers ``subtitles.control.respond`` with the outcome.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

ACTIONS = ("start", "stop", "status")


def subtitle_overlay_tool(
    action: str = "status",
    language: Optional[str] = None,
    target: Optional[str] = None,
    band_fraction: Optional[float] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Flip the live-subtitle session on or off, or report on it."""
    if callback is None:
        return tool_error("subtitle_overlay is only available in the Hermes desktop app.")

    verb = (action or "status").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")

    lang = (language or "").strip()
    if verb == "start" and not lang:
        return tool_error("start needs a language (e.g. 'pt', 'Portuguese', 'es').")

    if band_fraction is not None:
        is_number = isinstance(band_fraction, (int, float)) and not isinstance(band_fraction, bool)
        if not is_number or not 0 < band_fraction <= 0.5:
            return tool_error("band_fraction must be a number in (0, 0.5].")

    payload = {
        name: val
        for name, val in (
            ("action", verb),
            ("language", lang or None),
            ("target", (target or "").strip() or None),
            ("band_fraction", band_fraction),
        )
        if val is not None
    }

    try:
        raw = callback(payload)
    except Exception as exc:
        return tool_error(f"Failed to reach the subtitle session: {exc}")

    if not raw:
        return tool_error(
            "The subtitle request timed out, or no GUI window answered. "
            "The Hermes desktop app must be in the foreground of this session."
        )

    # The renderer answers with a JSON object; pass it through, else wrap it.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


SUBTITLE_OVERLAY_SCHEMA = {
    "name": "subtitle_overlay",
    "description": (
        "Live subtitle translation for whatever the user is watching. "
        "action='start' begins a background loop, run entirely by the desktop "
        "app, that reads the subtitle band of the target window, translates "
        "each new line, and paints the translation over the original — "
        "action='stop' ends it, action='status' reports on it. Start it once "
        "when the user asks to translate what they're watching and then get "
        "out of the way: the loop keeps running after your turn ends, handles "
        "every line by itself, and never needs you per line. Do NOT capture "
        "screenshots or call other tools per subtitle. `target` names the app "
        "to watch (e.g. 'Chrome', 'Netflix'); omit it to watch the window "
        "directly behind the Hermes window. A new start replaces any running "
        "session. If the user says the placement is wrong, restart with a "
        "different band_fraction (a larger value watches more of the window's "
        "bottom). Requires the movie's subtitles to be ON and visible."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "start: begin translating. stop: take it down. status: how it's going.",
            },
            "language": {
                "type": "string",
                "description": "Target language for start (e.g. 'pt', 'Portuguese', 'es', 'ja').",
            },
            "target": {
                "type": "string",
                "description": (
                    "App name of the window to watch (matched case-insensitively). "
                    "Omit to use the window directly behind the Hermes window."
                ),
            },
            "band_fraction": {
                "type": "number",
                "description": (
                    "How much of the window's bottom to watch for subtitles, 0-0.5. "
                    "Default 0.28. Only pass it to correct placement."
                ),
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="subtitle_overlay",
    toolset="desktop_ui",
    schema=SUBTITLE_OVERLAY_SCHEMA,
    handler=lambda args, **kw: subtitle_overlay_tool(
        action=args.get("action", "status"),
        language=args.get("language"),
        target=args.get("target"),
        band_fraction=args.get("band_fraction"),
        callback=kw.get("callback"),
    ),
    emoji="💬",
)
