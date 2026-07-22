#!/usr/bin/env python3
"""Open a URL, dev server, or file in the Hermes desktop GUI's preview pane.

The preview pane lives in the desktop renderer, so this tool bridges through a
gateway-injected emitter: the desktop ``tui_gateway`` wires ``set_preview_emitter``
at session start to emit a ``preview.open`` event the renderer handles (opening
the pane beside the chat, scoped to the window that asked). Like ``read_terminal``
and ``close_terminal`` it is gated on ``HERMES_DESKTOP`` so it never appears
outside the GUI. Fire-and-forget: the renderer never steals focus for a
background session.
"""

import json
import re
from typing import Callable, Optional

from gateway.session_context import get_session_env
from tools.registry import registry, tool_error
from utils import env_var_enabled

# Set by the desktop gateway (tui_gateway) to bridge this tool → a renderer
# event. ``None`` everywhere else, which is how the tool reports "desktop only".
_preview_emitter: Optional[Callable[[str, str, str], None]] = None


def set_preview_emitter(fn: Optional[Callable[[str, str, str], None]]) -> None:
    """Install the (sid, url, label) → emit sink. Called by the desktop gateway."""
    global _preview_emitter
    _preview_emitter = fn


def _normalize_target(raw: str) -> str:
    """Coax a bare host/domain into a fetchable URL; leave paths + schemes alone.

    ``www.cnn.com`` → ``https://www.cnn.com``; ``localhost:3000`` →
    ``http://localhost:3000``. File paths and explicit schemes pass through for
    the renderer's preview normalizer to classify.
    """
    v = raw.strip().strip("`").strip()
    if not v or "://" in v or v.startswith(("/", "./", "../", "~", "file:")):
        return v
    if re.match(r"^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(:\d+)?(/|$)", v, re.I):
        return "http://" + v
    if re.match(r"^[\w.-]+\.[a-z]{2,}(:\d+)?(/.*)?$", v, re.I):
        return "https://" + v
    return v


def open_preview_tool(url: str, label: str = "") -> str:
    """Ask the desktop GUI to show ``url`` in the preview pane beside the chat."""
    target = _normalize_target(url or "")
    if not target:
        return tool_error(
            "url is required — a web URL (https://…), a localhost dev server, or a "
            "file path to show in the preview pane."
        )

    emit = _preview_emitter
    if emit is None:
        return tool_error("The preview pane is only available in the Hermes desktop app.")

    label = (label or "").strip()
    try:
        emit(get_session_env("HERMES_UI_SESSION_ID", ""), target, label)
    except Exception as exc:
        return tool_error(f"Failed to open the preview pane: {exc}")

    return json.dumps({"success": True, "url": target, "label": label}, ensure_ascii=False)


def check_open_preview_requirements() -> bool:
    """Desktop GUI only — HERMES_DESKTOP is set on the gateway the app spawns."""
    return env_var_enabled("HERMES_DESKTOP")


OPEN_PREVIEW_SCHEMA = {
    "name": "open_preview",
    "description": (
        "Open something in the preview pane beside the chat in the Hermes desktop "
        "app. Use this when the user asks to see a page, dev server, or file in the "
        "preview pane — e.g. \"open cnn.com in the preview pane\" or \"preview "
        "localhost:3000\". Accepts a web URL (a bare domain like www.cnn.com is fine), "
        "a localhost dev-server URL, or a file path (HTML renders live; other files "
        "show their contents). The pane opens for the current window only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "What to preview: a web URL (https://… or a bare domain), a "
                    "localhost URL (localhost:3000), or a file path."
                ),
            },
            "label": {
                "type": "string",
                "description": "Optional tab label; defaults to the target's name.",
            },
        },
        "required": ["url"],
    },
}


registry.register(
    name="open_preview",
    toolset="terminal",
    schema=OPEN_PREVIEW_SCHEMA,
    handler=lambda args, **kw: open_preview_tool(url=args.get("url", ""), label=args.get("label", "")),
    check_fn=check_open_preview_requirements,
    emoji="🖼️",
)
