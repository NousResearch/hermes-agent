#!/usr/bin/env python3
"""Reveal/focus a pane in the Hermes desktop GUI (``pane.reveal`` via ``desktop_ui``).

The renderer runs each pane's own reveal path and only acts on the active window, so
a background turn never moves the user's focus. URLs/files go through `desktop_preview`.
"""

from tools import desktop_ui
from tools.registry import registry, tool_error

PANES = ("chat", "files", "terminal", "review", "sessions")


def focus_pane_tool(pane: str) -> str:
    """Ask the desktop GUI to reveal and focus ``pane``."""
    name = (pane or "").strip().lower()
    if name not in PANES:
        return tool_error(f"pane must be one of: {', '.join(PANES)}.")
    return desktop_ui.emit_or_error(
        "pane.reveal", {"pane": name}, f"Failed to focus the {name} pane: ",
        "Pane focus is only available in the Hermes desktop app.", {"success": True, "pane": name})


registry.register(
    name="focus_pane",
    toolset="desktop_ui",
    schema={
        "name": "focus_pane",
        "description": (
            "Reveal and focus a Hermes desktop pane when the user asks to see it: "
            "chat, files, terminal, review (git diff), or sessions. For URLs/"
            "files use the desktop_preview tool instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pane": {
                    "type": "string",
                    "enum": list(PANES),
                    "description": "Which pane to reveal.",
                },
            },
            "required": ["pane"],
        },
    },
    handler=lambda args, **kw: focus_pane_tool(pane=args.get("pane", "")),
    emoji="🪟",
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402

FOCUS_PANE_SCHEMA = {
    "name": "focus_pane",
    "description": (
        "Reveal and focus a Hermes desktop pane when the user asks to see it: "
        "chat, files, terminal, review (git diff), or sessions. For URLs/"
        "files use the desktop_preview tool instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pane": {
                "type": "string",
                "enum": list(PANES),
                "description": "Which pane to reveal.",
            },
        },
        "required": ["pane"],
    },
}
# ---- END PLUGIN-COMPAT ----
