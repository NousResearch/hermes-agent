#!/usr/bin/env python3
"""Apply a layout preset in the Hermes desktop GUI.

Lives in the ``desktop_ui`` toolset (like ``focus_pane``), which the GUI
gateway enables only for desktop-sourced sessions. Emits ``layout.apply``
through the shared ``desktop_ui`` bridge; the renderer resolves the preset id
against its layouts registry (core presets, plugin presets, and user-saved
presets are all the same list) and applies the tree through the exact code
path the layout picker uses. Only the active window's session may act — a
background turn never rearranges the user's desktop.

Preset ids are free-form on purpose: plugins and users mint their own. The
renderer answers with the applied preset's id/title on success and the list
of available ids when the id is unknown, so the model can self-correct
without a second registry-listing tool.
"""

import json

from tools import desktop_ui
from tools.registry import registry, tool_error

# Renderer answer arrives via the blocking-prompt bridge with this timeout;
# applying a layout is synchronous in the renderer, so this is generous.
_TIMEOUT_NOTE = "Layout apply is only available in the Hermes desktop app."


def apply_layout_tool(preset: str) -> str:
    """Ask the desktop GUI to apply layout preset ``preset``."""
    name = (preset or "").strip()
    if not name:
        return tool_error("preset is required — a layout preset id, e.g. 'default' or 'focus'.")

    try:
        ok = desktop_ui.emit("layout.apply", {"preset": name})
    except Exception as exc:
        return tool_error(f"Failed to apply layout '{name}': {exc}")
    if not ok:
        return tool_error(_TIMEOUT_NOTE)

    return json.dumps({"success": True, "preset": name}, ensure_ascii=False)


APPLY_LAYOUT_SCHEMA = {
    "name": "apply_layout",
    "description": (
        "Apply a saved layout preset to the Hermes desktop app when the user asks to "
        "rearrange the workspace — e.g. \"set up my layout for coding\", \"give me a "
        "focused view\", \"put the terminal front and center\". Built-in presets: "
        "default (chat + sidebars), focus (chat only), terminal-deck (terminal "
        "forward), quad (four zones). Plugin and user-saved presets are addressed by "
        "their id. To reveal a single pane without rearranging everything, use "
        "focus_pane instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "preset": {
                "type": "string",
                "description": "Layout preset id to apply (e.g. 'default', 'focus', 'terminal-deck', 'quad', or a user/plugin preset id).",
            },
        },
        "required": ["preset"],
    },
}


registry.register(
    name="apply_layout",
    toolset="desktop_ui",
    schema=APPLY_LAYOUT_SCHEMA,
    handler=lambda args, **kw: apply_layout_tool(preset=args.get("preset", "")),
    emoji="🧱",
)
