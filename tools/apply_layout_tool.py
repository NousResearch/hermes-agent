"""Apply a layout preset in the Hermes desktop GUI (``layout.apply`` via ``desktop_ui``).

The renderer resolves the id against its layouts registry (core, plugin and user
presets are one list); only the active window's session acts, so a background turn
never rearranges the desktop. Preset ids are free-form on purpose. The renderer
answers with the applied id/title, or the available ids when unknown, so the model
can self-correct without a registry-listing tool.
"""

from tools import desktop_ui
from tools.registry import registry, tool_error


def apply_layout_tool(preset: str) -> str:
    """Ask the desktop GUI to apply layout preset ``preset``."""
    name = (preset or "").strip()
    if not name:
        return tool_error("preset is required — a layout preset id, e.g. 'default' or 'focus'.")
    return desktop_ui.emit_or_error(
        "layout.apply", {"preset": name}, f"Failed to apply layout '{name}': ",
        "Layout apply is only available in the Hermes desktop app.", {"success": True, "preset": name},
    )


APPLY_LAYOUT_SCHEMA = {
    "name": "apply_layout",
    "description": (
        "Apply a saved layout preset to the Hermes desktop app when the user "
        "asks to rearrange the workspace. Built-ins: default (chat + "
        "sidebars), focus (chat only), terminal-deck, quad; plugin/user "
        "presets by id. To reveal ONE pane, use focus_pane instead."
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
