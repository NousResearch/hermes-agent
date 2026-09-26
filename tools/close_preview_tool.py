"""Close the Hermes desktop GUI's preview pane, or one tab (``preview.close``).
Registration lives in `desktop_preview`. The renderer drops the matching tab — or
the whole pane when no url is given — only for the window that asked."""

from tools import desktop_ui
from tools.open_preview_tool import _normalize_target


def close_preview_tool(url: str = "") -> str:
    """Ask the desktop GUI to close the preview pane, or the tab for ``url``."""
    target = _normalize_target(url or "")
    return desktop_ui.emit_or_error(
        "preview.close", {"url": target}, "Failed to close the preview pane: ",
        "The preview pane is only available in the Hermes desktop app.", {"success": True, "url": target},
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402

CLOSE_PREVIEW_SCHEMA = {
    "name": "close_preview",
    "description": (
        "Close the preview pane beside the chat in the Hermes desktop app, or one "
        "tab inside it. Use this when the user asks to close, hide, or dismiss the "
        "preview — e.g. \"close the preview pane\", \"close cnn.com\", \"hide the "
        "preview\". Omit url to close the whole pane (every tab). Pass a web URL, "
        "localhost address, or file path to close only that tab. Counterpart of "
        "open_preview."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "Optional. The tab to close: a web URL (https://… or a bare "
                    "domain), a localhost URL, or a file path. Omit to close the "
                    "whole preview pane."
                ),
            },
        },
    },
}


_PLUGIN_COMPAT_LAZY = {
    'registry': ('tools.registry', 'registry'),
    'tool_error': ('tools.registry', 'tool_error'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
