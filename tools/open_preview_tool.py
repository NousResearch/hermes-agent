#!/usr/bin/env python3
"""Open a URL, dev server, or file in the Hermes desktop GUI's preview pane.

Registration lives in the `desktop_preview` tool (``tools.preview_tool``); this module keeps
the normalizer + open action. Emits ``preview.open`` via ``desktop_ui``: the renderer opens
the pane for the window that asked and never steals focus for a background session.
"""

import re

from tools import desktop_ui
from tools.registry import tool_error


def _normalize_target(raw: str) -> str:
    """Coax a bare host/domain into a fetchable URL; leave paths + schemes alone.

    ``www.cnn.com`` -> ``https://www.cnn.com``; ``localhost:3000`` -> ``http://localhost:3000``.
    File paths and explicit schemes pass through for the renderer's preview normalizer.
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
            "file path to show in the preview pane.")
    label = (label or "").strip()
    return desktop_ui.emit_or_error(
        "preview.open",
        {"url": target, "label": label},
        "Failed to open the preview pane: ",
        "The preview pane is only available in the Hermes desktop app.",
        {"success": True, "url": target, "label": label})


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402

OPEN_PREVIEW_SCHEMA = {
    "name": "open_preview",
    "description": (
        "Open something in the preview pane beside the chat in the Hermes desktop "
        "app. Use this when the user asks to see a page, dev server, or file in the "
        "preview pane — e.g. \"open cnn.com in the preview pane\" or \"preview "
        "localhost:3000\". Accepts a web URL (a bare domain like www.cnn.com is fine), "
        "a localhost dev-server URL, or a file path (HTML renders live; other files "
        "show their contents). The pane opens for the current window only. To close "
        "the pane or a tab, use close_preview."
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


_PLUGIN_COMPAT_LAZY = {
    'registry': ('tools.registry', 'registry'),
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
