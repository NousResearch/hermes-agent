#!/usr/bin/env python3
"""Read the in-app browser / preview pane in the Hermes desktop GUI.

The preview's content lives in the renderer (a sandboxed ``<webview>``), so this
round-trips through the gateway's blocking-prompt bridge like ``read_terminal``
(``preview.read.request`` -> ``preview.read.respond``). Registered as action=read of
`desktop_preview`; the agent dispatches here with the injected callback.
"""

from typing import Callable, Optional

from tools.read_terminal_tool import read_pane


def read_preview_tool(
    start: Optional[int] = None, count: Optional[int] = None, callback: Optional[Callable] = None
) -> str:
    """Return the active preview tab's contents (+ metadata) as a JSON string."""
    return read_pane(callback, (("start", start, 0), ("count", count, 1)), (
        "read_preview is only available in the Hermes desktop app.",
        "start and count must be integers.",
        "Failed to read the preview pane: ",
        "No preview tab is open, or the read timed out."))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402

READ_PREVIEW_SCHEMA = {
    "name": "read_preview",
    "description": (
        "Read what's currently shown in the in-app browser / preview pane of the "
        "Hermes desktop GUI (the pane open_preview opens beside this chat). Call "
        "with no arguments for the first window of the active tab's content. "
        "Returns JSON {kind, url, title, text, start, end, total_chars, note?}: "
        "a URL (Browser) tab's text is the rendered page's visible text — page "
        "through longer pages with `start`/`count` (character offsets, capped "
        "per read); a file tab answers identity only (read the file with "
        "read_file); an artifact tab points back at the conversation. Use after "
        "open_preview, or whenever the user refers to what's on screen in the "
        "browser ('what does this page say?'). To close the pane, use close_preview."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start": {
                "type": "integer",
                "description": "0-indexed character offset into the page text. Omit for the start.",
            },
            "count": {
                "type": "integer",
                "description": "Characters to return from start. Defaults to (and is capped at) the per-read maximum.",
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
