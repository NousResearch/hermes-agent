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
