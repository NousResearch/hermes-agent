#!/usr/bin/env python3
"""Photograph the in-app browser / preview pane in the Hermes desktop GUI.

The preview is a sandboxed ``<webview>`` the renderer owns, so this round-trips
through the gateway's blocking-prompt bridge like ``read_preview``
(``preview.screenshot.request`` -> ``preview.screenshot.respond``) and the PNG is
written by the main process. Registered as action=screenshot of `desktop_preview`;
the agent dispatches here with the injected callback.
"""

from typing import Callable, Optional

from tools.read_terminal_tool import read_pane


def screenshot_preview_tool(callback: Optional[Callable] = None) -> str:
    """Return the active preview tab's PNG path (+ metadata) as a JSON string."""
    return read_pane(callback, (), (
        "preview screenshot is only available in the Hermes desktop app.",
        "",
        "Failed to photograph the preview pane: ",
        "No preview tab is open, or the capture timed out."))
