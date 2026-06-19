#!/usr/bin/env python3
"""Drive the in-app browser pane in the Hermes desktop GUI.

The embedded browser lives in the desktop renderer (an Electron BrowserView),
so these tools round-trip through the gateway's blocking-prompt bridge — the
same one `clarify` and `read_terminal` use. tui_gateway emits
``browser.navigate.request`` / ``browser.read.request`` and the renderer answers
with ``browser.navigate.respond`` / ``browser.read.respond``. These modules are
just schema + a thin dispatcher over the platform-injected callbacks.
"""

import json
import os
from typing import Callable, Optional

from tools.registry import registry, tool_error


def browser_navigate_tool(
    url: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Open/drive the in-app browser and return the resolved URL as JSON.

    With a ``url`` the renderer navigates there; with no ``url`` it simply opens
    the browser pane. The renderer answers ``browser.navigate.respond`` with the
    resolved URL once the load settles.
    """
    if callback is None:
        return tool_error("desktop_browser_navigate is only available in the Hermes desktop app.")

    try:
        raw = callback(url=url) if url else callback()
    except Exception as exc:
        return tool_error(f"Failed to navigate browser: {exc}")

    if not raw:
        return tool_error("The in-app browser did not open, or the navigation timed out.")

    return json.dumps({"url": str(raw)}, ensure_ascii=False)


def browser_read_tool(
    callback: Optional[Callable] = None,
) -> str:
    """Return the in-app browser's nav-state as a JSON string.

    The renderer answers ``browser.read.respond`` with a JSON object describing
    the current browser state: {open, url, title, loading, canGoBack,
    canGoForward, loadError}.
    """
    if callback is None:
        return tool_error("desktop_browser_read is only available in the Hermes desktop app.")

    try:
        raw = callback()
    except Exception as exc:
        return tool_error(f"Failed to read browser: {exc}")

    if not raw:
        return tool_error("No in-app browser is open, or the read timed out.")

    # Desktop answers with a JSON object; pass it through, else wrap the raw text.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


def check_browser_requirements() -> bool:
    """Desktop GUI only — HERMES_DESKTOP is set on the gateway the app spawns."""
    return (os.getenv("HERMES_DESKTOP") or "").strip().lower() in ("1", "true", "yes")


BROWSER_NAVIGATE_SCHEMA = {
    "name": "desktop_browser_navigate",
    "description": (
        "Open or drive the in-app browser pane of the Hermes desktop GUI (the "
        "embedded web view beside this chat). Pass a `url` to navigate the "
        "browser there; call with no arguments to just open the browser pane. "
        "Returns JSON: {url} — the resolved URL after the load settles."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to navigate to. Omit to just open the browser pane.",
            },
        },
    },
}


BROWSER_READ_SCHEMA = {
    "name": "desktop_browser_read",
    "description": (
        "Read the current state of the in-app browser pane of the Hermes "
        "desktop GUI (the embedded web view beside this chat). Takes no "
        "arguments. Returns JSON: "
        "{open, url, title, loading, canGoBack, canGoForward, loadError}."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


registry.register(
    name="desktop_browser_navigate",
    toolset="browser",
    schema=BROWSER_NAVIGATE_SCHEMA,
    handler=lambda args, **kw: browser_navigate_tool(
        url=args.get("url"),
        callback=kw.get("callback"),
    ),
    check_fn=check_browser_requirements,
    emoji="🌐",
)


registry.register(
    name="desktop_browser_read",
    toolset="browser",
    schema=BROWSER_READ_SCHEMA,
    handler=lambda args, **kw: browser_read_tool(
        callback=kw.get("callback"),
    ),
    check_fn=check_browser_requirements,
    emoji="🌐",
)
