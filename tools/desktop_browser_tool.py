#!/usr/bin/env python3
"""Drive the in-app browser pane in the Hermes desktop GUI.

The embedded browser lives in the desktop renderer (an Electron BrowserView),
so these tools round-trip through the gateway's blocking-prompt bridge — the
same one `clarify` and `read_terminal` use. tui_gateway emits
``browser.navigate.request`` / ``browser.read.request`` / ``browser.act.request``
/ ``browser.extract.request`` / ``browser.screenshot.request`` and the renderer
answers with matching ``.respond`` methods. These modules are just schema + a
thin dispatcher over the platform-injected callbacks.
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


def browser_act_tool(
    action: str,
    selector: Optional[str] = None,
    text: Optional[str] = None,
    key: Optional[str] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    direction: Optional[str] = None,
    amount: Optional[int] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Interact with the in-app browser page (click, type, scroll, key, hover).

    The renderer runs the action against the live webview via executeJavaScript.
    Returns JSON {ok: true, action, detail} or {ok: false, error}.
    """
    if callback is None:
        return tool_error("desktop_browser_act is only available in the Hermes desktop app.")

    try:
        raw = callback(
            action=action,
            selector=selector,
            text=text,
            key=key,
            x=x,
            y=y,
            direction=direction,
            amount=amount,
        )
    except Exception as exc:
        return tool_error(f"Failed to act in browser: {exc}")

    if not raw:
        return tool_error("No in-app browser is open, or the action timed out.")

    # Renderer returns a JSON object; pass it through.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"ok": False, "error": str(raw)}, ensure_ascii=False)


def browser_extract_tool(
    mode: str = "text",
    selector: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Extract content from the in-app browser page.

    Modes: 'text' (innerText), 'html' (outerHTML, capped), 'links' (array of
    {text, href}), 'a11y' (simplified role/name tree). Returns JSON.
    """
    if callback is None:
        return tool_error("desktop_browser_extract is only available in the Hermes desktop app.")

    try:
        raw = callback(mode=mode, selector=selector)
    except Exception as exc:
        return tool_error(f"Failed to extract from browser: {exc}")

    if not raw:
        return tool_error("No in-app browser is open, or the extraction timed out.")

    # Renderer returns a JSON object; pass it through.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


def browser_screenshot_tool(
    full_page: bool = False,
    callback: Optional[Callable] = None,
) -> str:
    """Capture a screenshot of the in-app browser panel.

    Returns JSON {image: <data-url>} or {image_path: <path>}.
    """
    if callback is None:
        return tool_error("desktop_browser_screenshot is only available in the Hermes desktop app.")

    try:
        raw = callback(full_page=full_page)
    except Exception as exc:
        return tool_error(f"Failed to screenshot browser: {exc}")

    if not raw:
        return tool_error("No in-app browser is open, or the screenshot timed out.")

    # Renderer returns a JSON object with image data; pass it through.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"error": str(raw)}, ensure_ascii=False)


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


BROWSER_ACT_SCHEMA = {
    "name": "desktop_browser_act",
    "description": (
        "Interact with the in-app browser page: click elements, type text, "
        "scroll, press keys, or hover. Returns JSON: {ok, action, detail} or "
        "{ok: false, error}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["click", "type", "scroll", "key", "hover"],
                "description": "The interaction to perform.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for click/type/hover actions.",
            },
            "text": {
                "type": "string",
                "description": "Text to type (for 'type' action).",
            },
            "key": {
                "type": "string",
                "description": "Key name for 'key' action (e.g., 'Enter', 'Tab', 'ArrowDown').",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate for click fallback (when no selector).",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate for click fallback (when no selector).",
            },
            "direction": {
                "type": "string",
                "enum": ["up", "down", "top", "bottom"],
                "description": "Scroll direction (for 'scroll' action).",
            },
            "amount": {
                "type": "integer",
                "description": "Scroll amount in pixels (for 'scroll' action).",
            },
        },
        "required": ["action"],
    },
}


BROWSER_EXTRACT_SCHEMA = {
    "name": "desktop_browser_extract",
    "description": (
        "Extract content from the in-app browser page. Modes: 'text' "
        "(innerText), 'html' (outerHTML), 'links' (array of {text, href}), "
        "'a11y' (role/name tree). Returns JSON with extracted content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["text", "html", "links", "a11y"],
                "description": "Extraction mode (default: 'text').",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector to scope extraction to a subtree (optional).",
            },
        },
    },
}


BROWSER_SCREENSHOT_SCHEMA = {
    "name": "desktop_browser_screenshot",
    "description": (
        "Capture a screenshot of the in-app browser panel. Returns JSON: "
        "{image: <data-url>}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "full_page": {
                "type": "boolean",
                "description": "Capture the full page height (default: false, visible viewport only).",
            },
        },
    },
}


registry.register(
    name="desktop_browser_act",
    toolset="browser",
    schema=BROWSER_ACT_SCHEMA,
    handler=lambda args, **kw: browser_act_tool(
        action=args.get("action"),
        selector=args.get("selector"),
        text=args.get("text"),
        key=args.get("key"),
        x=args.get("x"),
        y=args.get("y"),
        direction=args.get("direction"),
        amount=args.get("amount"),
        callback=kw.get("callback"),
    ),
    check_fn=check_browser_requirements,
    emoji="🌐",
)


registry.register(
    name="desktop_browser_extract",
    toolset="browser",
    schema=BROWSER_EXTRACT_SCHEMA,
    handler=lambda args, **kw: browser_extract_tool(
        mode=args.get("mode", "text"),
        selector=args.get("selector"),
        callback=kw.get("callback"),
    ),
    check_fn=check_browser_requirements,
    emoji="🌐",
)


registry.register(
    name="desktop_browser_screenshot",
    toolset="browser",
    schema=BROWSER_SCREENSHOT_SCHEMA,
    handler=lambda args, **kw: browser_screenshot_tool(
        full_page=args.get("full_page", False),
        callback=kw.get("callback"),
    ),
    check_fn=check_browser_requirements,
    emoji="🌐",
)
