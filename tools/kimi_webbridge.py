"""Kimi WebBridge integration for Hermes Agent.

Provides browser automation via the local Kimi WebBridge daemon. Unlike the
built-in ``browser`` toolset (which uses Playwright/Browserbase/Camofox),
this controls the user's REAL browser with their actual login sessions.

Configuration
-------------
Add to ``~/.hermes/config.yaml``::

    providers:
      kimi_webbridge:
        base_url: http://127.0.0.1:10086

If omitted, ``base_url`` defaults to ``http://127.0.0.1:10086``.

The toolset is off by default (``_DEFAULT_OFF_TOOLSETS``) because it requires
the Kimi WebBridge browser extension + daemon to be installed separately.
See https://www.kimi.com/features/webbridge for setup instructions.
"""

import base64
import json
import os
from pathlib import Path
from typing import Optional

import requests

from tools.registry import registry

_DEFAULT_DAEMON_URL = "http://127.0.0.1:10086"
_DEFAULT_SESSION = "hermes"
_COMMAND_ENDPOINT = "/command"


def _get_daemon_url() -> str:
    """Resolve daemon URL from config or fallback."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        url = cfg.get("providers", {}).get("kimi_webbridge", {}).get("base_url")
        if url:
            return url.rstrip("/")
    except Exception:
        pass
    return _DEFAULT_DAEMON_URL


def _bridge_call(action: str, args: Optional[dict] = None, session: Optional[str] = None) -> dict:
    """POST a command to the Kimi WebBridge daemon."""
    url = _get_daemon_url() + _COMMAND_ENDPOINT
    payload = {
        "action": action,
        "args": args or {},
        "session": session or _DEFAULT_SESSION,
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {"error": True, "message": str(exc)}


def _check_bridge() -> bool:
    """Return True if the Kimi WebBridge daemon is reachable."""
    url = _get_daemon_url() + _COMMAND_ENDPOINT
    try:
        resp = requests.post(
            url,
            json={"action": "list_tabs", "args": {}, "session": _DEFAULT_SESSION},
            timeout=3,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _validate_screenshot_path(output_path: Optional[str]) -> Path:
    """Ensure screenshot path is safe and within allowed directories."""
    if output_path is None:
        return Path(f"/tmp/kimi-webbridge-screenshots/{_DEFAULT_SESSION}_{os.getpid()}.png")

    path = Path(output_path).resolve()
    allowed_roots = [
        Path("/tmp").resolve(),
        Path.home().resolve(),
    ]
    if not any(str(path).startswith(str(root)) for root in allowed_roots):
        raise ValueError(f"Screenshot path must be under /tmp or home directory, got: {output_path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Tool handlers
# ═══════════════════════════════════════════════════════════════════════════════

def kimi_webbridge_navigate(
    url: str,
    new_tab: bool = True,
    group_title: Optional[str] = None,
    session: Optional[str] = None,
) -> str:
    """Navigate to a URL in the user's real browser."""
    args: dict = {"url": url, "newTab": new_tab}
    if group_title:
        args["group_title"] = group_title
    return json.dumps(_bridge_call("navigate", args, session))


def kimi_webbridge_find_tab(
    url: str,
    active: bool = False,
    session: Optional[str] = None,
) -> str:
    """Find and reuse an already-open tab by URL or domain."""
    return json.dumps(_bridge_call("find_tab", {"url": url, "active": active}, session))


def kimi_webbridge_snapshot(session: Optional[str] = None) -> str:
    """Get an accessibility tree snapshot of the current page with @e refs."""
    return json.dumps(_bridge_call("snapshot", {}, session))


def kimi_webbridge_click(selector: str, session: Optional[str] = None) -> str:
    """Click an element by @e ref or CSS selector."""
    return json.dumps(_bridge_call("click", {"selector": selector}, session))


def kimi_webbridge_fill(
    selector: str,
    value: str,
    session: Optional[str] = None,
) -> str:
    """Fill an input, textarea, or contenteditable element. Clears existing content."""
    return json.dumps(_bridge_call("fill", {"selector": selector, "value": value}, session))


def kimi_webbridge_evaluate(code: str, session: Optional[str] = None) -> str:
    """Evaluate JavaScript in the current page. Supports async/await."""
    return json.dumps(_bridge_call("evaluate", {"code": code}, session))


def kimi_webbridge_screenshot(
    format: str = "png",
    quality: int = 90,
    selector: Optional[str] = None,
    session: Optional[str] = None,
) -> str:
    """Take a screenshot.

    Returns a lightweight result. The actual image data is stripped from context
    to avoid flooding the token window; use ``kimi_webbridge_save_screenshot``
    when you need the file on disk.
    """
    args: dict = {"format": format, "quality": quality}
    if selector:
        args["selector"] = selector
    result = _bridge_call("screenshot", args, session)
    inner = result.get("data", {}) if isinstance(result.get("data"), dict) else result
    b64 = inner.get("data") if isinstance(inner, dict) else None
    if b64 and isinstance(b64, str) and len(b64) > 1000:
        inner["data"] = f"<base64 image data, {len(b64)} chars — use kimi_webbridge_save_screenshot instead>"
    return json.dumps(result)


def kimi_webbridge_save_screenshot(
    output_path: Optional[str] = None,
    format: str = "png",
    quality: int = 90,
    session: Optional[str] = None,
) -> str:
    """Take a screenshot and save it to disk, returning only the file path."""
    args: dict = {"format": format, "quality": quality}
    result = _bridge_call("screenshot", args, session)
    inner = result.get("data", {}) if isinstance(result.get("data"), dict) else result
    b64 = inner.get("data") if isinstance(inner, dict) else None
    if not b64 or not isinstance(b64, str):
        return json.dumps({"error": "screenshot failed", "details": result})

    try:
        path = _validate_screenshot_path(output_path)
    except ValueError as exc:
        return json.dumps({"error": "invalid path", "message": str(exc)})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))
    return json.dumps({"success": True, "path": str(path), "size_bytes": path.stat().st_size})


def kimi_webbridge_list_tabs(session: Optional[str] = None) -> str:
    """List all tabs in the current session."""
    return json.dumps(_bridge_call("list_tabs", {}, session))


def kimi_webbridge_close_tab(session: Optional[str] = None) -> str:
    """Close the current tab."""
    return json.dumps(_bridge_call("close_tab", {}, session))


def kimi_webbridge_close_session(session: Optional[str] = None) -> str:
    """Close all tabs in the session. Call at the end of a task."""
    return json.dumps(_bridge_call("close_session", {}, session))


def kimi_webbridge_save_pdf(
    paper_format: str = "letter",
    landscape: bool = False,
    scale: float = 1.0,
    print_background: bool = True,
    file_name: Optional[str] = None,
    session: Optional[str] = None,
) -> str:
    """Save the current page as a PDF."""
    args: dict = {
        "paper_format": paper_format,
        "landscape": landscape,
        "scale": scale,
        "print_background": print_background,
    }
    if file_name:
        args["file_name"] = file_name
    return json.dumps(_bridge_call("save_as_pdf", args, session))


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

_SCHEMA_BASE = {
    "type": "object",
    "properties": {
        "session": {"type": "string", "default": "hermes", "description": "Session name for tab isolation"},
    },
}


def _schema(required: list, extra_props: dict) -> dict:
    props = {**_SCHEMA_BASE["properties"], **extra_props}
    return {"type": "object", "properties": props, "required": required}


registry.register(
    name="kimi_webbridge_navigate",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_navigate",
        "description": "Navigate the user's real browser to a URL via Kimi WebBridge. Uses the user's actual login sessions.",
        "parameters": _schema(
            ["url"],
            {
                "url": {"type": "string", "description": "URL to navigate to"},
                "new_tab": {"type": "boolean", "description": "Open in a new tab", "default": True},
                "group_title": {"type": "string", "description": "Visible label for the tab group"},
            },
        ),
    },
    handler=lambda args, **kw: kimi_webbridge_navigate(
        url=args["url"],
        new_tab=args.get("new_tab", True),
        group_title=args.get("group_title"),
        session=args.get("session"),
    ),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_find_tab",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_find_tab",
        "description": "Find and reuse an already-open tab by URL or domain. Use when the user refers to an existing page.",
        "parameters": _schema(
            ["url"],
            {
                "url": {"type": "string", "description": "URL or domain to match"},
                "active": {"type": "boolean", "description": "Pick the currently-viewed tab", "default": False},
            },
        ),
    },
    handler=lambda args, **kw: kimi_webbridge_find_tab(
        url=args["url"],
        active=args.get("active", False),
        session=args.get("session"),
    ),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_snapshot",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_snapshot",
        "description": "Get an accessibility tree snapshot of the current page. Returns interactive elements with @e refs for clicking/filling.",
        "parameters": _schema([], {}),
    },
    handler=lambda args, **kw: kimi_webbridge_snapshot(session=args.get("session")),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_click",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_click",
        "description": "Click an element by @e ref or CSS selector. Use @e refs from kimi_webbridge_snapshot when available.",
        "parameters": _schema(
            ["selector"],
            {"selector": {"type": "string", "description": "@e ref (e.g. @e5) or CSS selector"}},
        ),
    },
    handler=lambda args, **kw: kimi_webbridge_click(
        selector=args["selector"],
        session=args.get("session"),
    ),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_fill",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_fill",
        "description": "Fill an input, textarea, or contenteditable element. Clears existing content before inserting.",
        "parameters": _schema(
            ["selector", "value"],
            {
                "selector": {"type": "string", "description": "@e ref or CSS selector"},
                "value": {"type": "string", "description": "Text to insert"},
            },
        ),
    },
    handler=lambda args, **kw: kimi_webbridge_fill(
        selector=args["selector"],
        value=args["value"],
        session=args.get("session"),
    ),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_evaluate",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_evaluate",
        "description": "Evaluate JavaScript in the current page. Supports async/await. Use for scrolling, extracting data, or complex interactions.",
        "parameters": _schema(
            ["code"],
            {"code": {"type": "string", "description": "JavaScript code to run"}},
        ),
    },
    handler=lambda args, **kw: kimi_webbridge_evaluate(
        code=args["code"],
        session=args.get("session"),
    ),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_screenshot",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_screenshot",
        "description": "Take a screenshot. Returns a lightweight result — base64 data is stripped to avoid context flooding.",
        "parameters": _schema(
            [],
            {
                "format": {"type": "string", "enum": ["png", "jpeg"], "default": "png"},
                "quality": {"type": "integer", "default": 90},
                "selector": {"type": "string", "description": "Optional CSS selector or @e ref to capture only that element"},
            },
        ),
    },
    handler=lambda args, **kw: kimi_webbridge_screenshot(
        format=args.get("format", "png"),
        quality=args.get("quality", 90),
        selector=args.get("selector"),
        session=args.get("session"),
    ),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_save_screenshot",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_save_screenshot",
        "description": "Take a screenshot and save it to disk. Returns only the file path — safe for context windows.",
        "parameters": _schema(
            [],
            {
                "output_path": {"type": "string", "description": "Where to save the image (default: auto-generated in /tmp)"},
                "format": {"type": "string", "enum": ["png", "jpeg"], "default": "png"},
                "quality": {"type": "integer", "default": 90},
            },
        ),
    },
    handler=lambda args, **kw: kimi_webbridge_save_screenshot(
        output_path=args.get("output_path"),
        format=args.get("format", "png"),
        quality=args.get("quality", 90),
        session=args.get("session"),
    ),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_list_tabs",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_list_tabs",
        "description": "List all tabs in the current session.",
        "parameters": _schema([], {}),
    },
    handler=lambda args, **kw: kimi_webbridge_list_tabs(session=args.get("session")),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_close_tab",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_close_tab",
        "description": "Close the current tab.",
        "parameters": _schema([], {}),
    },
    handler=lambda args, **kw: kimi_webbridge_close_tab(session=args.get("session")),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_close_session",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_close_session",
        "description": "Close all tabs in the session. Call at the end of a task.",
        "parameters": _schema([], {}),
    },
    handler=lambda args, **kw: kimi_webbridge_close_session(session=args.get("session")),
    check_fn=_check_bridge,
)

registry.register(
    name="kimi_webbridge_save_pdf",
    toolset="kimi_webbridge",
    schema={
        "name": "kimi_webbridge_save_pdf",
        "description": "Save the current page as a PDF.",
        "parameters": _schema(
            [],
            {
                "paper_format": {"type": "string", "default": "letter"},
                "landscape": {"type": "boolean", "default": False},
                "scale": {"type": "number", "default": 1.0},
                "print_background": {"type": "boolean", "default": True},
                "file_name": {"type": "string", "description": "Custom filename; defaults to page title"},
            },
        ),
    },
    handler=lambda args, **kw: kimi_webbridge_save_pdf(
        paper_format=args.get("paper_format", "letter"),
        landscape=args.get("landscape", False),
        scale=args.get("scale", 1.0),
        print_background=args.get("print_background", True),
        file_name=args.get("file_name"),
        session=args.get("session"),
    ),
    check_fn=_check_bridge,
)
