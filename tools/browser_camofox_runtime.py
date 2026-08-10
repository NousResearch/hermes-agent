"""Camofox exec-mode runtime — pre-imported helpers for browser_exec.

This module is NOT imported by the Hermes process. It is executed in a
fresh Python subprocess (``sys.executable -c <wrapper>``) with the model's
``code`` piped on stdin, mirroring how the Browser Use CLI 3.0 harness
runs. Keeping it stdlib-only (urllib, json) means the subprocess has no
dependency on the Hermes venv, config, or secret store.

The parent (``tools/browser_camofox_exec.py``) resolves the Camofox
session in-process — honoring managed persistence, identity overrides and
tab adoption — and hands it to us through environment variables:

    CAMOFOX_URL           Camofox server base URL (required)
    CAMOFOX_API_KEY       optional Bearer token (CAMOFOX_API_KEY secret)
    BH_USER_ID            stable userId the server maps to a profile
    BH_SESSION_KEY        listItemId grouping tabs for the task
    BH_TAB_ID             last known tab id, when the parent has one
    BH_AGENT_WORKSPACE    scratch dir (screenshots land here when set)

Tab lifecycle across calls: each call is a fresh interpreter, but the tab
lives on the server. On entry we reuse ``BH_TAB_ID``; if the server says
404 (tab garbage-collected), we create a fresh tab under the same
userId/sessionKey, so cookies and profile state survive. Every time the
tab is resolved we print ``CAMOFOX_TAB_ID=<id>`` on its own line; the
parent parses it and updates its in-memory session cache so the next call
starts from the new tab.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_BASE_URL = os.environ.get("CAMOFOX_URL", "").rstrip("/")
_API_KEY = os.environ.get("CAMOFOX_API_KEY", "") or ""
_USER_ID = os.environ.get("BH_USER_ID", "") or ""
_SESSION_KEY = os.environ.get("BH_SESSION_KEY", "") or ""
_TAB_ID = os.environ.get("BH_TAB_ID", "") or ""
_WORKSPACE = os.environ.get("BH_AGENT_WORKSPACE", "") or ""

if not _BASE_URL:
    raise RuntimeError(
        "CAMOFOX_URL is not set. The Camofox exec backend requires a running "
        "Camofox server (CAMOFOX_URL in ~/.hermes/.env)."
    )
if not _USER_ID:
    raise RuntimeError("BH_USER_ID is not set — Camofox session not resolved.")


class CamofoxHTTPError(RuntimeError):
    """A non-2xx response from the Camofox server (carries the status code)."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def _request(
    method: str,
    path: str,
    body: dict | None = None,
    params: dict | None = None,
    timeout: float = 30.0,
    raw: bool = False,
):
    """HTTP helper against the Camofox REST API."""
    url = _BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {}
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if raw:
                return resp.read()
            payload = resp.read()
            return json.loads(payload.decode("utf-8")) if payload else {}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise CamofoxHTTPError(
            f"Camofox {method} {path} failed: HTTP {exc.code} {detail}", exc.code
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Camofox at {_BASE_URL}: {exc.reason}"
        ) from exc


def _tab() -> str:
    """Return a live tab id, creating one when needed (or on 404)."""
    global _TAB_ID
    if _TAB_ID:
        try:
            _request("GET", f"/tabs/{_TAB_ID}/stats", params={"userId": _USER_ID}, timeout=10)
            print(f"CAMOFOX_TAB_ID={_TAB_ID}")
            return _TAB_ID
        except CamofoxHTTPError as exc:
            # A 404 (tab gone) or 410 (the real Camofox server's response
            # for a GC'd/restarted tab: "Tab no longer exists (browser was
            # restarted)") means "recreate". Anything else — 401, 5xx, or a
            # connection error — must surface: silently opening a fresh tab
            # would throw away the live page and hide the real failure.
            if exc.code not in (404, 410):
                raise
    data = _request(
        "POST",
        "/tabs",
        # No explicit url: the real server rejects the "about:" scheme
        # (HTTP 400 "Blocked URL scheme"), while omitting the key opens a
        # native blank page. new_tab() always passes a real http(s) URL.
        body={"userId": _USER_ID, "listItemId": _SESSION_KEY},
    )
    tab_id = str(data.get("tabId") or "")
    if not tab_id:
        raise RuntimeError(f"Camofox /tabs returned no tabId: {data!r}")
    _TAB_ID = tab_id
    print(f"CAMOFOX_TAB_ID={tab_id}")
    return tab_id


def ensure_real_tab() -> dict:
    """Recover from a stale/internal tab; returns the active tab id."""
    return {"ok": True, "tab_id": _tab()}


def new_tab(url: str) -> dict:
    """Open a URL in a fresh tab (first navigation of a session)."""
    global _TAB_ID
    data = _request(
        "POST",
        "/tabs",
        body={"userId": _USER_ID, "listItemId": _SESSION_KEY, "url": url},
        timeout=60,
    )
    tab_id = str(data.get("tabId") or "")
    if not tab_id:
        raise RuntimeError(f"Camofox /tabs returned no tabId: {data!r}")
    _TAB_ID = tab_id
    print(f"CAMOFOX_TAB_ID={tab_id}")
    return {"url": data.get("url", url), "title": data.get("title", "")}


def goto_url(url: str) -> dict:
    """Navigate the current tab to a URL."""
    data = _request(
        "POST",
        f"/tabs/{_tab()}/navigate",
        body={"userId": _USER_ID, "url": url},
        timeout=60,
    )
    return {"url": data.get("url", url), "title": data.get("title", "")}


def wait_for_load(timeout: int = 10000, wait_for_network: bool = True) -> dict:
    """Wait for the page to be ready (default 10s, network quiet)."""
    data = _request(
        "POST",
        f"/tabs/{_tab()}/wait",
        body={"userId": _USER_ID, "timeout": timeout, "waitForNetwork": wait_for_network},
        timeout=max(timeout / 1000 + 10, 30),
    )
    return {"ok": True, "ready": bool(data.get("ready"))}


def page_info() -> str:
    """Summarize the current page: URL plus the accessibility snapshot."""
    tab = _tab()
    try:
        snap = _request(
            "GET", f"/tabs/{tab}/snapshot", params={"userId": _USER_ID}, timeout=30
        )
        snapshot_text = str(snap.get("snapshot") or "")
        refs = snap.get("refsCount", 0)
    except RuntimeError:
        snapshot_text, refs = "", 0
    try:
        url = js("location.href")
    except RuntimeError:
        url = ""
    return (
        f"URL: {url}\n"
        f"Element refs: {refs}\n"
        + (snapshot_text if snapshot_text else "(no snapshot available)")
    )


def js(expr: str):
    """Evaluate a JavaScript expression in the page and return its value."""
    data = _request(
        "POST",
        f"/tabs/{_tab()}/evaluate",
        body={"userId": _USER_ID, "expression": expr},
        timeout=60,
    )
    return data.get("result")


def fill_input(selector: str, text: str, press_enter: bool = False) -> dict:
    """Type text into an input selected by CSS selector (fill mode)."""
    _request(
        "POST",
        f"/tabs/{_tab()}/type",
        body={
            "userId": _USER_ID,
            "selector": selector,
            "text": text,
            "pressEnter": press_enter,
        },
        timeout=60,
    )
    return {"ok": True, "selector": selector}


def type_into_ref(ref: str, text: str, press_enter: bool = False) -> dict:
    """Type text into a snapshot element ref (e.g. 'e3')."""
    _request(
        "POST",
        f"/tabs/{_tab()}/type",
        body={
            "userId": _USER_ID,
            "ref": ref.lstrip("@"),
            "text": text,
            "pressEnter": press_enter,
        },
        timeout=60,
    )
    return {"ok": True, "ref": ref.lstrip("@")}


def click_ref(ref: str) -> dict:
    """Click a snapshot element ref (e.g. 'e3')."""
    data = _request(
        "POST",
        f"/tabs/{_tab()}/click",
        body={"userId": _USER_ID, "ref": ref.lstrip("@")},
        timeout=60,
    )
    return {"ok": True, "clicked": ref.lstrip("@"), "url": data.get("url", "")}


def click_selector(selector: str) -> dict:
    """Click the first element matching a CSS selector."""
    data = _request(
        "POST",
        f"/tabs/{_tab()}/click",
        body={"userId": _USER_ID, "selector": selector},
        timeout=60,
    )
    return {"ok": True, "clicked": selector, "url": data.get("url", "")}


def click_at_xy(x: int, y: int) -> dict:
    """Click viewport coordinates via elementFromPoint (fallback path)."""
    expr = (
        f"(() => {{ const el = document.elementFromPoint({int(x)}, {int(y)}); "
        "if (!el) return {clicked: false, reason: 'no element at point'}; "
        "el.click(); return {clicked: true, tag: el.tagName}; })()"
    )
    result = js(expr)
    return {"ok": True, "result": result}


def press_key(key: str) -> dict:
    """Press a keyboard key (e.g. 'Enter', 'Tab', 'Escape')."""
    _request(
        "POST",
        f"/tabs/{_tab()}/press",
        body={"userId": _USER_ID, "key": key},
        timeout=30,
    )
    return {"ok": True, "pressed": key}


def scroll_page(direction: str = "down", amount: int = 500) -> dict:
    """Scroll the page (direction: up/down/left/right)."""
    _request(
        "POST",
        f"/tabs/{_tab()}/scroll",
        body={"userId": _USER_ID, "direction": direction, "amount": int(amount)},
        timeout=30,
    )
    return {"ok": True, "direction": direction}


def get_links(limit: int = 50) -> dict:
    """Return page links from the Camofox links endpoint."""
    data = _request(
        "GET",
        f"/tabs/{_tab()}/links",
        params={"userId": _USER_ID, "limit": int(limit)},
        timeout=30,
    )
    links = data.get("links", []) if isinstance(data, dict) else []
    return {"links": links, "count": len(links)}


def capture_screenshot(full_page: bool = False) -> str:
    """Save a screenshot PNG and return (and print) its absolute path.

    The path is printed with forward slashes so the Hermes parent can
    detect it (the screenshot detector regex expects POSIX-style paths)
    and attach the image for vision-capable models.
    """
    png = _request(
        "GET",
        f"/tabs/{_tab()}/screenshot",
        params={"userId": _USER_ID, "fullPage": "true" if full_page else "false"},
        timeout=60,
        raw=True,
    )
    out_dir = Path(_WORKSPACE) if _WORKSPACE else Path(tempdir())
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"camofox_{int(time.time() * 1000)}.png"
    path.write_bytes(png)
    # Native path marker first (the parent prefers it: the screenshot
    # detector regex for the CLI backend requires a leading "/", which
    # resolves wrong for Windows drive letters), then the POSIX-style
    # print as a fallback for the model and the CLI-style detector.
    print(f"CAMOFOX_SCREENSHOT={path}")
    posix = path.as_posix()
    print(posix)
    return posix


def tempdir() -> str:
    """Fallback scratch dir when no workspace env is present."""
    import tempfile

    return tempfile.gettempdir()


def cdp(method: str, **kwargs):
    """Raw-CDP shim for the Browser Use helper surface (limited).

    Camofox is a REST API, not a CDP endpoint, so only a small mapping is
    supported:

    * ``cdp('Page.navigate', url=...)``            -> goto_url
    * ``cdp('Runtime.evaluate', expression=...)``  -> js
    * ``cdp('Accessibility.getFullAXTree')``       -> page_info() text
    * ``cdp('DOM.getBoxModel', selector=...)``     -> bounding rect

    Prefer the typed helpers (click_ref, fill_input, js, …) — the snapshot
    already exposes stable element refs, so coordinate hacking is rarely
    needed.
    """
    if method == "Page.navigate":
        url = kwargs.get("url") or kwargs.get("url_")
        if not url:
            raise ValueError("cdp('Page.navigate') requires url=...")
        return goto_url(url)
    if method == "Runtime.evaluate":
        expr = kwargs.get("expression") or kwargs.get("expr")
        if not expr:
            raise ValueError("cdp('Runtime.evaluate') requires expression=...")
        return js(expr)
    if method == "Accessibility.getFullAXTree":
        return page_info()
    if method == "DOM.getBoxModel":
        selector = kwargs.get("selector")
        if not selector:
            raise ValueError(
                "cdp('DOM.getBoxModel') requires selector=...; refs are not "
                "resolvable over raw CDP in Camofox mode — use click_ref(ref)"
            )
        rect = js(
            f"(() => {{ const el = document.querySelector({json.dumps(selector)}); "
            "if (!el) return null; const r = el.getBoundingClientRect(); "
            "return {x: r.x, y: r.y, width: r.width, height: r.height}; })()"
        )
        return rect
    raise NotImplementedError(
        f"cdp({method!r}) is not available in Camofox exec mode. Use the typed "
        "helpers: new_tab, goto_url, page_info, js, fill_input, click_ref, "
        "click_selector, press_key, scroll_page, capture_screenshot."
    )


__all__ = [
    "ensure_real_tab",
    "new_tab",
    "goto_url",
    "wait_for_load",
    "page_info",
    "js",
    "fill_input",
    "type_into_ref",
    "click_ref",
    "click_selector",
    "click_at_xy",
    "press_key",
    "scroll_page",
    "get_links",
    "capture_screenshot",
    "cdp",
]


def _load_agent_helpers() -> None:
    """Auto-import ``agent_helpers.py`` from the workspace, as promised.

    The tool description tells the model it can "define reusable functions in
    agent_helpers.py there — the harness auto-imports it into every call"
    (the Browser Use CLI harness does). Without this, the model writes the
    file and every helper silently vanishes on the next call. Names are
    appended to ``__all__`` so the wrapper's ``import *`` re-exports them.
    """
    if not _WORKSPACE:
        return
    helpers = Path(_WORKSPACE) / "agent_helpers.py"
    if not helpers.is_file():
        return
    if _WORKSPACE not in sys.path:
        sys.path.insert(0, _WORKSPACE)
    try:
        import agent_helpers
    except Exception as exc:  # a broken helper file must not kill the call
        print(f"agent_helpers.py failed to import: {exc}", file=sys.stderr)
        return
    exported = getattr(agent_helpers, "__all__", None) or [
        name for name in vars(agent_helpers) if not name.startswith("_")
    ]
    for name in exported:
        if not hasattr(agent_helpers, name):
            continue
        globals()[name] = getattr(agent_helpers, name)
        if name not in __all__:
            __all__.append(name)


_load_agent_helpers()
