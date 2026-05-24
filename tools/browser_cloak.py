"""Native CloakBrowser backend for Hermes browser tools.

CloakBrowser is a stealth Chromium Playwright wrapper.  This module keeps the
same high-level contract as the regular browser tool backend while avoiding the
agent-browser CLI and the Camofox REST server.

Enable with either:
- CLOAKBROWSER_ENABLED=true
- browser.cloakbrowser.enabled: true in config.yaml

A live CDP override (BROWSER_CDP_URL) always takes precedence so /browser
connect keeps operating on the user-selected browser.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli.config import load_config
from tools.registry import tool_error
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 60_000
_SCROLL_PIXELS = 500
_ACTION_SELECTOR = ", ".join(
    [
        "a",
        "button",
        "input",
        "textarea",
        "select",
        "summary",
        "[role=button]",
        "[role=link]",
        "[role=checkbox]",
        "[role=radio]",
        "[role=tab]",
        "[role=menuitem]",
        "[contenteditable=true]",
        "[onclick]",
    ]
)

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


def _cfg_enabled() -> bool:
    try:
        browser_cfg = load_config().get("browser", {})
        cloak_cfg = browser_cfg.get("cloakbrowser", {}) if isinstance(browser_cfg, dict) else {}
        if isinstance(cloak_cfg, dict):
            return bool(cloak_cfg.get("enabled"))
    except Exception as exc:
        logger.debug("Could not read browser.cloakbrowser.enabled: %s", exc)
    return False


def is_cloakbrowser_mode() -> bool:
    """Return True when CloakBrowser backend is enabled.

    CDP override wins because it represents an explicit user/browser connect.
    """
    if os.getenv("BROWSER_CDP_URL", "").strip():
        return False
    return is_truthy_value(os.getenv("CLOAKBROWSER_ENABLED", "")) or _cfg_enabled()


def check_cloakbrowser_available() -> bool:
    """True when the Python package and browser binary are available."""
    if not is_cloakbrowser_mode():
        return False
    try:
        import cloakbrowser  # noqa: F401
        from cloakbrowser import ensure_binary

        ensure_binary()
        return True
    except Exception as exc:
        logger.debug("CloakBrowser unavailable: %s", exc)
        return False


def _get_session(task_id: Optional[str]) -> Dict[str, Any]:
    task_id = task_id or "default"
    with _sessions_lock:
        session = _sessions.get(task_id)
        if session:
            return session

    from cloakbrowser import launch

    headless = not is_truthy_value(os.getenv("CLOAKBROWSER_HEADFUL", ""))
    humanize = is_truthy_value(os.getenv("CLOAKBROWSER_HUMANIZE", "true"))
    browser = launch(headless=headless, humanize=humanize)
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(_DEFAULT_TIMEOUT_MS)
    session = {
        "browser": browser,
        "context": context,
        "page": page,
        "refs": {},
    }
    with _sessions_lock:
        existing = _sessions.get(task_id)
        if existing:
            try:
                browser.close()
            except Exception:
                pass
            return existing
        _sessions[task_id] = session
    return session


def _page(task_id: Optional[str]):
    return _get_session(task_id)["page"]


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _clean_ref(ref: str) -> str:
    return (ref or "").lstrip("@")


def _locator_for_ref(session: Dict[str, Any], ref: str):
    clean = _clean_ref(ref)
    if not clean:
        raise ValueError("Empty element ref")
    selector = session.get("refs", {}).get(clean) or f'[data-hermes-ref="{clean}"]'
    return session["page"].locator(selector).first()


def _refresh_refs(session: Dict[str, Any]) -> int:
    page = session["page"]
    refs = page.evaluate(
        """
        (selector) => {
          const nodes = Array.from(document.querySelectorAll(selector));
          const refs = {};
          let idx = 1;
          for (const el of nodes) {
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            const visible = rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            if (!visible) continue;
            const ref = `e${idx++}`;
            el.setAttribute('data-hermes-ref', ref);
            refs[ref] = `[data-hermes-ref="${ref}"]`;
          }
          return refs;
        }
        """,
        _ACTION_SELECTOR,
    )
    session["refs"] = refs or {}
    return len(session["refs"])


def _compact_refs_text(session: Dict[str, Any]) -> str:
    page = session["page"]
    items = page.evaluate(
        """
        () => Array.from(document.querySelectorAll('[data-hermes-ref]')).map(el => ({
          ref: el.getAttribute('data-hermes-ref'),
          tag: el.tagName.toLowerCase(),
          role: el.getAttribute('role') || '',
          text: (el.innerText || el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('alt') || el.value || '').trim().slice(0, 120),
          href: el.getAttribute('href') || ''
        }))
        """
    )
    lines = []
    for item in items or []:
        label = item.get("text") or item.get("href") or item.get("tag")
        role = item.get("role") or item.get("tag")
        lines.append(f'- {role} "{label}" [@{item.get("ref")}]')
    return "\n".join(lines)


def _snapshot_text(session: Dict[str, Any], full: bool = False) -> str:
    page = session["page"]
    try:
        snap = page.locator("body").aria_snapshot(timeout=10_000)
    except Exception:
        snap = page.content()
    ref_count = _refresh_refs(session)
    refs = _compact_refs_text(session)
    if refs:
        snap = f"{snap}\n\nInteractive elements:\n{refs}"
    if not full and len(snap) > 8000:
        snap = snap[:8000] + f"\n\n[Snapshot truncated; {ref_count} interactive elements indexed]"
    return snap


def cloakbrowser_navigate(url: str, task_id: Optional[str] = None) -> str:
    try:
        session = _get_session(task_id)
        page = session["page"]
        page.goto(url, wait_until="domcontentloaded", timeout=_DEFAULT_TIMEOUT_MS)
        try:
            page.wait_for_load_state("networkidle", timeout=5_000)
        except Exception:
            pass
        snapshot = _snapshot_text(session, full=False)
        return _json({
            "success": True,
            "backend": "cloakbrowser",
            "url": page.url,
            "title": page.title(),
            "snapshot": snapshot,
            "element_count": len(session.get("refs", {})),
        })
    except Exception as exc:
        return tool_error(f"CloakBrowser navigation failed: {exc}", success=False)


def cloakbrowser_snapshot(full: bool = False, task_id: Optional[str] = None, user_task: Optional[str] = None) -> str:
    try:
        session = _get_session(task_id)
        snapshot = _snapshot_text(session, full=full)
        return _json({"success": True, "snapshot": snapshot, "element_count": len(session.get("refs", {}))})
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_click(ref: str, task_id: Optional[str] = None) -> str:
    try:
        session = _get_session(task_id)
        loc = _locator_for_ref(session, ref)
        loc.click()
        return _json({"success": True, "clicked": _clean_ref(ref), "url": session["page"].url})
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    try:
        session = _get_session(task_id)
        loc = _locator_for_ref(session, ref)
        loc.fill("")
        loc.type(text)
        return _json({"success": True, "typed": text, "element": _clean_ref(ref)})
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_scroll(direction: str, task_id: Optional[str] = None) -> str:
    try:
        page = _page(task_id)
        delta = _SCROLL_PIXELS if direction == "down" else -_SCROLL_PIXELS
        page.mouse.wheel(0, delta)
        return _json({"success": True, "scrolled": direction})
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_back(task_id: Optional[str] = None) -> str:
    try:
        page = _page(task_id)
        page.go_back(wait_until="domcontentloaded")
        return _json({"success": True, "url": page.url})
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_press(key: str, task_id: Optional[str] = None) -> str:
    try:
        _page(task_id).keyboard.press(key)
        return _json({"success": True, "pressed": key})
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_get_images(task_id: Optional[str] = None) -> str:
    try:
        page = _page(task_id)
        images = page.evaluate(
            """
            () => Array.from(document.images).map(img => ({
              src: img.currentSrc || img.src || '',
              alt: img.alt || ''
            })).filter(x => x.src || x.alt)
            """
        )
        return _json({"success": True, "images": images or [], "count": len(images or [])})
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_console(clear: bool = False, expression: Optional[str] = None, task_id: Optional[str] = None) -> str:
    try:
        page = _page(task_id)
        result: Dict[str, Any] = {"success": True}
        if expression:
            result["result"] = page.evaluate(expression)
        else:
            result["messages"] = []
            result["errors"] = []
        return _json(result)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_screenshot(task_id: Optional[str] = None) -> str:
    session = _get_session(task_id)
    path = Path(tempfile.gettempdir()) / f"hermes_cloak_{os.getpid()}_{task_id or 'default'}.png"
    session["page"].screenshot(path=str(path), full_page=True)
    return str(path)


def cloakbrowser_vision(question: str, annotate: bool = False, task_id: Optional[str] = None) -> str:
    try:
        path = cloakbrowser_screenshot(task_id)
        from tools.vision_tool import analyze_image

        analysis = analyze_image(str(path), question)
        return _json({"success": True, "analysis": analysis, "screenshot_path": str(path)})
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_close(task_id: Optional[str] = None) -> str:
    task_id = task_id or "default"
    with _sessions_lock:
        session = _sessions.pop(task_id, None)
    if session:
        try:
            session["browser"].close()
        except Exception as exc:
            return _json({"success": True, "closed": True, "warning": str(exc)})
    return _json({"success": True, "closed": True})


def cloakbrowser_soft_cleanup(task_id: Optional[str] = None) -> bool:
    return False
