"""CloakBrowser stealth browser backend — local anti-detection Chromium via Playwright.

CloakBrowser is a Playwright-based Chromium wrapper with built-in fingerprint
spoofing, human-like input simulation, and anti-detection features.  Unlike the
Camofox backend (which communicates over a REST API), CloakBrowser is driven
directly via the ``cloakbrowser`` Python package which wraps Playwright.

When ``CLOAKBROWSER_MODE`` is set (e.g. ``local``) **or** ``browser.backend``
is configured as ``cloakbrowser`` in config.yaml, all browser operations route
through this module instead of the ``agent-browser`` CLI.

Routing priority:
    CDP → CloakBrowser → Camofox → Cloud → Local

Setup::

    pip install cloakbrowser

Then set ``CLOAKBROWSER_MODE=local`` in ``~/.hermes/.env`` **or** add
``browser.backend: cloakbrowser`` to ``~/.hermes/config.yaml``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from hermes_cli.config import cfg_get, load_config
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "combobox", "checkbox", "radio",
    "switch", "slider", "menuitem", "tab", "treeitem", "option",
    "searchbox", "spinbutton", "gridcell", "cell", "columnheader",
    "rowheader", "img", "dialog", "alertdialog",
})

_ARIA_BARE_LINE_RE = re.compile(
    r'^(?P<indent>\s*)- (?P<role>\w+)(?:\s+(?P<name>"[^"]*"))?'
    r'(?:\s+(?P<attrs>\[.*?\]))?'
    r'(?P<colon>:[\s]*)?$'
)


def _env_flag(name: str) -> Optional[bool]:
    """Parse a boolean-ish env var. Returns None when unset."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.debug("Ignoring invalid boolean env %s=%r", name, raw)
    return None


def is_cloakbrowser_mode() -> bool:
    """True when CloakBrowser backend is configured and no CDP override is active.

    When the user has explicitly connected to a live Chrome instance via
    ``/browser connect`` (which sets ``BROWSER_CDP_URL``), the CDP connection
    takes priority over CloakBrowser so the browser tools operate on the real
    browser instead of being silently routed to CloakBrowser.
    """
    if os.getenv("BROWSER_CDP_URL", "").strip():
        return False
    # Env var takes priority
    env = _env_flag("CLOAKBROWSER_MODE")
    if env is True:
        return True
    if env is False:
        return False
    # Fall back to config
    backend = ""
    try:
        backend = (load_config().get("browser", {}) or {}).get("backend", "") or ""
    except Exception:
        pass
    return backend.strip().lower() == "cloakbrowser"


def check_cloakbrowser_available() -> bool:
    """Verify the cloakbrowser package is importable and its binary exists."""
    try:
        import cloakbrowser  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "CloakBrowser mode is active but the 'cloakbrowser' package is not installed. "
            "Install with: pip install cloakbrowser"
        )
        return False


def _get_cloakbrowser_config() -> Dict[str, Any]:
    """Return the ``browser.cloakbrowser`` config block, or an empty dict."""
    try:
        cb_cfg = load_config().get("browser", {}).get("cloakbrowser", {})
    except Exception as exc:
        logger.warning("CloakBrowser config check failed, defaulting to disabled: %s", exc)
        return {}
    return cb_cfg if isinstance(cb_cfg, dict) else {}


def _is_headless() -> bool:
    """Return whether CloakBrowser should run headlessly (default True)."""
    env = _env_flag("CLOAKBROWSER_HEADLESS")
    if env is not None:
        return env
    return bool(_get_cloakbrowser_config().get("headless", True))


def _is_humanize() -> bool:
    """Return whether human-like input simulation is enabled (default False)."""
    env = _env_flag("CLOAKBROWSER_HUMANIZE")
    if env is not None:
        return env
    return bool(_get_cloakbrowser_config().get("humanize", False))


def _get_humanize_preset() -> str:
    """Return the humanize preset name (default 'default')."""
    env = os.getenv("CLOAKBROWSER_HUMANIZE_PRESET", "").strip()
    if env:
        return env
    return str(_get_cloakbrowser_config().get("humanize_preset", "default"))


def _get_fingerprint_seed() -> str:
    """Return the fingerprint seed (empty = random per session)."""
    env = os.getenv("CLOAKBROWSER_FINGERPRINT_SEED", "").strip()
    if env:
        return env
    return str(_get_cloakbrowser_config().get("fingerprint_seed", ""))


def _get_proxy() -> str:
    """Return proxy URL for CloakBrowser (empty = no proxy)."""
    env = os.getenv("CLOAKBROWSER_PROXY", "").strip()
    if env:
        return env
    return str(_get_cloakbrowser_config().get("proxy", ""))


def _is_geoip() -> bool:
    """Return whether GeoIP-based fingerprint matching is enabled (default False)."""
    env = _env_flag("CLOAKBROWSER_GEOIP")
    if env is not None:
        return env
    return bool(_get_cloakbrowser_config().get("geoip", False))


def _get_timezone() -> str:
    """Return the timezone override (empty = auto-detect or system default)."""
    env = os.getenv("CLOAKBROWSER_TIMEZONE", "").strip()
    if env:
        return env
    return str(_get_cloakbrowser_config().get("timezone", ""))


def _get_locale() -> str:
    """Return the locale override (empty = auto-detect or system default)."""
    env = os.getenv("CLOAKBROWSER_LOCALE", "").strip()
    if env:
        return env
    return str(_get_cloakbrowser_config().get("locale", ""))


# ---------------------------------------------------------------------------
# Browser / page lifecycle
# ---------------------------------------------------------------------------

_browser_lock = threading.Lock()
# ---------------------------------------------------------------------------
# Playwright thread dispatcher
# ---------------------------------------------------------------------------
# Playwright's sync API binds its greenlet/fiber dispatcher to the thread that
# called ``sync_playwright().__enter__()``.  When Hermes calls browser tools
# from inside an asyncio event loop, ``sync_playwright`` detects the running
# loop and raises.  Even if we launch in a background thread, all subsequent
# operations (new_page, goto, evaluate, etc.) must run on that *same* thread
# because the greenlet dispatcher is thread-bound.
#
# The fix: a dedicated daemon thread that owns the Playwright context and
# processes all browser operations via a ``queue.Queue``.  Every public
# function submits a callable to the queue and blocks on the result.

_playwright_thread: Optional[threading.Thread] = None
_playwright_queue: "queue.Queue" = None  # type: ignore[assignment]
_playwright_ready = threading.Event()


def _run_on_playwright_thread(fn: Callable, timeout: float = 120.0) -> Any:
    """Submit *fn* to the dedicated Playwright thread and return the result.

    Starts the thread on first call.  Raises any exception from *fn* in the
    caller's thread.

    If called from within the Playwright thread itself (e.g. nested calls
    from _ensure_page -> _ensure_browser), runs *fn* directly to avoid
    deadlock.
    """
    global _playwright_thread, _playwright_queue
    import queue as _queue_mod

    # If we're already on the Playwright thread, just call directly — no
    # deadlock risk since a single thread can't dequeue while executing.
    if threading.current_thread() is _playwright_thread:
        return fn()

    if _playwright_thread is None or not _playwright_thread.is_alive():
        _playwright_queue = _queue_mod.Queue()
        _playwright_ready.clear()
        _playwright_thread = threading.Thread(
            target=_playwright_main, daemon=True, name="cloakbrowser-pw"
        )
        _playwright_thread.start()
        _playwright_ready.wait(timeout=120)  # block until Playwright is ready

    result_container: list = []
    exc_container: list = []

    def _work():
        try:
            result_container.append(fn())
        except Exception as e:
            exc_container.append(e)

    _playwright_queue.put(_work)
    # Block until the work item is processed
    deadline = time.monotonic() + timeout
    while not result_container and not exc_container:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Playwright thread did not respond within {timeout}s")
        time.sleep(0.01)
    if exc_container:
        raise exc_container[0]
    return result_container[0]


def _playwright_main():
    """Persistent thread that owns the Playwright event loop."""
    # This thread has no asyncio loop, so sync_playwright() works without
    # the "using Playwright Sync API inside the asyncio loop" error.
    _playwright_ready.set()
    while True:
        work = _playwright_queue.get()
        if work is None:
            # Shutdown sentinel
            break
        try:
            work()
        except Exception:
            # Exceptions are captured by the caller via _run_on_playwright_thread;
            # we must not crash the thread.
            pass


_browser_instance = None  # Singleton cloakbrowser.Browser


def _ensure_browser():
    """Launch (or return the existing) singleton CloakBrowser instance.

    All Playwright operations are dispatched to a dedicated thread to avoid
    conflicts with asyncio event loops in the caller's thread.
    """
    global _browser_instance
    if _browser_instance is not None:
        return _browser_instance

    import cloakbrowser

    kwargs: Dict[str, Any] = {}
    kwargs["headless"] = _is_headless()

    extra_args: List[str] = []
    fingerprint_seed = _get_fingerprint_seed()
    if fingerprint_seed:
        extra_args.append(f"--fingerprint={fingerprint_seed}")

    proxy = _get_proxy()
    if proxy:
        kwargs["proxy"] = proxy

    if _is_humanize():
        kwargs["humanize"] = True
        kwargs["humanize_preset"] = _get_humanize_preset()

    if _is_geoip():
        kwargs["geoip"] = True

    binary_path = os.getenv("CLOAKBROWSER_BINARY_PATH", "").strip()
    if binary_path:
        kwargs["executable_path"] = binary_path

    timezone = _get_timezone()
    if timezone:
        kwargs["timezone"] = timezone

    locale = _get_locale()
    if locale:
        kwargs["locale"] = locale

    logger.info("Launching CloakBrowser (headless=%s, humanize=%s, geoip=%s, tz=%s, locale=%s, fp_seed=%s)",
                kwargs.get("headless", True),
                kwargs.get("humanize", False),
                kwargs.get("geoip", False),
                kwargs.get("timezone", ""),
                kwargs.get("locale", ""),
                fingerprint_seed or "random")
    if extra_args:
        kwargs["args"] = extra_args

    _browser_instance = _run_on_playwright_thread(lambda: cloakbrowser.launch(**kwargs))
    return _browser_instance


# Per-task pages and ref stores
_pages_lock = threading.Lock()
_pages: Dict[str, Any] = {}           # task_id → Playwright Page
_refs: Dict[str, Dict[str, str]] = {}  # task_id → {ref_id → locator_string}


def _ensure_page(task_id: Optional[str] = None):
    """Get or create a Playwright Page for the given task.

    Runs browser launch and page creation on the dedicated Playwright thread.
    """
    task_id = task_id or "default"
    with _pages_lock:
        if task_id in _pages:
            return _pages[task_id]

        def _create_page():
            browser = _ensure_browser()
            context = browser.new_context() if hasattr(browser, "new_context") else browser
            return context.new_page() if hasattr(context, "new_page") else browser.new_page()

        page = _run_on_playwright_thread(_create_page)
        _pages[task_id] = page
        _refs[task_id] = {}
        return page


# ---------------------------------------------------------------------------
# Aria snapshot → Hermes format with @eN refs
# ---------------------------------------------------------------------------

def _add_refs_to_aria_snapshot(raw_aria: str, task_id: str) -> str:
    """Parse Playwright aria_snapshot output and inject @eN refs for interactive elements.

    The raw aria_snapshot is a YAML-ish tree like::

        - main
          - heading "Welcome" [level=1]
          - link "Home" [/]
          - textbox "Search" []

    We inject ``@e0``, ``@e1``, ... labels on lines whose role is in
    ``_INTERACTIVE_ROLES`` and store a Playwright locator string in the
    per-task ``_refs`` dict for later click/type resolution.
    """
    refs = _refs.setdefault(task_id, {})
    lines = raw_aria.splitlines()
    result_lines = []
    ref_counter = 0

    for line in lines:
        m = _ARIA_BARE_LINE_RE.match(line)
        if m:
            role = m.group("role")
            name_raw = m.group("name")  # includes surrounding quotes, e.g. '"Home"'
            name = name_raw[1:-1] if name_raw else ""
            attrs = m.group("attrs") or ""
            indent = m.group("indent")
            colon = m.group("colon") or ""

            if role in _INTERACTIVE_ROLES:
                ref_id = f"e{ref_counter}"
                ref_counter += 1
                # Build a locator string for Playwright resolution
                parts = []
                parts.append(f"[role='{role}']")
                if name:
                    parts.append(f"text='{name}'")
                locator_str = " >> ".join(parts)
                refs[ref_id] = locator_str

                # Inject ref into the line
                ref_tag = f"@{ref_id}"
                # Reconstruct line with ref injected
                new_line = f"{indent}- {role}"
                if name_raw:
                    new_line += f" {name_raw}"
                if attrs:
                    new_line += f" {attrs}"
                new_line += f" {ref_tag}"
                if colon:
                    new_line += colon
                result_lines.append(new_line)
                continue

        # Non-interactive or non-matching line: pass through
        result_lines.append(line)

    return "\n".join(result_lines)


def _playwright_snapshot_to_hermes_format(page, task_id: str) -> Dict[str, Any]:
    """Take a Playwright aria_snapshot and convert it to the Hermes browser tool format."""
    try:
        raw_aria = page.aria_snapshot()
    except Exception as exc:
        logger.error("CloakBrowser aria_snapshot failed: %s", exc)
        return {"success": False, "error": str(exc), "snapshot": "", "element_count": 0}

    if not raw_aria:
        return {"success": True, "snapshot": "(empty page)", "element_count": 0}

    annotated = _add_refs_to_aria_snapshot(raw_aria, task_id)
    ref_count = len(_refs.get(task_id, {}))

    return {
        "success": True,
        "snapshot": annotated,
        "element_count": ref_count,
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def cloakbrowser_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to a URL via CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)

        def _do_navigate():
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            # Wait a moment for dynamic content to render
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass  # networkidle can timeout on complex pages; that's fine
            return {
                "success": True,
                "url": page.url,
                "title": page.title() if hasattr(page, "title") else "",
            }

        result = _run_on_playwright_thread(_do_navigate)

        # Auto-take a compact snapshot so the model can act immediately
        try:
            from tools.browser_tool import SNAPSHOT_SUMMARIZE_THRESHOLD, _truncate_snapshot
            snap = _run_on_playwright_thread(
                lambda: _playwright_snapshot_to_hermes_format(page, task_id)
            )
            snapshot_text = snap.get("snapshot", "")
            if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
                snapshot_text = _truncate_snapshot(snapshot_text)
            result["snapshot"] = snapshot_text
            result["element_count"] = snap.get("element_count", 0)
        except Exception:
            pass  # Navigation succeeded; snapshot is a bonus

        return json.dumps(result)
    except Exception as e:
        return tool_error(f"CloakBrowser navigation failed: {e}", success=False)


def cloakbrowser_snapshot(full: bool = False, task_id: Optional[str] = None,
                          user_task: Optional[str] = None) -> str:
    """Get accessibility tree snapshot from CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)
        snap = _run_on_playwright_thread(
            lambda: _playwright_snapshot_to_hermes_format(page, task_id)
        )

        snapshot = snap.get("snapshot", "")
        refs_count = snap.get("element_count", 0)

        # Apply same summarization logic as the main browser tool
        from tools.browser_tool import (
            SNAPSHOT_SUMMARIZE_THRESHOLD,
            _extract_relevant_content,
            _truncate_snapshot,
        )

        if len(snapshot) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            if user_task:
                snapshot = _extract_relevant_content(snapshot, user_task)
            else:
                snapshot = _truncate_snapshot(snapshot)

        return json.dumps({
            "success": True,
            "snapshot": snapshot,
            "element_count": refs_count,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def cloakbrowser_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click an element by ref via CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)
        clean_ref = ref.lstrip("@")

        def _do_click():
            locator_str = _refs.get(task_id, {}).get(clean_ref)
            if locator_str:
                try:
                    locator = page.locator(locator_str)
                    locator.first.click(timeout=5000)
                    return
                except Exception as exc:
                    logger.debug("CloakBrowser locator click failed, falling back to role: %s", exc)
                    locator_str = None  # Fall through to role-based resolution

            if not locator_str:
                # Fallback: re-parse snapshot to find role and name
                try:
                    raw_aria = page.aria_snapshot()
                    _add_refs_to_aria_snapshot(raw_aria, task_id)
                    locator_str = _refs.get(task_id, {}).get(clean_ref)
                    if locator_str:
                        locator = page.locator(locator_str)
                        locator.first.click(timeout=5000)
                    else:
                        # Last resort: try get_by_role with name from aria
                        _resolve_click_by_role(page, clean_ref, task_id)
                except Exception as exc2:
                    raise RuntimeError(f"CloakBrowser click failed for ref @{clean_ref}: {exc2}")

            return page.url

        url_after = _run_on_playwright_thread(_do_click)
        return json.dumps({
            "success": True,
            "clicked": clean_ref,
            "url": url_after,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def _resolve_click_by_role(page, ref_id: str, task_id: str):
    """Attempt to click an element using page.get_by_role()."""
    refs = _refs.get(task_id, {})
    locator_str = refs.get(ref_id, "")
    # Parse out role and name from locator_str like "[role='button'] >> text='Submit'"
    if locator_str:
        parts = locator_str.split(" >> ")
        role = ""
        name = ""
        for part in parts:
            if part.startswith("[role='") and part.endswith("']"):
                role = part[7:-2]
            elif part.startswith("text='") and part.endswith("'"):
                name = part[6:-1]
        if role:
            if name:
                page.get_by_role(role, name=name).first.click(timeout=5000)
            else:
                page.get_by_role(role).first.click(timeout=5000)


def cloakbrowser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type text into an element by ref via CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)
        clean_ref = ref.lstrip("@")

        def _do_type():
            locator_str = _refs.get(task_id, {}).get(clean_ref)
            if locator_str:
                try:
                    locator = page.locator(locator_str)
                    locator.first.fill(text, timeout=5000)
                    return
                except Exception as exc:
                    logger.debug("CloakBrowser locator fill failed, falling back to role: %s", exc)
                    # Try using get_by_role
                    refs = _refs.get(task_id, {})
                    ls = refs.get(clean_ref, "")
                    parts = ls.split(" >> ")
                    role = ""
                    name = ""
                    for part in parts:
                        if part.startswith("[role='") and part.endswith("']"):
                            role = part[7:-2]
                        elif part.startswith("text='") and part.endswith("'"):
                            name = part[6:-1]
                    if role:
                        if name:
                            page.get_by_role(role, name=name).first.fill(text, timeout=5000)
                        else:
                            page.get_by_role(role).first.fill(text, timeout=5000)
                    else:
                        raise
            else:
                # Re-parse snapshot to find the ref
                raw_aria = page.aria_snapshot()
                _add_refs_to_aria_snapshot(raw_aria, task_id)
                locator_str = _refs.get(task_id, {}).get(clean_ref)
                if locator_str:
                    locator = page.locator(locator_str)
                    locator.first.fill(text, timeout=5000)
                else:
                    raise ValueError(f"Could not resolve ref @{clean_ref}")

        _run_on_playwright_thread(_do_type)
        return json.dumps({
            "success": True,
            "typed": text,
            "element": clean_ref,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def cloakbrowser_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page via CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)
        scroll_amount = 500  # pixels per scroll step

        def _do_scroll():
            if direction == "up":
                page.mouse.wheel(0, -scroll_amount)
            elif direction == "down":
                page.mouse.wheel(0, scroll_amount)
            elif direction == "left":
                page.mouse.wheel(-scroll_amount, 0)
            elif direction == "right":
                page.mouse.wheel(scroll_amount, 0)
            else:
                # Default to down
                page.mouse.wheel(0, scroll_amount)

        _run_on_playwright_thread(_do_scroll)
        return json.dumps({"success": True, "scrolled": direction})
    except Exception as e:
        return tool_error(str(e), success=False)


def cloakbrowser_back(task_id: Optional[str] = None) -> str:
    """Navigate back via CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)

        def _do_back():
            page.go_back(timeout=10000)
            return page.url

        url_after = _run_on_playwright_thread(_do_back)
        return json.dumps({
            "success": True,
            "url": url_after,
        })
    except Exception as e:
        return tool_error(f"CloakBrowser back failed: {e}", success=False)


def cloakbrowser_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key via CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)

        def _do_press():
            page.keyboard.press(key)

        _run_on_playwright_thread(_do_press)
        return json.dumps({
            "success": True,
            "pressed": key,
        })
    except Exception as e:
        return tool_error(f"CloakBrowser press failed: {e}", success=False)


def cloakbrowser_console(clear: bool = False, expression: Optional[str] = None,
                         task_id: Optional[str] = None) -> str:
    """Evaluate JavaScript or get console output via CloakBrowser."""
    task_id = task_id or "default"
    if expression is not None:
        return _cloakbrowser_eval(expression, task_id)

    # Console output mode: Playwright doesn't have a built-in console history
    # so we can only evaluate JS to return recent messages if desired.
    try:
        page = _ensure_page(task_id)
        # Return a simple response; console log capture would need listener setup
        return json.dumps({
            "success": True,
            "console_messages": [],
            "js_errors": [],
            "total_messages": 0,
            "total_errors": 0,
            "note": "CloakBrowser console capture requires JS evaluation; use browser_console(expression=...) instead.",
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def cloakbrowser_get_images(task_id: Optional[str] = None) -> str:
    """Extract all image elements from the page via CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)

        def _get_images():
            return page.evaluate("""() => {
                const imgs = [];
                for (const img of document.querySelectorAll('img')) {
                    imgs.push({
                        src: img.src || '',
                        alt: img.alt || '',
                        width: img.naturalWidth || img.width || 0,
                        height: img.naturalHeight || img.height || 0,
                    });
                }
                for (const el of document.querySelectorAll('[role="img"], svg, canvas')) {
                    const ariaLabel = el.getAttribute('aria-label') || '';
                    const src = el.getAttribute('src') || el.toDataURL?.() || '';
                    if (src || ariaLabel) {
                        imgs.push({
                            src: src,
                            alt: ariaLabel,
                            width: el.offsetWidth || 0,
                            height: el.offsetHeight || 0,
                        });
                    }
                }
                return imgs;
            }""")

        images = _run_on_playwright_thread(_get_images)
        return json.dumps({
            "success": True,
            "images": images or [],
            "count": len(images) if images else 0,
        })
    except Exception as e:
        return tool_error(f"CloakBrowser get_images failed: {e}", success=False)


def cloakbrowser_vision(question: Optional[str] = None, annotate: bool = False,
                        task_id: Optional[str] = None) -> str:
    """Take a screenshot and optionally analyze it via CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)

        def _take_screenshot():
            return page.screenshot(full_page=False)

        screenshot_bytes = _run_on_playwright_thread(_take_screenshot)
        import base64
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("ascii")

        result = {
            "success": True,
            "screenshot_available": True,
            "screenshot_size_bytes": len(screenshot_bytes),
        }

        if question:
            # Vision analysis would need a multimodal model, which is handled
            # at the browser_tool level. Just return the screenshot info here.
            result["note"] = "Vision Q&A is handled by the calling layer; returning screenshot metadata."

        return json.dumps(result)
    except Exception as e:
        return tool_error(f"CloakBrowser vision failed: {e}", success=False)

def _cloakbrowser_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate a JavaScript expression in the page context via CloakBrowser."""
    task_id = task_id or "default"
    try:
        page = _ensure_page(task_id)

        def _do_eval():
            return page.evaluate(expression)

        raw_result = _run_on_playwright_thread(_do_eval)

        # Try to parse as JSON for structured return
        parsed = raw_result
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
            except (json.JSONDecodeError, ValueError):
                pass

        return json.dumps({
            "success": True,
            "result": parsed,
            "result_type": type(parsed).__name__,
        }, ensure_ascii=False, default=str)
    except Exception as e:
        error_msg = str(e)
        return json.dumps({
            "success": False,
            "error": error_msg,
        })


def cloakbrowser_close(task_id: Optional[str] = None) -> str:
    """Close the CloakBrowser page for a task, or the browser if no task_id."""
    global _browser_instance
    task_id = task_id or "default"

    with _pages_lock:
        page = _pages.pop(task_id, None)
        _refs.pop(task_id, None)

    if page is not None:
        try:
            _run_on_playwright_thread(page.close, timeout=10)
        except Exception as exc:
            logger.debug("CloakBrowser page close error for task %s: %s", task_id, exc)

    # If no more pages, shut down the whole browser
    with _pages_lock:
        remaining = len(_pages)

    if remaining == 0:
        with _browser_lock:
            if _browser_instance is not None:
                try:
                    _run_on_playwright_thread(_browser_instance.close, timeout=10)
                except Exception as exc:
                    logger.debug("CloakBrowser browser close error: %s", exc)
                _browser_instance = None
                # Signal the Playwright thread to exit
                _playwright_queue.put(None)

    return json.dumps({"success": True})