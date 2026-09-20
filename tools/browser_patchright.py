"""Native Patchright browser backend for Hermes.

This module implements the command contract consumed by ``tools.browser_tool``
without going through agent-browser or Camofox.  It uses the Python Patchright
sync API, keeps one isolated persistent context per Hermes task, and returns the
same ``{"success": bool, "data": {...}}`` envelopes as agent-browser.

The backend is selected with::

    browser:
      cloud_provider: patchright

The Patchright package is optional at install time.  If it is not installed,
commands return an actionable setup error instead of silently falling back to a
different browser.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from hermes_cli.config import cfg_get, read_raw_config

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_DEFAULT_NAV_TIMEOUT = 120
_DEFAULT_SNAPSHOT_MAX_CHARS = 80_000

_runtime_lock = threading.RLock()
_runtime = None
_sessions: Dict[str, Dict[str, Any]] = {}


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _patchright_config() -> Dict[str, Any]:
    try:
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if not isinstance(browser_cfg, dict):
            return {}
        value = browser_cfg.get("patchright", {})
        return value if isinstance(value, dict) else {}
    except Exception as exc:  # pragma: no cover - defensive config fallback
        logger.debug("Could not read browser.patchright config: %s", exc)
        return {}


def is_patchright_mode() -> bool:
    """Return whether Hermes normal browser tools should use Patchright."""
    env_value = os.getenv("HERMES_BROWSER_BACKEND", "").strip().lower()
    if env_value in {"patchright", "patch-right"}:
        return True
    if env_value in {"camofox", "agent-browser", "local", "chrome"}:
        return False

    try:
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict):
            provider = str(browser_cfg.get("cloud_provider", "") or "").strip().lower()
            return provider in {"patchright", "patch-right"}
    except Exception:
        pass
    return False


def _command_timeout() -> int:
    try:
        cfg = read_raw_config()
        value = cfg_get(cfg, "browser", "command_timeout", default=_DEFAULT_TIMEOUT)
        return max(int(value), 5)
    except Exception:
        return _DEFAULT_TIMEOUT


def _nav_timeout() -> int:
    try:
        value = _patchright_config().get("navigation_timeout", _DEFAULT_NAV_TIMEOUT)
        return max(int(value), 5)
    except Exception:
        return _DEFAULT_NAV_TIMEOUT


def _browser_cache_path() -> str:
    cfg = _patchright_config()
    value = (
        os.getenv("PATCHRIGHT_BROWSERS_PATH", "").strip()
        or str(cfg.get("browsers_path", "") or "").strip()
        or str(get_hermes_home() / "patchright" / "browsers")
    )
    return value


def _profile_root() -> Path:
    cfg = _patchright_config()
    value = (
        os.getenv("PATCHRIGHT_PROFILE_DIR", "").strip()
        or str(cfg.get("profile_dir", "") or "").strip()
        or str(get_hermes_home() / "patchright" / "profile")
    )
    return Path(value).expanduser()


def _profile_for_task(task_id: str) -> Path:
    """Give concurrent Hermes tasks independent Chromium profile locks."""
    digest = hashlib.sha256((task_id or "default").encode("utf-8")).hexdigest()[:20]
    root = _profile_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / digest


def _headless() -> bool:
    cfg = _patchright_config()
    env = os.getenv("PATCHRIGHT_HEADLESS", "").strip()
    if env:
        return _truthy(env, default=False)
    if "headless" in cfg:
        return _truthy(cfg.get("headless"), default=False)
    # Hermes's local browser defaults to headless; the configured Patchright
    # integration deliberately defaults to headed so the user can observe it.
    return False


def _launch_options() -> Dict[str, Any]:
    cfg = _patchright_config()
    options: Dict[str, Any] = {
        "headless": _headless(),
        "viewport": {"width": 1440, "height": 1000},
    }
    channel = str(cfg.get("channel", "") or os.getenv("PATCHRIGHT_CHANNEL", "")).strip()
    executable = str(
        cfg.get("executable_path", "") or os.getenv("PATCHRIGHT_EXECUTABLE_PATH", "")
    ).strip()
    if channel:
        options["channel"] = channel
    if executable:
        options["executable_path"] = executable

    args = cfg.get("args")
    if isinstance(args, list) and all(isinstance(item, str) for item in args):
        options["args"] = list(args)
    return options


def _load_sync_playwright():
    try:
        from patchright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Patchright is not installed in Hermes's Python environment. "
            "Install it with: hermes-agent\\venv\\Scripts\\python.exe -m pip "
            "install patchright==1.60.1"
        ) from exc
    return sync_playwright


def check_patchright_available() -> bool:
    """Return whether the Hermes runtime can import Patchright."""
    try:
        _load_sync_playwright()
        return True
    except RuntimeError as exc:
        logger.warning("Patchright browser backend unavailable: %s", exc)
        return False


def _ensure_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", _browser_cache_path())
    sync_playwright = _load_sync_playwright()
    _runtime = sync_playwright().start()
    return _runtime


def _message_text(message: Any) -> str:
    try:
        value = getattr(message, "text", "")
        return str(value() if callable(value) else value)
    except Exception:
        return str(message)


def _attach_page_hooks(session: Dict[str, Any], page: Any) -> None:
    page_id = id(page)
    if page_id in session["hooked_pages"]:
        return
    session["hooked_pages"].add(page_id)

    def on_console(message: Any) -> None:
        session["console"].append(
            {"type": str(getattr(message, "type", "log")), "text": _message_text(message)}
        )

    def on_page_error(error: Any) -> None:
        session["errors"].append({"message": str(error)})

    try:
        page.on("console", on_console)
        page.on("pageerror", on_page_error)
    except Exception as exc:  # pragma: no cover - browser-version defensive
        logger.debug("Could not attach Patchright page hooks: %s", exc)


def _active_page(session: Dict[str, Any]):
    context = session["context"]
    pages = [page for page in context.pages if not page.is_closed()]
    if not pages:
        page = context.new_page()
        pages = [page]
    for page in pages:
        _attach_page_hooks(session, page)
    page = pages[-1]
    session["page"] = page
    return page


def _new_session(task_id: str) -> Dict[str, Any]:
    runtime = _ensure_runtime()
    cfg = _patchright_config()
    cdp_url = str(
        os.getenv("PATCHRIGHT_CDP_URL", "")
        or cfg.get("cdp_url", "")
        or ""
    ).strip()

    if cdp_url:
        browser = runtime.chromium.connect_over_cdp(cdp_url)
        contexts = browser.contexts
        context = contexts[0] if contexts else browser.new_context()
        session = {
            "browser": browser,
            "context": context,
            "attached": True,
            "profile_dir": None,
            "page": None,
            "hooked_pages": set(),
            "console": [],
            "errors": [],
        }
    else:
        profile_dir = _profile_for_task(task_id)
        options = _launch_options()
        context = runtime.chromium.launch_persistent_context(str(profile_dir), **options)
        session = {
            "browser": None,
            "context": context,
            "attached": False,
            "profile_dir": str(profile_dir),
            "page": None,
            "hooked_pages": set(),
            "console": [],
            "errors": [],
        }
    _sessions[task_id] = session
    _active_page(session)
    return session


def _get_session(task_id: Optional[str]) -> Dict[str, Any]:
    key = task_id or "default"
    with _runtime_lock:
        session = _sessions.get(key)
        if session is not None:
            try:
                _active_page(session)
                return session
            except Exception:
                _close_session_locked(key, session)
        return _new_session(key)


def _close_session_locked(task_id: str, session: Dict[str, Any]) -> None:
    try:
        if session.get("attached"):
            browser = session.get("browser")
            if browser is not None:
                browser.close()
        else:
            context = session.get("context")
            if context is not None:
                context.close()
    except Exception as exc:
        logger.debug("Patchright close failed for %s: %s", task_id, exc)
    _sessions.pop(task_id, None)


def close_session(task_id: Optional[str] = None) -> Dict[str, Any]:
    key = task_id or "default"
    with _runtime_lock:
        session = _sessions.get(key)
        if session is None:
            return {"success": True, "closed": False}
        _close_session_locked(key, session)
        return {"success": True, "closed": True}


def close_all() -> None:
    global _runtime
    with _runtime_lock:
        for key, session in list(_sessions.items()):
            _close_session_locked(key, session)
        if _runtime is not None:
            try:
                _runtime.stop()
            except Exception:
                pass
            _runtime = None


def _ref_selector(ref: str) -> str:
    ref = str(ref or "").strip()
    if not ref.startswith("@"):
        ref = "@" + ref
    return f'[data-hermes-ref="{ref[1:]}"]'


def _snapshot(session: Dict[str, Any], compact: bool = False) -> Dict[str, Any]:
    page = _active_page(session)
    payload = page.evaluate(
        """
        ({compact, maxElements}) => {
          const selector = 'a,button,input,textarea,select,summary,' +
            '[role="button"],[role="link"],[role="textbox"],[contenteditable="true"]';
          const nodes = [...document.querySelectorAll(selector)]
            .filter(el => {
              const style = getComputedStyle(el);
              const rect = el.getBoundingClientRect();
              return style.display !== 'none' && style.visibility !== 'hidden' &&
                rect.width > 0 && rect.height > 0;
            }).slice(0, maxElements);
          const refs = {};
          const items = nodes.map((el, index) => {
            const ref = '@e' + (index + 1);
            el.setAttribute('data-hermes-ref', ref.slice(1));
            let role = el.getAttribute('role') || el.tagName.toLowerCase();
            if (el.tagName.toLowerCase() === 'input') {
              const type = (el.getAttribute('type') || 'text').toLowerCase();
              role = type === 'checkbox' || type === 'radio' ? type :
                (type === 'submit' || type === 'button' ? 'button' : 'textbox');
            }
            const label = el.getAttribute('aria-label') || el.getAttribute('placeholder') ||
              el.innerText || el.value || el.getAttribute('title') || '';
            const clean = String(label).replace(/\\s+/g, ' ').trim().slice(0, 240);
            refs[ref] = {role, name: clean, tag: el.tagName.toLowerCase()};
            return {ref, role, name: clean};
          });
          let text = document.body ? (document.body.innerText || '') : '';
          text = text.replace(/\\n{3,}/g, '\\n\\n').trim();
          return {url: location.href, title: document.title || '', text, items, refs,
            compact: Boolean(compact)};
        }
        """,
        {"compact": compact, "maxElements": 500},
    )
    items = payload.get("items", []) if isinstance(payload, dict) else []
    refs = payload.get("refs", {}) if isinstance(payload, dict) else {}
    lines = [f"URL: {payload.get('url', page.url)}", f"TITLE: {payload.get('title', '')}"]
    if items:
        lines.extend(["", "INTERACTIVE ELEMENTS:"])
        for item in items:
            name = f' "{item["name"]}"' if item.get("name") else ""
            lines.append(f'{item.get("ref")} {item.get("role", "element")}{name}')
    text = str(payload.get("text", "") or "")
    if text:
        lines.extend(["", "PAGE TEXT:", text])
    snapshot_text = "\n".join(lines)
    max_chars = _DEFAULT_SNAPSHOT_MAX_CHARS
    try:
        max_chars = max(1_000, int(_patchright_config().get("snapshot_max_chars", max_chars)))
    except Exception:
        pass
    return {
        "snapshot": snapshot_text[:max_chars],
        "refs": refs,
        "url": payload.get("url", page.url),
        "title": payload.get("title", ""),
    }


def _result(data: Optional[Dict[str, Any]] = None, **extra: Any) -> Dict[str, Any]:
    payload = {"success": True}
    if data is not None:
        payload["data"] = data
    payload.update(extra)
    return payload


def _failure(error: Any) -> Dict[str, Any]:
    return {"success": False, "error": str(error)}


def run_browser_command(
    task_id: str,
    command: str,
    args: Optional[list[str]] = None,
    timeout: Optional[int] = None,
    _engine_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one Hermes browser command through Patchright."""
    del _engine_override  # agent-browser compatibility parameter
    args = list(args or [])
    command_timeout = max(int(timeout or _command_timeout()), 5) * 1000
    with _runtime_lock:
        try:
            if command == "close":
                return close_session(task_id)
            session = _get_session(task_id)
            page = _active_page(session)
            page.set_default_timeout(command_timeout)
            if command == "open":
                if not args:
                    return _failure("open requires a URL")
                response = page.goto(args[0], wait_until="domcontentloaded", timeout=max(command_timeout, _nav_timeout() * 1000))
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=5_000)
                except Exception:
                    pass
                return _result({"url": page.url, "title": page.title(), "status": getattr(response, "status", None)})
            if command == "snapshot":
                return _result(_snapshot(session, compact="-c" in args or "--compact" in args))
            if command == "click":
                if not args:
                    return _failure("click requires an element reference")
                page.locator(_ref_selector(args[0])).first.click(timeout=command_timeout)
                return _result({"ref": args[0], "url": page.url})
            if command == "fill":
                if len(args) < 2:
                    return _failure("fill requires an element reference and text")
                page.locator(_ref_selector(args[0])).first.fill(args[1], timeout=command_timeout)
                return _result({"ref": args[0]})
            if command == "scroll":
                direction = args[0] if args else "down"
                amount = int(args[1]) if len(args) > 1 else 500
                delta = -abs(amount) if direction == "up" else abs(amount)
                page.evaluate("delta => window.scrollBy(0, delta)", delta)
                return _result({"direction": direction, "amount": amount})
            if command == "back":
                page.go_back(wait_until="domcontentloaded", timeout=command_timeout)
                return _result({"url": page.url, "title": page.title()})
            if command == "press":
                if not args:
                    return _failure("press requires a key")
                page.keyboard.press(args[0])
                return _result({"key": args[0]})
            if command == "eval":
                if not args:
                    return _failure("eval requires a JavaScript expression")
                return _result({"result": page.evaluate(args[0])})
            if command == "console":
                messages = list(session["console"])
                if "--clear" in args:
                    session["console"].clear()
                return _result({"messages": messages})
            if command == "errors":
                errors = list(session["errors"])
                if "--clear" in args:
                    session["errors"].clear()
                return _result({"errors": errors})
            if command == "screenshot":
                path = None
                for value in reversed(args):
                    if not value.startswith("--"):
                        path = value
                        break
                if not path:
                    path = str(get_hermes_home() / "cache" / "screenshots" / f"patchright_{int(time.time() * 1000)}.png")
                target = Path(path).expanduser()
                target.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(target), full_page=("--full" in args), timeout=command_timeout)
                return _result({"path": str(target), "url": page.url})
            if command == "record":
                return _failure("Patchright recording is not enabled by the native backend")
            return _failure(f"Unsupported Patchright browser command: {command}")
        except Exception as exc:
            logger.debug("Patchright browser command failed (%s): %s", command, exc, exc_info=True)
            return _failure(exc)


def get_images(task_id: Optional[str] = None) -> Dict[str, Any]:
    result = run_browser_command(
        task_id or "default",
        "eval",
        [
            "JSON.stringify([...document.images].map(img => ({src: img.src, alt: img.alt || '', width: img.naturalWidth, height: img.naturalHeight})).filter(img => img.src && !img.src.startsWith('data:')))"
        ],
    )
    if not result.get("success"):
        return result
    raw = result.get("data", {}).get("result", "[]")
    try:
        images = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        images = []
    return _result({"images": images, "count": len(images)})


@atexit.register
def _shutdown() -> None:
    close_all()
