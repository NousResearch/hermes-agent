"""Minimal CloakBrowser runtime wrapper.

This slice keeps the runtime narrowly scoped to the already-wired browser tool
surface: navigate, snapshot, click, type, scroll, back, and press.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import importlib.util
import inspect
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from hermes_cli import config as hermes_config
from hermes_constants import get_hermes_home, get_hermes_dir
from tools.tool_backend_helpers import normalize_browser_cloud_provider
from tools.registry import tool_error

# Timeouts in seconds for CloakBrowser async operations.
_CLOAKBROWSER_DEFAULT_TIMEOUT: float = 60.0
_CLOAKBROWSER_NAV_TIMEOUT: float = 90.0
_CLOAKBROWSER_LAUNCH_TIMEOUT: float = 30.0


def _config_cdp_url() -> str:
    """Return persistent ``browser.cdp_url`` from config, if any."""
    try:
        browser_cfg = hermes_config.read_raw_config().get("browser", {})
    except Exception:
        return ""
    if not isinstance(browser_cfg, dict):
        return ""
    return str(browser_cfg.get("cdp_url", "") or "").strip()


def _get_browser_config() -> dict[str, Any]:
    try:
        browser_cfg = hermes_config.load_config().get("browser", {})
    except Exception:
        return {}
    return browser_cfg if isinstance(browser_cfg, dict) else {}


def _get_cloakbrowser_config() -> dict[str, Any]:
    browser_cfg = _get_browser_config()
    cloak_cfg = browser_cfg.get("cloakbrowser", {})
    return cloak_cfg if isinstance(cloak_cfg, dict) else {}


def get_cloakbrowser_profile_dir() -> Path:
    """Return the profile-scoped default user-data directory."""
    return get_hermes_home() / "cloakbrowser_profile"


def is_cloakbrowser_mode() -> bool:
    """True when config selects CloakBrowser and no CDP override is active."""
    if os.environ.get("BROWSER_CDP_URL", "").strip():
        return False
    if _config_cdp_url():
        return False

    try:
        browser_cfg = hermes_config.read_raw_config().get("browser", {})
    except Exception:
        return False
    if not isinstance(browser_cfg, dict):
        return False

    return (
        normalize_browser_cloud_provider(browser_cfg.get("cloud_provider"))
        == "cloakbrowser"
        and not bool(browser_cfg.get("use_gateway"))
    )


def check_cloakbrowser_available() -> bool:
    """Return whether the Python package is importable.

    Deliberately avoids launching or probing the network in this minimal slice.
    """
    return importlib.util.find_spec("cloakbrowser") is not None


def build_cloakbrowser_launch_options() -> dict[str, Any]:
    """Normalize config into a minimal launch-options dict."""
    cloak_cfg = _get_cloakbrowser_config()
    user_data_dir = str(cloak_cfg.get("user_data_dir", "") or "").strip()
    if not user_data_dir:
        user_data_dir = str(get_cloakbrowser_profile_dir())

    options: dict[str, Any] = {
        "headless": bool(cloak_cfg.get("headless", False)),
        "humanize": bool(cloak_cfg.get("humanize", True)),
        "stealth_args": bool(cloak_cfg.get("stealth_args", True)),
        "user_data_dir": user_data_dir,
    }

    for key in (
        "proxy",
        "geoip",
        "locale",
        "timezone",
        "viewport_width",
        "viewport_height",
        "color_scheme",
        "user_agent",
    ):
        value = cloak_cfg.get(key)
        if value not in (None, ""):
            options[key] = value

    extra_args = cloak_cfg.get("extra_args")
    if isinstance(extra_args, list) and extra_args:
        options["extra_args"] = extra_args

    return options


_sessions: dict[str, dict[str, Any]] = {}
_sessions_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None


def _start_loop_thread(name: str) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        try:
            loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    thread = threading.Thread(target=_runner, name=name, daemon=True)
    thread.start()
    ready.wait()
    return loop, thread


def _task_key(task_id: str | None) -> str:
    return task_id or "default"


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        return _loop

    with _sessions_lock:
        if _loop is not None and _loop.is_running():
            return _loop

        loop, thread = _start_loop_thread("cloakbrowser-loop")
        _loop = loop
        _loop_thread = thread
        return loop


def _run_async(coro: Any, timeout: float | None = None) -> Any:
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    effective = timeout if timeout is not None else _CLOAKBROWSER_DEFAULT_TIMEOUT
    try:
        return future.result(timeout=effective)
    except concurrent.futures.TimeoutError:
        future.cancel()
        try:
            future.result(timeout=1)
        except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError):
            pass
        raise RuntimeError(
            f"CloakBrowser operation timed out after {effective}s"
        )


def _run_coro_in_thread(coro_factory: Any, timeout: float) -> Any:
    """Run a coroutine to completion in a worker thread with a hard timeout.

    Some CloakBrowser launch paths call synchronous helpers inside their async
    wrappers. If those helpers block, they can stall the shared event-loop
    thread before asyncio timeouts get a chance to fire. Running the whole
    launch coroutine in a dedicated worker thread preserves a hard timeout at
    the caller boundary.
    """

    def _runner() -> Any:
        return asyncio.run(coro_factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_runner)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(
                f"CloakBrowser operation timed out after {timeout}s"
            )


def _run_on_session_loop(
    session: dict[str, Any],
    coro: Any,
    timeout: float | None = None,
) -> Any:
    loop = session.get("_loop")
    if loop is None:
        return _run_async(coro, timeout=timeout)

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    effective = timeout if timeout is not None else _CLOAKBROWSER_DEFAULT_TIMEOUT
    try:
        return future.result(timeout=effective)
    except concurrent.futures.TimeoutError:
        future.cancel()
        try:
            future.result(timeout=1)
        except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError):
            pass
        _teardown_session(session)
        raise RuntimeError(
            f"CloakBrowser operation timed out after {effective}s"
        )


def _launch_session_on_dedicated_loop(timeout: float) -> dict[str, Any]:
    loop, thread = _start_loop_thread("cloakbrowser-session-loop")
    future = asyncio.run_coroutine_threadsafe(_launch_session(), loop)
    try:
        session = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        _shutdown_loop_thread(loop, thread)
        raise RuntimeError(
            f"CloakBrowser operation timed out after {timeout}s"
        )
    except Exception:
        _shutdown_loop_thread(loop, thread)
        raise

    if isinstance(session, dict):
        session["_loop"] = loop
        session["_thread"] = thread
    return session


def _shutdown_loop_thread(
    loop: asyncio.AbstractEventLoop | None,
    thread: threading.Thread | None,
    join_timeout: float = 2.0,
) -> None:
    if loop is not None and not loop.is_closed() and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=join_timeout)


async def _close_session_resources(session: dict[str, Any]) -> None:
    for name in ("page", "context", "browser"):
        resource = session.get(name)
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            continue


def _teardown_session(session: dict[str, Any], task_id: str | None = None) -> None:
    with _sessions_lock:
        if task_id is not None:
            key = _task_key(task_id)
            if _sessions.get(key) is session:
                _sessions.pop(key, None)
        else:
            for key, candidate in list(_sessions.items()):
                if candidate is session:
                    _sessions.pop(key, None)
                    break

    loop = session.get("_loop")
    if loop is not None and not loop.is_closed() and loop.is_running():
        try:
            future = asyncio.run_coroutine_threadsafe(_close_session_resources(session), loop)
            future.result(timeout=_CLOAKBROWSER_DEFAULT_TIMEOUT)
        except Exception:
            pass

    _shutdown_loop_thread(loop, session.get("_thread"))


def _import_cloakbrowser_api() -> tuple[Any, Any]:
    module = importlib.import_module("cloakbrowser")
    launch_async = getattr(module, "launch_async", None)
    launch_persistent_context_async = getattr(module, "launch_persistent_context_async", None)
    if launch_async is None and launch_persistent_context_async is None:
        raise RuntimeError("cloakbrowser package is missing launch_async/launch_persistent_context_async")
    return launch_async, launch_persistent_context_async


def _normalized_viewport(options: dict[str, Any]) -> dict[str, int] | None:
    width = options.get("viewport_width")
    height = options.get("viewport_height")
    if width in (None, "") or height in (None, ""):
        return None
    try:
        return {"width": int(width), "height": int(height)}
    except Exception:
        return None


async def _launch_session() -> dict[str, Any]:
    launch_async, launch_persistent_context_async = _import_cloakbrowser_api()
    options = build_cloakbrowser_launch_options()
    user_data_dir = str(options.get("user_data_dir") or "")
    viewport = _normalized_viewport(options)
    launch_kwargs = dict(options)
    launch_kwargs.pop("viewport_width", None)
    launch_kwargs.pop("viewport_height", None)
    if viewport is not None:
        launch_kwargs["viewport"] = viewport

    if user_data_dir:
        Path(user_data_dir).mkdir(parents=True, exist_ok=True)

    if launch_persistent_context_async is not None and user_data_dir:
        launch_kwargs.pop("user_data_dir", None)
        context = await launch_persistent_context_async(user_data_dir=user_data_dir, **launch_kwargs)
        page = None
        pages = getattr(context, "pages", None)
        if isinstance(pages, list) and pages:
            page = pages[0]
        if page is None:
            page = await context.new_page()
        return {"browser": None, "context": context, "page": page, "refs": {}, "persistent": True}

    if launch_async is None:
        raise RuntimeError("cloakbrowser launch_async unavailable")
    browser = await launch_async(**launch_kwargs)
    context = await browser.new_context(viewport=viewport) if viewport is not None else await browser.new_context()
    page = await context.new_page()
    return {"browser": browser, "context": context, "page": page, "refs": {}, "persistent": False}


async def _ensure_live_page(session: dict[str, Any]) -> Any:
    page = session.get("page")
    if page is None:
        raise RuntimeError("No browser page available")
    is_closed = getattr(page, "is_closed", None)
    if callable(is_closed) and is_closed():
        context = session.get("context")
        if context is None:
            raise RuntimeError("Browser context disappeared; navigate again to recreate it")
        page = await context.new_page()
        session["page"] = page
        session["refs"] = {}
    return page


async def _ensure_session_ready(session: dict[str, Any]) -> Any:
    page = await _ensure_live_page(session)
    _register_console_listeners(session)
    return page


def _ensure_session(task_id: str | None = None) -> dict[str, Any]:
    key = _task_key(task_id)
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _launch_session_on_dedicated_loop(timeout=_CLOAKBROWSER_LAUNCH_TIMEOUT)
            _sessions[key] = session
    try:
        _run_on_session_loop(session, _ensure_session_ready(session), timeout=_CLOAKBROWSER_DEFAULT_TIMEOUT)
        return session
    except Exception:
        _teardown_session(session, task_id=task_id)
        raise


def _console_buffer(session: dict[str, Any]) -> list[dict[str, Any]]:
    messages = session.get("console_messages")
    if not isinstance(messages, list):
        messages = []
        session["console_messages"] = messages
    return messages


def _page_error_buffer(session: dict[str, Any]) -> list[dict[str, Any]]:
    errors = session.get("page_errors")
    if not isinstance(errors, list):
        errors = []
        session["page_errors"] = errors
    return errors


def _listener_text(value: Any) -> str:
    text_attr = getattr(value, "text", None)
    if callable(text_attr):
        try:
            text_attr = text_attr()
        except Exception:
            text_attr = None
    if text_attr not in (None, ""):
        return str(text_attr)
    return str(value or "")


def _listener_type(value: Any) -> str:
    type_attr = getattr(value, "type", None)
    if callable(type_attr):
        try:
            type_attr = type_attr()
        except Exception:
            type_attr = None
    return str(type_attr or "log")


def _register_console_listeners(session: dict[str, Any]) -> None:
    page = session.get("page")
    _console_buffer(session)
    _page_error_buffer(session)
    if session.get("_console_listeners_page") is page:
        return
    if page is None or not callable(getattr(page, "on", None)):
        session["_console_listeners_page"] = page
        return

    def _on_console(message: Any) -> None:
        _console_buffer(session).append(
            {
                "type": _listener_type(message),
                "text": _listener_text(message),
                "source": "console",
            }
        )

    def _on_pageerror(error: Any) -> None:
        _page_error_buffer(session).append(
            {
                "message": _listener_text(error),
                "source": "exception",
            }
        )

    page.on("console", _on_console)
    page.on("pageerror", _on_pageerror)
    session["_console_listeners_page"] = page


async def _navigate_page(page: Any, url: str, timeout: float | None = None) -> None:
    goto_kwargs = {}
    if timeout is not None:
        goto_kwargs["timeout"] = int(timeout * 1000)
    await page.goto(url, **goto_kwargs)


async def _click_selector(page: Any, selector: str) -> None:
    await page.click(selector)


async def _type_into_selector(page: Any, selector: str, text: str) -> None:
    await page.fill(selector, text)


async def _snapshot_page(page: Any, full: bool = False, user_task: str | None = None) -> dict[str, Any]:
    del user_task
    payload = await page.evaluate(
        r"""
        ({ full }) => {
          const isVisible = (el) => {
            if (!el || !el.getBoundingClientRect) return false;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') return false;
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          };

          const quote = (value) => JSON.stringify((value || '').replace(/\s+/g, ' ').trim());
          const refs = {};
          const lines = [];
          let counter = 1;

          const interactiveSelector = [
            'a[href]', 'button', 'input', 'textarea', 'select',
            '[role="button"]', '[role="link"]', '[role="textbox"]',
            '[contenteditable="true"]'
          ].join(',');

          const elements = Array.from(document.querySelectorAll(interactiveSelector));
          for (const el of elements) {
            if (!isVisible(el)) continue;
            const tag = (el.tagName || '').toLowerCase();
            const text = el.innerText || el.textContent || el.getAttribute('aria-label') || el.value || '';
            const ref = `e${counter++}`;
            el.setAttribute('data-hermes-ref', ref);
            refs[ref] = {
              selector: `[data-hermes-ref="${ref}"]`,
              tag,
              text: (text || '').trim(),
            };
            lines.push(`- ${tag || 'element'} ${quote(text)} [@${ref}]`);
          }

          if (full) {
            const bodyText = (document.body?.innerText || '').replace(/\s+/g, ' ').trim();
            if (bodyText) lines.unshift(bodyText);
          }

          return { snapshot: lines.join('\n'), refs, element_count: Object.keys(refs).length };
        }
        """,
        {"full": full},
    )
    return payload if isinstance(payload, dict) else {"snapshot": "", "refs": {}, "element_count": 0}


def _resolve_ref(session: dict[str, Any], ref: str) -> tuple[str, str]:
    clean_ref = ref.lstrip("@")
    info = session.get("refs", {}).get(clean_ref)
    if not info or not info.get("selector"):
        raise RuntimeError(f"Unknown element ref '@{clean_ref}'. Call browser_snapshot first.")
    return clean_ref, str(info["selector"])


async def _capture_screenshot(page: Any, *, full: bool = True) -> bytes:
    screenshot = getattr(page, "screenshot", None)
    if not callable(screenshot):
        raise RuntimeError("Browser page does not support screenshots")
    payload = await screenshot(full_page=full)
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, str):
        candidate = Path(payload)
        if candidate.exists():
            return candidate.read_bytes()
    raise RuntimeError("CloakBrowser screenshot capture returned no image bytes")


def _annotation_payload_from_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    refs = snapshot.get("refs", {}) or {}
    annotations: list[dict[str, Any]] = []
    for ref, info in refs.items():
        clean_ref = str(ref).lstrip("@")
        tag = str(info.get("tag") or "element").strip() or "element"
        text = str(info.get("text") or "").strip()
        label = f"{tag} {json.dumps(text)}" if text else tag
        annotations.append({
            "ref": f"@{clean_ref}",
            "label": label,
            "tag": tag,
            "text": text,
            "selector": str(info.get("selector") or ""),
        })
    return annotations


def cloakbrowser_screenshot(
    task_id: str | None = None,
    *,
    annotate: bool = False,
    full: bool = True,
) -> str:
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        annotation_data = None
        if annotate:
            snapshot = _run_on_session_loop(session, _snapshot_page(page, full=False))
            session["refs"] = snapshot.get("refs", {}) or {}
            annotation_data = _annotation_payload_from_snapshot(snapshot)
        screenshot_bytes = _run_on_session_loop(session, _capture_screenshot(page, full=full))
        screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = screenshots_dir / f"browser_screenshot_{uuid.uuid4().hex}.png"
        screenshot_path.write_bytes(screenshot_bytes)
        data: dict[str, Any] = {"path": str(screenshot_path)}
        if annotation_data is not None:
            data["annotations"] = annotation_data
        return json.dumps({"success": True, "data": data}, ensure_ascii=False)
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_navigate(url: str, task_id: str | None = None) -> str:
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session), timeout=_CLOAKBROWSER_DEFAULT_TIMEOUT)
        _run_on_session_loop(
            session,
            _navigate_page(page, url, timeout=_CLOAKBROWSER_NAV_TIMEOUT),
            timeout=_CLOAKBROWSER_NAV_TIMEOUT,
        )
        title = _run_on_session_loop(session, page.title())
        response = {"success": True, "url": getattr(page, "url", url), "title": title}
        try:
            snap = _run_on_session_loop(session, _snapshot_page(page, full=False))
            session["refs"] = snap.get("refs", {}) or {}
            response["snapshot"] = snap.get("snapshot", "")
            response["element_count"] = int(snap.get("element_count", len(session["refs"])))
        except Exception as exc:
            response["snapshot_warning"] = str(exc)
        return json.dumps(response, ensure_ascii=False)
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_snapshot(full: bool = False, task_id: str | None = None, user_task: str | None = None) -> str:
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        snap = _run_on_session_loop(session, _snapshot_page(page, full=full, user_task=user_task))
        session["refs"] = snap.get("refs", {}) or {}
        return json.dumps(
            {
                "success": True,
                "snapshot": snap.get("snapshot", ""),
                "element_count": int(snap.get("element_count", len(session["refs"]))),
            },
            ensure_ascii=False,
        )
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_click(ref: str, task_id: str | None = None) -> str:
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        clean_ref, selector = _resolve_ref(session, ref)
        _run_on_session_loop(session, _click_selector(page, selector))
        return json.dumps({"success": True, "clicked": f"@{clean_ref}", "url": getattr(page, "url", "")}, ensure_ascii=False)
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_type(ref: str, text: str, task_id: str | None = None) -> str:
    try:
        from agent.display import redact_browser_typed_text_for_display, redact_tool_args_for_display

        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        clean_ref, selector = _resolve_ref(session, ref)
        _run_on_session_loop(session, _type_into_selector(page, selector, text))
        display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {"text": text})["text"]
        response = {"success": True, "typed": display_text, "element": f"@{clean_ref}"}
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response, ensure_ascii=False)
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_scroll(direction: str, task_id: str | None = None) -> str:
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        delta_y = 500 if direction == "down" else -500
        _run_on_session_loop(session, page.mouse.wheel(0, delta_y))
        return json.dumps({"success": True, "scrolled": direction}, ensure_ascii=False)
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_back(task_id: str | None = None) -> str:
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        _run_on_session_loop(session, page.go_back())
        return json.dumps({"success": True, "url": getattr(page, "url", "")}, ensure_ascii=False)
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_press(key: str, task_id: str | None = None) -> str:
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        _run_on_session_loop(session, page.keyboard.press(key))
        return json.dumps({"success": True, "pressed": key}, ensure_ascii=False)
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_current_url(task_id: str | None = None) -> str | None:
    """Return the current page URL for the task's live CloakBrowser session."""
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        return getattr(page, "url", None)
    except Exception:
        return None


def cloakbrowser_get_images(task_id: str | None = None) -> str:
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        images = _run_on_session_loop(
            session,
            page.evaluate(
                r"""
                () => [...document.images]
                  .map((img) => ({
                    src: img.src,
                    alt: img.alt || '',
                    width: img.naturalWidth,
                    height: img.naturalHeight,
                  }))
                  .filter((img) => img.src && !img.src.startsWith('data:'))
                """
            ),
        )
        if not isinstance(images, list):
            images = []
        return json.dumps(
            {"success": True, "images": images, "count": len(images)},
            ensure_ascii=False,
        )
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_eval(expression: str, task_id: str | None = None) -> str:
    try:
        session = _ensure_session(task_id)
        page = _run_on_session_loop(session, _ensure_live_page(session))
        raw_result = _run_on_session_loop(session, page.evaluate(expression))
        parsed = raw_result
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
            except (json.JSONDecodeError, ValueError):
                pass
        return json.dumps(
            {
                "success": True,
                "result": parsed,
                "result_type": type(parsed).__name__,
                "method": "cloakbrowser_native",
            },
            ensure_ascii=False,
            default=str,
        )
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_console(clear: bool = False, task_id: str | None = None) -> str:
    try:
        session = _ensure_session(task_id)
        messages = list(_console_buffer(session))
        errors = list(_page_error_buffer(session))
        if clear:
            session["console_messages"] = []
            session["page_errors"] = []
        return json.dumps(
            {
                "success": True,
                "console_messages": messages,
                "js_errors": errors,
                "total_messages": len(messages),
                "total_errors": len(errors),
            },
            ensure_ascii=False,
        )
    except ModuleNotFoundError:
        return tool_error("CloakBrowser Python package is not installed", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def cloakbrowser_close(task_id: str | None = None) -> bool:
    key = _task_key(task_id)
    with _sessions_lock:
        session = _sessions.pop(key, None)
    if session is None:
        return False
    _teardown_session(session)
    return True


def cloakbrowser_close_all() -> int:
    """Close all live CloakBrowser sessions and return the count closed."""
    with _sessions_lock:
        items = list(_sessions.items())
        _sessions.clear()
    for _, session in items:
        _teardown_session(session)
    return len(items)
