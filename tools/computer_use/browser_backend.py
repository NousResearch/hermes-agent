"""Playwright Chromium backend for computer_use browser-use (#113850).

Same ComputerUseBackend surface as cua-driver. Headless by default.
Optional HERMES_CU_BROWSER_URL / HERMES_CU_BROWSER_HTML, or pass `url` on
capture/run_goal/navigate. Does not change approval/safety paths.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement

_QUERY = "button, input, select, textarea, a[href], [role='button'], [role='textbox'], [role='searchbox']"


def _as_goto(target: str) -> str:
    raw = (target or "").strip() or "about:blank"
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https", "file", "about"}:
        return raw
    return Path(raw).expanduser().resolve().as_uri()


class BrowserBackend(ComputerUseBackend):
    """Headless Chromium with a tab list (window_id = 1-based tab index)."""

    def __init__(self, target: str | Path = "about:blank", *, app_name: str = "browser") -> None:
        self._target = str(target) if str(target).strip() else "about:blank"
        self._app_name = app_name
        self._playwright = None
        self._browser = None
        self._pages: list[Any] = []
        self._page = None
        self._elements: list[UIElement] = []
        self._last_element: Optional[int] = None

    def start(self) -> None:
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        headless = os.environ.get("HERMES_CU_BROWSER_HEADED", "").strip() not in {"1", "true", "yes"}
        self._browser = self._playwright.chromium.launch(headless=headless)
        page = self._browser.new_page(viewport={"width": 1280, "height": 720})
        self._pages = [page]
        self._page = page
        page.goto(_as_goto(self._target), wait_until="domcontentloaded")
        self._refresh_elements()

    def stop(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        self._pages = []
        self._page = None
        self._elements = []

    def is_available(self) -> bool:
        return True

    def _select_tab(self, window_id: Optional[int]) -> None:
        if window_id is None or not self._pages:
            return
        idx = int(window_id) - 1
        if 0 <= idx < len(self._pages):
            self._page = self._pages[idx]
            self._last_element = None

    def navigate(self, url: str, *, new_tab: bool = False) -> ActionResult:
        target = _as_goto(url)
        if self._page is None:
            return ActionResult(ok=False, action="navigate", message="browser not started")
        if new_tab and self._browser is not None:
            page = self._browser.new_page(viewport={"width": 1280, "height": 720})
            self._pages.append(page)
            self._page = page
        try:
            self._page.goto(target, wait_until="domcontentloaded")
        except Exception as exc:
            return ActionResult(ok=False, action="navigate", message=str(exc))
        self._last_element = None
        self._refresh_elements()
        n = len(self._pages)
        return ActionResult(ok=True, action="navigate", message=f"opened {target} tab={n}")

    def _refresh_elements(self) -> list[dict[str, Any]]:
        if self._page is None:
            self._elements = []
            return []
        raw = self._page.evaluate(
            f"""() => {{
                const nodes = document.querySelectorAll({_QUERY!r});
                return Array.from(nodes).map((el, i) => ({{
                    index: i + 1,
                    role: (el.getAttribute('role') || el.tagName || '').toLowerCase(),
                    label: (el.getAttribute('aria-label') || el.getAttribute('placeholder')
                        || el.value || el.textContent || '').trim().slice(0, 120),
                    enabled: !el.disabled,
                }}));
            }}"""
        )
        elements: list[UIElement] = []
        for row in raw or []:
            elements.append(
                UIElement(
                    index=int(row["index"]),
                    role=str(row.get("role") or "button"),
                    label=str(row.get("label") or ""),
                    app=self._app_name,
                    pid=1,
                    window_id=self._pages.index(self._page) + 1 if self._page in self._pages else 1,
                )
            )
        self._elements = elements
        return raw or []

    def capture(
        self,
        mode: str = "som",
        app: Optional[str] = None,
        pid: Optional[int] = None,
        window_id: Optional[int] = None,
    ) -> CaptureResult:
        del app, pid
        self._select_tab(window_id)
        self._refresh_elements()
        title = ""
        if self._page is not None:
            try:
                title = self._page.title() or ""
            except Exception:
                title = ""
        return CaptureResult(
            mode=mode,
            width=1280,
            height=720,
            elements=list(self._elements),
            app=self._app_name,
            window_title=title,
        )

    def _with_node(self, element: int, js: str, extra: Any = None) -> Any:
        assert self._page is not None
        return self._page.evaluate(
            f"""([idx, extra]) => {{
                const nodes = document.querySelectorAll({_QUERY!r});
                const el = nodes[idx - 1];
                if (!el) return null;
                {js}
            }}""",
            [element, extra],
        )

    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        click_count: int = 1,
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        del x, y, button, click_count, modifiers, delivery_mode, bring_to_front
        if element is None or element < 1:
            return ActionResult(ok=False, action="click", message=f"invalid element {element!r}")
        try:
            clicked = self._with_node(element, "el.click(); return true;")
            if not clicked:
                return ActionResult(ok=False, action="click", message=f"element #{element} not found")
            if self._page is not None:
                self._page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception as exc:
            return ActionResult(ok=False, action="click", message=str(exc))
        self._last_element = element
        self._refresh_elements()
        return ActionResult(ok=True, action="click", message=f"clicked element #{element}")

    def drag(self, **kwargs: Any) -> ActionResult:
        del kwargs
        return ActionResult(ok=False, action="drag", message="drag not supported in browser backend")

    def scroll(
        self,
        *,
        direction: str,
        amount: int = 3,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        del element, x, y, modifiers, delivery_mode, bring_to_front
        if self._page is None:
            return ActionResult(ok=False, action="scroll", message="browser not started")
        delta = max(int(amount), 1) * 120
        dx, dy = 0, 0
        if direction == "down":
            dy = delta
        elif direction == "up":
            dy = -delta
        elif direction == "right":
            dx = delta
        elif direction == "left":
            dx = -delta
        else:
            return ActionResult(ok=False, action="scroll", message=f"bad direction {direction!r}")
        try:
            self._page.mouse.wheel(dx, dy)
        except Exception as exc:
            return ActionResult(ok=False, action="scroll", message=str(exc))
        self._refresh_elements()
        return ActionResult(ok=True, action="scroll", message=f"scroll {direction} x{amount}")

    def type_text(self, text: str, **kwargs: Any) -> ActionResult:
        element = kwargs.get("element") if kwargs.get("element") is not None else self._last_element
        if element is None:
            return ActionResult(ok=False, action="type", message="type requires a focused element")
        try:
            filled = self._with_node(
                int(element),
                """
                el.focus();
                if ('value' in el) { el.value = extra; el.dispatchEvent(new Event('input', {bubbles:true})); }
                else { el.textContent = extra; }
                return true;
                """,
                text,
            )
            if not filled:
                return ActionResult(ok=False, action="type", message=f"element #{element} not found")
        except Exception as exc:
            return ActionResult(ok=False, action="type", message=str(exc))
        self._last_element = int(element)
        self._refresh_elements()
        return ActionResult(ok=True, action="type", message=f"typed {len(text)} chars into #{element}")

    def key(self, keys: str, **kwargs: Any) -> ActionResult:
        del kwargs
        if self._page is None:
            return ActionResult(ok=False, action="key", message="browser not started")
        mapping = {"return": "Enter", "enter": "Enter", "esc": "Escape", "escape": "Escape", "tab": "Tab"}
        try:
            self._page.keyboard.press(mapping.get(keys.strip().lower(), keys))
        except Exception as exc:
            return ActionResult(ok=False, action="key", message=str(exc))
        self._refresh_elements()
        return ActionResult(ok=True, action="key", message=f"key {keys!r}")

    def list_apps(self) -> List[Dict[str, Any]]:
        return [{"name": self._app_name, "pid": 1, "windows": len(self._pages) or 1}]

    def list_windows(self) -> List[Dict[str, Any]]:
        out = []
        for i, page in enumerate(self._pages, start=1):
            title = ""
            try:
                title = page.title() or page.url or f"tab {i}"
            except Exception:
                title = f"tab {i}"
            out.append({"title": title, "pid": 1, "window_id": i, "app": self._app_name})
        if not out:
            out.append({"title": self._target, "pid": 1, "window_id": 1, "app": self._app_name})
        return out

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        del raise_window
        raw = (app or "").strip()
        if raw.lower().startswith("tab:"):
            try:
                self._select_tab(int(raw.split(":", 1)[1]))
            except ValueError:
                return ActionResult(ok=False, action="focus_app", message=f"bad tab ref {app!r}")
            self._refresh_elements()
            return ActionResult(ok=True, action="focus_app", message=f"focused {raw}")
        return ActionResult(ok=True, action="focus_app", message=f"focused {app or self._app_name}")

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        if element is None:
            return ActionResult(ok=False, action="set_value", message="set_value requires element")
        return self.type_text(value, element=element)
