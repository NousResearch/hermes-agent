"""Playwright Chromium backend for computer_use browser-use (#113850).

Same ComputerUseBackend surface as cua-driver. Headless by default.
Set HERMES_CU_BROWSER_URL or HERMES_CU_BROWSER_HTML. Does not change
approval/safety paths.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement

_QUERY = "button, input, select, textarea, a[href], [role='button'], [role='textbox'], [role='searchbox']"


class BrowserBackend(ComputerUseBackend):
    """Headless Chromium over a URL or local HTML file."""

    def __init__(self, target: str | Path, *, app_name: str = "browser") -> None:
        self._target = str(target)
        self._app_name = app_name
        self._playwright = None
        self._browser = None
        self._page = None
        self._elements: list[UIElement] = []
        self._last_element: Optional[int] = None

    def _goto_url(self) -> str:
        raw = self._target
        parsed = urlparse(raw)
        if parsed.scheme in {"http", "https", "file"}:
            return raw
        path = Path(raw).expanduser().resolve()
        return path.as_uri()

    def start(self) -> None:
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        headless = os.environ.get("HERMES_CU_BROWSER_HEADED", "").strip() not in {"1", "true", "yes"}
        self._browser = self._playwright.chromium.launch(headless=headless)
        self._page = self._browser.new_page(viewport={"width": 1280, "height": 720})
        self._page.goto(self._goto_url(), wait_until="domcontentloaded")
        self._refresh_elements()

    def stop(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        self._page = None
        self._elements = []

    def is_available(self) -> bool:
        return True

    def _refresh_elements(self) -> list[dict[str, Any]]:
        assert self._page is not None
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
                    window_id=1,
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
        del app, pid, window_id
        self._refresh_elements()
        title = self._page.title() if self._page is not None else ""
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
        except Exception as exc:
            return ActionResult(ok=False, action="click", message=str(exc))
        self._last_element = element
        if self._page is not None:
            self._page.wait_for_timeout(120)
        self._refresh_elements()
        return ActionResult(ok=True, action="click", message=f"clicked element #{element}")

    def drag(self, **kwargs: Any) -> ActionResult:
        del kwargs
        return ActionResult(ok=False, action="drag", message="drag not supported in browser backend")

    def scroll(self, **kwargs: Any) -> ActionResult:
        del kwargs
        return ActionResult(ok=False, action="scroll", message="scroll not supported in browser backend")

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
        return [{"name": self._app_name, "pid": 1, "windows": 1}]

    def list_windows(self) -> List[Dict[str, Any]]:
        title = self._page.title() if self._page else self._target
        return [{"title": title, "pid": 1, "window_id": 1, "app": self._app_name}]

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        del raise_window
        return ActionResult(ok=True, action="focus_app", message=f"focused {app or self._app_name}")

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        if element is None:
            return ActionResult(ok=False, action="set_value", message="set_value requires element")
        return self.type_text(value, element=element)
