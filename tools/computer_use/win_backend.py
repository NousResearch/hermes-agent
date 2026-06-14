"""Windows computer-use backend via pyautogui + UI Automation (UIA).

Self-contained implementation of the ``ComputerUseBackend`` ABC using
widely-available open-source libraries:

  * **pyautogui**     — screenshot, mouse-move/click/drag/scroll, keyboard
  * **Pillow**        — PNG encoding, SOM overlay drawing (idx boxes + labels)
  * **uiautomation**  — enumerate interactable elements (UIA tree), app list,
                        window focus

All three are cross-platform / pure Python with pre-built wheels on PyPI
for every supported Windows version (8.1+). No external binary drivers.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import sys
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — availability checks
# ---------------------------------------------------------------------------

_WIN_DEP_MESSAGE = (
    "Windows computer-use dependencies are missing. "
    "Install with:\n"
    '  uv pip install pyautogui uiautomation\n'
    "Or enable the Computer Use toolset via `hermes tools`."
)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _check_pyautogui() -> bool:
    try:
        import pyautogui  # noqa: F401
        return True
    except ImportError:
        return False


def _check_uiautomation() -> bool:
    try:
        import uiautomation  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Coordinate helpers — DPI-aware scaling
# ---------------------------------------------------------------------------

def _get_scale_factor() -> float:
    """Return the DPI scaling factor for the primary monitor.

    On high-DPI displays (150 %, 200 %, …) pyautogui operates in physical
    pixels while UIA reports logical coordinates.  We multiply UIA coordinates
    by this scale to convert to physical.
    """
    try:
        # ctypes approach: GetDpiForWindow on the desktop window
        import ctypes
        user32 = ctypes.windll.user32
        # Windows 10 1607+: GetDpiForWindow(GetDesktopWindow())
        hwnd = user32.GetDesktopWindow()
        dpi = ctypes.c_uint()
        # prefer GetDpiForWindow (Win 10 1607+)
        try:
            _GetDpiForWindow = user32.GetDpiForWindow
            _GetDpiForWindow.argtypes = [ctypes.c_void_p]
            _GetDpiForWindow.restype = ctypes.c_uint
            dpi_val = _GetDpiForWindow(hwnd)
        except AttributeError:
            # Fallback to GetDeviceCaps + LOGPIXELSX for older Windows
            hdc = user32.GetDC(0)
            import ctypes.wintypes
            LOGPIXELSX = 88
            dpi_val = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
            user32.ReleaseDC(0, hdc)
        if dpi_val and dpi_val > 0:
            return dpi_val / 96.0
    except Exception:
        pass
    return 1.0


# ---------------------------------------------------------------------------
# Element provider — UI Automation tree walk
# ---------------------------------------------------------------------------

_UIA_ELEMENT_ROLES: Dict[int, str] = {
    50000: "Button",
    50004: "CheckBox",
    50005: "ComboBox",
    50006: "DataGrid",
    50007: "DataItem",
    50008: "Document",
    50009: "Edit",
    50019: "Hyperlink",
    50020: "Image",
    50025: "ListItem",
    50026: "List",
    50027: "Menu",
    50028: "MenuBar",
    50029: "MenuItem",
    50030: "ProgressBar",
    50032: "RadioButton",
    50033: "ScrollBar",
    50036: "Slider",
    50038: "SplitButton",
    50040: "Tab",
    50041: "TabItem",
    50042: "Text",
    50048: "Tree",
    50049: "TreeItem",
    50050: "Window",
    50003: "Calendar",
    50010: "Group",
    50012: "Header",
    50013: "HeaderItem",
    50018: "HotKeyField",
    50023: "MenuButton",
    50031: "RangeSpinner",
    50034: "SemanticZoom",
    50039: "Spinner",
    50043: "Thumb",
    50044: "TitleBar",
    50045: "ToggleButton",
    50046: "ToggleSwitch",
    50047: "ToolBar",
    50051: "Custom",
}


class _UIAProvider:
    """Lightweight UI Automation element scanner.

    Wraps ``uiautomation`` to enumerate interactable controls, build
    SOM element lists, and provide app-list / focus helpers.  All methods
    are synchronous.
    """

    def __init__(self) -> None:
        self._uia: Any = None

    def _ensure_uia(self) -> Any:
        if self._uia is not None:
            return self._uia
        import uiautomation as uia
        self._uia = uia
        return uia

    # -- element scanning -------------------------------------------------

    def enumerate_elements(
        self,
        top_window: Optional[Any] = None,
    ) -> List[Tuple[Dict[str, Any], Any]]:
        """Walk the UIA tree and return ``(info_dict, uia_control)`` pairs.

        Only controls that are *interactable* (enabled, not off-screen,
        with a non-zero bounding rect) are included.

        When *top_window* is ``None`` the **foreground (active) top-level
        window** is used automatically so that element lists correspond to
        the app the user is currently working with — not the entire desktop.
        """
        uia = self._ensure_uia()
        if top_window is None:
            try:
                # Start from the focused control and walk up to its
                # top-level window so we only enumerate the active app.
                focused = uia.GetFocusedControl()
                if focused is not None:
                    current = focused
                    while current is not None:
                        parent = current.GetParentControl()
                        # Reached the desktop root → current is the top-level
                        if parent is None:
                            break
                        try:
                            if parent is uia.GetRootControl():
                                break
                        except Exception:
                            break
                        current = parent
                    top_window = current
            except Exception:
                pass
        if top_window is None:
            return []

        results: List[Tuple[Dict[str, Any], Any]] = []
        self._walk(uia, top_window, results)
        return results

    def _walk(
        self,
        uia: Any,
        control: Any,
        results: List[Tuple[Dict[str, Any], Any]],
        depth: int = 0,
    ) -> None:
        if depth > 20:
            return
        try:
            # Skip invisible / off-screen / zero-size controls
            if not control.BoundingRectangle:
                return
            rect = control.BoundingRectangle
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w <= 0 or h <= 0:
                return
            try:
                if not control.IsEnabled:
                    return
            except Exception:
                pass
            try:
                if control.IsOffscreen:
                    return
            except Exception:
                pass

            role_id = control.ControlType
            role = _UIA_ELEMENT_ROLES.get(role_id, f"UIA_{role_id}")
            try:
                name = control.Name or ""
            except Exception:
                name = ""
            try:
                class_name = control.ClassName or ""
            except Exception:
                class_name = ""
            label = name or class_name or role

            pid = 0
            try:
                pid = control.ProcessId or 0
            except Exception:
                pass

            info: Dict[str, Any] = {
                "role": role,
                "label": label,
                "bounds": (rect.left, rect.top, w, h),
                "pid": pid,
                "control_type": role_id,
                "class_name": class_name,
            }
            results.append((info, control))
        except Exception:
            return

        # Recurse children — capped at 20 levels deep
        try:
            children = control.GetChildren()
        except Exception:
            return
        for child in children:
            try:
                self._walk(uia, child, results, depth + 1)
            except Exception:
                pass

    # -- app listing ------------------------------------------------------

    def list_running_apps(self) -> List[Dict[str, Any]]:
        """Return foreground visible windows as a simplified app list."""
        uia = self._ensure_uia()
        apps: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}
        try:
            root = uia.GetRootControl()
            for child in root.GetChildren():
                try:
                    if not child.IsTopmost and child.IsOffscreen:
                        continue
                except Exception:
                    continue
                try:
                    name = child.Name or ""
                except Exception:
                    name = ""
                try:
                    pid = child.ProcessId or 0
                except Exception:
                    pid = 0
                if name:
                    seen[name] = seen.get(name, 0) + 1
        except Exception:
            pass
        for app_name, win_count in seen.items():
            apps.append({"name": app_name, "window_count": win_count})
        apps.sort(key=lambda a: a["name"].lower())
        return apps

    def focus_window(self, app_name: str, raise_window: bool = True) -> bool:
        """Find a window by partial name match and bring it to foreground."""
        uia = self._ensure_uia()
        try:
            root = uia.GetRootControl()
            for child in root.GetChildren():
                try:
                    if not child.Name:
                        continue
                except Exception:
                    continue
                if app_name.lower() in child.Name.lower():
                    if raise_window:
                        try:
                            child.SetFocus()
                        except Exception:
                            pass
                        try:
                            child.SetTopmost(True)
                            _time.sleep(0.1)
                            child.SetTopmost(False)
                        except Exception:
                            pass
                    return True
        except Exception:
            pass
        return False

    def get_focused_element_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """Return bounds of the currently focused element, if any."""
        uia = self._ensure_uia()
        try:
            focused = uia.GetFocusedControl()
            if focused and focused.BoundingRectangle:
                r = focused.BoundingRectangle
                return r.left, r.top, r.right - r.left, r.bottom - r.top
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# PyAutoGUI mouse / keyboard provider
# ---------------------------------------------------------------------------

class _PyAutoGUIProvider:
    """Wraps pyautogui for screenshot, pointer, and keyboard operations.

    Public methods mirror what ``ComputerUseBackend`` needs; all coordinates
    are physical pixels (pyautogui's native coordinate space).
    """

    def __init__(self, scale: float = 1.0) -> None:
        self._scale = scale
        self._pg: Any = None

    def _ensure_pg(self) -> Any:
        if self._pg is not None:
            return self._pg
        import pyautogui as pg
        pg.FAILSAFE = False
        pg.PAUSE = 0.0
        self._pg = pg
        return pg

    # -- screenshot -------------------------------------------------------

    def screenshot(self) -> bytes:
        """Capture full virtual screen as raw PNG bytes."""
        pg = self._ensure_pg()
        img = pg.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @property
    def screen_size(self) -> Tuple[int, int]:
        pg = self._ensure_pg()
        return pg.size()

    # -- mouse ------------------------------------------------------------

    def move_to(self, x: int, y: int) -> None:
        pg = self._ensure_pg()
        pg.moveTo(x, y)

    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        pg = self._ensure_pg()
        pg.click(x, y, button=button, clicks=clicks)

    def drag(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        button: str = "left",
    ) -> None:
        pg = self._ensure_pg()
        pg.moveTo(from_x, from_y)
        pg.drag(to_x - from_x, to_y - from_y, button=button, duration=0.3)

    def scroll(self, clicks: int, x: int, y: int) -> None:
        pg = self._ensure_pg()
        pg.moveTo(x, y)
        pg.scroll(clicks, x=x, y=y)

    def hscroll(self, clicks: int, x: int, y: int) -> None:
        pg = self._ensure_pg()
        pg.moveTo(x, y)
        pg.hscroll(clicks, x=x, y=y)

    # -- keyboard ---------------------------------------------------------

    def typewrite(self, text: str, interval: float = 0.0) -> None:
        pg = self._ensure_pg()
        pg.typewrite(text, interval=interval)

    def hotkey(self, *keys: str) -> None:
        pg = self._ensure_pg()
        pg.hotkey(*keys)

    def keyDown(self, key: str) -> None:
        pg = self._ensure_pg()
        pg.keyDown(key)

    def keyUp(self, key: str) -> None:
        pg = self._ensure_pg()
        pg.keyUp(key)

    # -- coord conversion -------------------------------------------------

    def logical_to_physical(
        self, x: int, y: int, w: int, h: int
    ) -> Tuple[int, int, int, int]:
        """Convert UIA logical coordinates → pyautogui physical."""
        s = self._scale
        if s == 1.0:
            return x, y, w, h
        return int(x * s), int(y * s), int(w * s), int(h * s)


# ---------------------------------------------------------------------------
# Windows Computer-Use Backend
# ---------------------------------------------------------------------------

class WinComputerUseBackend(ComputerUseBackend):
    """Windows desktop automation via pyautogui + UIA.

    Environment variables
    ---------------------
    ``HERMES_CU_WIN_SCALE`` — force DPI scale factor (float; default auto-detect)
    """

    def __init__(self) -> None:
        super().__init__()
        scale_str = os.environ.get("HERMES_CU_WIN_SCALE", "")
        self._scale: float = float(scale_str) if scale_str else _get_scale_factor()
        self._uia = _UIAProvider()
        self._pg = _PyAutoGUIProvider(scale=self._scale)
        self._started = False
        self._element_cache: Dict[int, Tuple[Dict[str, Any], Any]] = {}

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._element_cache.clear()
        self._started = False

    def is_available(self) -> bool:
        return _is_windows() and _check_pyautogui() and _check_uiautomation()

    # ── Capture ─────────────────────────────────────────────────────

    def capture(
        self, mode: str = "som", app: Optional[str] = None
    ) -> CaptureResult:
        png_b64: Optional[str] = None
        elements: List[UIElement] = []

        if mode in ("vision", "som"):
            raw_png = self._pg.screenshot()
            png_b64 = base64.b64encode(raw_png).decode("ascii")

        # Enumerate elements for "som" / "ax" modes
        if mode in ("som", "ax"):
            try:
                pairs = self._uia.enumerate_elements()
            except Exception as exc:
                logger.debug("UIA element enumeration failed: %s", exc)
                pairs = []

            self._element_cache.clear()
            for idx, (info, ctrl) in enumerate(pairs, start=1):
                lx, ly, lw, lh = info["bounds"]
                px, py, pw, ph = self._pg.logical_to_physical(lx, ly, lw, lh)

                elem = UIElement(
                    index=idx,
                    role=info["role"],
                    label=info["label"][:120],
                    bounds=(px, py, pw, ph),
                    pid=info.get("pid", 0),
                    attributes={
                        "control_type": info.get("control_type"),
                        "class_name": info.get("class_name", ""),
                    },
                )
                elements.append(elem)
                self._element_cache[idx] = (info, ctrl)

            # SOM: draw numbered overlay on screenshot
            if mode == "som" and png_b64:
                png_b64 = _draw_som_overlay(
                    png_b64, elements, self._pg.screen_size
                )

        w, h = self._pg.screen_size
        png_bytes_len = len(base64.b64decode(png_b64)) if png_b64 else 0

        return CaptureResult(
            mode=mode,
            width=w,
            height=h,
            png_b64=png_b64,
            elements=elements,
            app=app or "",
            window_title="",
            png_bytes_len=png_bytes_len,
        )

    # ── Pointer actions ──────────────────────────────────────────────

    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        click_count: int = 1,
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult:
        px, py = self._resolve_coords(element, x, y)
        if px is None:
            return ActionResult(ok=False, action="click",
                               message="no target coordinates")

        if modifiers:
            for m in modifiers:
                self._pg.keyDown(m)
            _time.sleep(0.02)
        self._pg.click(px, py, button=button, clicks=click_count)
        if modifiers:
            for m in reversed(modifiers):
                self._pg.keyUp(m)
        return ActionResult(ok=True, action="click",
                           message=f"clicked {button} at ({px}, {py})")

    def drag(
        self,
        *,
        from_element: Optional[int] = None,
        to_element: Optional[int] = None,
        from_xy: Optional[Tuple[int, int]] = None,
        to_xy: Optional[Tuple[int, int]] = None,
        button: str = "left",
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult:
        fx, fy = self._resolve_coords(from_element, from_xy[0] if from_xy else None, from_xy[1] if from_xy else None)
        tx, ty = self._resolve_coords(to_element, to_xy[0] if to_xy else None, to_xy[1] if to_xy else None)
        if fx is None or fy is None or tx is None or ty is None:
            return ActionResult(ok=False, action="drag",
                               message="missing drag target")
        if from_xy and fx is not None and fy is not None:
            self._pg.move_to(fx, fy)
        self._pg.drag(fx, fy, tx, ty, button=button)
        return ActionResult(ok=True, action="drag",
                           message=f"dragged from ({fx},{fy}) to ({tx},{ty})")

    def scroll(
        self,
        *,
        direction: str,
        amount: int = 3,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult:
        px, py = self._resolve_coords(element, x, y)
        if px is None:
            return ActionResult(ok=False, action="scroll",
                               message="no scroll target")

        clicks = amount if direction in ("up", "left") else -amount
        if direction in ("down", "up"):
            self._pg.scroll(clicks, px, py)
        elif direction in ("left", "right"):
            self._pg.hscroll(clicks, px, py)
        else:
            return ActionResult(ok=False, action="scroll",
                               message=f"unknown direction: {direction}")
        return ActionResult(ok=True, action="scroll",
                           message=f"scrolled {direction} {amount} ticks at ({px}, {py})")

    # ── Keyboard ────────────────────────────────────────────────────

    def type_text(self, text: str) -> ActionResult:
        self._pg.typewrite(text)
        return ActionResult(ok=True, action="type",
                           message=f"typed {len(text)} chars")

    def key(self, keys: str) -> ActionResult:
        # Normalise: "ctrl+s" / "ctrl+alt+t" / "return" / "enter"
        parts = [k.strip().lower() for k in keys.split("+")]
        translated = [_translate_key(k) for k in parts]
        self._pg.hotkey(*translated)
        return ActionResult(ok=True, action="key",
                           message=f"pressed {keys}")

    # ── Introspection ───────────────────────────────────────────────

    def list_apps(self) -> List[Dict[str, Any]]:
        return self._uia.list_running_apps()

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        ok = self._uia.focus_window(app, raise_window=raise_window)
        return ActionResult(ok=ok, action="focus_app",
                           message=f"focus {'ok' if ok else 'failed'} for '{app}'")

    # ── Native-value mutation ────────────────────────────────────────

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        if element is None or element not in self._element_cache:
            return ActionResult(ok=False, action="set_value",
                               message="element not found in cache — run capture first")
        _, ctrl = self._element_cache[element]
        try:
            ctrl.SetValue(value)
            return ActionResult(ok=True, action="set_value",
                               message=f"set value on element {element}")
        except Exception as exc:
            return ActionResult(ok=False, action="set_value",
                               message=str(exc))

    # ── Helpers ─────────────────────────────────────────────────────

    def _resolve_coords(
        self,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Resolve target coordinates from element index or explicit xy."""
        if element is not None and element in self._element_cache:
            info, _ctrl = self._element_cache[element]
            lx, ly, lw, lh = info["bounds"]
            px, py, pw, ph = self._pg.logical_to_physical(lx, ly, lw, lh)
            return px + pw // 2, py + ph // 2
        if x is not None and y is not None:
            return int(x), int(y)
        return None, None


# ---------------------------------------------------------------------------
# SOM overlay — draw numbered boxes on screenshot
# ---------------------------------------------------------------------------

_SOM_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F1948A", "#82E0AA", "#F8C471", "#AED6F1", "#D7BDE2",
    "#A3E4D7", "#FAD7A0", "#ABEBC6", "#D5F5E3", "#FADBD8",
]


def _draw_som_overlay(
    png_b64: str,
    elements: List[UIElement],
    screen_size: Tuple[int, int],
    max_elements: int = 80,
) -> str:
    """Draw numbered boxes + labels on a base64 PNG screenshot.

    Returns a new base64 PNG string with SOM overlays applied.  Elements
    beyond ``max_elements`` are silently dropped; their bounds are tiny
    and the visual noise hurts more than it helps.
    """
    from PIL import Image, ImageDraw, ImageFont

    raw = base64.b64decode(png_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(10, min(img.width, img.height) // 60)
    try:
        if sys.platform == "win32":
            font = ImageFont.truetype("arial.ttf", font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for i, elem in enumerate(elements[:max_elements]):
        idx = elem.index
        color_str = _SOM_COLORS[(idx - 1) % len(_SOM_COLORS)]
        color = _hex_to_rgba(color_str, alpha=180)

        x, y, w, h = elem.bounds
        # Clamp to screen
        sw, sh = screen_size
        x = max(0, min(x, sw - 1))
        y = max(0, min(y, sh - 1))
        w = max(1, min(w, sw - x))
        h = max(1, min(h, sh - y))

        # Filled rectangle (tinted box)
        draw.rectangle([x, y, x + w, y + h], fill=color)
        # Thin outline
        outline_color = color[:3] + (255,)
        draw.rectangle([x, y, x + w, y + h], outline=outline_color, width=1)
        # Index label — white text on black background for contrast
        label = str(idx)
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
        except AttributeError:
            # Pillow < 8.0 fallback
            tw, th = font.getsize(label)
        lx = x + 2
        ly = max(0, y - th - 2)
        draw.rectangle([lx, ly, lx + tw + 4, ly + th + 2], fill=(0, 0, 0, 200))
        draw.text((lx + 2, ly + 1), label, fill=(255, 255, 255, 255), font=font)

    merged = Image.alpha_composite(img, overlay)
    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
        alpha,
    )


# ---------------------------------------------------------------------------
# Key translation — normalize key names for pyautogui
# ---------------------------------------------------------------------------

_KEY_MAP: Dict[str, str] = {
    "cmd": "win",
    "command": "win",
    "super": "win",
    "option": "alt",
    "opt": "alt",
    "return": "enter",
    "escape": "esc",
    "pageup": "pgup",
    "pagedown": "pgdn",
    "capslock": "capslock",
    "backspace": "backspace",
    "delete": "delete",
    "tab": "tab",
    "space": "space",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "home": "home",
    "end": "end",
    "insert": "insert",
    "printscreen": "printscreen",
    "scrolllock": "scrolllock",
    "pause": "pause",
    "f1": "f1",
    "f2": "f2",
    "f3": "f3",
    "f4": "f4",
    "f5": "f5",
    "f6": "f6",
    "f7": "f7",
    "f8": "f8",
    "f9": "f9",
    "f10": "f10",
    "f11": "f11",
    "f12": "f12",
}


def _translate_key(key: str) -> str:
    return _KEY_MAP.get(key, key)
