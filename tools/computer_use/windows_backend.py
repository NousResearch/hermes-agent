"""Windows implementation of ComputerUseBackend — UI Automation + SendInput.

The Windows analogue of cua_backend.py. Element discovery and set_value go
through UI Automation (the Windows counterpart of the macOS AX tree, via the
`uiautomation` package); screenshots through PIL.ImageGrab; mouse/keyboard
through SendInput.

One behavioural difference from the macOS backend: Windows has no supported
way to post input to a background window, so pointer/keyboard actions bring
the target window to the foreground first. `set_value` is the exception — it
mutates element values through UIA patterns and works without focus.

All coordinates are physical pixels; start() opts the process into
per-monitor DPI awareness so UIA bounds, ImageGrab and SendInput agree.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import io
import json
import logging
import os
import socket
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)

_IMPORT_ERROR: Optional[Exception] = None
try:
    import uiautomation as _auto
    import win32api
    import win32con
    import win32gui
    import win32process
    from PIL import Image, ImageDraw, ImageGrab
except Exception as _e:  # pragma: no cover - exercised via availability check
    _IMPORT_ERROR = _e


def windows_backend_available() -> bool:
    """True iff this host can run the Windows UIA backend."""
    return sys.platform == "win32" and _IMPORT_ERROR is None


# ---------------------------------------------------------------------------
# DPI awareness
# ---------------------------------------------------------------------------

_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


def _set_dpi_awareness() -> None:
    """Opt into per-monitor-v2 DPI awareness, best-effort with fallbacks."""
    user32 = ctypes.windll.user32
    try:
        if user32.SetProcessDpiAwarenessContext(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2):
            return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return
    except Exception:
        pass
    try:
        user32.SetProcessDPIAware()
    except Exception:
        logger.warning("could not set DPI awareness; coordinates may be scaled")


# ---------------------------------------------------------------------------
# SendInput layer
# ---------------------------------------------------------------------------

_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_ABSOLUTE = 0x8000
_MOUSEEVENTF_VIRTUALDESK = 0x4000
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_HWHEEL = 0x1000
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_KEYEVENTF_EXTENDEDKEY = 0x0001
_WHEEL_DELTA = 120

_BUTTON_FLAGS = {
    "left": (0x0002, 0x0004),     # MOUSEEVENTF_LEFTDOWN / LEFTUP
    "right": (0x0008, 0x0010),    # RIGHTDOWN / RIGHTUP
    "middle": (0x0020, 0x0040),   # MIDDLEDOWN / MIDDLEUP
}

_ULONG_PTR = ctypes.c_size_t


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.wintypes.DWORD), ("u", _INPUTUNION)]


def _send_inputs(inputs: List[_INPUT]) -> int:
    if not inputs:
        return 0
    arr = (_INPUT * len(inputs))(*inputs)
    sent = ctypes.windll.user32.SendInput(len(inputs), arr, ctypes.sizeof(_INPUT))
    if sent != len(inputs):
        raise OSError(f"SendInput injected {sent}/{len(inputs)} events "
                      f"(error {ctypes.get_last_error()})")
    return sent


def _mouse_input(dx: int = 0, dy: int = 0, data: int = 0, flags: int = 0) -> _INPUT:
    inp = _INPUT(type=_INPUT_MOUSE)
    inp.mi = _MOUSEINPUT(dx=dx, dy=dy, mouseData=data & 0xFFFFFFFF, dwFlags=flags,
                         time=0, dwExtraInfo=0)
    return inp


def _key_input(vk: int = 0, scan: int = 0, flags: int = 0) -> _INPUT:
    inp = _INPUT(type=_INPUT_KEYBOARD)
    inp.ki = _KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
    return inp


def _abs_coords(x: int, y: int) -> Tuple[int, int]:
    """Normalize physical screen coords to 0..65535 across the virtual desktop."""
    user32 = ctypes.windll.user32
    vx = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
    vy = user32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
    vw = user32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
    vh = user32.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN
    nx = round((x - vx) * 65535 / max(1, vw - 1))
    ny = round((y - vy) * 65535 / max(1, vh - 1))
    return max(0, min(65535, nx)), max(0, min(65535, ny))


def _mouse_move(x: int, y: int) -> None:
    nx, ny = _abs_coords(x, y)
    _send_inputs([_mouse_input(
        dx=nx, dy=ny,
        flags=_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK)])


def _mouse_button(button: str, down: bool) -> None:
    flags = _BUTTON_FLAGS.get(button)
    if flags is None:
        raise ValueError(f"unknown button {button!r}")
    _send_inputs([_mouse_input(flags=flags[0] if down else flags[1])])


def _mouse_wheel(ticks: int, horizontal: bool = False) -> None:
    flag = _MOUSEEVENTF_HWHEEL if horizontal else _MOUSEEVENTF_WHEEL
    _send_inputs([_mouse_input(data=ticks * _WHEEL_DELTA, flags=flag)])


# Virtual-key map. 'cmd' deliberately aliases to CTRL: models carry macOS
# habits ("cmd+s" = save) and Ctrl is the Windows equivalent. The Windows key
# is reachable as 'win' (tool.py also aliases windows/super/meta to it).
_VK_MAP: Dict[str, int] = {
    "ctrl": 0x11, "control": 0x11, "cmd": 0x11, "command": 0x11,
    "alt": 0x12, "option": 0x12,
    "shift": 0x10,
    "win": 0x5B, "windows": 0x5B, "super": 0x5B, "meta": 0x5B,
    "enter": 0x0D, "return": 0x0D,
    "esc": 0x1B, "escape": 0x1B,
    "tab": 0x09, "space": 0x20,
    "backspace": 0x08,
    "delete": 0x2E, "del": 0x2E,
    "insert": 0x2D,
    "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pgup": 0x21, "pagedown": 0x22, "pgdn": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "arrowleft": 0x25, "arrowup": 0x26, "arrowright": 0x27, "arrowdown": 0x28,
    "capslock": 0x14, "numlock": 0x90, "printscreen": 0x2C,
    "apps": 0x5D, "menu": 0x5D,
}
for _i in range(1, 25):
    _VK_MAP[f"f{_i}"] = 0x70 + _i - 1
for _c in "abcdefghijklmnopqrstuvwxyz0123456789":
    _VK_MAP[_c] = ord(_c.upper())

# Keys that need KEYEVENTF_EXTENDEDKEY for correct scan codes.
_EXTENDED_VKS = frozenset({0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
                           0x2C, 0x2D, 0x2E, 0x5B, 0x5D, 0x90})

_MODIFIER_VKS = frozenset({0x10, 0x11, 0x12, 0x5B})


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.wintypes.UINT), ("dwTime", ctypes.wintypes.DWORD)]


def _seconds_since_user_input() -> float:
    """Seconds since the user's last keyboard/mouse input (session-wide)."""
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return float("inf")
    # GetTickCount wraps at 49.7 days; the unsigned subtraction below stays
    # correct across a single wrap.
    delta = (ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF
    return delta / 1000.0


def _wait_for_user_idle() -> None:
    """Hold injected input briefly while the user is actively typing/mousing.

    Synthetic input lands in whatever has focus; colliding with a human
    mid-keystroke sprays input across both parties' targets. Wait for
    HERMES_COMPUTER_USE_IDLE_WAIT seconds (default 1.5, 0 disables) of user
    idle, but never longer than ~8s total — the agent should yield, not
    deadlock behind a user who is working.
    """
    try:
        threshold = float(os.environ.get("HERMES_COMPUTER_USE_IDLE_WAIT", "1.5"))
    except ValueError:
        threshold = 1.5
    if threshold <= 0:
        return
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if _seconds_since_user_input() >= threshold:
            return
        time.sleep(0.2)


def _vk_for_key(name: str) -> Optional[int]:
    """Map a key name to a virtual-key code; None when unknown."""
    name = name.strip().lower()
    if not name:
        return None
    vk = _VK_MAP.get(name)
    if vk is not None:
        return vk
    if len(name) == 1:
        # Punctuation etc. — layout-dependent lookup.
        res = ctypes.windll.user32.VkKeyScanW(ord(name))
        if res != -1:
            return res & 0xFF
    return None


def _key_event(vk: int, down: bool) -> _INPUT:
    flags = 0 if down else _KEYEVENTF_KEYUP
    if vk in _EXTENDED_VKS:
        flags |= _KEYEVENTF_EXTENDEDKEY
    scan = ctypes.windll.user32.MapVirtualKeyW(vk, 0)  # MAPVK_VK_TO_VSC
    return _key_input(vk=vk, scan=scan, flags=flags)


def _press_combo(vks: List[int]) -> None:
    """Hold all but the last code as modifiers, tap the last, release."""
    mods, tap = vks[:-1], vks[-1]
    seq = [_key_event(vk, True) for vk in mods]
    seq += [_key_event(tap, True), _key_event(tap, False)]
    seq += [_key_event(vk, False) for vk in reversed(mods)]
    _send_inputs(seq)


def _switch_desktop_via_keybd(direction: str, overlay_client) -> bool:
    """Switch virtual desktop via Ctrl+Win+Left/Right SendInput.

    The overlay subprocess (full-screen tkinter window) is killed by any
    virtual-desktop transition, so we stop it before switching and restart
    it on the new desktop.
    """
    if direction not in ("left", "right"):
        return False

    VK_CONTROL = 0x11
    VK_LWIN = 0x5B
    VK_LEFT = 0x25
    VK_RIGHT = 0x27

    vk_dir = VK_LEFT if direction == "left" else VK_RIGHT

    # Single-batch SendInput matching _press_combo semantics:
    # hold modifiers → tap arrow → release, so the system input thread
    # sees the same event order as a physical keyboard.
    seq = [_key_event(VK_CONTROL, True),
           _key_event(VK_LWIN, True),
           _key_event(vk_dir, True),
           _key_event(vk_dir, False),
           _key_event(VK_LWIN, False),
           _key_event(VK_CONTROL, False)]

    try:
        overlay_client.stop()
        _send_inputs(seq)
        # The virtual-desktop transition is asynchronous — a brief delay
        # before restarting the overlay avoids racing the DWM compositor.
        time.sleep(0.15)
        overlay_client._dead = False
        overlay_client.start()
        return True
    except Exception as e:
        logger.warning("switch_desktop SendInput failed: %s", e)
        try:
            overlay_client._dead = False
            overlay_client.start()
        except Exception as e2:
            logger.warning("switch_desktop overlay restart also failed: %s", e2)
        return False


def _type_unicode(text: str) -> None:
    """Type text via KEYEVENTF_UNICODE; newlines become Return taps."""
    batch: List[_INPUT] = []
    for ch in text.replace("\r\n", "\n"):
        if ch in ("\n", "\r"):
            batch.append(_key_event(0x0D, True))
            batch.append(_key_event(0x0D, False))
            continue
        units = ch.encode("utf-16-le")
        for i in range(0, len(units), 2):
            code = units[i] | (units[i + 1] << 8)
            batch.append(_key_input(scan=code, flags=_KEYEVENTF_UNICODE))
            batch.append(_key_input(scan=code, flags=_KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP))
        if len(batch) >= 100:
            _send_inputs(batch)
            batch = []
            time.sleep(0.01)
    _send_inputs(batch)


# ---------------------------------------------------------------------------
# Window / UIA helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# On-screen overlay (visible "PC use mode") — optional, best-effort
# ---------------------------------------------------------------------------

class _OverlayClient:
    """Drives the overlay subprocess (tools/computer_use/overlay.py).

    Strictly fire-and-forget: every failure disables the overlay silently;
    desktop-control actions must never be affected by overlay problems.
    Disable entirely with HERMES_COMPUTER_USE_OVERLAY=0.
    """

    def __init__(self) -> None:
        self._proc = None
        self._sock: Optional[socket.socket] = None
        self._addr: Optional[Tuple[str, int]] = None
        self._dead = os.environ.get("HERMES_COMPUTER_USE_OVERLAY", "1") == "0"

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def start(self) -> None:
        if self._dead or self._proc is not None:
            return
        try:
            import subprocess
            overlay_py = os.path.join(os.path.dirname(__file__), "overlay.py")
            self._proc = subprocess.Popen(
                [sys.executable, "-u", overlay_py],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            line = ""
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                line = (self._proc.stdout.readline() or b"").decode("utf-8", "ignore").strip()
                if line.startswith("PORT "):
                    break
            if not line.startswith("PORT "):
                raise RuntimeError(f"overlay did not report a port (got {line!r})")
            self._addr = ("127.0.0.1", int(line.split()[1]))
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.send({"cmd": "banner", "text": "HERMES — DESKTOP CONTROL",
                       "state": "active"})
        except Exception as e:
            logger.warning("computer_use overlay unavailable: %s", e)
            self._shutdown()
            self._dead = True

    def send(self, msg: Dict[str, Any]) -> None:
        if self._dead or self._sock is None or self._addr is None:
            return
        try:
            self._sock.sendto(json.dumps(msg).encode("utf-8"), self._addr)
        except Exception:
            self._dead = True
            self._shutdown()

    def stop(self) -> None:
        self.send({"cmd": "bye"})
        self._shutdown()

    def _shutdown(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        try:
            if self._proc is not None:
                try:
                    self._proc.stdin.close()  # stdin EOF → overlay exits
                except Exception:
                    pass
                self._proc.terminate()
        except Exception:
            pass
        self._proc = None


_DWMWA_CLOAKED = 14
_DWMWA_EXTENDED_FRAME_BOUNDS = 9

# Control types we surface as interactable. Mirrors the macOS AX role list.
_INTERACTABLE_TYPES = frozenset({
    "Button", "CheckBox", "ComboBox", "Edit", "Hyperlink", "ListItem",
    "MenuItem", "RadioButton", "Slider", "Spinner", "SplitButton",
    "TabItem", "TreeItem", "DataItem", "Document",
})


def _window_rect(hwnd: int) -> Tuple[int, int, int, int]:
    """(x, y, w, h) of the window, preferring DWM extended frame bounds."""
    rect = ctypes.wintypes.RECT()
    try:
        hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.wintypes.HWND(hwnd), _DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rect), ctypes.sizeof(rect))
        if hr == 0 and rect.right > rect.left and rect.bottom > rect.top:
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        pass
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return left, top, right - left, bottom - top


def _is_cloaked(hwnd: int) -> bool:
    cloaked = ctypes.wintypes.DWORD(0)
    try:
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.wintypes.HWND(hwnd), _DWMWA_CLOAKED,
            ctypes.byref(cloaked), ctypes.sizeof(cloaked))
    except Exception:
        return False
    return cloaked.value != 0


def _exe_for_pid(pid: int) -> str:
    try:
        handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
        try:
            return os.path.basename(win32process.GetModuleFileNameEx(handle, 0))
        finally:
            handle.close()
    except Exception:
        return "unknown"


class WindowsUIABackend(ComputerUseBackend):
    """Desktop control through UI Automation + SendInput."""

    def __init__(self) -> None:
        self._elements: Dict[int, UIElement] = {}
        self._last_app: Optional[str] = None
        self._target_hwnd: Optional[int] = None
        self._target_pid: Optional[int] = None
        # Window rect at the time of the last capture — element bounds are
        # absolute screen coords, so if the window moves between capture and
        # click we translate by the origin delta instead of clicking stale
        # pixels (see _element_offset).
        self._capture_rect: Optional[Tuple[int, int, int, int]] = None
        self._started = False
        self._overlay = _OverlayClient()

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        if self._started:
            return
        if not windows_backend_available():
            raise RuntimeError(f"Windows backend unavailable: {_IMPORT_ERROR}")
        _set_dpi_awareness()
        self._overlay.start()
        self._started = True

    def stop(self) -> None:
        self._elements.clear()
        self._overlay.stop()
        self._started = False

    def is_available(self) -> bool:
        return windows_backend_available()

    # ── Window enumeration ─────────────────────────────────────────
    def _enum_top_windows(self) -> List[Dict[str, Any]]:
        """Visible, titled, non-cloaked, non-tool top-level windows."""
        windows: List[Dict[str, Any]] = []

        def handler(hwnd: int, _arg: Any) -> bool:
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                if win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) & win32con.WS_EX_TOOLWINDOW:
                    return True
                title = win32gui.GetWindowText(hwnd)
                if not title or _is_cloaked(hwnd):
                    return True
                _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
                windows.append({"hwnd": hwnd, "title": title, "pid": pid,
                                "exe": _exe_for_pid(pid)})
            except Exception:
                pass
            return True

        win32gui.EnumWindows(handler, None)
        return windows

    def _find_window(self, app: str) -> Optional[Dict[str, Any]]:
        needle = app.lower()
        for w in self._enum_top_windows():
            if needle in w["exe"].lower() or needle in w["title"].lower():
                return w
        return None

    def list_apps(self) -> List[Dict[str, Any]]:
        apps: Dict[str, Dict[str, Any]] = {}
        for w in self._enum_top_windows():
            entry = apps.setdefault(w["exe"], {
                "app": w["exe"], "pid": w["pid"], "windows": [], "window_count": 0,
            })
            entry["windows"].append(w["title"])
            entry["window_count"] += 1
        return list(apps.values())

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        target = self._find_window(app)
        if target is None:
            return ActionResult(ok=False, action="focus_app",
                                message=f"No window matching {app!r}. "
                                        f"Use list_apps to see what is running.")
        self._target_hwnd = target["hwnd"]
        self._target_pid = target["pid"]
        self._last_app = target["exe"]
        if raise_window:
            ok = self._bring_to_foreground(target["hwnd"])
            return ActionResult(
                ok=ok, action="focus_app",
                message=(f"Raised {target['exe']} ({target['title']!r})." if ok
                         else f"Targeted {target['exe']} but could not raise it."),
                meta={"hwnd": target["hwnd"], "pid": target["pid"]})
        return ActionResult(
            ok=True, action="focus_app",
            message=(f"Targeted {target['exe']} ({target['title']!r}). Note: on "
                     "Windows, pointer/keyboard actions raise the target window "
                     "when they run."),
            meta={"hwnd": target["hwnd"], "pid": target["pid"]})

    @staticmethod
    def _bring_to_foreground(hwnd: int) -> bool:
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            if win32gui.GetForegroundWindow() == hwnd:
                return True
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                # Foreground-lock workaround: an injected no-op ALT tap makes
                # our process the "last input" owner, unlocking the call.
                _send_inputs([_key_event(0x12, True), _key_event(0x12, False)])
                win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.15)
            return win32gui.GetForegroundWindow() == hwnd
        except Exception as e:
            logger.warning("SetForegroundWindow(%s) failed: %s", hwnd, e)
            return False

    def _ensure_target_foreground(self) -> None:
        # Called exactly once at the top of every input-injecting action —
        # the idle guard lives here so all of click/drag/scroll/type/key
        # yield to an actively-working user before touching the desktop.
        _wait_for_user_idle()
        if self._target_hwnd and win32gui.IsWindow(self._target_hwnd):
            self._bring_to_foreground(self._target_hwnd)

    # ── Capture ─────────────────────────────────────────────────────
    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        hwnd: Optional[int] = None
        if app:
            target = self._find_window(app)
            if target is None:
                return CaptureResult(mode=mode, width=0, height=0,
                                     app=app, window_title="(no matching window)")
            hwnd = target["hwnd"]
            self._target_hwnd = hwnd
            self._target_pid = target["pid"]
            self._last_app = target["exe"]
        elif self._target_hwnd and win32gui.IsWindow(self._target_hwnd):
            hwnd = self._target_hwnd
        else:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid == self._overlay.pid:
                    # Never capture our own overlay; fall back to the first
                    # real top-level window.
                    wins = self._enum_top_windows()
                    if wins:
                        hwnd, pid = wins[0]["hwnd"], wins[0]["pid"]
                    else:
                        hwnd = None
                if hwnd:
                    self._target_hwnd, self._target_pid = hwnd, pid
                    self._last_app = _exe_for_pid(pid)

        if not hwnd:
            return CaptureResult(mode=mode, width=0, height=0)

        x, y, w, h = _window_rect(hwnd)
        self._capture_rect = (x, y, w, h)
        window_title = win32gui.GetWindowText(hwnd)

        img = None
        png_b64: Optional[str] = None
        png_bytes_len = 0
        if mode != "ax":
            try:
                img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
            except Exception as e:
                logger.warning("screenshot failed: %s", e)

        elements = self._walk_elements(hwnd, (x, y, w, h)) if mode in ("som", "ax") else []
        self._elements = {e.index: e for e in elements}

        if img is not None and mode == "som" and elements:
            self._draw_som_overlay(img, elements, origin=(x, y))
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            png_b64 = base64.b64encode(png_bytes).decode("ascii")
            png_bytes_len = len(png_bytes)

        # Mirror what Hermes sees onto the user's screen (sent AFTER the
        # grab, so the boxes are never part of the screenshot itself).
        self._overlay.send({
            "cmd": "elements",
            "items": [{"index": e.index, "bounds": list(e.bounds)} for e in elements],
            "ttl": 4.0,
        })
        self._overlay.send({"cmd": "flash",
                            "text": f"capture · {len(elements)} elements", "ttl": 2.0})

        return CaptureResult(
            mode=mode,
            width=img.width if img is not None else w,
            height=img.height if img is not None else h,
            png_b64=png_b64,
            elements=elements,
            app=self._last_app or "",
            window_title=window_title,
            png_bytes_len=png_bytes_len,
        )

    def _iter_interactable(
        self,
        hwnd: int,
        win_rect: Tuple[int, int, int, int],
        max_nodes: int = 1500,
        max_depth: int = 60,
        time_budget: float = 4.0,
    ):
        """Yield (ctrl, role, label, rect) for interactable controls under
        `hwnd`, in breadth-first discovery order.

        This is the single source of element discovery order *and* the
        interactability/visibility filter. Both the capture walk
        (`_walk_elements`) and set_value's re-find (`_control_at_index`)
        consume it, so element #N resolves to the same control in both — if
        the two ever drifted, set_value would act on a different control than
        the capture advertised under that index. The caller owns the
        UIAutomation COM apartment (the live `ctrl`/`rect` are only valid for
        the duration of the iteration).
        """
        wx, wy, ww, wh = win_rect
        root = _auto.ControlFromHandle(hwnd)
        if root is None:
            return
        deadline = time.monotonic() + time_budget
        queue: deque = deque([(root, 0)])
        yielded = 0
        while queue and yielded < max_nodes and time.monotonic() < deadline:
            ctrl, depth = queue.popleft()
            try:
                role = ctrl.ControlTypeName
                if role.endswith("Control"):
                    role = role[: -len("Control")]
                interactable = role in _INTERACTABLE_TYPES
                if not interactable and role == "Text":
                    interactable = any(
                        ctrl.GetPattern(pid) is not None
                        for pid in (_auto.PatternId.ValuePattern,
                                    _auto.PatternId.InvokePattern))
                if interactable and ctrl.IsEnabled and not ctrl.IsOffscreen:
                    r = ctrl.BoundingRectangle
                    if (r.right > r.left and r.bottom > r.top
                            and r.left < wx + ww and r.right > wx
                            and r.top < wy + wh and r.bottom > wy):
                        label = (ctrl.Name or ctrl.AutomationId or "")
                        if len(label) > 120:
                            label = label[:120]
                        yielded += 1
                        yield ctrl, role, label, r
            except Exception:
                pass
            if depth < max_depth:
                try:
                    queue.extend((c, depth + 1) for c in ctrl.GetChildren())
                except Exception:
                    pass

    def _walk_elements(
        self,
        hwnd: int,
        win_rect: Tuple[int, int, int, int],
        max_nodes: int = 1500,
        max_depth: int = 60,
        time_budget: float = 4.0,
    ) -> List[UIElement]:
        """Collect interactable elements under `hwnd` for a capture.

        Indices follow `_iter_interactable`'s discovery order, which
        set_value re-walks (via `_control_at_index`) to resolve an element
        back to a live control.
        """
        elements: List[UIElement] = []
        try:
            with _auto.UIAutomationInitializerInThread():
                for ctrl, role, label, r in self._iter_interactable(
                        hwnd, win_rect, max_nodes=max_nodes,
                        max_depth=max_depth, time_budget=time_budget):
                    elements.append(UIElement(
                        index=len(elements) + 1,
                        role=role,
                        label=label,
                        bounds=(r.left, r.top, r.right - r.left, r.bottom - r.top),
                        app=self._last_app or "",
                        pid=self._target_pid or 0,
                        window_id=hwnd,
                    ))
        except Exception as e:
            logger.warning("UIA element walk failed: %s", e)
        return elements

    @staticmethod
    def _draw_som_overlay(img: "Image.Image", elements: List[UIElement],
                          origin: Tuple[int, int]) -> None:
        ox, oy = origin
        draw = ImageDraw.Draw(img)
        for e in elements:
            ex, ey, ew, eh = e.bounds
            ix, iy = ex - ox, ey - oy
            draw.rectangle([ix, iy, ix + ew, iy + eh], outline=(255, 0, 0), width=2)
            text = str(e.index)
            badge_w = 7 * len(text) + 6
            draw.rectangle([ix, iy, ix + badge_w, iy + 14], fill=(255, 0, 0))
            draw.text((ix + 3, iy + 1), text, fill=(255, 255, 255))

    # ── Pointer actions ─────────────────────────────────────────────
    def _element_offset(self, el: UIElement) -> Tuple[int, int]:
        """Origin shift to apply to cached element bounds.

        If the element's window moved since the capture, the cached absolute
        coords are stale by exactly the window-origin delta — translate.
        A resize invalidates interior layout, so that demands a re-capture.
        """
        hwnd = el.window_id
        if not (hwnd and self._capture_rect and win32gui.IsWindow(hwnd)):
            return 0, 0
        try:
            cx, cy, cw, ch = self._capture_rect
            nx, ny, nw, nh = _window_rect(hwnd)
        except Exception:
            return 0, 0
        if (nw, nh) != (cw, ch):
            raise ValueError(
                "the target window was resized since the last capture — "
                "re-run capture(mode='som') for fresh element positions")
        return nx - cx, ny - cy

    def _resolve_point(self, element: Optional[int], x: Optional[int],
                       y: Optional[int]) -> Tuple[int, int, str]:
        """Return (x, y, what) for an action target; raises ValueError."""
        if element is not None:
            el = self._elements.get(element)
            if el is None:
                raise ValueError(
                    f"element #{element} is not in the last capture — re-run "
                    "capture(mode='som') and use a fresh index")
            dx, dy = self._element_offset(el)
            cx, cy = el.center()
            what = f"element #{element} ({el.role} {el.label!r})"
            if dx or dy:
                what += f" [window moved {dx:+d},{dy:+d} since capture]"
            return cx + dx, cy + dy, what
        if x is None or y is None:
            raise ValueError("requires element= or coordinate [x, y]")
        return int(x), int(y), f"({x}, {y})"

    def _with_modifiers(self, modifiers: Optional[List[str]]):
        """Return (down_inputs, up_inputs) for a modifier list."""
        vks: List[int] = []
        for m in modifiers or []:
            vk = _vk_for_key(m)
            if vk is None or vk not in _MODIFIER_VKS:
                raise ValueError(f"unknown modifier {m!r}")
            vks.append(vk)
        down = [_key_event(vk, True) for vk in vks]
        up = [_key_event(vk, False) for vk in reversed(vks)]
        return down, up

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
        try:
            px, py, what = self._resolve_point(element, x, y)
            mods_down, mods_up = self._with_modifiers(modifiers)
        except ValueError as e:
            return ActionResult(ok=False, action="click", message=str(e))
        if button not in _BUTTON_FLAGS:
            return ActionResult(ok=False, action="click",
                                message=f"unknown button {button!r}")
        try:
            self._overlay.send({"cmd": "click", "x": px, "y": py})
            self._overlay.send({"cmd": "flash", "text": f"click · {what}", "ttl": 1.5})
            self._ensure_target_foreground()
            old_pos = win32api.GetCursorPos()
            try:
                _send_inputs(mods_down)
                _mouse_move(px, py)
                time.sleep(0.03)
                for i in range(max(1, click_count)):
                    _mouse_button(button, True)
                    _mouse_button(button, False)
                    if i + 1 < click_count:
                        time.sleep(0.05)
            finally:
                # Release modifiers + restore the cursor even if an injection
                # above raised — otherwise a failed click strands Ctrl/Alt/Shift
                # in the held-down state for the user at the keyboard.
                _send_inputs(mods_up)
                _mouse_move(*old_pos)
            return ActionResult(ok=True, action="click",
                                message=f"{button}-clicked {what}"
                                        + (f" x{click_count}" if click_count > 1 else ""))
        except Exception as e:
            return ActionResult(ok=False, action="click", message=f"click failed: {e}")

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
        try:
            fx, fy, src = self._resolve_point(
                from_element, *(from_xy or (None, None)))
            tx, ty, dst = self._resolve_point(
                to_element, *(to_xy or (None, None)))
            mods_down, mods_up = self._with_modifiers(modifiers)
        except ValueError as e:
            return ActionResult(ok=False, action="drag", message=str(e))
        if button not in _BUTTON_FLAGS:
            return ActionResult(ok=False, action="drag",
                                message=f"unknown button {button!r}")
        try:
            self._overlay.send({"cmd": "drag", "from": [fx, fy], "to": [tx, ty]})
            self._overlay.send({"cmd": "flash", "text": f"drag · {src} → {dst}", "ttl": 1.5})
            self._ensure_target_foreground()
            old_pos = win32api.GetCursorPos()
            try:
                _send_inputs(mods_down)
                _mouse_move(fx, fy)
                time.sleep(0.05)
                _mouse_button(button, True)
                steps = 12
                for i in range(1, steps + 1):
                    _mouse_move(fx + (tx - fx) * i // steps, fy + (ty - fy) * i // steps)
                    time.sleep(0.01)
                _mouse_button(button, False)
            finally:
                # A mid-drag failure must not strand the button or modifiers
                # held down — a stuck primary button turns every later move
                # into a drag-select. The button-up is idempotent on success.
                _mouse_button(button, False)
                _send_inputs(mods_up)
                _mouse_move(*old_pos)
            return ActionResult(ok=True, action="drag",
                                message=f"dragged {src} -> {dst}")
        except Exception as e:
            return ActionResult(ok=False, action="drag", message=f"drag failed: {e}")

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
        if direction not in {"up", "down", "left", "right"}:
            return ActionResult(ok=False, action="scroll",
                                message=f"bad direction {direction!r}")
        amount = max(1, min(50, int(amount)))
        try:
            if element is not None or (x is not None and y is not None):
                px, py, what = self._resolve_point(element, x, y)
            elif self._target_hwnd and win32gui.IsWindow(self._target_hwnd):
                wx, wy, ww, wh = _window_rect(self._target_hwnd)
                px, py, what = wx + ww // 2, wy + wh // 2, "window center"
            else:
                return ActionResult(ok=False, action="scroll",
                                    message="no target — pass element/coordinate "
                                            "or capture first")
            mods_down, mods_up = self._with_modifiers(modifiers)
        except ValueError as e:
            return ActionResult(ok=False, action="scroll", message=str(e))
        try:
            self._overlay.send({"cmd": "flash",
                                "text": f"scroll {direction} x{amount}", "ttl": 1.2})
            self._ensure_target_foreground()
            old_pos = win32api.GetCursorPos()
            try:
                _send_inputs(mods_down)
                _mouse_move(px, py)
                time.sleep(0.03)
                if direction in ("up", "down"):
                    _mouse_wheel(amount if direction == "up" else -amount)
                else:
                    _mouse_wheel(amount if direction == "right" else -amount,
                                 horizontal=True)
            finally:
                # Release modifiers + restore the cursor even if the wheel
                # injection raised, so a failed scroll can't strand a modifier.
                _send_inputs(mods_up)
                _mouse_move(*old_pos)
            return ActionResult(ok=True, action="scroll",
                                message=f"scrolled {direction} x{amount} at {what}")
        except Exception as e:
            return ActionResult(ok=False, action="scroll", message=f"scroll failed: {e}")

    # ── Keyboard ────────────────────────────────────────────────────
    def type_text(self, text: str) -> ActionResult:
        if len(text) > 20000:
            return ActionResult(ok=False, action="type",
                                message=f"text too long ({len(text)} chars; max 20000)")
        try:
            self._overlay.send({"cmd": "flash",
                                "text": f"typing · {len(text)} chars", "ttl": 2.0})
            self._ensure_target_foreground()
            _type_unicode(text)
            return ActionResult(ok=True, action="type",
                                message=f"typed {len(text)} characters")
        except Exception as e:
            return ActionResult(ok=False, action="type", message=f"type failed: {e}")

    def key(self, keys: str) -> ActionResult:
        parts = [p.strip() for p in keys.split("+") if p.strip()]
        if not parts:
            return ActionResult(ok=False, action="key", message="empty key combo")
        vks: List[int] = []
        for part in parts:
            vk = _vk_for_key(part)
            if vk is None:
                return ActionResult(ok=False, action="key",
                                    message=f"unknown key {part!r} in {keys!r}")
            vks.append(vk)
        try:
            self._overlay.send({"cmd": "flash", "text": f"key · {keys}", "ttl": 1.5})
            self._ensure_target_foreground()
            _press_combo(vks)
            return ActionResult(ok=True, action="key", message=f"pressed {keys}")
        except Exception as e:
            return ActionResult(ok=False, action="key", message=f"key failed: {e}")

    def switch_desktop(self, direction: str) -> ActionResult:
        """Switch to adjacent virtual desktop.

        Temporarily stops the overlay subprocess before switching and
        restarts it on the new desktop, because any SendInput-based
        virtual-desktop transition kills the full-screen tkinter window.
        """
        if direction not in ("left", "right"):
            return ActionResult(ok=False, action="switch_desktop",
                                message=f"unknown direction {direction!r}")
        self._overlay.send({"cmd": "flash", "text": f"switch desktop · {direction}", "ttl": 1.0})
        ok = _switch_desktop_via_keybd(direction, self._overlay)
        return ActionResult(
            ok=ok, action="switch_desktop",
            message=(f"switched to {direction} virtual desktop"
                     if ok else "virtual desktop switch failed"))

    # ── Native-value mutation ───────────────────────────────────────
    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        if element is None:
            return ActionResult(ok=False, action="set_value",
                                message="set_value requires element=")
        cached = self._elements.get(element)
        if cached is None:
            return ActionResult(ok=False, action="set_value",
                                message=f"element #{element} is not in the last "
                                        "capture — re-run capture first")
        hwnd = cached.window_id
        if not (hwnd and win32gui.IsWindow(hwnd)):
            return ActionResult(ok=False, action="set_value",
                                message="target window is gone — re-run capture")
        self._overlay.send({"cmd": "elements",
                            "items": [{"index": cached.index, "bounds": list(cached.bounds)}],
                            "ttl": 2.0})
        self._overlay.send({"cmd": "flash",
                            "text": f"set_value · #{cached.index}", "ttl": 1.5})
        try:
            with _auto.UIAutomationInitializerInThread():
                ctrl = self._refind_control(hwnd, cached)
                if ctrl is None:
                    return ActionResult(ok=False, action="set_value",
                                        message=f"element #{element} no longer "
                                                "matches the UI — re-run capture")
                return self._apply_value(ctrl, cached, value)
        except Exception as e:
            return ActionResult(ok=False, action="set_value",
                                message=f"set_value failed: {e}")

    def _refind_control(self, hwnd: int, cached: UIElement) -> Optional[Any]:
        """Re-walk the tree (same traversal as capture) to the cached index.

        Live COM pointers cannot be safely reused across tool calls (the COM
        apartment is torn down when each call's initializer exits), so we
        re-discover the control and verify role+label still match.
        """
        ctrl = self._control_at_index(hwnd, cached.index)
        if ctrl is None:
            return None
        try:
            role = ctrl.ControlTypeName
            if role.endswith("Control"):
                role = role[: -len("Control")]
            label = (ctrl.Name or ctrl.AutomationId or "")[:120]
        except Exception:
            return None
        if role != cached.role or label != cached.label:
            return None
        return ctrl

    def _control_at_index(self, hwnd: int, index: int) -> Optional[Any]:
        """Return the live control at 1-based `index` in capture discovery order.

        Re-walks via the same `_iter_interactable` traversal the capture used,
        so `index` resolves to the control capture advertised under that
        number. The caller owns the UIAutomation COM apartment.
        """
        for pos, (ctrl, _role, _label, _rect) in enumerate(
                self._iter_interactable(hwnd, _window_rect(hwnd)), start=1):
            if pos == index:
                return ctrl
        return None

    @staticmethod
    def _apply_value(ctrl: Any, cached: UIElement, value: str) -> ActionResult:
        # 1. ValuePattern — text fields and many custom controls.
        try:
            pattern = ctrl.GetPattern(_auto.PatternId.ValuePattern)
            if pattern is not None:
                pattern.SetValue(value)
                return ActionResult(ok=True, action="set_value",
                                    message=f"set value via ValuePattern on "
                                            f"#{cached.index} ({cached.label!r})")
        except Exception:
            pass
        # 2. ComboBox — expand, select matching item, collapse.
        if cached.role == "ComboBox":
            try:
                expand = ctrl.GetPattern(_auto.PatternId.ExpandCollapsePattern)
                if expand is not None:
                    expand.Expand()
                    time.sleep(0.2)
                item = ctrl.ListItemControl(Name=value)
                if item.Exists(1, 0.1):
                    sel = item.GetPattern(_auto.PatternId.SelectionItemPattern)
                    if sel is not None:
                        sel.Select()
                        if expand is not None:
                            try:
                                expand.Collapse()
                            except Exception:
                                pass
                        return ActionResult(ok=True, action="set_value",
                                            message=f"selected {value!r} in combo "
                                                    f"#{cached.index}")
                if expand is not None:
                    try:
                        expand.Collapse()
                    except Exception:
                        pass
            except Exception:
                pass
        # 3. RangeValuePattern — sliders, spinners.
        try:
            rng = ctrl.GetPattern(_auto.PatternId.RangeValuePattern)
            if rng is not None:
                rng.SetValue(float(value))
                return ActionResult(ok=True, action="set_value",
                                    message=f"set range value {value} on "
                                            f"#{cached.index}")
        except Exception:
            pass
        return ActionResult(ok=False, action="set_value",
                            message=f"element #{cached.index} ({cached.role}) does "
                                    "not accept a value via UIA patterns — try "
                                    "click + type instead")
