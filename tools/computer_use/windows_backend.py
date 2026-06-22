"""Windows computer-use backend.

This backend uses only built-in Windows APIs and PowerShell/.NET. It is a
foreground backend: pointer and keyboard actions operate on the active desktop
session and can move the user's cursor/focus.
"""

from __future__ import annotations

import base64
import ctypes
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend


def windows_backend_available() -> bool:
    """Return True when the built-in Windows backend can run."""
    if sys.platform != "win32":
        return False
    if not shutil.which("powershell.exe"):
        return False
    try:
        ctypes.windll.user32.GetSystemMetrics(0)
        return True
    except Exception:
        return False


def _run_powershell(script: str, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class WindowsComputerUseBackend(ComputerUseBackend):
    """Foreground desktop control for Windows."""

    _LEFT_DOWN = 0x0002
    _LEFT_UP = 0x0004
    _RIGHT_DOWN = 0x0008
    _RIGHT_UP = 0x0010
    _MIDDLE_DOWN = 0x0020
    _MIDDLE_UP = 0x0040
    _WHEEL = 0x0800
    _HWHEEL = 0x01000

    _KEYEVENTF_KEYUP = 0x0002
    _KEYEVENTF_UNICODE = 0x0004

    _VK_MODIFIERS = {
        "ctrl": 0x11,
        "control": 0x11,
        "shift": 0x10,
        "alt": 0x12,
        "option": 0x12,
        "cmd": 0x5B,
        "win": 0x5B,
    }

    _VK_KEYS = {
        "enter": 0x0D,
        "return": 0x0D,
        "escape": 0x1B,
        "esc": 0x1B,
        "tab": 0x09,
        "space": 0x20,
        "backspace": 0x08,
        "delete": 0x2E,
        "del": 0x2E,
        "insert": 0x2D,
        "home": 0x24,
        "end": 0x23,
        "pageup": 0x21,
        "pagedown": 0x22,
        "up": 0x26,
        "down": 0x28,
        "left": 0x25,
        "right": 0x27,
    }
    _VK_KEYS.update({f"f{i}": 0x70 + i - 1 for i in range(1, 25)})

    def start(self) -> None:
        if not self.is_available():
            raise RuntimeError("Windows computer-use backend is not available")

    def stop(self) -> None:
        return None

    def is_available(self) -> bool:
        return windows_backend_available()

    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        if mode not in {"som", "vision", "ax"}:
            raise ValueError(f"unsupported capture mode: {mode}")

        temp_path = Path(tempfile.gettempdir()) / f"hermes-screen-{time.time_ns()}.png"
        temp_literal = _ps_quote(str(temp_path))
        should_capture = "$true" if mode != "ax" else "$false"
        script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$result = [ordered]@{{ width = $bounds.Width; height = $bounds.Height; path = {temp_literal} }}
if ({should_capture}) {{
  $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
  $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
  $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
  $bitmap.Save({temp_literal}, [System.Drawing.Imaging.ImageFormat]::Png)
  $graphics.Dispose()
  $bitmap.Dispose()
}}
$result | ConvertTo-Json -Compress
"""
        proc = _run_powershell(script)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "screen capture failed").strip())
        data = json.loads(proc.stdout)
        png_b64: Optional[str] = None
        png_bytes_len = 0
        if mode != "ax":
            raw = temp_path.read_bytes()
            png_b64 = base64.b64encode(raw).decode("ascii")
            png_bytes_len = len(raw)
            try:
                temp_path.unlink()
            except OSError:
                pass
        return CaptureResult(
            mode=mode,
            width=int(data["width"]),
            height=int(data["height"]),
            png_b64=png_b64,
            elements=[],
            app=app or "",
            window_title="",
            png_bytes_len=png_bytes_len,
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
    ) -> ActionResult:
        if element is not None:
            return ActionResult(
                ok=False,
                action="click",
                message="Windows backend does not expose element indexes; pass coordinate=[x,y].",
            )
        if x is None or y is None:
            return ActionResult(ok=False, action="click", message="click requires coordinate=[x,y]")
        if button not in {"left", "right", "middle"}:
            return ActionResult(ok=False, action="click", message=f"unsupported mouse button: {button}")

        down, up = {
            "left": (self._LEFT_DOWN, self._LEFT_UP),
            "right": (self._RIGHT_DOWN, self._RIGHT_UP),
            "middle": (self._MIDDLE_DOWN, self._MIDDLE_UP),
        }[button]
        self._with_modifiers(modifiers, lambda: self._click_at(x, y, down, up, click_count))
        return ActionResult(ok=True, action="click", message=f"{button} click at ({x}, {y})")

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
        if from_element is not None or to_element is not None:
            return ActionResult(
                ok=False,
                action="drag",
                message="Windows backend does not expose element indexes; pass coordinates.",
            )
        if not from_xy or not to_xy:
            return ActionResult(ok=False, action="drag", message="drag requires from_coordinate and to_coordinate")
        if button != "left":
            return ActionResult(ok=False, action="drag", message="Windows drag supports left button only")

        def _drag() -> None:
            ctypes.windll.user32.SetCursorPos(int(from_xy[0]), int(from_xy[1]))
            ctypes.windll.user32.mouse_event(self._LEFT_DOWN, 0, 0, 0, 0)
            for step in range(1, 21):
                nx = int(from_xy[0] + (to_xy[0] - from_xy[0]) * step / 20)
                ny = int(from_xy[1] + (to_xy[1] - from_xy[1]) * step / 20)
                ctypes.windll.user32.SetCursorPos(nx, ny)
                time.sleep(0.01)
            ctypes.windll.user32.mouse_event(self._LEFT_UP, 0, 0, 0, 0)

        self._with_modifiers(modifiers, _drag)
        return ActionResult(ok=True, action="drag", message=f"dragged from {from_xy} to {to_xy}")

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
        if element is not None:
            return ActionResult(
                ok=False,
                action="scroll",
                message="Windows backend does not expose element indexes; pass coordinate=[x,y].",
            )
        if direction not in {"up", "down", "left", "right"}:
            return ActionResult(ok=False, action="scroll", message=f"unsupported scroll direction: {direction}")
        ticks = max(1, min(int(amount), 30))
        wheel_delta = 120 * ticks
        if direction in {"down", "left"}:
            wheel_delta *= -1
        flag = self._HWHEEL if direction in {"left", "right"} else self._WHEEL

        def _scroll() -> None:
            if x is not None and y is not None:
                ctypes.windll.user32.SetCursorPos(int(x), int(y))
            ctypes.windll.user32.mouse_event(flag, 0, 0, wheel_delta, 0)

        self._with_modifiers(modifiers, _scroll)
        return ActionResult(ok=True, action="scroll", message=f"scrolled {direction} x{ticks}")

    def type_text(self, text: str) -> ActionResult:
        for char in text:
            code = ord(char)
            if code > 0xFFFF:
                return ActionResult(ok=False, action="type", message="Windows backend supports BMP Unicode only")
            self._unicode_key(code)
        return ActionResult(ok=True, action="type", message=f"typed {len(text)} character(s)")

    def key(self, keys: str) -> ActionResult:
        parts = [part.strip().lower() for part in keys.split("+") if part.strip()]
        if not parts:
            return ActionResult(ok=False, action="key", message="key requires a key or combo")
        modifiers = [p for p in parts if p in self._VK_MODIFIERS]
        main_keys = [p for p in parts if p not in self._VK_MODIFIERS]
        if len(main_keys) != 1:
            return ActionResult(ok=False, action="key", message=f"could not parse key combo: {keys!r}")
        vk = self._virtual_key(main_keys[0])
        if vk is None:
            return ActionResult(ok=False, action="key", message=f"unsupported key: {main_keys[0]}")

        def _press() -> None:
            self._key_down(vk)
            self._key_up(vk)

        self._with_modifiers(modifiers, _press)
        return ActionResult(ok=True, action="key", message=f"sent key combo {keys!r}")

    def list_apps(self) -> List[Dict[str, Any]]:
        script = r"""
$items = @(Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle } |
  Sort-Object ProcessName, Id |
  ForEach-Object {
    [ordered]@{
      app_name = $_.ProcessName
      pid = $_.Id
      window_count = 1
      title = $_.MainWindowTitle
    }
  })
$items | ConvertTo-Json -Compress
"""
        proc = _run_powershell(script)
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        data = json.loads(proc.stdout)
        return data if isinstance(data, list) else [data]

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        app_literal = _ps_quote(app)
        script = f"""
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class NativeWin {{
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
}}
"@
$needle = {app_literal}
$target = Get-Process | Where-Object {{
  $_.MainWindowHandle -ne 0 -and
  ($_.ProcessName -like "*$needle*" -or $_.MainWindowTitle -like "*$needle*")
}} | Select-Object -First 1
if ($null -eq $target) {{
  [ordered]@{{ ok = $false; message = "No visible window matched '$needle'" }} | ConvertTo-Json -Compress
  exit 0
}}
[NativeWin]::ShowWindowAsync($target.MainWindowHandle, 9) | Out-Null
$ok = [NativeWin]::SetForegroundWindow($target.MainWindowHandle)
[ordered]@{{ ok = $ok; app_name = $target.ProcessName; pid = $target.Id; title = $target.MainWindowTitle }} | ConvertTo-Json -Compress
"""
        proc = _run_powershell(script)
        if proc.returncode != 0:
            return ActionResult(ok=False, action="focus_app", message=(proc.stderr or proc.stdout).strip())
        data = json.loads(proc.stdout)
        if not data.get("ok"):
            return ActionResult(ok=False, action="focus_app", message=data.get("message", "focus failed"))
        return ActionResult(
            ok=True,
            action="focus_app",
            message=f"focused {data.get('app_name')} pid={data.get('pid')}",
            meta={"title": data.get("title", ""), "raise_window": bool(raise_window)},
        )

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        return ActionResult(
            ok=False,
            action="set_value",
            message="Windows backend does not expose accessibility element value setting; use click/type/key.",
        )

    def _click_at(self, x: int, y: int, down: int, up: int, click_count: int) -> None:
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
        for _ in range(max(1, min(click_count, 5))):
            ctypes.windll.user32.mouse_event(down, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(up, 0, 0, 0, 0)
            time.sleep(0.05)

    def _with_modifiers(self, modifiers: Optional[List[str]], callback) -> None:
        vks = [self._VK_MODIFIERS[m.lower()] for m in (modifiers or []) if m.lower() in self._VK_MODIFIERS]
        for vk in vks:
            self._key_down(vk)
        try:
            callback()
        finally:
            for vk in reversed(vks):
                self._key_up(vk)

    def _virtual_key(self, key: str) -> Optional[int]:
        if len(key) == 1 and key.isalpha():
            return ord(key.upper())
        if len(key) == 1 and key.isdigit():
            return ord(key)
        return self._VK_KEYS.get(key)

    def _key_down(self, vk: int) -> None:
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)

    def _key_up(self, vk: int) -> None:
        ctypes.windll.user32.keybd_event(vk, 0, self._KEYEVENTF_KEYUP, 0)

    def _unicode_key(self, code_unit: int) -> None:
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.c_void_p),
            ]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("union", INPUT_UNION)]

        for flags in (self._KEYEVENTF_UNICODE, self._KEYEVENTF_UNICODE | self._KEYEVENTF_KEYUP):
            event = INPUT(type=1, union=INPUT_UNION(ki=KEYBDINPUT(0, code_unit, flags, 0, None)))
            sent = ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event))
            if sent != 1:
                raise RuntimeError("SendInput failed")
