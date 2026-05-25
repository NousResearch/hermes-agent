"""Linux X11 backend for the computer_use toolset.

Uses standard X11 utilities — ``xdotool`` for input synthesis,
``scrot`` for screen capture, and ``wmctrl`` for window/app
enumeration. Vision-mode captures are fully supported; SOM and AX
modes gracefully degrade to vision (the macOS backend uses Accessibility
APIs to build a numbered element overlay — Linux has AT-SPI but
coverage varies wildly by toolkit, so we deliberately skip element
indexing here and let the agent click by pixel coordinates).

Wayland is intentionally excluded: by design, Wayland compositors do
not let arbitrary clients synthesize input system-wide. Users who
want desktop automation on Linux today must run an X11 session
(``echo $XDG_SESSION_TYPE`` should print ``x11``). XWayland clients
under a Wayland session would partially work for X11 apps but not for
native Wayland windows, and the asymmetry is more confusing than
useful — we surface a clear "not available" signal in that case.

The backend is intentionally subprocess-based with no global state
beyond resolved tool paths, so tests can mock ``subprocess.run`` and
``shutil.which`` without monkey-patching the module.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)


# Tools the backend depends on. xdotool + scrot are hard requirements;
# wmctrl is used for list_apps/focus_app and the backend degrades to
# xdotool-only window search if it's missing.
_REQUIRED_TOOLS: Tuple[str, ...] = ("xdotool", "scrot")
_OPTIONAL_TOOLS: Tuple[str, ...] = ("wmctrl",)

# Map of macOS-style modifier names → X11 keysyms, so the agent can keep
# using the same schema ("cmd+s", "option+f4") on either platform.
_MODIFIER_MAP: Dict[str, Optional[str]] = {
    "cmd": "super",
    "command": "super",
    "shift": "shift",
    "option": "alt",
    "alt": "alt",
    "ctrl": "ctrl",
    "control": "ctrl",
    # "fn" has no portable X11 equivalent — drop silently.
    "fn": None,
}

# X11 button numbers for mouse actions.
_BUTTON_NUMS: Dict[str, str] = {"left": "1", "middle": "2", "right": "3"}
# Scroll wheel: in X11 scroll directions are synthesized as button presses.
_SCROLL_BUTTONS: Dict[str, str] = {"up": "4", "down": "5", "left": "6", "right": "7"}


def linux_backend_available() -> bool:
    """True iff this host can run the Linux X11 backend right now.

    Conditions checked:
      * platform is Linux,
      * an X11 display is exported (``$DISPLAY``) and the session is
        not Wayland (``$XDG_SESSION_TYPE != wayland``),
      * required tools (``xdotool``, ``scrot``) are on PATH.

    Optional tools (``wmctrl``) are not required — their absence only
    degrades ``list_apps``/``focus_app``.
    """
    if sys.platform != "linux":
        return False
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        return False
    if not os.environ.get("DISPLAY"):
        return False
    return all(shutil.which(tool) for tool in _REQUIRED_TOOLS)


class LinuxBackend(ComputerUseBackend):
    """X11 desktop control via xdotool + scrot + wmctrl.

    Element-index targeting (``capture(mode="som")`` returning numbered
    overlays) is not implemented — capture always returns the plain
    screenshot and an empty element list, and actions that accept an
    ``element`` index fall back to an error message asking for ``x``/``y``.
    """

    def __init__(self) -> None:
        self._started = False
        # Resolved at start(); empty until then.
        self._tools: Dict[str, str] = {}

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        for tool in _REQUIRED_TOOLS + _OPTIONAL_TOOLS:
            path = shutil.which(tool)
            if path:
                self._tools[tool] = path
        # Missing required tools surface through is_available(); we don't
        # raise here so the noop/cua paths keep working in mixed envs.
        self._started = True
        logger.debug(
            "linux_backend started — resolved tools: %s",
            {k: v for k, v in self._tools.items()},
        )

    def stop(self) -> None:
        self._started = False
        self._tools.clear()

    def is_available(self) -> bool:
        return linux_backend_available()

    # ── Subprocess helper ───────────────────────────────────────────

    def _run(self, *args: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
        """Run a CLI tool as an argv list (never via shell).

        Returns a CompletedProcess with ``returncode``/``stdout``/``stderr``
        populated. Times out cleanly — the caller decides what to do
        with a non-zero rc.
        """
        try:
            return subprocess.run(
                list(args),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning("linux_backend: %s timed out after %.1fs", args[0], timeout)
            return subprocess.CompletedProcess(
                args=list(args),
                returncode=124,
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=f"timeout after {timeout}s",
            )
        except (OSError, FileNotFoundError) as exc:
            logger.warning("linux_backend: %s failed to start: %s", args[0], exc)
            return subprocess.CompletedProcess(
                args=list(args), returncode=127, stdout="", stderr=str(exc)
            )

    def _tool(self, name: str) -> str:
        """Return resolved path for a tool, falling back to PATH lookup."""
        return self._tools.get(name) or name

    # ── Capture ─────────────────────────────────────────────────────

    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        # Linux has no first-class AX tree; degrade SOM → vision.
        out_mode = "vision" if mode in ("som", "vision") else mode
        width, height = self._screen_size()
        window_title = self._active_window_title()

        if out_mode == "ax":
            # AT-SPI integration is out of scope for the initial Linux
            # backend; return an empty AX result so the upper layers
            # don't crash.
            return CaptureResult(
                mode="ax",
                width=width,
                height=height,
                png_b64=None,
                elements=[],
                app=app or "",
                window_title=window_title,
            )

        if app:
            # Best-effort focus so the screenshot shows the requested app.
            self._focus_window(app)

        png_bytes = self._screenshot_png()
        if not png_bytes:
            return CaptureResult(
                mode=out_mode,
                width=width,
                height=height,
                png_b64=None,
                elements=[],
                app=app or "",
                window_title=window_title,
            )

        return CaptureResult(
            mode=out_mode,
            width=width,
            height=height,
            png_b64=base64.b64encode(png_bytes).decode("ascii"),
            elements=[],
            app=app or "",
            window_title=window_title,
            png_bytes_len=len(png_bytes),
        )

    def _screenshot_png(self) -> bytes:
        """Capture the full virtual display to a PNG byte string."""
        fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="hermes-cu-")
        os.close(fd)
        try:
            # ``-o`` overwrite, ``-z`` compress. scrot defaults to the
            # whole virtual display when given no geometry.
            result = self._run(self._tool("scrot"), "-o", "-z", tmp_path, timeout=15.0)
            if result.returncode != 0:
                logger.warning(
                    "scrot failed (rc=%s): %s", result.returncode, result.stderr.strip()
                )
                return b""
            path = Path(tmp_path)
            if not path.exists():
                return b""
            return path.read_bytes()
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    def _screen_size(self) -> Tuple[int, int]:
        result = self._run(self._tool("xdotool"), "getdisplaygeometry")
        if result.returncode == 0 and result.stdout.strip():
            try:
                parts = result.stdout.split()
                return int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                pass
        # Safe default — width/height are advisory in vision mode.
        return 1920, 1080

    def _active_window_title(self) -> str:
        result = self._run(
            self._tool("xdotool"), "getactivewindow", "getwindowname"
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def _focus_window(self, needle: str) -> bool:
        """Try to focus a window matching ``needle`` (title or wm_class).

        Returns True on success. Prefers ``wmctrl`` when available — it
        matches against the window title and class without requiring an
        exact match.
        """
        wmctrl = self._tools.get("wmctrl")
        if wmctrl:
            if self._run(wmctrl, "-a", needle).returncode == 0:
                return True
        xdotool = self._tool("xdotool")
        for selector in ("--class", "--name"):
            if self._run(xdotool, "search", selector, needle, "windowactivate").returncode == 0:
                return True
        return False

    # ── Pointer actions ─────────────────────────────────────────────

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
                message=(
                    "element-index targeting requires an accessibility tree; "
                    "the Linux backend does not build one — pass x and y instead"
                ),
            )
        if x is None or y is None:
            return ActionResult(
                ok=False, action="click", message="click requires x and y"
            )

        btn = _BUTTON_NUMS.get(button)
        if btn is None:
            return ActionResult(
                ok=False, action="click", message=f"unknown button: {button!r}"
            )

        xdotool = self._tool("xdotool")
        self._run(xdotool, "mousemove", "--sync", str(int(x)), str(int(y)))

        held = self._press_modifiers(modifiers)
        try:
            count = max(1, int(click_count))
            for _ in range(count):
                rc = self._run(xdotool, "click", btn).returncode
                if rc != 0:
                    return ActionResult(
                        ok=False,
                        action="click",
                        message=f"xdotool click rc={rc}",
                    )
        finally:
            self._release_modifiers(held)

        return ActionResult(
            ok=True,
            action="click",
            message=f"clicked {button} at ({x},{y}) x{click_count}",
        )

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
                message="element-index targeting unsupported on linux — use from_xy/to_xy",
            )
        if not from_xy or not to_xy:
            return ActionResult(
                ok=False, action="drag", message="drag requires from_xy and to_xy"
            )

        btn = _BUTTON_NUMS.get(button)
        if btn is None:
            return ActionResult(
                ok=False, action="drag", message=f"unknown button: {button!r}"
            )

        xdotool = self._tool("xdotool")
        x1, y1 = from_xy
        x2, y2 = to_xy

        held = self._press_modifiers(modifiers)
        try:
            self._run(xdotool, "mousemove", "--sync", str(int(x1)), str(int(y1)))
            self._run(xdotool, "mousedown", btn)
            self._run(xdotool, "mousemove", "--sync", str(int(x2)), str(int(y2)))
            self._run(xdotool, "mouseup", btn)
        finally:
            self._release_modifiers(held)

        return ActionResult(
            ok=True,
            action="drag",
            message=f"dragged ({x1},{y1}) → ({x2},{y2}) with {button}",
        )

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
        btn = _SCROLL_BUTTONS.get(direction.lower())
        if not btn:
            return ActionResult(
                ok=False, action="scroll", message=f"unknown direction: {direction!r}"
            )
        xdotool = self._tool("xdotool")
        if x is not None and y is not None:
            self._run(xdotool, "mousemove", "--sync", str(int(x)), str(int(y)))

        held = self._press_modifiers(modifiers)
        try:
            ticks = max(1, int(amount))
            for _ in range(ticks):
                self._run(xdotool, "click", btn)
        finally:
            self._release_modifiers(held)

        return ActionResult(
            ok=True, action="scroll", message=f"scrolled {direction} x{amount}"
        )

    # ── Modifier helpers ────────────────────────────────────────────

    def _press_modifiers(self, modifiers: Optional[List[str]]) -> List[str]:
        if not modifiers:
            return []
        held: List[str] = []
        xdotool = self._tool("xdotool")
        for mod in modifiers:
            key = _MODIFIER_MAP.get(mod.lower())
            if key is None:
                continue
            if self._run(xdotool, "keydown", key).returncode == 0:
                held.append(key)
        return held

    def _release_modifiers(self, held: List[str]) -> None:
        if not held:
            return
        xdotool = self._tool("xdotool")
        for key in reversed(held):
            self._run(xdotool, "keyup", key)

    # ── Keyboard ────────────────────────────────────────────────────

    def type_text(self, text: str) -> ActionResult:
        if not text:
            return ActionResult(ok=True, action="type", message="empty input — noop")
        # The trailing "--" stops xdotool from parsing the text as flags;
        # the delay is xdotool's default but explicit for clarity.
        result = self._run(
            self._tool("xdotool"), "type", "--delay", "12", "--", text, timeout=30.0
        )
        if result.returncode != 0:
            return ActionResult(
                ok=False,
                action="type",
                message=result.stderr.strip() or "xdotool type failed",
            )
        return ActionResult(
            ok=True, action="type", message=f"typed {len(text)} chars"
        )

    def key(self, keys: str) -> ActionResult:
        if not keys:
            return ActionResult(ok=False, action="key", message="empty key combo")
        # Translate any macOS-style names ("cmd+s") to X11 keysyms
        # ("super+s") so the same prompts work on both platforms.
        translated_parts: List[str] = []
        for part in keys.split("+"):
            cleaned = part.strip()
            if not cleaned:
                continue
            mapped = _MODIFIER_MAP.get(cleaned.lower(), cleaned)
            if mapped:
                translated_parts.append(mapped)
        if not translated_parts:
            return ActionResult(ok=False, action="key", message=f"invalid combo: {keys!r}")
        combo = "+".join(translated_parts)
        result = self._run(self._tool("xdotool"), "key", "--", combo)
        if result.returncode != 0:
            return ActionResult(
                ok=False,
                action="key",
                message=result.stderr.strip() or f"xdotool key {combo!r} failed",
            )
        return ActionResult(ok=True, action="key", message=f"sent key combo: {combo}")

    # ── Introspection ───────────────────────────────────────────────

    def list_apps(self) -> List[Dict[str, Any]]:
        wmctrl = self._tools.get("wmctrl")
        if not wmctrl:
            return []
        # ``-l`` list windows, ``-p`` include PID, ``-x`` include wm_class.
        # Output columns: <wid> <desktop> <pid> <wm_class> <host> <title>
        result = self._run(wmctrl, "-l", "-p", "-x")
        if result.returncode != 0:
            return []

        apps: Dict[str, Dict[str, Any]] = {}
        for raw in result.stdout.splitlines():
            parts = raw.split(None, 5)
            if len(parts) < 6:
                continue
            _wid, _desk, pid_str, wm_class, _host, title = parts
            try:
                pid = int(pid_str)
            except ValueError:
                pid = 0
            # wm_class is "instance.Class"; use the class half as app name.
            app_name = wm_class.rsplit(".", 1)[-1] if "." in wm_class else wm_class
            entry = apps.setdefault(
                app_name,
                {
                    "name": app_name,
                    "pid": pid,
                    "windows": 0,
                    "wm_class": wm_class,
                    "titles": [],
                },
            )
            entry["windows"] += 1
            if title and title not in entry["titles"]:
                entry["titles"].append(title)
        return list(apps.values())

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        if not app:
            return ActionResult(
                ok=False, action="focus_app", message="app name required"
            )
        # On X11 the distinction between "focus" and "raise" is muddier
        # than on macOS — wmctrl -a raises by default on most WMs, and we
        # accept that for compatibility. raise_window is honored as a
        # soft hint; we don't try to suppress the raise on Linux.
        if self._focus_window(app):
            return ActionResult(
                ok=True, action="focus_app", message=f"focused {app!r}"
            )
        return ActionResult(
            ok=False,
            action="focus_app",
            message=f"window matching {app!r} not found",
        )

    # ── Native-value mutation ───────────────────────────────────────

    def set_value(
        self, value: str, element: Optional[int] = None
    ) -> ActionResult:
        # AT-SPI value-set integration is not implemented in the initial
        # Linux backend. The agent can usually achieve the same effect
        # via focus + select-all + type. Returning a clear, actionable
        # error keeps the agent unblocked.
        return ActionResult(
            ok=False,
            action="set_value",
            message=(
                "set_value is unsupported on linux (no AT-SPI integration); "
                "fall back to click → key('ctrl+a') → type_text"
            ),
        )
