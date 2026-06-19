"""Cua-driver backend (macOS only), CLI/app-daemon transport.

The native Hermes tool surface routes through the approved CuaDriver.app daemon.
This backend shells out to ``cua-driver call ...`` so all actions share the same
macOS permission context that the Swift app can onboard, monitor, and stop.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement

logger = logging.getLogger(__name__)

PINNED_CUA_DRIVER_VERSION = os.environ.get("HERMES_CUA_DRIVER_VERSION", "0.5.0")
_CUA_DRIVER_CMD = os.environ.get("HERMES_CUA_DRIVER_CMD", "cua-driver")
_CUA_DRIVER_ARGS = ["mcp"]

_WINDOW_LINE_RE = re.compile(r'^-\s+(.+?)\s+\(pid\s+(\d+)\)\s+.*\[window_id:\s+(\d+)\]', re.MULTILINE)
_ELEMENT_LINE_RE = re.compile(
    r'^\s*(?:-\s+)?\[(\d+)\]\s+(\w+)(?:\s+"([^"]*)"|(?:\s+\(\d+\))?\s+id=([^\s\[\]]*))?',
    re.MULTILINE,
)


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_arm_mac() -> bool:
    return _is_macos() and platform.machine() == "arm64"


def cua_driver_executable() -> Optional[str]:
    """Resolve cua-driver even when Hermes runs with a sparse macOS PATH."""
    resolved = shutil.which(_CUA_DRIVER_CMD)
    if resolved:
        return resolved
    if os.path.isabs(_CUA_DRIVER_CMD) and os.path.exists(_CUA_DRIVER_CMD):
        return _CUA_DRIVER_CMD
    for candidate in (
        "/opt/homebrew/bin/cua-driver",
        "/usr/local/bin/cua-driver",
        os.path.expanduser("~/.local/bin/cua-driver"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def cua_driver_binary_available() -> bool:
    return bool(cua_driver_executable())


def _version_tuple(version: str) -> Tuple[int, ...]:
    nums = re.findall(r"\d+", version or "")
    return tuple(int(n) for n in nums[:3]) or (0,)


def cua_driver_version(timeout: float = 5.0) -> str:
    exe = cua_driver_executable()
    if not exe:
        return ""
    try:
        return subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def cua_driver_version_status() -> Dict[str, Any]:
    actual = cua_driver_version()
    minimum = PINNED_CUA_DRIVER_VERSION
    ok = bool(actual) and _version_tuple(actual) >= _version_tuple(minimum)
    return {"ok": ok, "actual": actual, "minimum": minimum}


def cua_driver_install_hint() -> str:
    return (
        "cua-driver is not installed. Install with `hermes computer-use install` "
        "or run `hermes tools` and enable the Computer Use toolset."
    )


def cua_driver_permissions_status(timeout: float = 2.0) -> Dict[str, Any]:
    """Probe macOS TCC status through the approved CuaDriver app context."""
    try:
        exe = cua_driver_executable()
        if not exe:
            return {"available": False, "ok": False, "message": cua_driver_install_hint()}
        if _is_macos():
            open_bin = shutil.which("open") or "/usr/bin/open"
            subprocess.run(
                [open_bin, "-g", "-a", "CuaDriver", "--args", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
            time.sleep(0.5)
        last_timeout: Optional[float] = None
        try:
            proc = subprocess.run(
                [exe, "call", "check_permissions", "{}"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            last_timeout = float(e.timeout or timeout)
            return {"available": True, "ok": None, "message": f"permission probe timed out after {last_timeout}s; CuaDriver may still be starting"}
    except Exception as e:
        return {"available": True, "ok": None, "message": str(e)}
    text = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
    lower = text.lower()
    unreliable = "not running inside the cua-driver daemon" in lower or "results may be inaccurate" in lower
    ok = proc.returncode == 0 and "granted" in lower and "denied" not in lower and "not granted" not in lower and not unreliable
    return {"available": True, "ok": ok, "message": text, "returncode": proc.returncode, "unreliable": unreliable}


def _parse_json_or_text(text: str) -> Any:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Some CLIs print logs before a final JSON object. Grab the last plausible JSON line.
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if line.startswith(("{", "[")):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return stripped


def _parse_windows_from_text(text: str) -> List[Dict[str, Any]]:
    windows = []
    for m in _WINDOW_LINE_RE.finditer(text):
        windows.append({
            "app_name": m.group(1).strip(),
            "pid": int(m.group(2)),
            "window_id": int(m.group(3)),
            "off_screen": "[off-screen]" in m.group(0),
        })
    return windows


def _label_from_element_match(m: re.Match[str]) -> str:
    return (m.group(3) or m.group(4) or "").strip()


def _image_dimensions_from_bytes(raw: bytes) -> Tuple[int, int]:
    """Best-effort PNG/JPEG dimension sniffing without extra dependencies."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        if width > 0 and height > 0:
            return width, height

    if raw.startswith(b"\xff\xd8"):
        i = 2
        n = len(raw)
        while i + 9 < n:
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            i += 2
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > n:
                break
            segment_len = int.from_bytes(raw[i:i + 2], "big")
            if segment_len < 2 or i + segment_len > n:
                break
            if marker in {
                0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
            }:
                if segment_len >= 7:
                    height = int.from_bytes(raw[i + 3:i + 5], "big")
                    width = int.from_bytes(raw[i + 5:i + 7], "big")
                    if width > 0 and height > 0:
                        return width, height
                break
            i += segment_len

    return 0, 0


def _parse_elements_from_tree(markdown: str, *, app: str = "", pid: int = 0, window_id: int = 0) -> List[UIElement]:
    elements = []
    for m in _ELEMENT_LINE_RE.finditer(markdown or ""):
        elements.append(UIElement(
            index=int(m.group(1)),
            role=m.group(2),
            label=_label_from_element_match(m),
            app=app,
            pid=pid or 0,
            window_id=window_id or 0,
        ))
    return elements


def _split_tree_text(full_text: str) -> Tuple[str, str]:
    lines = (full_text or "").split("\n", 1)
    return lines[0], lines[1] if len(lines) > 1 else ""


class _AsyncBridge:
    """Compatibility bridge for MCP-session tests; production uses the daemon CLI path."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True, name="cua-driver-loop")
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("cua-driver asyncio bridge failed to start")

    def run(self, coro, timeout: Optional[float] = 30.0) -> Any:
        from agent.async_utils import safe_schedule_threadsafe
        if not self._loop or not self._thread or not self._thread.is_alive():
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("cua-driver bridge not started")
        fut = safe_schedule_threadsafe(coro, self._loop)
        if fut is None:
            raise RuntimeError("cua-driver bridge not started")
        return fut.result(timeout=timeout)

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._loop = None


class _CuaDriverSession:
    """Legacy MCP session shim kept for tests and external imports."""

    def __init__(self, bridge: _AsyncBridge) -> None:
        self._bridge = bridge
        self._session = None
        self._exit_stack = None
        self._lock = threading.Lock()
        self._started = False

    async def _aenter(self) -> None:
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from tools.environments.local import _sanitize_subprocess_env

        if not cua_driver_binary_available():
            raise RuntimeError(cua_driver_install_hint())

        params = StdioServerParameters(
            command=_CUA_DRIVER_CMD,
            args=_CUA_DRIVER_ARGS,
            env=_sanitize_subprocess_env(dict(os.environ)),
        )
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._exit_stack = stack
        self._session = session

    async def _aexit(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning("cua-driver shutdown error: %s", e)
        self._exit_stack = None
        self._session = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._bridge.start()
            self._bridge.run(self._aenter(), timeout=15.0)
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            try:
                self._bridge.run(self._aexit(), timeout=5.0)
            finally:
                self._started = False

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError("cua-driver session not started")
        result = self._bridge.run(self._session.call_tool(name, args), timeout=timeout)
        data = getattr(result, "structuredContent", None)
        return {"data": data, "images": [], "structuredContent": data, "isError": bool(getattr(result, "isError", False))}


def _parse_key_combo(keys: str) -> Tuple[Optional[str], List[str]]:
    modifiers = []
    key = None
    aliases = {"command": "cmd", "alt": "option", "control": "ctrl"}
    for part in [p.strip().lower() for p in re.split(r'[+\-]', keys or "") if p.strip()]:
        normalized = aliases.get(part, part)
        if normalized in {"cmd", "shift", "option", "ctrl", "fn"}:
            modifiers.append(normalized)
        else:
            key = part
    return key, modifiers



class CuaDriverBackend(ComputerUseBackend):
    """Default computer-use backend using ``cua-driver call``."""

    def __init__(self) -> None:
        self._active_pid: Optional[int] = None
        self._active_window_id: Optional[int] = None
        self._active_app: str = ""
        self._last_app: Optional[str] = None
        self._app_cache: List[Dict[str, Any]] = []
        self._app_cache_at: float = 0.0

    def start(self) -> None:
        if not cua_driver_binary_available():
            raise RuntimeError(cua_driver_install_hint())
        if _is_macos():
            open_bin = shutil.which("open") or "/usr/bin/open"
            subprocess.run([open_bin, "-n", "-g", "-a", "CuaDriver", "--args", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        try:
            self.apply_runtime_config()
        except Exception:
            logger.debug("apply_runtime_config failed during start", exc_info=True)

    def stop(self) -> None:
        if not _is_macos():
            return
        # Best-effort daemon shutdown so a fresh `start()` re-applies runtime config
        # against a clean process. The CuaDriver app is otherwise quiet.
        try:
            subprocess.run(
                ["/usr/bin/pkill", "-x", "CuaDriver"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3,
            )
        except Exception:
            logger.debug("CuaDriver stop pkill failed", exc_info=True)

    def is_available(self) -> bool:
        return _is_macos() and cua_driver_binary_available()

    def permissions_status(self) -> Dict[str, Any]:
        return cua_driver_permissions_status()

    def _call(self, name: str, args: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        # Test compatibility: older MCP-backed tests patch ``_session.call_tool``.
        # The production path below remains CLI/app-daemon transport.
        session = getattr(self, "_session", None)
        if session is not None and hasattr(session, "call_tool"):
            return session.call_tool(name, args)

        exe = cua_driver_executable()
        if not exe:
            raise RuntimeError(cua_driver_install_hint())
        cmd = [exe, "call", name, json.dumps(args or {})]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        parsed = _parse_json_or_text(proc.stdout)
        if proc.returncode != 0:
            return {"data": parsed or proc.stderr.strip(), "images": [], "structuredContent": None, "isError": True}
        images: List[str] = []
        structured = parsed if isinstance(parsed, dict) else None
        data: Any = parsed
        if isinstance(parsed, dict):
            for key in ("image", "png_b64", "screenshot", "data"):
                val = parsed.get(key)
                if isinstance(val, str) and len(val) > 100 and re.match(r"^[A-Za-z0-9+/=]+$", val[:120]):
                    images.append(val)
                    break
            data = parsed.get("data", parsed.get("text", parsed))
        return {"data": data, "images": images, "structuredContent": structured, "isError": False}

    def _app_catalog(self, ttl: float = 5.0) -> List[Dict[str, Any]]:
        now = time.monotonic()
        if self._app_cache and now - self._app_cache_at < ttl:
            return self._app_cache
        out = self._call("list_apps", {})
        data = out["data"]
        if isinstance(data, dict):
            apps = data.get("apps", [])
        elif isinstance(data, list):
            apps = data
        elif isinstance(data, str):
            apps = []
            for line in data.splitlines():
                m = re.search(r'(.+?)\s+\(pid\s+(\d+)\)', line)
                if m:
                    apps.append({"name": m.group(1).strip(), "pid": int(m.group(2))})
        else:
            apps = []
        self._app_cache = apps
        self._app_cache_at = now
        return apps

    def _app_names_by_bundle(self) -> Dict[str, str]:
        return {
            str(app.get("bundle_id") or app.get("bundleId") or "").lower(): str(app.get("name") or "")
            for app in self._app_catalog()
            if app.get("bundle_id") or app.get("bundleId")
        }

    def _windows(self) -> List[Dict[str, Any]]:
        out = self._call("list_windows", {"on_screen_only": True})
        data = out.get("data")
        structured = out.get("structuredContent") or {}
        raw_windows = None
        if isinstance(structured, dict):
            raw_windows = structured.get("windows")
        if raw_windows is None and isinstance(data, dict):
            raw_windows = data.get("windows")
        if raw_windows:
            windows = []
            apps_by_bundle: Dict[str, str] = {}
            for w in raw_windows:
                app_name = w.get("app_name") or w.get("app") or w.get("name") or ""
                bundle = str(w.get("bundle_id") or w.get("bundleId") or "")
                if bundle and (not app_name or app_name == bundle):
                    if not apps_by_bundle:
                        apps_by_bundle = self._app_names_by_bundle()
                    app_name = apps_by_bundle.get(bundle.lower(), app_name)
                windows.append({
                    "app_name": app_name,
                    "bundle_id": bundle,
                    "pid": int(w.get("pid") or 0),
                    "window_id": int(w.get("window_id") or w.get("windowId") or 0),
                    "off_screen": not w.get("is_on_screen", True),
                    "title": w.get("title", ""),
                    "bounds": w.get("bounds") or {},
                    "z_index": w.get("z_index", 0),
                })
            return sorted(windows, key=lambda w: w.get("z_index", 0))
        if isinstance(data, str):
            return _parse_windows_from_text(data)
        return []

    def _select_window(self, app: Optional[str] = None) -> Optional[Dict[str, Any]]:
        windows = self._windows()
        if app:
            needle = app.lower()
            def matches(w: Dict[str, Any]) -> bool:
                haystack = " ".join([
                    str(w.get("app_name") or ""),
                    str(w.get("bundle_id") or ""),
                ]).lower()
                return needle in haystack
            matched = [w for w in windows if matches(w)]
            if not matched:
                return None
            windows = matched
        target = next((w for w in windows if not w.get("off_screen")), windows[0] if windows else None)
        if target:
            self._active_pid = int(target.get("pid") or 0)
            self._active_window_id = int(target.get("window_id") or 0)
            self._active_app = str(target.get("app_name") or "")
            self._last_app = self._active_app or None
        return target

    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        target = self._select_window(app)
        if not target:
            diagnostic = ""
            if app:
                diagnostic = (
                    f"<no on-screen window matched app={app!r}; call list_apps to see available app names "
                    "(macOS reports localized names, e.g. '計算機' instead of 'Calculator')>"
                )
            return CaptureResult(mode=mode, width=0, height=0, elements=[], app="", window_title=diagnostic)
        pid, window_id = self._active_pid, self._active_window_id
        png_b64: Optional[str] = None
        elements: List[UIElement] = []
        window_title = str(target.get("title") or "")
        bounds = target.get("bounds") or {}
        width = int(bounds.get("width") or target.get("width") or 0) if isinstance(bounds, dict) else 0
        height = int(bounds.get("height") or target.get("height") or 0) if isinstance(bounds, dict) else 0
        if mode == "vision":
            out = self._call("screenshot", {"window_id": window_id, "format": "jpeg", "quality": 85})
            if out["images"]:
                png_b64 = out["images"][0]
        else:
            call_args: Dict[str, Any] = {"pid": pid, "window_id": window_id}
            tmp_path: Optional[str] = None
            try:
                if mode == "som":
                    tmp = tempfile.NamedTemporaryFile(prefix="hermes-cua-", suffix=".jpg", delete=False)
                    tmp.close()
                    tmp_path = tmp.name
                    call_args["screenshot_out_file"] = tmp_path
                out = self._call("get_window_state", call_args)
                structured = out.get("structuredContent") if isinstance(out, dict) else None
                data = out["data"]
                if isinstance(data, dict):
                    tree = str(data.get("tree_markdown") or data.get("tree") or data.get("markdown") or "")
                    window_title = str(data.get("window_title") or data.get("title") or window_title)
                    width = int(data.get("width") or width or 0)
                    height = int(data.get("height") or height or 0)
                elif isinstance(structured, dict):
                    tree = str(structured.get("tree_markdown") or structured.get("tree") or "")
                else:
                    text = data if isinstance(data, str) else json.dumps(data)
                    _summary, tree = _split_tree_text(text)
                elements = _parse_elements_from_tree(tree, app=self._active_app, pid=int(pid or 0), window_id=int(window_id or 0))
                wt = re.search(r'AXWindow\s+"([^"]+)"', tree)
                if wt:
                    window_title = wt.group(1)
                if out["images"]:
                    png_b64 = out["images"][0]
                elif mode == "som" and tmp_path and os.path.exists(tmp_path):
                    with open(tmp_path, "rb") as fh:
                        png_b64 = base64.b64encode(fh.read()).decode("ascii")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        png_bytes_len = 0
        if png_b64:
            try:
                raw = base64.b64decode(png_b64, validate=False)
                png_bytes_len = len(raw)
                detected_width, detected_height = _image_dimensions_from_bytes(raw)
                if detected_width and detected_height:
                    width = detected_width
                    height = detected_height
            except Exception:
                png_bytes_len = len(png_b64) * 3 // 4
        return CaptureResult(mode=mode, width=width, height=height, png_b64=png_b64, elements=elements, app=self._active_app, window_title=window_title, png_bytes_len=png_bytes_len)

    def _action(self, name: str, args: Dict[str, Any]) -> ActionResult:
        try:
            out = self._call(name, args)
        except Exception as e:
            logger.exception("cua-driver %s call failed", name)
            return ActionResult(ok=False, action=name, message=f"cua-driver error: {e}")
        data = out["data"]
        message = data.get("message", "") if isinstance(data, dict) else str(data or "")
        return ActionResult(ok=not out["isError"], action=name, message=message, meta=data if isinstance(data, dict) else {})

    def _require_pid(self, action: str) -> Optional[ActionResult]:
        if self._active_pid is None:
            return ActionResult(ok=False, action=action, message="No active window — pass app=... or call computer_use_get_app_state(app=...) first.")
        return None

    def click(self, *, element: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", click_count: int = 1, modifiers: Optional[List[str]] = None) -> ActionResult:
        if err := self._require_pid("click"):
            return err
        if button not in {"left", "right"}:
            return ActionResult(ok=False, action="click", message=f"unsupported mouse button {button!r}; use left or right")
        if click_count not in {1, 2}:
            return ActionResult(ok=False, action="click", message="click_count must be 1 or 2")
        tool = "right_click" if button == "right" else ("double_click" if click_count == 2 else "click")
        args: Dict[str, Any] = {"pid": self._active_pid}
        if element is not None:
            args.update({"window_id": self._active_window_id, "element_index": element})
        elif x is not None and y is not None:
            args.update({"x": x, "y": y})
        else:
            return ActionResult(ok=False, action=tool, message="click requires element or coordinate")
        if modifiers:
            args["modifier"] = modifiers
        return self._action(tool, args)

    def drag(self, *, from_element: Optional[int] = None, to_element: Optional[int] = None, from_xy: Optional[Tuple[int, int]] = None, to_xy: Optional[Tuple[int, int]] = None, button: str = "left", modifiers: Optional[List[str]] = None) -> ActionResult:
        if err := self._require_pid("drag"):
            return err
        args: Dict[str, Any] = {"pid": self._active_pid, "window_id": self._active_window_id}
        if from_element is not None:
            args["from_element_index"] = from_element
        if to_element is not None:
            args["to_element_index"] = to_element
        if from_xy:
            args["from_x"], args["from_y"] = from_xy
        if to_xy:
            args["to_x"], args["to_y"] = to_xy
        return self._action("drag", args)

    def scroll(self, *, direction: str, amount: int = 3, pages: Optional[float] = None, element: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None, modifiers: Optional[List[str]] = None) -> ActionResult:
        if err := self._require_pid("scroll"):
            return err
        if pages is not None:
            # Codex uses page-ish distances; cua-driver scroll takes wheel ticks.
            amount = max(1, int(round(abs(float(pages)) * 6)))
        args: Dict[str, Any] = {"pid": self._active_pid, "direction": direction, "amount": max(1, min(50, amount))}
        if element is not None:
            args.update({"window_id": self._active_window_id, "element_index": element})
        elif x is not None and y is not None:
            args.update({"x": x, "y": y})
        return self._action("scroll", args)

    def type_text(self, text: str) -> ActionResult:
        if err := self._require_pid("type_text"):
            return err
        return self._action("type_text_chars", {"pid": self._active_pid, "text": text})

    def key(self, keys: str) -> ActionResult:
        if err := self._require_pid("key"):
            return err
        key_name, modifiers = _parse_key_combo(keys)
        if not key_name:
            return ActionResult(ok=False, action="key", message=f"Could not parse key from {keys!r}.")
        if modifiers:
            return self._action("hotkey", {"pid": self._active_pid, "keys": modifiers + [key_name]})
        return self._action("press_key", {"pid": self._active_pid, "key": key_name})

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        if err := self._require_pid("set_value"):
            return err
        if element is None:
            return ActionResult(ok=False, action="set_value", message="set_value requires element")
        return self._action("set_value", {"pid": self._active_pid, "window_id": self._active_window_id, "element_index": element, "value": value})

    def perform_secondary_action(self, element: Optional[int] = None, secondary_action: str = "AXShowMenu") -> ActionResult:
        if err := self._require_pid("perform_secondary_action"):
            return err
        if element is None:
            return ActionResult(ok=False, action="perform_secondary_action", message="secondary action requires element")
        return self._action("perform_secondary_action", {"pid": self._active_pid, "window_id": self._active_window_id, "element_index": element, "action": secondary_action})

    def select_text(self, element: Optional[int] = None, text: str = "", selection: str = "all", prefix: str = "", suffix: str = "", cursor: Optional[str] = None) -> ActionResult:
        if err := self._require_pid("select_text"):
            return err
        args: Dict[str, Any] = {"pid": self._active_pid, "window_id": self._active_window_id, "selection": selection}
        if element is not None:
            args["element_index"] = element
        if text:
            args["text"] = text
        if prefix:
            args["prefix"] = prefix
        if suffix:
            args["suffix"] = suffix
        if cursor:
            args["cursor"] = cursor
        return self._action("select_text", args)

    def list_apps(self) -> List[Dict[str, Any]]:
        return self._app_catalog(ttl=0.0)

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        target = self._select_window(app)
        if target:
            return ActionResult(ok=True, action="focus_app", message=f"Targeted {target.get('app_name')} (pid {self._active_pid}, window {self._active_window_id}) without raising window.")
        return ActionResult(ok=False, action="focus_app", message=f"No on-screen window found for app {app!r}.")

    def _resolve_launch_bundle_id(self, app: str = "", bundle_id: str = "") -> Tuple[str, str, Optional[str]]:
        if bundle_id:
            return bundle_id.strip(), app.strip() or bundle_id.strip(), None
        target = app.strip()
        if not target:
            return "", "", "launch_app requires app or bundle_id"
        expanded = os.path.expanduser(target)
        if expanded.endswith(".app") or os.path.sep in expanded:
            info_plist = os.path.join(expanded, "Contents", "Info.plist")
            try:
                with open(info_plist, "rb") as fh:
                    info = plistlib.load(fh)
                resolved = str(info.get("CFBundleIdentifier") or "").strip()
            except Exception as e:
                return "", target, f"Could not read bundle id from {expanded!r}: {e}"
            if not resolved:
                return "", target, f"No CFBundleIdentifier found in {info_plist!r}."
            return resolved, os.path.basename(expanded).removesuffix(".app"), None

        needle = target.lower()
        apps = self._app_catalog()
        exact = []
        fuzzy = []
        for candidate in apps:
            name = str(candidate.get("name") or "").strip()
            candidate_bundle = str(candidate.get("bundle_id") or candidate.get("bundleId") or "").strip()
            if not candidate_bundle:
                continue
            if needle in {name.lower(), candidate_bundle.lower()}:
                exact.append((candidate_bundle, name or candidate_bundle))
            elif needle in name.lower():
                fuzzy.append((candidate_bundle, name or candidate_bundle))
        matches = exact or fuzzy
        unique: Dict[str, str] = {bundle: name for bundle, name in matches}
        if len(unique) == 1:
            bundle, name = next(iter(unique.items()))
            return bundle, name, None
        if len(unique) > 1:
            choices = ", ".join(f"{name} ({bundle})" for bundle, name in list(unique.items())[:5])
            return "", target, f"Ambiguous app name {target!r}; use a bundle_id. Matches: {choices}"
        return "", target, f"No bundle id found for app {target!r}; call computer_use_list_apps to inspect installed apps."

    def launch_app(self, app: str = "", bundle_id: str = "", background: bool = True) -> ActionResult:
        target, display, error = self._resolve_launch_bundle_id(app=app, bundle_id=bundle_id)
        if error:
            return ActionResult(ok=False, action="launch_app", message=error)

        res = self._action("launch_app", {"bundle_id": target})
        if res.ok:
            # Refresh the target window/pid if the launched app has a window. Some
            # menu-bar/background apps launch successfully without a layer-0 window.
            self._app_cache_at = 0.0
            launched_name = str(res.meta.get("name") or display or target)
            self._select_window(target) or self._select_window(launched_name)
            res.message = res.message or f"Launched {launched_name or target}."
            res.meta = {**res.meta, "app": launched_name, "bundle_id": target, "background": background}
        return res

    # ── Daemon lifecycle / runtime config ──────────────────────────────
    def _show_cursor_setting(self) -> Optional[bool]:
        """Resolve the `display.show_cursor` config flag.

        Order: HERMES_CUA_SHOW_CURSOR env (1/true/on / 0/false/off) → user config
        `computer_use.show_cursor` → None (leave driver default).
        """
        env = os.environ.get("HERMES_CUA_SHOW_CURSOR")
        if env is not None:
            return env.strip().lower() in {"1", "true", "yes", "on"}
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly() or {}
        except Exception:
            cfg = {}
        section = (cfg.get("computer_use") or {}) if isinstance(cfg, dict) else {}
        if isinstance(section, dict) and "show_cursor" in section:
            return bool(section["show_cursor"])
        return None

    def apply_runtime_config(self) -> None:
        show_cursor = self._show_cursor_setting()
        if show_cursor is not None:
            try:
                self._action("set_agent_cursor_enabled", {"enabled": bool(show_cursor)})
            except Exception:
                logger.debug("set_agent_cursor_enabled failed", exc_info=True)

    def daemon_status(self) -> Dict[str, Any]:
        exe = cua_driver_executable()
        version_status = cua_driver_version_status() if exe else {"actual": "", "minimum": PINNED_CUA_DRIVER_VERSION, "ok": False}
        permissions = cua_driver_permissions_status() if exe else {"available": False, "ok": False, "message": cua_driver_install_hint()}
        running = False
        if _is_macos() and exe:
            try:
                proc = subprocess.run(["/usr/bin/pgrep", "-x", "CuaDriver"], capture_output=True, text=True, timeout=2)
                running = proc.returncode == 0 and bool(proc.stdout.strip())
            except Exception:
                running = False
        return {
            "binary_installed": bool(exe),
            "binary_path": exe or "",
            "version": version_status.get("actual") or "",
            "minimum_version": version_status.get("minimum") or "",
            "version_ok": bool(version_status.get("ok")),
            "running": running,
            "permissions": "ok" if permissions.get("ok") is True else ("not_ready" if permissions.get("ok") is False else "unknown"),
            "permissions_message": (permissions.get("message") or "").strip(),
            "show_cursor": self._show_cursor_setting(),
        }
