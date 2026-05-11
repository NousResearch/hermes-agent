"""General desktop computer-use tools for macOS.

This module provides a small, model-friendly control surface for operating the
local desktop when browser/API tools are insufficient. It intentionally uses
macOS-native primitives plus `cliclick` instead of a long-running daemon:

- `screencapture` for screenshots
- AppleScript/System Events for app/window metadata and launching apps
- `cliclick` for mouse and keyboard events

The tools are conservative: they report permission problems clearly and return
paths/structured data so the agent can verify each step with vision/snapshots.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.registry import registry, tool_error, tool_result


SCREENSHOT_SCHEMA = {
    "name": "desktop_screenshot",
    "description": "Take a screenshot of the local macOS desktop or active display and return a file path for vision analysis. Use before/after desktop actions to verify UI state.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional output path. Defaults to ~/.hermes/screenshots/desktop_<timestamp>.png",
            },
            "display": {
                "type": "integer",
                "description": "Optional display number for screencapture -D. Omit for all displays.",
            },
        },
    },
}

DESKTOP_ACTION_SCHEMA = {
    "name": "desktop_action",
    "description": "Perform local desktop mouse/keyboard/app actions on macOS. Use with desktop_screenshot + vision to click/type/drag/key/open apps and verify results.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "move", "click", "double_click", "right_click", "drag",
                    "type", "key", "hotkey", "open_app", "wait", "position",
                ],
                "description": "Desktop action to perform.",
            },
            "x": {"type": "integer", "description": "X screen coordinate for mouse actions."},
            "y": {"type": "integer", "description": "Y screen coordinate for mouse actions."},
            "to_x": {"type": "integer", "description": "Destination X coordinate for drag."},
            "to_y": {"type": "integer", "description": "Destination Y coordinate for drag."},
            "text": {"type": "string", "description": "Text to type for action=type."},
            "key": {"type": "string", "description": "Key name for action=key/hotkey, e.g. return, tab, esc, space, arrow-down, c."},
            "modifiers": {
                "type": "array",
                "items": {"type": "string", "enum": ["cmd", "shift", "alt", "ctrl", "fn"]},
                "description": "Modifier keys for action=hotkey.",
            },
            "app": {"type": "string", "description": "Application name for action=open_app, e.g. Safari, Google Chrome, System Settings."},
            "duration_ms": {"type": "integer", "description": "Wait duration or drag pacing in milliseconds."},
        },
        "required": ["action"],
    },
}

DESKTOP_STATUS_SCHEMA = {
    "name": "desktop_status",
    "description": "Inspect macOS desktop automation readiness, active app/window, screen size, mouse position, and required permissions/tools.",
    "parameters": {"type": "object", "properties": {}},
}


def _run(cmd: list[str], timeout: float = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _osascript(script: str, timeout: float = 10) -> subprocess.CompletedProcess[str]:
    return _run(["osascript", "-e", script], timeout=timeout)


def _check_macos() -> bool:
    return sys_platform() == "darwin" and shutil.which("screencapture") is not None


def sys_platform() -> str:
    import sys
    return sys.platform


def _cliclick() -> str | None:
    return shutil.which("cliclick")


def _desktop_status(args: dict[str, Any] | None = None, **kw) -> str:
    cliclick = _cliclick()
    screen_bounds = None
    active_app = None
    status_errors: list[str] = []

    try:
        screen = _osascript('tell application "Finder" to get bounds of window of desktop', timeout=3)
        if screen.returncode == 0:
            screen_bounds = screen.stdout.strip()
        elif screen.stderr.strip():
            status_errors.append(f"screen_bounds: {screen.stderr.strip()}")
    except subprocess.TimeoutExpired:
        status_errors.append("screen_bounds: osascript timed out")

    try:
        active = _osascript('tell application "System Events" to get name of first application process whose frontmost is true', timeout=3)
        if active.returncode == 0:
            active_app = active.stdout.strip()
        elif active.stderr.strip():
            status_errors.append(f"active_app: {active.stderr.strip()}")
    except subprocess.TimeoutExpired:
        status_errors.append("active_app: osascript timed out")

    position = None
    if cliclick:
        try:
            pos = _run([cliclick, "p"], timeout=5)
            if pos.returncode == 0:
                position = pos.stdout.strip()
            elif pos.stderr.strip():
                status_errors.append(f"mouse_position: {pos.stderr.strip()}")
        except subprocess.TimeoutExpired:
            status_errors.append("mouse_position: cliclick timed out")

    return tool_result({
        "success": True,
        "platform": sys_platform(),
        "screencapture": shutil.which("screencapture"),
        "osascript": shutil.which("osascript"),
        "cliclick": cliclick,
        "screen_bounds": screen_bounds,
        "active_app": active_app,
        "mouse_position": position,
        "status_errors": status_errors,
        "permissions_note": "Mouse/keyboard actions require Accessibility permission for the process running Hermes (Terminal/iTerm/Python/launchd gateway). Screenshots may require Screen Recording permission.",
        "ready": bool(shutil.which("screencapture") and shutil.which("osascript") and cliclick),
    })


def _desktop_screenshot(args: dict[str, Any], **kw) -> str:
    if not shutil.which("screencapture"):
        return tool_error("screencapture not found; this tool currently supports macOS.")
    out = args.get("path")
    if out:
        path = Path(os.path.expanduser(out))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path.home() / ".hermes" / "screenshots" / f"desktop_{ts}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["screencapture", "-x"]
    if args.get("display") is not None:
        cmd += ["-D", str(args["display"])]
    cmd.append(str(path))
    res = _run(cmd, timeout=20)
    if res.returncode != 0:
        return tool_error(res.stderr.strip() or res.stdout.strip() or "screenshot failed")
    return tool_result(success=True, path=str(path), media=f"MEDIA:{path}")


def _require_cliclick() -> str | None:
    cc = _cliclick()
    if not cc:
        return None
    return cc


def _coord(args: dict[str, Any], xkey="x", ykey="y") -> str | None:
    x, y = args.get(xkey), args.get(ykey)
    if x is None or y is None:
        return None
    return f"{int(x)},{int(y)}"


def _cliclick_key_command(key: str) -> str:
    """Return the right cliclick command for a key.

    cliclick's `kp:` command only accepts named non-printing keys (return, tab,
    arrows, etc.). Printable one-character keys must be sent with `t:`; this is
    especially important for shortcuts like Cmd-N or Cmd-L.
    """
    key = str(key)
    if len(key) == 1 and key.isprintable():
        return f"t:{key}"
    return f"kp:{key}"


def _desktop_action(args: dict[str, Any], **kw) -> str:
    action = args.get("action")
    cc = _require_cliclick()
    if action in {"move", "click", "double_click", "right_click", "drag", "type", "key", "hotkey", "position"} and not cc:
        return tool_error("cliclick is not installed. Install with: brew install cliclick")

    try:
        if action == "open_app":
            app = args.get("app")
            if not app:
                return tool_error("app is required for open_app")
            res = _run(["open", "-a", app], timeout=20)
            if res.returncode != 0:
                return tool_error(res.stderr.strip() or res.stdout.strip() or f"failed to open {app}")
            return tool_result(success=True, action=action, app=app)

        if action == "wait":
            ms = int(args.get("duration_ms") or 1000)
            time.sleep(max(0, ms) / 1000.0)
            return tool_result(success=True, action=action, duration_ms=ms)

        if action == "position":
            res = _run([cc, "p"], timeout=5)
            if res.returncode != 0:
                return tool_error(res.stderr.strip() or res.stdout.strip() or "position failed")
            return tool_result(success=True, position=res.stdout.strip())

        commands: list[str]
        if action == "move":
            coord = _coord(args)
            if not coord:
                return tool_error("x and y are required for move")
            commands = [f"m:{coord}"]
        elif action == "click":
            coord = _coord(args)
            if not coord:
                return tool_error("x and y are required for click")
            commands = [f"c:{coord}"]
        elif action == "double_click":
            coord = _coord(args)
            if not coord:
                return tool_error("x and y are required for double_click")
            commands = [f"dc:{coord}"]
        elif action == "right_click":
            coord = _coord(args)
            if not coord:
                return tool_error("x and y are required for right_click")
            commands = [f"rc:{coord}"]
        elif action == "drag":
            start = _coord(args)
            end = _coord(args, "to_x", "to_y")
            if not start or not end:
                return tool_error("x, y, to_x, and to_y are required for drag")
            wait = int(args.get("duration_ms") or 250)
            commands = [f"dd:{start}", f"w:{wait}", f"dm:{end}", f"du:{end}"]
        elif action == "type":
            text = args.get("text")
            if text is None:
                return tool_error("text is required for type")
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
                f.write("t:" + shlex.quote(str(text)) + "\n")
                fname = f.name
            try:
                res = _run([cc, "-f", fname], timeout=max(30, min(180, len(str(text)) / 2 + 15)))
            finally:
                try:
                    os.unlink(fname)
                except OSError:
                    pass
            if res.returncode != 0:
                return tool_error(res.stderr.strip() or res.stdout.strip() or "type failed")
            return tool_result(success=True, action=action, chars=len(str(text)))
        elif action == "key":
            key = args.get("key")
            if not key:
                return tool_error("key is required for key")
            commands = [_cliclick_key_command(key)]
        elif action == "hotkey":
            key = args.get("key")
            mods = args.get("modifiers") or []
            if not key or not mods:
                return tool_error("key and modifiers are required for hotkey")
            modstr = ",".join(mods)
            commands = [f"kd:{modstr}", _cliclick_key_command(key), f"ku:{modstr}"]
        else:
            return tool_error(f"unknown desktop action: {action}")

        res = _run([cc, *commands], timeout=30)
        if res.returncode != 0:
            return tool_error(res.stderr.strip() or res.stdout.strip() or f"{action} failed")
        return tool_result(success=True, action=action, commands=commands)
    except subprocess.TimeoutExpired:
        return tool_error(f"desktop action timed out: {action}")
    except Exception as e:
        return tool_error(f"{type(e).__name__}: {e}")


registry.register(
    name="desktop_screenshot",
    toolset="computer",
    schema=SCREENSHOT_SCHEMA,
    handler=lambda args, **kw: _desktop_screenshot(args or {}, **kw),
    check_fn=_check_macos,
    emoji="🖥️",
    max_result_size_chars=20_000,
)

registry.register(
    name="desktop_action",
    toolset="computer",
    schema=DESKTOP_ACTION_SCHEMA,
    handler=lambda args, **kw: _desktop_action(args or {}, **kw),
    check_fn=_check_macos,
    emoji="🖱️",
    max_result_size_chars=20_000,
)

registry.register(
    name="desktop_status",
    toolset="computer",
    schema=DESKTOP_STATUS_SCHEMA,
    handler=lambda args, **kw: _desktop_status(args or {}, **kw),
    check_fn=_check_macos,
    emoji="🧭",
    max_result_size_chars=20_000,
)
