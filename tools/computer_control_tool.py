"""Minimal macOS desktop-control tool for Hermes Agent.

Provides a small, explicit action surface for controlling the local Mac:
- take screenshots
- activate an application
- open a file/folder/URL
- send keystrokes
- inspect the current frontmost app/window
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.registry import registry

_REQUIRED_COMMANDS = ("osascript", "open", "screencapture")
_SPECIAL_KEY_CODES = {
    "return": 36,
    "enter": 76,
    "tab": 48,
    "space": 49,
    "escape": 53,
    "esc": 53,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "delete": 51,
    "forward_delete": 117,
    "home": 115,
    "end": 119,
    "page_up": 116,
    "page_down": 121,
}
_MODIFIER_ALIASES = {
    "cmd": "command",
    "command": "command",
    "ctrl": "control",
    "control": "control",
    "alt": "option",
    "opt": "option",
    "option": "option",
    "shift": "shift",
}
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

COMPUTER_CONTROL_SCHEMA = {
    "name": "computer_control",
    "description": "Control this macOS desktop in a minimal, explicit way: take a screenshot, activate an app, open a file/folder/URL, send keystrokes, or inspect the current frontmost app/window.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["screenshot", "activate_app", "open", "keystroke", "frontmost_app"],
                "description": "Desktop action to perform.",
            },
            "app_name": {
                "type": "string",
                "description": "Application name for action='activate_app' (for example 'Safari' or 'Finder').",
            },
            "target": {
                "type": "string",
                "description": "File path, folder path, or URL for action='open'.",
            },
            "output_path": {
                "type": "string",
                "description": "Optional output path for action='screenshot'. Defaults to a timestamped PNG under the Hermes home directory.",
            },
            "window_id": {
                "type": "integer",
                "description": "Optional macOS window id for action='screenshot'. When provided, capture that specific window instead of the whole display.",
            },
            "text": {
                "type": "string",
                "description": "Literal text to type for action='keystroke'.",
            },
            "key": {
                "type": "string",
                "description": "Named key or single character for action='keystroke' (for example 'return', 'tab', 'escape', 'l').",
            },
            "modifiers": {
                "type": "array",
                "description": "Optional modifier keys for action='keystroke'. Supported: command, control, option, shift.",
                "items": {
                    "type": "string",
                    "enum": ["command", "cmd", "control", "ctrl", "option", "opt", "alt", "shift"],
                },
            },
        },
        "required": ["action"],
    },
}


def _check_computer_control_available() -> bool:
    if sys.platform != "darwin":
        return False
    return all(shutil.which(cmd) for cmd in _REQUIRED_COMMANDS)


def _default_screenshot_path() -> Path:
    root = get_hermes_home() / "computer-control"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"desktop-screenshot-{int(time.time())}.png"


def _escape_applescript_string(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _run_command(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return (proc.stdout or proc.stderr or "").strip()


def _run_osascript(script: str) -> str:
    return _run_command(["osascript", "-e", script])


def _modifier_clause(modifiers: list[str] | None) -> str:
    if not modifiers:
        return ""
    normalized: list[str] = []
    for raw in modifiers:
        name = _MODIFIER_ALIASES.get(str(raw).strip().lower())
        if not name:
            raise ValueError(f"Unsupported modifier: {raw}")
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        return ""
    return " using {" + ", ".join(f"{name} down" for name in normalized) + "}"


def _build_keystroke_script(*, text: str | None = None, key: str | None = None, modifiers: list[str] | None = None) -> str:
    clause = _modifier_clause(modifiers)
    if text:
        escaped = _escape_applescript_string(text)
        return f'tell application "System Events" to keystroke "{escaped}"{clause}'
    if not key:
        raise ValueError("keystroke action requires text or key")
    key_name = str(key).strip().lower()
    if key_name in _SPECIAL_KEY_CODES:
        return f'tell application "System Events" to key code {_SPECIAL_KEY_CODES[key_name]}{clause}'
    if len(str(key)) == 1:
        escaped = _escape_applescript_string(str(key))
        return f'tell application "System Events" to keystroke "{escaped}"{clause}'
    raise ValueError(f"Unknown action key: {key}")


def _frontmost_app_script() -> str:
    return """
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    set appName to name of frontApp
    set winName to ""
    try
        if (count of windows of frontApp) > 0 then
            set winName to name of front window of frontApp
        end if
    end try
    return appName & linefeed & winName
end tell
""".strip()


def _window_helper_source() -> Path:
    return Path(__file__).with_name("mac_window_info.swift")


def _window_helper_binary() -> Path:
    root = get_hermes_home() / "computer-control" / "bin"
    root.mkdir(parents=True, exist_ok=True)
    return root / "mac-window-info"


def _ensure_window_helper_binary() -> Path:
    source = _window_helper_source()
    swiftc = shutil.which("swiftc")
    if not swiftc or not source.exists():
        raise RuntimeError("Window helper source or swiftc is unavailable.")
    binary = _window_helper_binary()
    needs_build = not binary.exists() or source.stat().st_mtime > binary.stat().st_mtime
    if needs_build:
        subprocess.run([swiftc, str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_window_bounds(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    x = _coerce_int(value.get("x"))
    if x is None:
        x = _coerce_int(value.get("X"))
    y = _coerce_int(value.get("y"))
    if y is None:
        y = _coerce_int(value.get("Y"))
    width = _coerce_int(value.get("width"))
    if width is None:
        width = _coerce_int(value.get("Width"))
    height = _coerce_int(value.get("height"))
    if height is None:
        height = _coerce_int(value.get("Height"))
    if width is None or height is None:
        return None
    return {
        "x": x or 0,
        "y": y or 0,
        "width": width,
        "height": height,
    }


def _normalize_helper_window(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    window_id = _coerce_int(value.get("window_id"))
    if window_id is None:
        window_id = _coerce_int(value.get("id"))
    bounds = _normalize_window_bounds(value.get("window_bounds") or value.get("bounds"))
    if window_id is None or not bounds:
        return None
    return {
        "window_title": str(value.get("window_title") or value.get("title") or ""),
        "window_id": window_id,
        "window_bounds": bounds,
    }


def _window_area(window: dict[str, object]) -> int:
    bounds = window.get("window_bounds")
    if not isinstance(bounds, dict):
        return 0
    width = _coerce_int(bounds.get("width")) or 0
    height = _coerce_int(bounds.get("height")) or 0
    return max(width, 0) * max(height, 0)


def _window_looks_fragmentary(window: dict[str, object]) -> bool:
    bounds = window.get("window_bounds")
    if not isinstance(bounds, dict):
        return True
    width = _coerce_int(bounds.get("width")) or 0
    height = _coerce_int(bounds.get("height")) or 0
    area = _window_area(window)
    title = str(window.get("window_title") or "").strip()
    return not title and (width < 120 or height < 80 or area < 10_000)


def _select_frontmost_window(windows: object) -> dict[str, object] | None:
    if not isinstance(windows, list):
        return None
    candidates = [candidate for candidate in (_normalize_helper_window(item) for item in windows) if candidate]
    if not candidates:
        return None
    if not _window_looks_fragmentary(candidates[0]):
        return candidates[0]
    for candidate in candidates[1:]:
        if not _window_looks_fragmentary(candidate):
            return candidate
    titled = [candidate for candidate in candidates if str(candidate.get("window_title") or "").strip()]
    if titled:
        return titled[0]
    return candidates[0]


def _frontmost_window_info() -> dict[str, object]:
    try:
        helper = _ensure_window_helper_binary()
        raw = _run_command([str(helper)])
        data = json.loads(raw)
        if isinstance(data, dict):
            info: dict[str, object] = {
                "app_name": str(data.get("app_name") or ""),
                "bundle_id": str(data.get("bundle_id") or ""),
                "bundle_name": str(data.get("bundle_name") or ""),
                "process_id": _coerce_int(data.get("process_id")),
                "window_title": str(data.get("window_title") or ""),
                "window_id": _coerce_int(data.get("window_id")),
                "window_bounds": _normalize_window_bounds(data.get("window_bounds")),
            }
            selected = _select_frontmost_window(data.get("windows"))
            if selected:
                info.update(selected)
            return info
    except Exception:
        pass

    raw = _run_osascript(_frontmost_app_script())
    parts = raw.splitlines()
    app = parts[0].strip() if parts else ""
    window = parts[1].strip() if len(parts) > 1 else ""
    return {
        "app_name": app,
        "bundle_id": "",
        "bundle_name": "",
        "process_id": None,
        "window_title": window,
        "window_id": None,
        "window_bounds": None,
    }


def _screencapture_command(path: Path, *, window_id: int | None = None) -> list[str]:
    cmd = ["screencapture", "-x"]
    if window_id is not None:
        cmd.extend(["-o", f"-l{int(window_id)}"])
    cmd.append(str(path))
    return cmd


def _humanize_os_error(message: str) -> str:
    text = (message or "").strip()
    low = text.lower()
    if "-1743" in text or "not authorized" in low or "automation" in low:
        return "macOS blocked Automation/System Events access. Grant Hermes/Terminal the needed Automation or Accessibility permission in System Settings."
    if "could not create image from display" in low:
        return "macOS could not capture the display. This usually means Screen Recording permission is missing, the display is unavailable, or the current runtime cannot access the GUI session."
    if "could not create image from window" in low:
        return "macOS could not capture that specific window. The window may be unavailable, private, or unsupported for window-level capture from the current session."
    if "not permitted" in low or "screen recording" in low:
        return "macOS blocked screen capture. Grant Hermes/Terminal Screen Recording permission in System Settings."
    return text or "Desktop control command failed."


def computer_control(*, action: str, app_name: str | None = None, target: str | None = None,
                     output_path: str | None = None, window_id: int | None = None, text: str | None = None,
                     key: str | None = None, modifiers: list[str] | None = None) -> str:
    action_name = str(action or "").strip().lower()
    screenshot_path: Path | None = None
    try:
        if action_name == "screenshot":
            path = Path(output_path).expanduser() if output_path else _default_screenshot_path()
            screenshot_path = path
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized_window_id = int(window_id) if window_id is not None else None
            _run_command(_screencapture_command(path, window_id=normalized_window_id))
            payload = {"success": True, "action": action_name, "path": str(path)}
            if normalized_window_id is not None:
                payload["window_id"] = normalized_window_id
            if " " not in str(path):
                payload["media_tag"] = f"MEDIA:{path}"
            return json.dumps(payload, ensure_ascii=False)

        if action_name == "activate_app":
            if not app_name:
                raise ValueError("activate_app action requires app_name")
            _run_osascript(f'tell application "{_escape_applescript_string(app_name)}" to activate')
            return json.dumps({"success": True, "action": action_name, "app_name": app_name}, ensure_ascii=False)

        if action_name == "open":
            if not target:
                raise ValueError("open action requires target")
            resolved = str(Path(target).expanduser()) if not _URL_RE.match(str(target)) else str(target)
            _run_command(["open", resolved])
            return json.dumps({"success": True, "action": action_name, "target": resolved}, ensure_ascii=False)

        if action_name == "keystroke":
            script = _build_keystroke_script(text=text, key=key, modifiers=modifiers)
            _run_osascript(script)
            return json.dumps({
                "success": True,
                "action": action_name,
                "text": text,
                "key": key,
                "modifiers": modifiers or [],
            }, ensure_ascii=False)

        if action_name == "frontmost_app":
            info = _frontmost_window_info()
            return json.dumps({
                "success": True,
                "action": action_name,
                "app_name": str(info.get("app_name") or ""),
                "bundle_id": str(info.get("bundle_id") or ""),
                "bundle_name": str(info.get("bundle_name") or ""),
                "process_id": info.get("process_id"),
                "window_title": str(info.get("window_title") or ""),
                "window_id": info.get("window_id"),
                "window_bounds": info.get("window_bounds"),
            }, ensure_ascii=False)

        raise ValueError(f"Unknown action: {action}")
    except ValueError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except subprocess.CalledProcessError as e:
        if screenshot_path and screenshot_path.exists() and screenshot_path.stat().st_size == 0:
            try:
                screenshot_path.unlink()
            except OSError:
                pass
        detail = _humanize_os_error((e.stderr or e.stdout or str(e)))
        return json.dumps({"error": detail}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Desktop control failed: {type(e).__name__}: {e}"}, ensure_ascii=False)


def _handle_computer_control(args, **_kw):
    return computer_control(
        action=args.get("action", ""),
        app_name=args.get("app_name"),
        target=args.get("target"),
        output_path=args.get("output_path"),
        window_id=args.get("window_id"),
        text=args.get("text"),
        key=args.get("key"),
        modifiers=args.get("modifiers"),
    )


registry.register(
    name="computer_control",
    toolset="computer",
    schema=COMPUTER_CONTROL_SCHEMA,
    handler=_handle_computer_control,
    check_fn=_check_computer_control_available,
    emoji="🖥️",
    max_result_size_chars=100_000,
)
